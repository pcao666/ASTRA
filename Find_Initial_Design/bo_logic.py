# -*- coding: utf-8 -*-
from botorch.optim import optimize_acqf

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from simulation_OTA_two import OTA_two_simulation_gmid_pro
from lut_utils import (
    calculate_w_linear_NMOS_pro,
    calculate_w_linear_PMOS_pro,
)
from constraint_utils import check_feasibility
from config import STAGE1_PARAM_RANGES, BIAS, BO_HYPERPARAMS
import numpy as np
import random
import time
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition import ScalarizedObjective, UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
import copy
import logging


def normalize(tensor, bounds):
    """Normalizes the tensor to the [0, 1] range."""
    lower, upper = bounds.unbind(0)
    tensor[0] = (tensor[0] - lower) / (upper[0] - lower)
    return tensor


def unnormalize(tensor, bounds):
    """Unnormalizes the tensor from the [0, 1] range back to the original scale."""
    lower, upper = bounds.unbind(0)
    tensor[0] = tensor[0] * (upper - lower) + lower
    return tensor



class BayesianOptimization():
    def __init__(self, n):
        self.n = n  # n is the number of optimization iterations

    def find(self, gmid1, gmid2, gmid3, gmid4, gmid5, file_path_x, file_path_y, logger):

        # --- Setup ---
        logger.info("--- Bayesian Optimization (Stage 1) find method started ---")
        # 9 parameters for GP training (Cap, k1, k2, L1-L5, R) — from config
        param_ranges_9dim = STAGE1_PARAM_RANGES

        # 9-dim bounds (for optimize_acqf) — derived from config ranges
        lower_bounds = [r[0] for r in STAGE1_PARAM_RANGES]
        upper_bounds = [r[1] for r in STAGE1_PARAM_RANGES]
        bound = torch.log(torch.tensor([lower_bounds, upper_bounds], dtype=torch.float64))

        # In-memory data for GP training (log X, raw Y)
        dbx_alter = torch.empty(0, 9, dtype=torch.float64)  # 9 log space inputs
        dby_alter = torch.empty(0, 4, dtype=torch.float64)  # 4 raw outputs

        # --- Streaming Output: Write CSV Headers (12 Output Parameters) ---
        # Output parameters: Cap, L1-L5, R, W1-W5
        headers_x_out = ['iter_times', 'cap', 'L1', 'L2', 'L3', 'L4', 'L5', 'r', 'W1', 'W2', 'W3', 'W4', 'W5']
        headers_y = ['iter_times', 'gain', 'dc_current', 'phase', 'GBW']

        try:
            # Write 12-parameter X header
            with open(file_path_x, 'w', encoding='utf-8') as f:
                f.write(','.join(headers_x_out) + '\n')
            # Write Y header
            with open(file_path_y, 'w', encoding='utf-8') as f:
                f.write(','.join(headers_y) + '\n')
            logger.info(f"CSV headers written: {file_path_x} (12-param) and {file_path_y}")
        except Exception as e:
            logger.error(f"Error: Failed to write CSV headers: {e}")
            return torch.empty(0), torch.empty(0)

        # --- Initial Sampling Loop (10 iterations) ---
        initial_samples_collected = 0
        initial_attempts = 0
        max_initial_attempts = 50
        total_iter_count = 0

        logger.info("Starting initial sampling (Find Feasible Initial Design)...")

        while initial_samples_collected < 10 and initial_attempts < max_initial_attempts:
            initial_attempts += 1

            # Generate random parameters in log space
            random_params = [np.random.uniform(np.log(low), np.log(high)) for low, high in param_ranges_9dim]

            step_x = np.empty((1, 9))
            step_x[0] = random_params
            dbx_alter_re_9dim = torch.exp(torch.tensor(step_x))  # X in real scale (9 dimensions)

            # Unpack 9-dimensional parameters
            cap, k1, k2, l1, l2, l3, l4, l5, r = dbx_alter_re_9dim[0].tolist()

            try:
                # Simulation (uses 9-dimensional input; W is calculated internally by LUT)
                # Pass user-provided gmid values
                dby_alter_OTA = OTA_two_simulation_gmid_pro(dbx_alter_re_9dim, gmid1, gmid2, gmid3, gmid4, gmid5)

                if torch.isnan(dby_alter_OTA).any() or torch.isinf(dby_alter_OTA).any() or (dby_alter_OTA == 0).all():
                    logger.warning(f"  - Simulation returned invalid values (NaN/Inf/Zero), skipping this point.")
                    continue

            except Exception as e:
                logger.warning(f"  - Initial sampling simulation failed: {e}. Skipping this point.")
                continue

            total_iter_count += 1

            # --- Core Fix: Calculate W and write 12-dimensional data ---
            try:
                # 1. Calculate Ws (using user-provided gmid values)
                I_ref = BIAS["I_ref"]
                I_s2 = I_ref * BIAS["stage2_factor"]
                w1 = calculate_w_linear_NMOS_pro(l1, I_ref * k1, gmid1, logger)  # M1 (NMOS)
                w2 = calculate_w_linear_PMOS_pro(l2, I_s2 * k2, gmid2, logger)  # M3 (PMOS)
                w3 = calculate_w_linear_PMOS_pro(l3, I_ref * k1, gmid3, logger)  # M5 (PMOS)
                w4 = calculate_w_linear_NMOS_pro(l4, I_s2 * k1, gmid4, logger)  # M7 (NMOS)
                w5 = calculate_w_linear_NMOS_pro(l5, I_s2 * k2, gmid5, logger)  # M9 (NMOS)

                # 2. Prepare 12-dimensional X data row (Cap, L1-L5, R, W1-W5)
                x_data_list_12dim = (
                        [total_iter_count] +
                        [cap, l1, l2, l3, l4, l5, r] +  # 7 physical parameters (Cap, L1-L5, R)
                        [w1, w2, w3, w4, w5]  # 5 widths (W1-W5)
                )
                x_data_str = ','.join(map(str, x_data_list_12dim))
                with open(file_path_x, 'a', encoding='utf-8') as f:
                    f.write(x_data_str + '\n')

                # 3. Prepare Y data row (transformed)
                dby_alter_re = copy.deepcopy(dby_alter_OTA)
                dby_alter_re[0][0] *= 20
                dby_alter_re[0][1] = torch.exp(dby_alter_re[0][1])
                dby_alter_re[0][2] = torch.exp(dby_alter_re[0][2])
                dby_alter_re[0][3] = torch.exp(dby_alter_re[0][3])

                y_data_list = [total_iter_count] + dby_alter_re[0].tolist()
                y_data_str = ','.join(map(str, y_data_list))
                with open(file_path_y, 'a', encoding='utf-8') as f:
                    f.write(y_data_str + '\n')
                logger.info(f"  - Initial sample point {initial_samples_collected + 1}/10 successfully written.")

            except Exception as e:
                logger.error(f"  - Error: Failed to write initial sample data to CSV (W calculation or write): {e}")
                continue

            # Store data for GP training (log X - 9 dims, raw Y - 4 dims)
            new_X_log_9dim = torch.log(dbx_alter_re_9dim)
            dbx_alter = torch.cat([dbx_alter, new_X_log_9dim.to(dtype=torch.float64)])
            dby_alter = torch.cat([dby_alter, dby_alter_OTA.to(dtype=torch.float64)])

            initial_samples_collected += 1

        if initial_samples_collected < 1:
            logger.error("Error: Initial sampling failed, unable to collect any valid data points. Task terminated.")
            return torch.empty(0), torch.empty(0)

        logger.info(f"\nInitial sampling complete, {initial_samples_collected} valid points obtained.\n")

        # --- Main Optimization Loop (self.n iterations) ---
        flag = 0
        while flag < self.n:
            current_opt_iter = flag + 1

            logger.info(f"--- Bayesian Optimization Iteration {current_opt_iter}/{self.n} started ---")

            try:
                gp = SingleTaskGP(dbx_alter, dby_alter)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                logger.info("  - Training Gaussian Process (GP)...")
                fit_gpytorch_mll(mll)
                logger.info("  - GP training complete.")
            except Exception as e:
                logger.warning(f"  - GP training failed (fit_gpytorch_mll): {e}. Skipping this iteration.")
                flag += 1
                continue

            objective = ScalarizedObjective(
                weights=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
            posterior_transform = Standardize(m=dby_alter.shape[1])
            ucb = UpperConfidenceBound(gp, beta=BO_HYPERPARAMS["stage1"]["beta"], objective=objective, posterior_transform=posterior_transform)

            logger.info("  - Optimizing Acquisition Function (UCB)...")
            try:
                new_X_log_q_9dim, _ = optimize_acqf(  # new_X_log_q_9dim is log scale (9 dimensions)
                    acq_function=ucb,
                    bounds=bound,
                    q=1,
                    num_restarts=BO_HYPERPARAMS["stage1"]["num_restarts"],
                    raw_samples=BO_HYPERPARAMS["stage1"]["raw_samples"],
                )
                logger.info("  - Acquisition function optimization complete.")
            except Exception as e:
                logger.warning(f"  - Acquisition function (optimize_acqf) failed: {e}. Skipping this iteration.")
                flag += 1
                continue

            new_X_real_9dim = torch.exp(new_X_log_q_9dim)  # Convert back to real scale (9 dimensions)
            logger.info(f"  - Suggested new input point (9D real scale): {new_X_real_9dim}")

            # Unpack 9-dimensional parameters
            cap, k1, k2, l1, l2, l3, l4, l5, r = new_X_real_9dim[0].tolist()

            try:
                # Simulation (uses 9-dimensional input)
                new_Y_raw = OTA_two_simulation_gmid_pro(new_X_real_9dim, gmid1, gmid2, gmid3, gmid4, gmid5)

                if torch.isnan(new_Y_raw).any() or torch.isinf(new_Y_raw).any() or (new_Y_raw == 0).all():
                    logger.warning(f"  - Simulation returned invalid values (NaN/Inf/Zero), skipping this point.")
                    flag += 1
                    continue
                logger.info(f"  - Simulation successful (raw Y): {new_Y_raw}")

            except Exception as e:
                logger.warning(f"  - Simulation failed: {e}. Skipping this iteration.")
                flag += 1
                continue

            total_iter_count += 1

            # --- Streaming Output: Write Optimization Data (12-dimensional X) ---
            try:
                # 1. Calculate Ws (using user-provided gmid values)
                I_ref = BIAS["I_ref"]
                I_s2 = I_ref * BIAS["stage2_factor"]
                w1 = calculate_w_linear_NMOS_pro(l1, I_ref * k1, gmid1, logger)  # M1 (NMOS)
                w2 = calculate_w_linear_PMOS_pro(l2, I_s2 * k2, gmid2, logger)  # M3 (PMOS)
                w3 = calculate_w_linear_PMOS_pro(l3, I_ref * k1, gmid3, logger)  # M5 (PMOS)
                w4 = calculate_w_linear_NMOS_pro(l4, I_s2 * k1, gmid4, logger)  # M7 (NMOS)
                w5 = calculate_w_linear_NMOS_pro(l5, I_s2 * k2, gmid5, logger)  # M9 (NMOS)

                # 2. Prepare 12-dimensional X data row
                x_data_list_12dim = (
                        [total_iter_count] +
                        [cap, l1, l2, l3, l4, l5, r] +  # 7 physical parameters (Cap, L1-L5, R)
                        [w1, w2, w3, w4, w5]  # 5 widths (W1-W5)
                )
                x_data_str = ','.join(map(str, x_data_list_12dim))
                with open(file_path_x, 'a', encoding='utf-8') as f:
                    f.write(x_data_str + '\n')

                # 3. Prepare Y data row (transformed)
                test_Y = torch.empty(1, 4)
                test_Y[0][0] = torch.tensor(new_Y_raw[0][0] * 20)
                test_Y[0][1] = torch.exp(new_Y_raw[0][1])
                test_Y[0][2] = torch.exp(new_Y_raw[0][2])
                test_Y[0][3] = torch.exp(new_Y_raw[0][3])

                y_data_list = [total_iter_count] + test_Y[0].tolist()
                y_data_str = ','.join(map(str, y_data_list))
                with open(file_path_y, 'a', encoding='utf-8') as f:
                    f.write(y_data_str + '\n')

                logger.info(f"  - Transformed performance (Y): {test_Y[0]}")

            except Exception as e:
                logger.error(f"  - Error: Failed to write optimization data to CSV: {e}")

            # Check constraints (using unified check from constraint_utils)
            if check_feasibility(test_Y):
                logger.info("******************************************")
                logger.info("*** Feasible solution found! Task terminated early. ***")
                logger.info("******************************************")
                # Return log scale X (9 dims) and real scale Y (4 dims)
                return new_X_log_q_9dim[0], test_Y[0]

            # Update GP training data (log X - 9 dims, raw Y - 4 dims)
            dbx_alter = torch.cat([dbx_alter, new_X_log_q_9dim.to(dtype=torch.float64)])
            dby_alter = torch.cat([dby_alter, new_Y_raw.to(dtype=torch.float64)])
            flag += 1

            if flag == self.n:
                logger.info("-----------------------------------------------------------------")
                logger.info("--- Maximum iterations reached, no fully constrained point found. ---")
                logger.info("-----------------------------------------------------------------")

        # Loop finished, return the result of the last iteration
        if 'test_Y' in locals() and 'new_X_log_q_9dim' in locals():
            return new_X_log_q_9dim[0], test_Y[0]
        else:
            return torch.empty(0), torch.empty(0)

    def print_results(self, dbx, dby, logger):
        # Check if dbx, dby are empty
        if dbx.nelement() == 0 or dby.nelement() == 0:
            logger.warning("Unable to print results: No valid data found.")
            return

        dbx = torch.exp(dbx)
        dbx_np = dbx.numpy()
        dby_np = dby.numpy()
        logger.info("--- Final Results (print_results) ---")
        logger.info(f"Parameters (9D real scale): {dbx_np}")
        logger.info(f"Performance (4D real scale): {dby_np}")
