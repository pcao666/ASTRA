"""
Shared LUT (Look-Up Table) utilities for gm/ID based transistor width calculation.

Provides a single implementation of:
  - find_closest_points_indices  — find bracketing indices in L array
  - calculate_zero               — linear interpolation for ID/W
  - calculate_w_linear_NMOS_pro  — NMOS W from L, I, gm/ID via LUT
  - calculate_w_linear_PMOS_pro  — PMOS W from L, I, gm/ID via LUT
"""

import os
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from config import PROCESS, PATHS

_GMID_LUT_DIR = PATHS["gmid_lut_dir"]
_MIN_W = PROCESS["min_w"]


def find_closest_points_indices(lst, aim_L):
    """Finds the indices of points in lst closest to aim_L for linear interpolation.

    Returns (idx_below, idx_above) where lst[idx_below] <= aim_L <= lst[idx_above].
    """
    arr = np.array(lst)

    below_indices = np.where(arr <= aim_L)[0]
    above_indices = np.where(arr >= aim_L)[0]

    if len(below_indices) > 0:
        idx_below = below_indices[-1]
    else:
        idx_below = above_indices[0] if len(above_indices) > 0 else 0

    if len(above_indices) > 0:
        idx_above = above_indices[0]
    else:
        idx_above = below_indices[-1] if len(below_indices) > 0 else 0

    if idx_below == idx_above and idx_above < len(arr) - 1:
        idx_above += 1
    elif idx_below == idx_above and idx_below > 0:
        idx_below -= 1

    return int(idx_below), int(idx_above)


def calculate_zero(L_below, L_above, idoverw_below, idoverw_above, aim_L):
    """Calculates ID/W at aim_L via linear interpolation between two bracketing points."""
    if L_below == L_above:
        return idoverw_below

    interp_func = interp1d([L_below, L_above], [idoverw_below, idoverw_above], fill_value="extrapolate")
    return interp_func(aim_L).item()


def calculate_w_linear_NMOS_pro(aim_L, aim_I, gmid, logger=None):
    """Calculates NMOS width from L, drain current, and gm/ID using LUT lookup.

    Args:
        aim_L: Target channel length (m).
        aim_I: Target drain current (A).
        gmid: gm/ID ratio (integer).
        logger: Optional logger for error reporting.

    Returns:
        Calculated width (m), clamped to process minimum.
    """
    lut_file_path = os.path.join(_GMID_LUT_DIR, f'nmos_gmid{int(gmid)}.csv')
    try:
        df = pd.read_csv(lut_file_path)

        L_values = df[f'L (GM/ID=ID/W (GM/ID={int(gmid)}))'].values
        idoverw_values = df['ID/W'].values

        idx_below, idx_above = find_closest_points_indices(L_values, aim_L)
        L_below, L_above = L_values[idx_below], L_values[idx_above]
        idoverw_below, idoverw_above = idoverw_values[idx_below], idoverw_values[idx_above]

        result_idoverw = calculate_zero(L_below, L_above, idoverw_below, idoverw_above, aim_L)
        result_w = aim_I / result_idoverw

        return max(result_w, _MIN_W)

    except Exception as e:
        if logger:
            logger.error(f"W_NMOS ({gmid}) LUT lookup failed (L={aim_L:.2e}, I={aim_I:.2e}). "
                         f"File: {lut_file_path}. Error: {e}")
        return _MIN_W


def calculate_w_linear_PMOS_pro(aim_L, aim_I, gmid, logger=None):
    """Calculates PMOS width from L, drain current, and gm/ID using LUT lookup.

    Args:
        aim_L: Target channel length (m).
        aim_I: Target drain current (A).
        gmid: gm/ID ratio (integer).
        logger: Optional logger for error reporting.

    Returns:
        Calculated width (m), clamped to process minimum.
    """
    lut_file_path = os.path.join(_GMID_LUT_DIR, f'pmos_gmid{int(gmid)}.csv')
    try:
        df = pd.read_csv(lut_file_path)

        L_values = df[f'L (GM/ID=ID/W (GM/ID={int(gmid)}))'].values
        idoverw_values = df['ID/W'].values

        idx_below, idx_above = find_closest_points_indices(L_values, aim_L)
        L_below, L_above = L_values[idx_below], L_values[idx_above]
        idoverw_below, idoverw_above = idoverw_values[idx_below], idoverw_values[idx_above]

        result_idoverw = calculate_zero(L_below, L_above, idoverw_below, idoverw_above, aim_L)
        result_w = aim_I / result_idoverw

        return max(result_w, _MIN_W)

    except Exception as e:
        if logger:
            logger.error(f"W_PMOS ({gmid}) LUT lookup failed (L={aim_L:.2e}, I={aim_I:.2e}). "
                         f"File: {lut_file_path}. Error: {e}")
        return _MIN_W
