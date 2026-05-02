"""
ASTRA Unified Configuration
All circuit-specific and optimization-specific parameters in one place.
"""

import os

# ---------------------------------------------------------------------------
# Project Paths (resolved relative to this file's location = project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "gmid_lut_dir": os.path.join(_PROJECT_ROOT, "gmid_LUT"),
    "netlist_dir": os.path.join(_PROJECT_ROOT, "examples", "netlists"),
    "store_dir": os.path.join(_PROJECT_ROOT, "store"),
    "database_dir": os.path.join(_PROJECT_ROOT, "database"),
}

# ---------------------------------------------------------------------------
# Process / Technology Node
# ---------------------------------------------------------------------------
PROCESS = {
    "min_w": 180e-9,          # Minimum transistor width (m)
    "min_l": 180e-9,          # Minimum transistor length (m)
    "vdd": 3.3,               # Supply voltage (V)
    "dc_current_offset": 8e-5, # Offset subtracted from simulated branch current
}

# ---------------------------------------------------------------------------
# Design Constraints (single source of truth)
# ---------------------------------------------------------------------------
CONSTRAINTS = {
    "gain": 60,               # Gain > 60 dB
    "current_limit": 1e-3,    # DC Current upper bound (A)
    "current_multiplier": 1.8, # effective limit = current * multiplier < current_limit
    "phase": 60,              # Phase Margin > 60 deg
    "gbw": 4e6,               # GBW > 4 MHz
}

# ---------------------------------------------------------------------------
# Transistor Parameter Names (12-dim for Two-Stage OTA)
# ---------------------------------------------------------------------------
PARAM_NAMES = ["cap", "L1", "L2", "L3", "L4", "L5", "r", "W1", "W2", "W3", "W4", "W5"]
INPUT_DIM = len(PARAM_NAMES)  # 12

# ---------------------------------------------------------------------------
# Initial Parameter Values & Ranges (center ±50%)
# ---------------------------------------------------------------------------
PARAM_INITIAL = {
    "cap": 4.66e-11,
    "L1": 1.52e-6,
    "L2": 4.32e-7,
    "L3": 1.33e-6,
    "L4": 1e-06,
    "L5": 1e-06,
    "r": 9056,
    "W1": 1.9608e-5,
    "W2": 1.944e-5,
    "W3": 8.5785e-5,
    "W4": 2.58e-5,
    "W5": 9e-6,
}

RANGE_FACTOR = 0.5  # ±50% around initial value

# ---------------------------------------------------------------------------
# Circuit Topology — Bias Current Definitions
# ---------------------------------------------------------------------------
# Used by W calculation: I_drain = I_ref * k_ratio
# M1,M2 (NMOS diff pair): I = I_ref * k1
# M3,M4 (PMOS load):      I = I_ref * k2 * stage2_factor
# M5,M6 (PMOS tail):      I = I_ref * k1
# M7,M8 (NMOS stage2):    I = I_ref * k1 * stage2_factor
# M9   (NMOS bias):        I = I_ref * k2 * stage2_factor
BIAS = {
    "I_ref": 20e-6,           # Reference tail current (A)
    "stage2_factor": 2,       # I_stage2 = I_ref * stage2_factor (= 40e-6)
}

# ---------------------------------------------------------------------------
# Stage 1 (FastInitial) — 9-Dim Search Space
# ---------------------------------------------------------------------------
STAGE1_PARAM_RANGES = [
    (0.5e-12, 4e-11),   # cap (0)
    (0.3, 8),            # k1  (1)
    (0.3, 8),            # k2  (2)
    (1.8e-7, 5e-6),     # l1  (3)
    (1.8e-7, 5e-6),     # l2  (4)
    (1.8e-7, 5e-6),     # l3  (5)
    (1.8e-7, 5e-6),     # l4  (6)
    (1.8e-7, 5e-6),     # l5  (7)
    (100, 10000),        # r   (8)
]

STAGE1_GMID_RANGE = (2, 25)  # Valid gm/ID values for LUT lookup

# ---------------------------------------------------------------------------
# Bayesian Optimization Hyperparameters
# ---------------------------------------------------------------------------
BO_HYPERPARAMS = {
    "stage1": {
        "beta": 0.1,
        "num_restarts": 1,
        "raw_samples": 20,
    },
    "stage2": {
        "beta": 2.0,
        "num_restarts": 10,
        "raw_samples": 100,
    },
}

# ---------------------------------------------------------------------------
# FocalOpt (Stage 2) — Multi-Stage Configuration
# ---------------------------------------------------------------------------
FOCALOPT = {
    "init_num": 20,             # Initial random samples for full-unbinding BO
    "stage_init_num": 20,       # Initial samples per sub-stage
    "stage1_iters": 50,         # Full 12-dim unbinding
    "stage2_iters": 200,        # Top-4 parameters
    "stage3_iters": 200,        # Top-8 parameters
    "stage4_iters": 250,        # All 12 parameters
    "w_llm_init": 0.5,          # Initial LLM weight
    "w_mi_init": 0.5,           # Initial MI weight
    "group_size": 4,            # Parameters per group (12 / 4 = 3 groups)
    "early_stop_count": 20,     # Consecutive no-improvements before stage termination
}

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
SEED = 5
