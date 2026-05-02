import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import CONSTRAINTS, PARAM_INITIAL, RANGE_FACTOR, PARAM_NAMES


def set_bounds(value: float, factor: float = RANGE_FACTOR) -> list:
    return [value * (1 - factor), value * (1 + factor)]


def init_OTA_two(logger=None):
    """
    Initializes parameter ranges and constraint thresholds for the OTA Two-Stage Op-Amp.
    Reads values from config.py instead of hardcoding.
    """
    param_ranges = [set_bounds(PARAM_INITIAL[name]) for name in PARAM_NAMES]

    thresholds = {
        'gain': CONSTRAINTS['gain'],
        'i_multiplier': CONSTRAINTS['current_multiplier'],
        'i': CONSTRAINTS['current_limit'],
        'phase': CONSTRAINTS['phase'],
        'gbw': CONSTRAINTS['gbw'],
    }

    if logger:
        logger.info("Loaded FocalOpt (OTA Two) parameter ranges and constraints from config.py.")

    return param_ranges, thresholds
