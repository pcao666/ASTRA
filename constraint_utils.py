"""
Unified constraint checking and Figure of Merit (FoM) calculation.

Single source of truth for all constraint-related logic, reading from config.py.
"""

import torch
from config import CONSTRAINTS


def is_sim_failure(test_y: torch.Tensor) -> bool:
    """Check if simulation output indicates a failure (NaN, Inf, or PM < 2)."""
    return bool(torch.isnan(test_y).any() or torch.isinf(test_y).any() or test_y[:, 2] < 2)


def check_feasibility(test_y: torch.Tensor, constraints: dict = None) -> bool:
    """Check if a simulation result satisfies all design constraints.

    Args:
        test_y: Tensor of shape (1, 4) with [Gain(dB), DC_Current(A), PM(deg), GBW(Hz)] in real scale.
        constraints: Constraint dict (defaults to config.CONSTRAINTS).

    Returns:
        True if all constraints are satisfied.
    """
    c = constraints or CONSTRAINTS
    return bool(
        test_y[:, 0] > c['gain'] and
        test_y[:, 1] * c['current_multiplier'] < c['current_limit'] and
        test_y[:, 2] > c['phase'] and
        test_y[:, 3] > c['gbw']
    )


def check_individual_metrics(test_y: torch.Tensor, constraints: dict = None) -> dict:
    """Check each metric individually against its constraint.

    Args:
        test_y: Tensor of shape (1, 4) — [Gain(dB), DC_Current(A), PM(deg), GBW(Hz)].

    Returns:
        Dict with boolean pass/fail for each metric.
    """
    c = constraints or CONSTRAINTS
    return {
        'gain': bool(test_y[0][0].item() > c['gain']),
        'current': bool(test_y[0][1].item() * c['current_multiplier'] < c['current_limit']),
        'phase': bool(test_y[0][2].item() > c['phase']),
        'gbw': bool(test_y[0][3].item() > c['gbw']),
    }


def calculate_fom(test_y_row: torch.Tensor, constraints: dict = None) -> float:
    """Calculate FoM for GP training in constrained optimization mode.

    Feasible points get FoM = 3 + (target_current / actual_current) * 50
    Infeasible points get FoM = 0

    Args:
        test_y_row: 1-D tensor [Gain(dB), DC_Current(A), PM(deg), GBW(Hz)] in real scale.
        constraints: Constraint dict (defaults to config.CONSTRAINTS).

    Returns:
        FoM value (float).
    """
    c = constraints or CONSTRAINTS
    current = test_y_row[1].item()

    feasible = (
        test_y_row[0].item() > c['gain'] and
        current * c['current_multiplier'] < c['current_limit'] and
        test_y_row[2].item() > c['phase'] and
        test_y_row[3].item() > c['gbw']
    )

    if feasible:
        return 3.0 + (c['current_limit'] / current) * 50.0
    return 0.0
