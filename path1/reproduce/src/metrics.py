"""
Utilities to compute performance bound and consolidate metrics.
"""

from __future__ import annotations

import numpy as np


def performance_bound(cost_drl: float, cost_opt: float) -> float:
    """
    Performance bound as defined in the paper:
    (C_drl - C_opt) / C_opt, lower is better (0 means optimal).
    """
    if cost_opt == 0:
        return np.nan
    return (cost_drl - cost_opt) / cost_opt


__all__ = ["performance_bound"]

