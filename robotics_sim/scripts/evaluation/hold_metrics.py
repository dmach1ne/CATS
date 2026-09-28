"""Pure distance-based hold metrics shared by Reach and Lift evaluators."""

from __future__ import annotations

import numpy as np


def max_contiguous_duration(distances, threshold_m: float, control_dt: float) -> float:
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    if threshold_m <= 0.0 or control_dt <= 0.0:
        raise ValueError("threshold_m and control_dt must be positive")
    below = np.isfinite(values) & (values < threshold_m)
    best = current = 0
    for is_below in below:
        current = current + 1 if is_below else 0
        best = max(best, current)
    return float(best * control_dt)


def hold_success(
    distances, threshold_m: float, required_duration_s: float, control_dt: float
) -> bool:
    if required_duration_s <= 0.0:
        raise ValueError("required_duration_s must be positive")
    return max_contiguous_duration(distances, threshold_m, control_dt) + 1.0e-9 >= required_duration_s
