"""Coefficient schedules shared by CATS variants."""


def _clamp_completed_fraction(completed_fraction: float) -> float:
    return min(1.0, max(0.0, float(completed_fraction)))


def linear_ramp_up_coefficient(peak_coefficient: float, completed_fraction: float) -> float:
    """Increase linearly from zero to the peak coefficient."""
    return float(peak_coefficient) * _clamp_completed_fraction(completed_fraction)


def linear_ramp_down_coefficient(peak_coefficient: float, completed_fraction: float) -> float:
    """Decrease linearly from the peak coefficient to zero."""
    return float(peak_coefficient) * (1.0 - _clamp_completed_fraction(completed_fraction))
