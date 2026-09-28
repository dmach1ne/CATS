from __future__ import annotations

from rl_games.algos_torch import a2c_continuous

from .configured_adaptive_scheduler import ConfiguredAdaptiveSchedulerMixin
from .finite_gradient_guard import FiniteGradientGuardMixin


class FiniteGradientA2CAgent(
    ConfiguredAdaptiveSchedulerMixin,
    FiniteGradientGuardMixin,
    a2c_continuous.A2CAgent,
):
    """Stock continuous PPO with non-finite gradient updates skipped."""
