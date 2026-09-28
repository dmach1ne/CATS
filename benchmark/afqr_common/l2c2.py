from __future__ import annotations

import torch as th
from torch.distributions import Normal


def half_squared_l2_distance(
    values: th.Tensor,
    neighbor_values: th.Tensor,
) -> th.Tensor:
    """Return the per-sample value distance from L2C2 Eq. (17)."""
    difference = values - neighbor_values
    if difference.ndim == 1:
        difference = difference.unsqueeze(-1)
    return 0.5 * difference.flatten(start_dim=1).pow(2).sum(dim=1)


def distance_dependent_gains(
    base_policy_gain: th.Tensor,
    distance: th.Tensor,
    value_scale: float,
) -> tuple[th.Tensor, th.Tensor]:
    """Normalize both L2C2 gains once by the local state distance."""
    policy_gain = base_policy_gain / distance
    value_gain = value_scale * policy_gain
    return policy_gain, value_gain


def monte_carlo_squared_hellinger(
    policy: Normal,
    neighbor_policy: Normal,
    num_samples: int,
) -> th.Tensor:
    """Estimate squared Hellinger distance using common samples from policy."""
    samples = policy.rsample((num_samples,))
    log_policy = policy.log_prob(samples).sum(dim=-1)
    log_neighbor = neighbor_policy.log_prob(samples).sum(dim=-1)
    log_density_ratio = 0.5 * (log_neighbor - log_policy)
    affinity = th.exp(log_density_ratio.clamp(min=-20.0, max=20.0)).mean(dim=0)
    return (1.0 - affinity).clamp(min=0.0)
