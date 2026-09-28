from __future__ import annotations

import torch

try:
    from .cats_layout import resolve_action_layout
except ImportError:  # Supports the repository's standalone contract-test loader.
    from cats_layout import resolve_action_layout


def linear_ramp(completed_steps: int, total_steps: int, alpha_max: float) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    progress = min(max(float(completed_steps) / float(total_steps), 0.0), 1.0)
    return float(alpha_max) * progress


def temporal_action_loss(
    current_mu: torch.Tensor,
    next_mu: torch.Tensor,
    valid_mask: torch.Tensor,
    arm_dims: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    if current_mu.shape != next_mu.shape:
        raise ValueError(f"Policy mean shapes differ: {current_mu.shape} and {next_mu.shape}")
    if current_mu.ndim != 2 or current_mu.shape[1] < arm_dims:
        raise ValueError(f"Expected at least {arm_dims} action dimensions, got {current_mu.shape}")

    valid_mask = valid_mask.to(device=current_mu.device, dtype=torch.bool)
    if valid_mask.ndim != 1 or valid_mask.shape[0] != current_mu.shape[0]:
        raise ValueError(f"Invalid transition mask shape: {valid_mask.shape}")

    squared = (current_mu[:, :arm_dims] - next_mu[:, :arm_dims]).square()
    if valid_mask.any():
        per_dim = 0.5 * squared[valid_mask].mean(dim=0)
    else:
        per_dim = squared.sum(dim=0) * 0.0
    return per_dim.mean(), per_dim


def transition_indices(horizon: int, num_envs: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    if horizon < 1 or num_envs < 1:
        raise ValueError("horizon and num_envs must be positive")
    rows = torch.arange(horizon * num_envs, device=device).reshape(num_envs, horizon)
    next_indices = torch.empty_like(rows)
    next_indices[:, :-1] = rows[:, 1:]
    next_indices[:, -1] = 0
    final_envs = torch.full_like(rows, -1)
    final_envs[:, -1] = torch.arange(num_envs, device=device)
    return next_indices.reshape(-1), final_envs.reshape(-1)


def resolve_next_observations(
    rollout_obs: torch.Tensor,
    final_obs: torch.Tensor,
    next_indices: torch.Tensor,
    final_envs: torch.Tensor,
) -> torch.Tensor:
    next_indices = next_indices.to(device=rollout_obs.device, dtype=torch.long)
    final_envs = final_envs.to(device=rollout_obs.device, dtype=torch.long)
    result = rollout_obs[next_indices].clone()
    uses_final = final_envs >= 0
    if uses_final.any():
        result[uses_final] = final_obs[final_envs[uses_final]]
    return result
