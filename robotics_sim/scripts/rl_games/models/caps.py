from __future__ import annotations

import inspect
import math

import torch


IMAGE_SIZE = 84 * 84 * 3


def resolve_target_overlay_projection(observation_term) -> dict:
    """Resolve the active observation's explicit values and callable defaults."""
    if observation_term is None or not callable(observation_term.func):
        raise ValueError("CAPS overlay preservation requires an image_proprio observation")
    params = observation_term.params
    if not params.get("include_goal", False):
        raise ValueError("CAPS overlay preservation requires include_goal=True")
    signature = inspect.signature(observation_term.func)
    shape_parameter = signature.parameters.get("image_shape")
    shape = tuple(params.get("image_shape", shape_parameter.default if shape_parameter is not None else (84, 84, 3)))
    if shape not in ((84, 84, 3), (128, 128, 3)):
        raise ValueError(f"Unsupported CAPS image_shape: {shape}")
    projection = {"image_height": shape[0], "image_width": shape[1]}
    for key in (
        "camera_translation", "camera_rotation_xyz_deg", "camera_fx_px", "camera_fy_px",
        "camera_cx_px", "camera_cy_px", "clipping_range", "target_marker_radius_m",
    ):
        parameter = signature.parameters.get(key)
        value = params.get(key, parameter.default if parameter is not None else inspect.Parameter.empty)
        if value is inspect.Parameter.empty or value is None:
            raise ValueError(f"Missing CAPS overlay parameter: {key}")
        size = {"camera_translation": 3, "camera_rotation_xyz_deg": 3, "clipping_range": 2}.get(key)
        if size is not None:
            if not isinstance(value, (tuple, list)) or len(value) != size:
                raise ValueError(f"Invalid CAPS overlay parameter: {key}")
            value = [float(item) for item in value]
            finite = all(math.isfinite(item) for item in value)
        else:
            value = float(value)
            finite = math.isfinite(value)
        if not finite:
            raise ValueError(f"Non-finite CAPS overlay parameter: {key}")
        projection["marker_radius_m" if key == "target_marker_radius_m" else key] = value
    if min(projection["camera_fx_px"], projection["camera_fy_px"], projection["marker_radius_m"]) <= 0:
        raise ValueError("CAPS overlay focal lengths and radius must be positive")
    near, far = projection["clipping_range"]
    if not 0 < near < far:
        raise ValueError("Invalid CAPS overlay clipping range")
    return projection


def perturb_goal_fixed_scale_observations(
    observations: torch.Tensor,
    sigma: float,
    action_dims: int,
    standard_normal: torch.Tensor | None = None,
    overlay_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Perturb measured inputs with equal noise scale in policy-input space."""
    if observations.ndim != 2:
        raise ValueError(f"Expected a flat observation batch, got {observations.shape}")
    if action_dims not in (7, 8):
        raise ValueError(f"GoalFixedScale requires 7 or 8 actions, got {action_dims}")
    image_size = observations.shape[1] - 18 - action_dims
    side = math.isqrt(max(0, image_size // 3))
    if side not in (84, 128) or image_size != side * side * 3:
        raise ValueError(
            f"GoalFixedScale requires 84 or 128 RGB plus {18 + action_dims} proprio entries, "
            f"got {observations.shape[1]}"
        )
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("CAPS sigma must be finite and positive")

    if standard_normal is None:
        standard_normal = torch.randn_like(observations)
    elif standard_normal.shape != observations.shape:
        raise ValueError("Standard-normal sample must match the observation shape")
    else:
        standard_normal = standard_normal.to(
            device=observations.device, dtype=observations.dtype
        )

    delta = torch.zeros_like(observations)
    q_start = image_size
    dq_start = q_start + 7
    previous_action_start = dq_start + 7
    ee_height_index = previous_action_start + action_dims

    # Invert the network's fixed scales so every selected policy input receives
    # an independent N(0, sigma^2) perturbation after preprocessing.
    delta[:, :image_size] = standard_normal[:, :image_size] * (sigma * 255.0)
    if overlay_mask is not None:
        if overlay_mask.shape != (observations.shape[0], side, side) or overlay_mask.dtype != torch.bool:
            raise ValueError(f"Overlay mask must be boolean with shape (N, {side}, {side})")
        delta[:, :image_size].view(-1, side, side, 3).masked_fill_(overlay_mask.unsqueeze(-1), 0.0)
    delta[:, q_start:dq_start] = standard_normal[:, q_start:dq_start] * (sigma * math.pi)
    delta[:, dq_start:previous_action_start] = (
        standard_normal[:, dq_start:previous_action_start] * (sigma / 0.5)
    )
    delta[:, ee_height_index : ee_height_index + 1] = (
        standard_normal[:, ee_height_index : ee_height_index + 1] * (sigma / 2.0)
    )
    return observations + delta


def spatial_action_loss(
    current_mu: torch.Tensor,
    perturbed_mu: torch.Tensor,
    arm_dims: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    if current_mu.shape != perturbed_mu.shape:
        raise ValueError(
            f"Policy mean shapes differ: {current_mu.shape} and {perturbed_mu.shape}"
        )
    if current_mu.ndim != 2 or current_mu.shape[1] < arm_dims:
        raise ValueError(f"Expected at least {arm_dims} action dimensions, got {current_mu.shape}")

    per_dim = 0.5 * (
        current_mu[:, :arm_dims] - perturbed_mu[:, :arm_dims]
    ).square().mean(dim=0)
    return per_dim.mean(), per_dim
