from __future__ import annotations

import math

import torch


def _usd_rotate_xyz_matrix(
    rotation_deg: tuple[float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rx, ry, rz = (math.radians(value) for value in rotation_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotate_x = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)),
        device=device,
        dtype=dtype,
    )
    rotate_y = torch.tensor(
        ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)),
        device=device,
        dtype=dtype,
    )
    rotate_z = torch.tensor(
        ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)),
        device=device,
        dtype=dtype,
    )
    return rotate_x @ rotate_y @ rotate_z


def project_target_to_image(
    target_position: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    camera_translation: tuple[float, float, float],
    camera_rotation_xyz_deg: tuple[float, float, float],
    camera_fx_px: float,
    camera_fy_px: float,
    camera_cx_px: float,
    camera_cy_px: float,
    clipping_range: tuple[float, float],
    marker_radius_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project base-frame Reach targets using the ASAP_real camera model."""
    if target_position.ndim != 2 or target_position.shape[1] != 3:
        raise ValueError(f"target_position must have shape (N, 3), got {tuple(target_position.shape)}")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if min(camera_fx_px, camera_fy_px) <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    if not all(math.isfinite(value) for value in (camera_cx_px, camera_cy_px)):
        raise ValueError("camera principal point must be finite")
    if not 0.0 < clipping_range[0] < clipping_range[1]:
        raise ValueError("clipping_range must satisfy 0 < near < far")
    if marker_radius_m <= 0.0:
        raise ValueError("marker_radius_m must be positive")

    dtype = target_position.dtype
    device = target_position.device
    camera_to_world = _usd_rotate_xyz_matrix(
        camera_rotation_xyz_deg,
        device=device,
        dtype=dtype,
    )
    translation = torch.tensor(camera_translation, device=device, dtype=dtype)
    camera_point = (target_position - translation) @ camera_to_world
    depth = -camera_point[:, 2]

    safe_depth = depth.clamp_min(torch.finfo(dtype).eps)
    pixel_x = camera_cx_px + camera_fx_px * camera_point[:, 0] / safe_depth
    pixel_y = camera_cy_px - camera_fy_px * camera_point[:, 1] / safe_depth
    radius = torch.round(
        0.5 * (camera_fx_px + camera_fy_px) * marker_radius_m / safe_depth
    ).to(torch.int64).clamp_min(1)
    center = torch.stack((torch.round(pixel_x), torch.round(pixel_y)), dim=-1).to(torch.int64)

    near, far = clipping_range
    visible = (depth >= near) & (depth <= far)
    visible &= pixel_x + radius >= 0
    visible &= pixel_x - radius < image_width
    visible &= pixel_y + radius >= 0
    visible &= pixel_y - radius < image_height
    return center, radius, visible


def circle_coverage(
    centers: torch.Tensor,
    radii: torch.Tensor,
    visible: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Return the exact anti-aliased coverage used by the policy overlay."""
    batch_size = centers.shape[0]
    if centers.shape != (batch_size, 2) or radii.shape != (batch_size,) or visible.shape != (batch_size,):
        raise ValueError("circle parameters must match the image batch")

    work_dtype = torch.float32
    grid_y = torch.arange(image_height, device=centers.device, dtype=work_dtype).view(1, image_height, 1)
    grid_x = torch.arange(image_width, device=centers.device, dtype=work_dtype).view(1, 1, image_width)
    center_x = centers[:, 0].to(work_dtype).view(batch_size, 1, 1)
    center_y = centers[:, 1].to(work_dtype).view(batch_size, 1, 1)
    radius = radii.to(work_dtype).view(batch_size, 1, 1)
    distance = torch.sqrt((grid_x - center_x).square() + (grid_y - center_y).square())
    coverage = (radius + 0.5 - distance).clamp(0.0, 1.0)
    coverage *= visible.to(work_dtype).view(batch_size, 1, 1)
    return coverage


def target_overlay_mask(target_position: torch.Tensor, **projection) -> torch.Tensor:
    """Include every overlay pixel, including partially covered edge pixels."""
    centers, radii, visible = project_target_to_image(target_position, **projection)
    return circle_coverage(
        centers, radii, visible,
        image_height=projection["image_height"],
        image_width=projection["image_width"],
    ) > 0.0


def draw_filled_circles(
    images: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    visible: torch.Tensor,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
) -> torch.Tensor:
    """Composite anti-aliased circles over a batch of NHWC RGB images."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"images must have shape (N, H, W, 3), got {tuple(images.shape)}")
    if centers.shape[0] != images.shape[0]:
        raise ValueError("circle parameters must match the image batch")
    coverage = circle_coverage(
        centers, radii, visible, image_height=images.shape[1], image_width=images.shape[2]
    )

    alpha = coverage.unsqueeze(-1)
    color_tensor = torch.tensor(color, device=images.device, dtype=torch.float32).view(1, 1, 1, 3)
    composited = images.to(torch.float32) * (1.0 - alpha) + color_tensor * alpha
    if images.dtype == torch.uint8:
        return composited.round().clamp(0.0, 255.0).to(torch.uint8)
    return composited.to(images.dtype)
