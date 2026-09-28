"""Pure frame-layout helpers for policy-input and high-resolution videos."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FULL_WIDTH = 960
FULL_HEIGHT = 540
POLICY_VIEW_SIDE = FULL_HEIGHT
GUTTER = 16
POLICY_CROP_X = (FULL_WIDTH - POLICY_VIEW_SIDE) // 2
DUAL_WIDTH = POLICY_VIEW_SIDE + GUTTER + FULL_WIDTH
POLICY_INPUT_WIDTH = 84
POLICY_IMAGE_SIZE = POLICY_INPUT_WIDTH * POLICY_INPUT_WIDTH * 3
CUBE_PROPRIO_SIZE = 23
CUBE_FLAT_OBSERVATION_SIZE = POLICY_IMAGE_SIZE + CUBE_PROPRIO_SIZE
VISUAL_REACH_PROPRIO_SIZE = 22
VISUAL_REACH_FLAT_OBSERVATION_SIZE = POLICY_IMAGE_SIZE + VISUAL_REACH_PROPRIO_SIZE
VISUAL_PROPRIO_FLAT_OBSERVATION_SIZES = {
    VISUAL_REACH_FLAT_OBSERVATION_SIZE,
    CUBE_FLAT_OBSERVATION_SIZE,
    POLICY_IMAGE_SIZE + 25,
    POLICY_IMAGE_SIZE + 26,
}
POLICY_INPUT_UPSCALE = 6
POLICY_INPUT_SIDE = POLICY_INPUT_WIDTH * POLICY_INPUT_UPSCALE
INPUT_HEADER_HEIGHT = 42
INPUT_ROW_HEIGHT = INPUT_HEADER_HEIGHT + FULL_HEIGHT
STANDARD_VIDEO_WIDTH = POLICY_INPUT_SIDE + GUTTER + FULL_WIDTH
POLICY_INPUT_ORIGIN = (0, INPUT_HEADER_HEIGHT + (FULL_HEIGHT - POLICY_INPUT_SIDE) // 2)
FULL_CAMERA_ORIGIN = (POLICY_INPUT_SIDE + GUTTER, INPUT_HEADER_HEIGHT)
POLICY_FOV_FULL_SIDE = FULL_HEIGHT
POLICY_FOV_FULL_X = (FULL_WIDTH - POLICY_FOV_FULL_SIDE) // 2
POLICY_FOV_BOUNDARY_THICKNESS = 2
POLICY_FOV_BOUNDARY_COLOR = (218, 224, 232)
ACTION_TRACE_HEIGHT = 240
CONTROL_PANEL_HEIGHT = ACTION_TRACE_HEIGHT
ACTION_TRACE_COLORS = (
    (66, 133, 244),
    (219, 68, 55),
    (244, 180, 0),
    (15, 157, 88),
    (171, 71, 188),
    (0, 172, 193),
    (255, 112, 67),
    (128, 203, 196),
)
CONTROL_COLORS = ACTION_TRACE_COLORS


def high_res_views(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frame.shape != (FULL_HEIGHT, FULL_WIDTH, 3):
        raise ValueError(
            f"expected a {FULL_WIDTH}x{FULL_HEIGHT} RGB frame, got {frame.shape}"
        )
    overview = frame
    policy_view = frame[:, POLICY_CROP_X : POLICY_CROP_X + POLICY_VIEW_SIDE]
    return policy_view, overview


def compose_high_res_dual(frame: np.ndarray) -> np.ndarray:
    policy_view, overview = high_res_views(frame)
    canvas = np.full((FULL_HEIGHT, DUAL_WIDTH, 3), (15, 18, 22), dtype=np.uint8)
    canvas[:, :POLICY_VIEW_SIDE] = policy_view
    canvas[:, POLICY_VIEW_SIDE + GUTTER :] = overview
    return canvas


def latest_policy_rgb(observation: np.ndarray) -> np.ndarray:
    """Return the most recent RGB image from a policy observation tensor.

    Camera observations are channel-last in this project. Visual manipulation
    stacks three RGB frames along that final channel axis, while Reach supplies
    one RGB frame. Reach and Place Cube proprio policies flatten one RGB frame before
    appending their task-specific state vector. A channel-first tensor is also
    accepted.
    """
    values = np.asarray(observation)
    if (
        values.ndim in (1, 2)
        and values.shape[-1] in VISUAL_PROPRIO_FLAT_OBSERVATION_SIZES
    ):
        if values.ndim == 2:
            if values.shape[0] < 1:
                raise ValueError("policy observation batch must contain at least one environment")
            values = values[0]
        values = values[:POLICY_IMAGE_SIZE].reshape(
            POLICY_INPUT_WIDTH, POLICY_INPUT_WIDTH, 3
        )
    if values.ndim == 4:
        if values.shape[0] < 1:
            raise ValueError("policy observation batch must contain at least one environment")
        values = values[0]
    if values.ndim != 3:
        raise ValueError(f"expected a 3-D policy image tensor, got {values.shape}")

    if (
        values.shape[:2] == (POLICY_INPUT_WIDTH, POLICY_INPUT_WIDTH)
        and values.shape[-1] >= 3
        and values.shape[-1] % 3 == 0
    ):
        latest = values[..., -3:]
    elif (
        values.shape[1:] == (POLICY_INPUT_WIDTH, POLICY_INPUT_WIDTH)
        and values.shape[0] >= 3
        and values.shape[0] % 3 == 0
    ):
        latest = np.moveaxis(values[-3:, ...], 0, -1)
    else:
        raise ValueError(
            "policy observation must contain RGB channels in a channel-first or channel-last axis"
        )
    if latest.shape != (POLICY_INPUT_WIDTH, POLICY_INPUT_WIDTH, 3):
        raise ValueError(
            f"expected latest policy RGB shape {(POLICY_INPUT_WIDTH, POLICY_INPUT_WIDTH, 3)}, got {latest.shape}"
        )
    return latest


def with_policy_fov_boundary(full_frame: np.ndarray) -> np.ndarray:
    """Mark the square policy field of view inside the wider recording frame."""
    if full_frame.shape != (FULL_HEIGHT, FULL_WIDTH, 3):
        raise ValueError(
            f"expected a {FULL_WIDTH}x{FULL_HEIGHT} RGB frame, got {full_frame.shape}"
        )
    if full_frame.dtype != np.uint8:
        raise ValueError("video input frames must be uint8 RGB arrays")

    outlined = full_frame.copy()
    x0 = POLICY_FOV_FULL_X
    x1 = x0 + POLICY_FOV_FULL_SIDE
    thickness = POLICY_FOV_BOUNDARY_THICKNESS
    outlined[:thickness, x0:x1] = POLICY_FOV_BOUNDARY_COLOR
    outlined[-thickness:, x0:x1] = POLICY_FOV_BOUNDARY_COLOR
    outlined[:, x0 : x0 + thickness] = POLICY_FOV_BOUNDARY_COLOR
    outlined[:, x1 - thickness : x1] = POLICY_FOV_BOUNDARY_COLOR
    return outlined


def compose_policy_input_high_res(policy_observation: np.ndarray, full_frame: np.ndarray) -> np.ndarray:
    """Place the latest exact policy RGB frame beside the raw full camera frame."""
    policy_frame = latest_policy_rgb(policy_observation)
    if full_frame.shape != (FULL_HEIGHT, FULL_WIDTH, 3):
        raise ValueError(
            f"expected a {FULL_WIDTH}x{FULL_HEIGHT} RGB frame, got {full_frame.shape}"
        )
    if policy_frame.dtype != np.uint8 or full_frame.dtype != np.uint8:
        raise ValueError("video input frames must be uint8 RGB arrays")

    canvas = Image.new("RGB", (STANDARD_VIDEO_WIDTH, INPUT_ROW_HEIGHT), (15, 18, 22))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    draw.rectangle((0, 0, STANDARD_VIDEO_WIDTH, INPUT_HEADER_HEIGHT), fill=(10, 12, 16))
    draw.text((10, 12), "POLICY INPUT: LATEST 84x84 FRAME", fill=(235, 235, 235), font=font)
    draw.text(
        (FULL_CAMERA_ORIGIN[0] + 10, 12),
        "HIGH-RES CAMERA (POLICY FOV OUTLINED)",
        fill=(235, 235, 235),
        font=font,
    )
    policy_image = Image.fromarray(policy_frame).resize(
        (POLICY_INPUT_SIDE, POLICY_INPUT_SIDE), Image.Resampling.NEAREST
    )
    canvas.paste(policy_image, POLICY_INPUT_ORIGIN)
    canvas.paste(Image.fromarray(with_policy_fov_boundary(full_frame)), FULL_CAMERA_ORIGIN)
    return np.asarray(canvas)


def redraw_high_res_header(frame: np.ndarray) -> np.ndarray:
    """Redraw the two static view labels after decoding a lossy video frame."""
    if frame.shape != (FULL_HEIGHT, DUAL_WIDTH, 3):
        raise ValueError(
            f"expected a {DUAL_WIDTH}x{FULL_HEIGHT} RGB frame, got {frame.shape}"
        )
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    draw.rectangle((0, 0, DUAL_WIDTH, 41), fill=(10, 12, 16))
    draw.text((10, 12), "HIGH-RES POLICY FOV", fill=(235, 235, 235), font=font)
    draw.text(
        (POLICY_VIEW_SIDE + GUTTER + 10, 12),
        "FULL CAMERA VIEW",
        fill=(235, 235, 235),
        font=font,
    )
    return np.asarray(image)


def full_to_policy_pixel(pixel: tuple[float, float] | None) -> tuple[float, float] | None:
    if pixel is None:
        return None
    return pixel[0] - POLICY_CROP_X, pixel[1]


def compose_action_trace(
    video_frame: np.ndarray,
    actions: np.ndarray,
    frame_index: int,
    action_bound: float | None = None,
    action_labels: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Place a synchronized task-specific policy-action trace below a frame."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"expected non-empty actions with shape (time, action_dim), got {values.shape}")
    if values.shape[1] > len(ACTION_TRACE_COLORS):
        raise ValueError(f"at most {len(ACTION_TRACE_COLORS)} action dimensions are supported")
    if video_frame.ndim != 3 or video_frame.shape[2] != 3:
        raise ValueError(f"expected an RGB video frame, got {video_frame.shape}")
    labels = action_labels or tuple(f"A{index + 1}" for index in range(values.shape[1]))
    if len(labels) != values.shape[1]:
        raise ValueError("action_labels must match the policy action dimension")

    index = min(max(int(frame_index), 0), values.shape[0] - 1)
    height, width = video_frame.shape[:2]
    canvas = Image.new("RGB", (width, height + ACTION_TRACE_HEIGHT), (15, 18, 22))
    canvas.paste(Image.fromarray(video_frame), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)

    left, right = 66, width - 18
    title_y = height + 10
    legend_y = height + 34
    top, bottom = height + 60, height + ACTION_TRACE_HEIGHT - 26
    bound = max(1.0, float(np.nanmax(np.abs(values)))) if action_bound is None else float(action_bound)
    if not np.isfinite(bound) or bound <= 0 or np.nanmax(np.abs(values)) > bound:
        raise ValueError("action_bound must be positive and cover every action value")
    draw.text(
        (18, title_y),
        f"ACTION TRACE ({values.shape[1]}D)",
        fill=(235, 235, 235),
        font=font,
    )
    draw.line((left, (top + bottom) // 2, right, (top + bottom) // 2), fill=(80, 86, 96), width=1)
    draw.rectangle((left, top, right, bottom), outline=(72, 78, 88), width=1)

    count = values.shape[0]
    xs = np.linspace(left, right, count)
    legend_width = max(1, (right - 18) // values.shape[1])
    for joint, color in enumerate(ACTION_TRACE_COLORS[: values.shape[1]]):
        trace = values[: index + 1, joint]
        ys = top + (bound - trace) * (bottom - top) / (2.0 * bound)
        points = [(float(xs[i]), float(ys[i])) for i in range(index + 1)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        legend_x = 18 + joint * legend_width
        draw.line((legend_x, legend_y, legend_x + 13, legend_y), fill=color, width=3)
        draw.text((legend_x + 17, legend_y - 7), labels[joint], fill=(220, 224, 230), font=font)

    current_x = float(xs[index])
    draw.line((current_x, top, current_x, bottom), fill=(245, 245, 245), width=1)
    draw.text((18, top - 5), f"+{bound:.2f}", fill=(165, 172, 184), font=font)
    draw.text((18, bottom - 7), f"-{bound:.2f}", fill=(165, 172, 184), font=font)
    return np.asarray(canvas)


def compose_control_trace(
    video_frame: np.ndarray,
    actions: np.ndarray,
    frame_index: int,
    action_bound: float | None = None,
) -> np.ndarray:
    """Backward-compatible seven-axis Reach action-trace wrapper."""
    return compose_action_trace(
        video_frame,
        actions,
        frame_index,
        action_bound=action_bound,
        action_labels=tuple(f"J{index + 1}" for index in range(7)),
    )
