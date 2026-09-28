from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.observations import image
from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

from .frame_stack import FrameStackBuffer
from .target_overlay_geometry import draw_filled_circles, project_target_to_image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def image_with_target_overlay(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    command_name: str = "ee_pose",
    camera_translation: tuple[float, float, float] = (1.29457219, -0.00127079938, 0.89823765),
    camera_rotation_xyz_deg: tuple[float, float, float] = (2.25576561, 45.25221894, 88.25015104),
    camera_fx_px: float = 104.5640380859375,
    camera_fy_px: float = 104.5640380859375,
    camera_cx_px: float = 42.0,
    camera_cy_px: float = 42.0,
    clipping_range: tuple[float, float] = (0.05, 20.0),
    target_marker_radius_m: float = 0.025,
) -> torch.Tensor:
    """Return policy RGB with the controller-known Reach target on top."""
    rendered = image(
        env=env,
        sensor_cfg=sensor_cfg,
        data_type=data_type,
        normalize=False,
    )
    command: UniformPoseCommand = env.command_manager.get_term(command_name)
    center, radius, visible = project_target_to_image(
        command.pose_command_b[:, :3],
        image_height=rendered.shape[1],
        image_width=rendered.shape[2],
        camera_translation=camera_translation,
        camera_rotation_xyz_deg=camera_rotation_xyz_deg,
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
        clipping_range=clipping_range,
        marker_radius_m=target_marker_radius_m,
    )
    return draw_filled_circles(rendered, center, radius, visible)


class image_stack_with_target_overlay(ManagerTermBase):
    """Stack policy frames after compositing the current command target."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.stack_num = int(cfg.params.get("stack_num", 3))
        self.frame_buffer = FrameStackBuffer(self.stack_num)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        stack_num: int = 3,
        command_name: str = "object_pose",
        camera_translation: tuple[float, float, float] = (1.29457219, -0.00127079938, 0.89823765),
        camera_rotation_xyz_deg: tuple[float, float, float] = (2.25576561, 45.25221894, 88.25015104),
        camera_fx_px: float = 104.5640380859375,
        camera_fy_px: float = 104.5640380859375,
        camera_cx_px: float = 42.0,
        camera_cy_px: float = 42.0,
        clipping_range: tuple[float, float] = (0.05, 20.0),
        target_marker_radius_m: float = 0.025,
    ) -> torch.Tensor:
        if stack_num != self.stack_num:
            raise ValueError(
                f"Configured stack_num changed after initialization: {self.stack_num} -> {stack_num}"
            )
        overlaid = image_with_target_overlay(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            command_name=command_name,
            camera_translation=camera_translation,
            camera_rotation_xyz_deg=camera_rotation_xyz_deg,
            camera_fx_px=camera_fx_px,
            camera_fy_px=camera_fy_px,
            camera_cx_px=camera_cx_px,
            camera_cy_px=camera_cy_px,
            clipping_range=clipping_range,
            target_marker_radius_m=target_marker_radius_m,
        )
        reset_mask = env.episode_length_buf == 0
        return self.frame_buffer.update(overlaid, reset_mask=reset_mask)
