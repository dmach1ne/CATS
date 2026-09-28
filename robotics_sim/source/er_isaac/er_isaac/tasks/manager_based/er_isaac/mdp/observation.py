from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.envs.mdp.observations import image

from .frame_stack import FrameStackBuffer
from .target_overlay import image_with_target_overlay

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)


CUBE_IMAGE_SIZE = 84 * 84 * 3
CUBE_PROPRIO_SIZE = 23
CUBE_OBSERVATION_SIZE = CUBE_IMAGE_SIZE + CUBE_PROPRIO_SIZE
VISUAL_REACH_IMAGE_SIZE = 84 * 84 * 3
VISUAL_REACH_PROPRIO_SIZE = 22
VISUAL_REACH_OBSERVATION_SIZE = VISUAL_REACH_IMAGE_SIZE + VISUAL_REACH_PROPRIO_SIZE


def visual_reach_image_proprio(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
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
    include_goal: bool = False,
) -> torch.Tensor:
    """Return one target-overlaid RGB frame and raw Reach proprioception."""
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
    if overlaid.shape[1:] != (84, 84, 3):
        raise ValueError(
            "Visual Reach proprio policy requires NHWC RGB frames with shape "
            f"(N, 84, 84, 3), got {tuple(overlaid.shape)}"
        )

    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, robot_cfg.joint_ids]
    last_action = env.action_manager.action
    ee_height = (
        robot.data.body_pos_w[:, ee_cfg.body_ids[0], 2]
        - env.scene.env_origins[:, 2]
    ).unsqueeze(-1)

    proprio = torch.cat((joint_pos, joint_vel, last_action, ee_height), dim=-1)
    if proprio.shape[-1] != VISUAL_REACH_PROPRIO_SIZE:
        raise ValueError(
            f"Expected {VISUAL_REACH_PROPRIO_SIZE} proprio values, got {proprio.shape[-1]}"
        )

    image_flat = overlaid.to(dtype=torch.float32).reshape(overlaid.shape[0], -1)
    observation = torch.cat((image_flat, proprio.to(dtype=torch.float32)), dim=-1)
    if observation.shape[-1] != VISUAL_REACH_OBSERVATION_SIZE:
        raise ValueError(
            f"Expected observation width {VISUAL_REACH_OBSERVATION_SIZE}, "
            f"got {observation.shape[-1]}"
        )
    if include_goal:
        # UniformPoseCommand exposes the actual pose in the robot base frame.
        goal = env.command_manager.get_command(command_name)[:, :3]
        if goal.shape != (observation.shape[0], 3):
            raise ValueError("Expected one robot-base XYZ goal per environment")
        observation = torch.cat((observation, goal.to(dtype=torch.float32)), dim=-1)
    return observation


def cube_image_proprio(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
    data_type: str = "rgb",
    command_name: str = "object_pose",
    camera_translation: tuple[float, float, float] = (1.29457219, -0.00127079938, 0.89823765),
    camera_rotation_xyz_deg: tuple[float, float, float] = (2.25576561, 45.25221894, 88.25015104),
    camera_fx_px: float = 104.5640380859375,
    camera_fy_px: float = 104.5640380859375,
    camera_cx_px: float = 42.0,
    camera_cy_px: float = 42.0,
    clipping_range: tuple[float, float] = (0.05, 20.0),
    target_marker_radius_m: float = 0.025,
    include_goal: bool = False,
    image_shape: tuple[int, int, int] = (84, 84, 3),
) -> torch.Tensor:
    """Return one target-overlaid RGB frame followed by raw robot proprioception."""
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
    if tuple(image_shape) not in ((84, 84, 3), (128, 128, 3)):
        raise ValueError(f"Unsupported Cube manipulation image_shape: {image_shape}")
    if overlaid.shape[1:] != tuple(image_shape):
        raise ValueError(
            "Cube manipulation proprio policy requires NHWC RGB frames with shape "
            f"(N, {image_shape[0]}, {image_shape[1]}, 3), got {tuple(overlaid.shape)}"
        )

    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, robot_cfg.joint_ids]
    last_action = env.action_manager.action
    ee_height = (
        robot.data.body_pos_w[:, ee_cfg.body_ids[0], 2]
        - env.scene.env_origins[:, 2]
    ).unsqueeze(-1)

    proprio = torch.cat((joint_pos, joint_vel, last_action, ee_height), dim=-1)
    if proprio.shape[-1] != CUBE_PROPRIO_SIZE:
        raise ValueError(
            f"Expected {CUBE_PROPRIO_SIZE} proprio values, got {proprio.shape[-1]}"
        )

    image_flat = overlaid.to(dtype=torch.float32).reshape(overlaid.shape[0], -1)
    observation = torch.cat((image_flat, proprio.to(dtype=torch.float32)), dim=-1)
    expected_size = image_shape[0] * image_shape[1] * image_shape[2] + CUBE_PROPRIO_SIZE
    if observation.shape[-1] != expected_size:
        raise ValueError(
            f"Expected observation width {expected_size}, "
            f"got {observation.shape[-1]}"
        )
    if include_goal:
        # UniformPoseCommand exposes the actual pose in the robot base frame.
        goal = env.command_manager.get_command(command_name)[:, :3]
        if goal.shape != (observation.shape[0], 3):
            raise ValueError("Expected one robot-base XYZ goal per environment")
        observation = torch.cat((observation, goal.to(dtype=torch.float32)), dim=-1)
    return observation


class image_stack(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.stack_num: int = cfg.params.get("stack_num", 3)
        self.frame_buffer = FrameStackBuffer(self.stack_num)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        stack_num: int = 3,
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
    ) -> torch.Tensor:
        if stack_num != self.stack_num:
            raise ValueError(f"Configured stack_num changed after initialization: {self.stack_num} -> {stack_num}")
        # obtain the images from the sensor
        img = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,  # we pre-process based on model
        )
        reset_mask = env.episode_length_buf == 0
        return self.frame_buffer.update(img, reset_mask=reset_mask)
