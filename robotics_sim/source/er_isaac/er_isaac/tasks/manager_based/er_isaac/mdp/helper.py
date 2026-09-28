from __future__ import annotations

import torch
# import numpy as np
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# from isaaclab.envs.utils.io_descriptors import (
#     generic_io_descriptor,
#     record_shape,
# )
from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.assets import RigidObject, AssetBase
import isaaclab.sim as sim_utils
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from isaacsim.core.utils.stage import get_current_stage


def set_prim_pose_from_command(
        env: ManagerBasedRLEnv, 
        env_ids: torch.Tensor | None,
        command_name: str,
        camera_pos_raw: tuple[float, float] = (3.0, 0.0, 0.6),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("target_viz")):
    # print("set_prim_pose_from_command 호출")
    cmd: UniformPoseCommand = env.command_manager.get_term(command_name)
    object_pos_w = cmd.pose_command_b[:, :3].clone()

    robot: RigidObject = env.scene[robot_cfg.name]
    entity: RigidObject = env.scene[asset_cfg.name]
    camera_pos_w = torch.tensor(camera_pos_raw, device=object_pos_w.device, dtype=object_pos_w.dtype).view(1, 3).expand_as(object_pos_w)
    # object_pos_w = entity.data.root_pos_w[:, :3].clone()
    new_pos_w = camera_pos_w + (object_pos_w - camera_pos_w) * 0.3

    object_pos_b, object_quat = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, new_pos_w)

    entity.write_root_link_pose_to_sim(torch.cat([object_pos_b, object_quat], dim=-1), env_ids)


def set_marker_pose_from_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target_viz"),
):
    """Place a visual marker at the commanded pose without changing its projection depth."""
    cmd: UniformPoseCommand = env.command_manager.get_term(command_name)
    pose_command_b = cmd.pose_command_b
    robot: RigidObject = env.scene[robot_cfg.name]
    marker: RigidObject = env.scene[asset_cfg.name]
    marker_pos_w, marker_quat_w = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        pose_command_b[:, :3],
        pose_command_b[:, 3:7],
    )
    marker.write_root_link_pose_to_sim(torch.cat([marker_pos_w, marker_quat_w], dim=-1), env_ids)
