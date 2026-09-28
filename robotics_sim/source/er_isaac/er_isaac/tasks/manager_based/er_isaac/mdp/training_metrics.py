from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MinimumObjectGoalDistanceLogger(ManagerTermBase):
    """Track the minimum object-to-goal distance without changing the reward."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._min_object_goal_distance = torch.full(
            (env.num_envs,), torch.inf, dtype=torch.float, device=env.device
        )
        self._has_sample = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)

        completed = self._has_sample[env_ids]
        if bool(completed.any()):
            window_minima = self._min_object_goal_distance[env_ids][completed]
            self._env.extras.setdefault("log", {})[
                "Metrics/object_pose/min_object_goal_distance_m"
            ] = window_minima.mean().item()

        self._min_object_goal_distance[env_ids] = torch.inf
        self._has_sample[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "object_pose",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        robot: RigidObject = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        goal_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            command[:, :3],
        )
        distance = torch.linalg.vector_norm(obj.data.root_pos_w[:, :3] - goal_pos_w, dim=1)

        self._min_object_goal_distance = torch.minimum(self._min_object_goal_distance, distance)
        self._has_sample[:] = True
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
