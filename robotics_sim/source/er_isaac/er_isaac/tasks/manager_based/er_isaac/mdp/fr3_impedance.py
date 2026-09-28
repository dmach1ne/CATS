from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


ACTION_SCALE = 0.5
POSITION_SCALE = 0.2
Q_GOAL_RATE_LIMIT = 0.4
VELOCITY_FILTER_ALPHA = 0.99
TORQUE_LIMIT = 84.0
TORQUE_RATE_LIMIT = 999.999
FR3_Q_GOAL_MIN = (-2.6437, -1.6837, -2.8007, -2.9421, -2.7065, 0.6445, -2.9159)
FR3_Q_GOAL_MAX = (2.6437, 1.6837, 2.8007, -0.2518, 2.7065, 4.4169, 2.9159)


class Fr3ImpedanceAction(JointPositionAction):
    """Map policy actions to the bounded, rate-limited FR3 joint target."""

    cfg: Fr3ImpedanceActionCfg

    def __init__(self, cfg: Fr3ImpedanceActionCfg, env):
        super().__init__(cfg, env)
        if self.action_dim != 7:
            raise ValueError(f"FR3 impedance action requires seven joints, got {self.action_dim}")
        if cfg.position_scale <= 0.0 or cfg.position_scale > 1.0:
            raise ValueError("position_scale must be in (0, 1]")
        if cfg.q_goal_rate_limit <= 0.0 or cfg.physics_dt <= 0.0:
            raise ValueError("q_goal_rate_limit and physics_dt must be positive")

        self._q_goal_min = torch.tensor(cfg.q_goal_min, device=self.device).view(1, -1)
        self._q_goal_max = torch.tensor(cfg.q_goal_max, device=self.device).view(1, -1)
        if self._q_goal_min.shape[1] != self.action_dim or self._q_goal_max.shape[1] != self.action_dim:
            raise ValueError("q_goal_min and q_goal_max must contain seven values")
        if not torch.all(self._q_goal_min < self._q_goal_max):
            raise ValueError("every q_goal_min entry must be smaller than q_goal_max")

        current = self._asset.data.joint_pos[:, self._joint_ids]
        self._commanded_goal = current.clone()
        self._filtered_goal = current.clone()
        self._zero_velocity = torch.zeros_like(current)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        current = self._asset.data.joint_pos[:, self._joint_ids]
        q_goal = current + self.cfg.position_scale * (self._processed_actions - current)
        self._commanded_goal = torch.clamp(q_goal, min=self._q_goal_min, max=self._q_goal_max)
        self._processed_actions = self._commanded_goal

    def apply_actions(self):
        max_step = self.cfg.q_goal_rate_limit * self.cfg.physics_dt
        step = torch.clamp(self._commanded_goal - self._filtered_goal, min=-max_step, max=max_step)
        self._filtered_goal += step
        self._asset.set_joint_position_target(self._filtered_goal, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(self._zero_velocity, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        current = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        self._commanded_goal[env_ids] = current
        self._filtered_goal[env_ids] = current
        self._processed_actions[env_ids] = current
        self._zero_velocity[env_ids] = 0.0


@configclass
class Fr3ImpedanceActionCfg(JointPositionActionCfg):
    class_type: type = Fr3ImpedanceAction
    joint_names: list[str] = MISSING
    scale: float = ACTION_SCALE
    use_default_offset: bool = True
    position_scale: float = POSITION_SCALE
    q_goal_rate_limit: float = Q_GOAL_RATE_LIMIT
    physics_dt: float = 1.0 / 60.0
    q_goal_min: tuple[float, ...] = FR3_Q_GOAL_MIN
    q_goal_max: tuple[float, ...] = FR3_Q_GOAL_MAX


class Fr3ImpedanceActuator(IdealPDActuator):
    """Explicit impedance actuator matching the FR3 real controller law."""

    cfg: Fr3ImpedanceActuatorCfg

    def __init__(self, cfg: Fr3ImpedanceActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.physics_dt <= 0.0 or cfg.torque_rate_limit <= 0.0:
            raise ValueError("physics_dt and torque_rate_limit must be positive")
        if cfg.velocity_filter_alpha < 0.0 or cfg.velocity_filter_alpha > 1.0:
            raise ValueError("velocity_filter_alpha must be in [0, 1]")
        self._dq_filtered = torch.zeros_like(self.computed_effort)
        self._previous_effort = torch.zeros_like(self.computed_effort)

    def reset(self, env_ids: Sequence[int]):
        self._dq_filtered[env_ids] = 0.0
        self._previous_effort[env_ids] = 0.0

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            raise RuntimeError("FR3 impedance actuator requires joint position targets")

        alpha = self.cfg.velocity_filter_alpha
        self._dq_filtered = (1.0 - alpha) * self._dq_filtered + alpha * joint_vel
        raw_effort = self.stiffness * (control_action.joint_positions - joint_pos)
        raw_effort -= self.damping * self._dq_filtered
        absolute_limited = self._clip_effort(raw_effort)

        max_delta = self.cfg.torque_rate_limit * self.cfg.physics_dt
        delta = torch.clamp(absolute_limited - self._previous_effort, min=-max_delta, max=max_delta)
        applied_effort = self._previous_effort + delta

        self.computed_effort = raw_effort
        self.applied_effort = applied_effort
        self._previous_effort = applied_effort.clone()
        control_action.joint_efforts = applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class Fr3ImpedanceActuatorCfg(IdealPDActuatorCfg):
    class_type: type = Fr3ImpedanceActuator
    velocity_filter_alpha: float = VELOCITY_FILTER_ALPHA
    torque_rate_limit: float = TORQUE_RATE_LIMIT
    physics_dt: float = 1.0 / 60.0
