"""Vectorized placement contracts. Only PyTorch is required, including on CPU."""
import math

import torch


REST_HEIGHT = 0.0505
CUBE_EDGE = 0.049
LIFT_RISE = 0.05
OPEN_WIDTH = 0.06
EPS = 1e-7


def placement_goal_from_angle(angle):
    """Uniform angle gives a uniform circumference, not a uniform disk."""
    return torch.stack((0.45 + 0.25 * angle.cos(), 0.25 * angle.sin(),
                        torch.full_like(angle, REST_HEIGHT)), dim=-1)


def initial_grasp_valid(height, xy_displacement, width, table, left, right,
                        linear_speed, angular_speed, hand_tilt, finger_table_force,
                        pose_stable=None):
    """A resting bilateral grasp, not merely a closed joint target."""
    stable = ((linear_speed <= 0.02) & (angular_speed <= 0.2)
              if pose_stable is None else pose_stable)
    return ((height - REST_HEIGHT).abs().le(0.005 + EPS)
            & (xy_displacement <= 0.01) & (width >= CUBE_EDGE - 0.006)
            & (width <= CUBE_EDGE + 0.006) & table & left & right
            & torch.isfinite(linear_speed) & torch.isfinite(angular_speed) & stable
            & (hand_tilt <= math.radians(5)) & (finger_table_force <= 0.1))


class PoseWindowStability:
    """Require a full window of bounded measured positions and orientations."""

    def __init__(self, window, num_envs, device):
        if window < 2:
            raise ValueError('Pose stability needs at least two samples')
        self.positions = torch.zeros(window, num_envs, 3, device=device)
        self.quaternions = torch.zeros(window, num_envs, 4, device=device)
        self.count = 0
        self.position_span = torch.full((num_envs,), float('inf'), device=device)
        self.orientation_span = self.position_span.clone()

    def update(self, position, quaternion):
        index = self.count % len(self.positions)
        self.positions[index] = position
        self.quaternions[index] = quaternion
        self.count += 1
        if self.count < len(self.positions):
            return torch.zeros(position.shape[0], dtype=torch.bool, device=position.device)
        self.position_span = (self.positions.amax(0) - self.positions.amin(0)).norm(dim=-1)
        q = self.quaternions / self.quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dots = (q * q[0:1]).sum(-1).abs().clamp(0, 1)
        self.orientation_span = (2 * dots.acos()).amax(0)
        finite = torch.isfinite(self.positions).all(dim=(0, 2)) & torch.isfinite(q).all(dim=(0, 2))
        nonzero = (self.quaternions.norm(dim=-1) > 1e-6).all(0)
        return finite & nonzero & (self.position_span <= .001) & (self.orientation_span <= math.radians(1))


class OpeningState:
    """Opening is latched by a nonnegative float (or True), until trial reset."""

    def __init__(self, num_envs, device):
        self.latched = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.target = torch.zeros(num_envs, 2, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.latched[ids] = False
        self.target[ids] = 0

    def process(self, actions, finger_positions):
        opening = actions[:, 0] if actions.dtype == torch.bool else actions[:, 0] >= 0
        newly_open = opening & ~self.latched
        self.target[newly_open] = finger_positions[newly_open]
        self.latched |= opening

    def advance(self, dt, close_target, open_target):
        if dt <= 0:
            raise ValueError("Opening requires a positive physics dt")
        self.target[:] = torch.where(
            self.latched[:, None], torch.minimum(self.target + 0.1 * dt, open_target), close_target)
        return self.target


class PlacementState:
    """Physics-rate stage and continuous-hold state; no Isaac dependencies."""

    def __init__(self, num_envs, device):
        self.initial_height = torch.full((num_envs,), REST_HEIGHT, device=device)
        self.lowering = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.max_rise = torch.zeros(num_envs, device=device)
        self.min_xy_error = torch.full((num_envs,), float('inf'), device=device)
        self.episode_max_reward = torch.zeros(num_envs, device=device)
        self.reward_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.hold = torch.zeros(num_envs, 3, dtype=torch.float64, device=device)
        self.ever_success = torch.zeros(num_envs, 3, dtype=torch.bool, device=device)
        self.elapsed = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.values = {
            name: torch.zeros(num_envs, device=device)
            for name in ("reward", "goal_xy_error_m", "height_m", "rise_m", "width_m")
        }
        self.values.update({
            name: torch.zeros(num_envs, dtype=torch.bool, device=device)
            for name in ("table_contact", "left_finger_contact", "right_finger_contact", "resting")
        })

    def reset(self, env_ids=None, initial_height=None):
        ids = slice(None) if env_ids is None else env_ids
        self.initial_height[ids] = REST_HEIGHT if initial_height is None else initial_height
        self.lowering[ids] = False
        self.max_rise[ids] = 0
        self.min_xy_error[ids] = float('inf')
        self.episode_max_reward[ids] = 0
        self.reward_samples[ids] = 0
        self.hold[ids] = 0
        self.ever_success[ids] = False
        self.elapsed[ids] = 0
        for value in self.values.values():
            value[ids] = 0

    def record_policy_reward(self):
        # Only rewards emitted at control steps count, not intermediate physics ticks.
        self.episode_max_reward[:] = torch.maximum(self.episode_max_reward, self.values['reward'])
        self.reward_samples += 1

    def update(self, height, xy_error, width, table, left, right, dt, *,
               tcp_distance, linear_speed, angular_speed, robot_speed, max_width):
        if dt <= 0:
            raise ValueError("Placement requires a positive physics dt")
        rise = height - self.initial_height
        self.max_rise[:] = torch.maximum(self.max_rise, rise)
        self.min_xy_error[:] = torch.minimum(self.min_xy_error, xy_error)
        resting = (height - REST_HEIGHT).abs().le(0.005 + EPS) & table
        at_goal = xy_error <= 0.05 + EPS
        grasped = left & right
        # Latch lowering only after arriving at XY while actually lifted 10cm.
        self.lowering |= grasped & at_goal & (rise >= 0.10 - EPS)
        # Success requires the valid elevated-arrival phase before release.
        released = self.lowering & resting & (width >= OPEN_WIDTH - EPS) & ~left & ~right
        conditions = released[:, None] & (xy_error[:, None] <= xy_error.new_tensor([0.02, 0.05, 0.10]) + EPS)
        self.hold[:] = torch.where(conditions, self.hold + dt, 0)
        success = self.hold >= 1.0 - 1e-9
        self.ever_success |= success
        self.elapsed += dt

        # ManiSkill PlaceSphere dense reward, adapted to our resting cube and
        # contact-based release/continuous hold contract; normalized by 13.
        grasped = left & right
        transport_height = self.initial_height + 0.10
        desired_height = torch.where(self.lowering, torch.full_like(height, REST_HEIGHT), transport_height)
        goal_distance = torch.sqrt(xy_error.square() + (height - desired_height).square())
        reward = 2 * (1 - torch.tanh(5 * tcp_distance))
        # Away from the goal, unlock movement reward over the first 10cm.
        # After elevated arrival, keep it unlocked during lowering.
        movement_weight = (rise / 0.10).clamp(0.0, 1.0)
        movement_weight = torch.where(self.lowering, torch.ones_like(movement_weight), movement_weight)
        movement_reward = 5 - torch.tanh(5 * goal_distance)
        blended_reward = reward + movement_weight * (movement_reward - reward)
        reward = torch.where(grasped, blended_reward, reward)
        ungrasp = torch.where(grasped, (width / max_width.clamp_min(EPS)).clamp(0, 1), 16.)
        static = 1 - torch.tanh(10 * linear_speed + angular_speed)
        robot_static = (robot_speed < 0.2).float()
        reward = torch.where(self.lowering & at_goal & resting, 6 + (ungrasp + static + robot_static) / 3, reward)
        reward = torch.where(success[:, 1], 13., reward) / 13
        for key, value in dict(reward=reward, goal_xy_error_m=xy_error, height_m=height,
                               rise_m=rise, width_m=width, table_contact=table,
                               left_finger_contact=left, right_finger_contact=right,
                               resting=resting).items():
            self.values[key][:] = value

    def snapshot(self):
        result = {key: value.clone() for key, value in self.values.items()}
        result['lowering_phase'] = self.lowering.clone()
        result['episode_max_reward'] = self.episode_max_reward.clone()
        result['episode_min_goal_xy_error_m'] = self.min_xy_error.clone()
        result.update(max_rise_m=self.max_rise.clone(), ever_lifted=(self.max_rise >= LIFT_RISE - EPS),
                      hold_2cm_s=self.hold[:, 0].clone(), hold_5cm_s=self.hold[:, 1].clone(),
                      hold_10cm_s=self.hold[:, 2].clone(),
                      success_2cm=(self.hold[:, 0] >= 1 - 1e-9),
                      success_5cm=(self.hold[:, 1] >= 1 - 1e-9),
                      success_10cm=(self.hold[:, 2] >= 1 - 1e-9),
                      ever_success_2cm=self.ever_success[:, 0].clone(),
                      ever_success_5cm=self.ever_success[:, 1].clone(),
                      ever_success_10cm=self.ever_success[:, 2].clone(), elapsed_s=self.elapsed.clone())
        return result
