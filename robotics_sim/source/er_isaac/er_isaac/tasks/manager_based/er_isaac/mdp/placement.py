"""Opt-in physical pregrasp preparation and placement runtime (IsaacLab 2.2.1)."""
from contextlib import contextmanager
import math

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import BinaryJointPositionAction, BinaryJointPositionActionCfg
from isaaclab.envs.mdp.commands import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import compute_pose_error, quat_apply, subtract_frame_transforms

from .placement_math import (OpeningState, PlacementState, REST_HEIGHT,
                             PoseWindowStability, initial_grasp_valid, placement_goal_from_angle)


class PlacementGoalCommand(UniformPoseCommand):
    def _resample_command(self, env_ids):
        angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * math.pi
        self.pose_command_b[env_ids, :3] = placement_goal_from_angle(angle)
        self.pose_command_b[env_ids, 3:] = self.pose_command_b.new_tensor([1., 0., 0., 0.])

    def compute(self, dt: float):
        # CommandTerm.reset is the only sampler. Never run its timed resampling path.
        self._update_metrics()
        self._update_command()


@configclass
class PlacementGoalCommandCfg(UniformPoseCommandCfg):
    class_type: type = PlacementGoalCommand


class PlacementGripperAction(BinaryJointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not isinstance(env, PregraspedPlacementEnv):
            raise TypeError("Placement requires mdp.placement:PregraspedPlacementEnv as Gym entry_point")
        if self._num_joints != 2:
            raise ValueError("Placement requires exactly two Panda finger joints")
        self.opening = OpeningState(self.num_envs, self.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.opening.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._processed_actions[ids] = self._close_command

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self.opening.process(actions, self._asset.data.joint_pos[:, self._joint_ids])

    def apply_actions(self):
        # apply_action is called before physics; consume the previous tick once.
        self._env.placement_state.sample(self._env._sim_step_counter - 1)
        self._processed_actions[:] = self.opening.advance(
            self._env.physics_dt, self._close_command, self._open_command)
        super().apply_actions()


@configclass
class PlacementGripperActionCfg(BinaryJointPositionActionCfg):
    class_type: type = PlacementGripperAction


def _pair_force(env, name):
    matrix = env.scene[name].data.force_matrix_w
    if matrix is None or matrix.shape[1] != 1 or matrix.shape[2] == 0:
        raise RuntimeError(f"{name}: missing single-body filtered pairwise contacts")
    if not bool(torch.isfinite(matrix).all()):
        raise RuntimeError(f"{name}: non-finite contact force")
    # Sum magnitudes, not vectors: opposing pair forces must not cancel.
    return matrix.norm(dim=-1).sum(dim=(1, 2))


def _measure(env):
    robot, obj = env.scene["robot"], env.scene["object"]
    pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                         obj.data.root_pos_w, obj.data.root_quat_w)
    goal = env.command_manager.get_command("object_pose")[:, :3]
    threshold = env.cfg.placement_contact_threshold_n
    return dict(height=pos_b[:, 2], xy_error=(pos_b[:, :2] - goal[:, :2]).norm(dim=-1),
                width=robot.data.joint_pos[:, env._placement_finger_ids].sum(-1),
                table=_pair_force(env, "placement_cube_table") > threshold,
                left=_pair_force(env, "placement_cube_left") > threshold,
                right=_pair_force(env, "placement_cube_right") > threshold,
                tcp_distance=(obj.data.root_pos_w - env.scene['ee_frame'].data.target_pos_w[:, 0]).norm(dim=-1),
                linear_speed=obj.data.root_lin_vel_w.norm(dim=-1),
                angular_speed=obj.data.root_ang_vel_w.norm(dim=-1),
                robot_speed=robot.data.joint_vel[:, env._placement_arm_ids].norm(dim=-1),
                max_width=env.action_manager.get_term('gripper_action')._open_command.sum(-1))


class PlacementRuntime:
    """Live state is resettable; last_metrics retains the pre-auto-reset sample."""

    def __init__(self, env):
        self.env = env
        self.state = PlacementState(env.num_envs, env.device)
        self.initial_grasp_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_tick = env._sim_step_counter
        self.last_metrics = {}

    def sample(self, tick):
        # After reset prep the first apply requests the already-consumed policy
        # tick. Return before reading prep-contaminated contact caches. The next
        # tick follows fresh policy physics + scene.update(dt); zero-period
        # contact sensors then refresh on access. Prep never advances this clock.
        if tick <= self.last_tick:
            return
        if tick != self.last_tick + 1:
            raise RuntimeError("Placement metrics missed a physics tick; continuous hold cannot be certified")
        self.state.update(**_measure(self.env), dt=self.env.physics_dt)
        self.last_tick = tick

    def publish(self):
        self.last_metrics = self.state.snapshot()
        self.last_metrics["initial_grasp_valid"] = self.initial_grasp_valid.clone()
        self.last_metrics["opening_latched"] = self.env.action_manager.get_term(
            "gripper_action").opening.latched.clone()
        self.last_metrics["goal_position_b"] = self.env.command_manager.get_command("object_pose")[:, :3].clone()
        robot = self.env.scene["robot"]
        self.last_metrics["arm_applied_torque"] = robot.data.applied_torque[:, self.env._placement_arm_ids].clone()
        self.last_metrics["arm_joint_velocity"] = robot.data.joint_vel[:, self.env._placement_arm_ids].clone()
        self.env.extras["placement"] = {key: value.clone() for key, value in self.last_metrics.items()}

    def reset(self, env_ids, initial_height):
        self.state.reset(env_ids, initial_height)
        self.initial_grasp_valid[env_ids] = True
        # Deliberately do not publish: the evaluator needs the terminal trial.


def placement_metrics(env):
    """Read the latest pre-reset snapshot without advancing any state or timer."""
    return {key: value.clone() for key, value in env.placement_state.last_metrics.items()}


def get_eval_snapshot(env):
    """Parent evaluator API: call after reward_manager.compute, before auto-reset."""
    result = placement_metrics(env)
    for alias, name in (("xy_error_m", "goal_xy_error_m"),
                        ("success_2cm_1s", "ever_success_2cm"),
                        ("success_5cm_1s", "ever_success_5cm"),
                        ("success_10cm_1s", "ever_success_10cm"),
                        ("cube_rise_m", "rise_m"), ("gripper_width_m", "width_m")):
        if name in result:
            result[alias] = result[name].clone()
    return result


def placement_reward(env):
    env.placement_state.sample(env._sim_step_counter)
    env.placement_state.state.record_policy_reward()
    env.placement_state.publish()
    return env.placement_state.last_metrics["reward"].clone()


def placement_dropped(env):
    robot, obj = env.scene["robot"], env.scene["object"]
    pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                         obj.data.root_pos_w, obj.data.root_quat_w)
    # The cube center crossing the actual work-surface plane is a below-table drop.
    return (pos_b[:, 2] < 0.026) | ~torch.isfinite(pos_b).all(dim=-1)


class PregraspedPlacementEnv(ManagerBasedRLEnv):
    """Reset-time physics is outside the six-second policy episode.

    Physically validate each environment's grasp once, then restore its snapshot
    without stepping physics at episode resets. No constraint attaches the cube.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        robot = self.scene["robot"]
        if not robot.is_fixed_base:
            raise ValueError("Placement IK requires a fixed-base Panda")
        self._placement_arm_ids = robot.find_joints(
            [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)[0]
        self._placement_finger_ids = robot.find_joints(
            ["panda_finger_joint1", "panda_finger_joint2"], preserve_order=True)[0]
        self._placement_hand_id = robot.find_bodies("panda_hand")[0][0]
        if len(self._placement_arm_ids) != 7 or len(self._placement_finger_ids) != 2:
            raise ValueError("Placement expects Panda arm7/fingers2")
        self._placement_ik_q = robot.data.default_joint_pos[:, self._placement_arm_ids].clone()
        self._placement_ik_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._placement_grasp_snapshot = None
        self._placement_initial_xy = torch.tensor(
            self.cfg.scene.object.init_state.pos[:2], device=self.device)
        self.placement_state = PlacementRuntime(self)

    def step(self, action):
        if not bool(self.placement_state.initial_grasp_valid.all()):
            raise RuntimeError("Call reset() and pass initial grasp validation before placement policy steps")
        return super().step(action)

    def _reset_idx(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if not env_ids.numel():
            return
        if self._placement_grasp_snapshot is None and (
                env_ids.numel() != self.num_envs or env_ids.unique().numel() != self.num_envs):
            raise RuntimeError("Placement first reset must initialize all environments together")
        super()._reset_idx(env_ids)
        state = self.placement_state.state
        completed = env_ids[state.reward_samples[env_ids] > 0]
        if completed.numel():
            # Completed episodes only, before subset reset clears accumulators.
            metrics = {
                'episode_max_reward': state.episode_max_reward,
                'episode_min_goal_xy_error_m': state.min_xy_error,
                'episode_max_rise_m': state.max_rise,
                'success_2cm_1s': state.ever_success[:, 0].float(),
                'success_5cm_1s': state.ever_success[:, 1].float(),
                'success_10cm_1s': state.ever_success[:, 2].float(),
            }
            for key, values in metrics.items():
                self.extras.setdefault('log', {})[f'Placement/{key}'] = values[completed].mean()
        self.placement_state.initial_grasp_valid[env_ids] = False
        if self._placement_grasp_snapshot is None:
            self._prepare_grasp(env_ids)
            self._cache_grasp()
        else:
            self._restore_grasp(env_ids)
        # Contact/solver caches are NOT serialized. This flag records the snapshot's
        # initial certification; policy-time contact/success uses fresh physics.
        self.placement_state.reset(env_ids, self._placement_grasp_snapshot["height"][env_ids])
        # No policy, reward, command, or episode counter was advanced by prep.
        self.episode_length_buf[env_ids] = 0

    def _cache_grasp(self):
        state = {
            kind: {name: {key: value.clone() for key, value in values.items()}
                   for name, values in assets.items()}
            for kind, assets in self.scene.get_state().items()
        }
        targets = {
            name: (asset.data.joint_pos_target.clone(), asset.data.joint_vel_target.clone(),
                   asset.data.joint_effort_target.clone())
            for name, asset in self.scene.articulations.items()
        }
        self._placement_grasp_snapshot = {
            "state": state, "targets": targets, "height": _measure(self)["height"].clone(),
        }

    def _restore_grasp(self, env_ids):
        snapshot = self._placement_grasp_snapshot
        state = {
            kind: {name: {key: value[env_ids].clone() for key, value in values.items()}
                   for name, values in assets.items()}
            for kind, assets in snapshot["state"].items()
        }
        self.scene.reset_to(state, env_ids=env_ids)
        # reset_to defaults PD targets to q/dq; retain the validated closing force
        # and arm targets instead. Only selected environment rows are changed.
        for name, (q, dq, effort) in snapshot["targets"].items():
            asset = self.scene[name]
            asset.set_joint_position_target(q[env_ids], env_ids=env_ids)
            asset.set_joint_velocity_target(dq[env_ids], env_ids=env_ids)
            asset.set_joint_effort_target(effort[env_ids], env_ids=env_ids)
        self.scene.write_data_to_sim()

    @contextmanager
    def _isolate_preparation(self, env_ids):
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = False
        other_ids = mask.nonzero().flatten()
        state = self.scene.get_state()
        other_state = {
            kind: {name: {key: value[other_ids].clone() for key, value in values.items()}
                   for name, values in assets.items()}
            for kind, assets in state.items()
        }
        targets = {
            name: (asset.data.joint_pos_target[other_ids].clone(),
                   asset.data.joint_vel_target[other_ids].clone(),
                   asset.data.joint_effort_target[other_ids].clone())
            for name, asset in self.scene.articulations.items()
        }

        def freeze_certified(ids):
            nonlocal other_ids
            if not len(ids):
                return
            if bool(torch.isin(ids, other_ids).any()):
                raise RuntimeError('Placement attempted to certify an already frozen environment')
            current = self.scene.get_state()
            for kind, assets in other_state.items():
                for name, values in assets.items():
                    for key in values:
                        values[key] = torch.cat((values[key], current[kind][name][key][ids].clone()))
            for name, asset in self.scene.articulations.items():
                q, dq, effort = targets[name]
                targets[name] = (torch.cat((q, asset.data.joint_pos_target[ids].clone())),
                                 torch.cat((dq, asset.data.joint_vel_target[ids].clone())),
                                 torch.cat((effort, asset.data.joint_effort_target[ids].clone())))
            other_ids = torch.cat((other_ids, ids))

        def restore_others():
            if other_ids.numel():
                self.scene.reset_to(other_state, env_ids=other_ids)
                for name, (q, dq, effort) in targets.items():
                    asset = self.scene[name]
                    asset.set_joint_position_target(q, env_ids=other_ids)
                    asset.set_joint_velocity_target(dq, env_ids=other_ids)
                    asset.set_joint_effort_target(effort, env_ids=other_ids)
                self.scene.write_data_to_sim()

        def step():
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            restore_others()

        step.freeze_certified = freeze_certified

        try:
            yield step
        finally:
            restore_others()
            # Sensor caches/timestamps and PhysX solver caches are not restored.
            # Runtime metrics must wait for fresh policy physics (sample's tick
            # guard), not consume these contacts as a new policy-time sample.

    def _ik_controller(self, env_ids):
        controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=len(env_ids), device=self.device)
        robot = self.scene["robot"]
        target = torch.zeros(len(env_ids), 7, device=self.device)
        frames = [frame for frame in self.cfg.scene.ee_frame.target_frames
                  if frame.prim_path.endswith("/panda_hand")]
        expected = (0., 0., self.cfg.placement_hand_offset_m)
        if (len(frames) != 1 or tuple(frames[0].offset.pos) != expected
                or tuple(frames[0].offset.rot) != (1., 0., 0., 0.)):
            raise ValueError("Placement requires the inherited Panda hand-to-grasp offset (0,0,0.1034)")
        # wxyz Rx(pi): R=diag(1,-1,-1), so local +Z points downward.
        target[:, 4] = 1.0
        offset_local = target.new_tensor(frames[0].offset.pos).expand(len(env_ids), -1)
        cube_center_w = robot.data.root_pos_w[env_ids] + target.new_tensor([0.45, 0., REST_HEIGHT])
        target[:, :3] = cube_center_w - quat_apply(target[:, 3:], offset_local)
        controller.set_command(target)
        return controller

    def _ik_delta(self, controller, env_ids):
        robot = self.scene["robot"]
        hand, joints = self._placement_hand_id, self._placement_arm_ids
        q = robot.data.joint_pos[env_ids][:, joints]
        jacobian = robot.root_physx_view.get_jacobians()[env_ids, hand - 1][:, :, joints]
        desired = controller.compute(robot.data.body_link_pos_w[env_ids, hand],
                                     robot.data.body_link_quat_w[env_ids, hand], jacobian, q)
        return desired - q

    def _solve_open_posture(self, env_ids):
        if not len(env_ids):
            return
        robot = self.scene["robot"]
        joints = self._placement_arm_ids
        controller = self._ik_controller(env_ids)
        limits = robot.data.soft_joint_pos_limits[env_ids][:, joints]
        for _ in range(self.cfg.placement_ik_iterations):
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=0.0)
            q = robot.data.joint_pos[env_ids][:, joints]
            q = (q + self._ik_delta(controller, env_ids).clamp(-0.1, 0.1)).clamp(
                limits[:, :, 0], limits[:, :, 1])
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=joints, env_ids=env_ids)
            robot.set_joint_position_target(q, joint_ids=joints, env_ids=env_ids)
        self.sim.forward()
        self.scene.update(dt=0.0)
        hand = self._placement_hand_id
        position_error, rotation_error = compute_pose_error(
            robot.data.body_link_pos_w[env_ids, hand], robot.data.body_link_quat_w[env_ids, hand],
            controller.ee_pos_des, controller.ee_quat_des)
        good = (position_error.norm(dim=-1) <= 0.002) & (rotation_error.norm(dim=-1) <= 0.03)
        if not bool(good.all()):
            raise RuntimeError(f"Placement initial IK failed for envs {env_ids[~good].tolist()}; "
                               f"position errors={position_error.norm(dim=-1).tolist()}")
        self._placement_ik_q[env_ids] = robot.data.joint_pos[env_ids][:, joints]
        self._placement_ik_ready[env_ids] = True

    def _grasp_guard(self, env_ids):
        robot, obj = self.scene["robot"], self.scene["object"]
        measured = _measure(self)
        pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                             obj.data.root_pos_w, obj.data.root_quat_w)
        axis = torch.zeros(self.num_envs, 3, device=self.device)
        axis[:, 2] = 1
        hand_axis = quat_apply(robot.data.body_link_quat_w[:, self._placement_hand_id], axis)
        tilt = (-hand_axis[:, 2]).clamp(-1, 1).acos()
        pose_stable = self._placement_pose_window.update(pos_b, obj.data.root_quat_w)
        # Reset-only support evidence shares the pose window; success uses raw contact.
        index = (self._placement_pose_window.count - 1) % len(self._placement_table_window)
        self._placement_table_window[index] = measured["table"]
        table_supported = (self._placement_table_window.any(dim=0)
                           & (measured["height"] - REST_HEIGHT).abs().le(0.002))
        valid = initial_grasp_valid(
            measured["height"], (pos_b[:, :2] - self._placement_initial_xy).norm(dim=-1),
            measured["width"], table_supported, measured["left"], measured["right"],
            obj.data.root_lin_vel_w.norm(dim=-1), obj.data.root_ang_vel_w.norm(dim=-1), tilt,
            _pair_force(self, "placement_left_table") + _pair_force(self, "placement_right_table"),
            pose_stable=pose_stable)
        self._placement_grasp_diagnostics = {
            **measured,
            "initial_table_window_contact": self._placement_table_window.any(dim=0),
            "initial_table_supported": table_supported,
            "cube_position_base": pos_b,
            "cube_linear_speed": obj.data.root_lin_vel_w.norm(dim=-1),
            "cube_angular_speed": obj.data.root_ang_vel_w.norm(dim=-1),
            "pose_window_position_span_m": self._placement_pose_window.position_span,
            "pose_window_orientation_span_rad": self._placement_pose_window.orientation_span,
            "hand_tilt_rad": tilt,
            "left_force_n": _pair_force(self, "placement_cube_left"),
            "right_force_n": _pair_force(self, "placement_cube_right"),
            "table_force_n": _pair_force(self, "placement_cube_table"),
            "finger_table_force_n": _pair_force(self, "placement_left_table") + _pair_force(self, "placement_right_table"),
            "arm_position_rad": robot.data.joint_pos[:, self._placement_arm_ids],
            "arm_target_rad": robot.data.joint_pos_target[:, self._placement_arm_ids],
        }
        return valid[env_ids]

    def _restore_cube_before_closure(self, env_ids):
        """Remove displacement acquired while settling the open arm, only at reset."""
        cube = self.scene["object"]
        pose = cube.data.default_root_state[env_ids, :7].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        cube.write_root_pose_to_sim(pose, env_ids=env_ids)
        cube.write_root_velocity_to_sim(torch.zeros_like(cube.data.default_root_state[env_ids, 7:13]),
                                        env_ids=env_ids)
        # Reset contact caches only, not the scene/articulation or their PD targets.
        for name in ("placement_cube_left", "placement_cube_right", "placement_cube_table",
                     "placement_left_table", "placement_right_table"):
            self.scene[name].reset(env_ids=env_ids)

    def _prepare_grasp(self, env_ids):
        if not len(env_ids):
            return
        # One initial attempt plus three reset-only retries, never policy steps.
        for attempt in range(1, 5):
            print(f"placement_grasp_attempt={attempt}/4 num_envs={len(env_ids)}", flush=True)
            try:
                return self._prepare_grasp_once(env_ids)
            except RuntimeError as error:
                if not str(error).startswith("Initial physical placement grasp invalid in envs "):
                    raise
                print(f"placement_grasp_attempt_failed={attempt}/4 num_envs={len(env_ids)}", flush=True)
                if attempt == 4:
                    raise

    def _prepare_grasp_once(self, env_ids):
        if not len(env_ids):
            return
        if self.cfg.placement_initial_arm_positions is not None:
            return self._prepare_measured_grasp(env_ids)
        robot = self.scene["robot"]
        joints, fingers = self._placement_arm_ids, self._placement_finger_ids
        # 13 observations cover 12 intervals (0.2s at the unchanged 60Hz physics).
        self._placement_pose_window = PoseWindowStability(
            self.cfg.placement_grasp_guard_steps + 1, self.num_envs, self.device)
        self._placement_table_window = torch.zeros(
            self.cfg.placement_grasp_guard_steps + 1, self.num_envs,
            dtype=torch.bool, device=self.device)
        gripper = self.action_manager.get_term("gripper_action")
        open_q = gripper._open_command.expand(len(env_ids), -1)
        robot.write_joint_state_to_sim(open_q, torch.zeros_like(open_q), joint_ids=fingers, env_ids=env_ids)
        robot.set_joint_position_target(open_q, joint_ids=fingers, env_ids=env_ids)
        with self._isolate_preparation(env_ids) as prep_step:
            self._solve_open_posture(env_ids[~self._placement_ik_ready[env_ids]])
            q = self._placement_ik_q[env_ids].clone()
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=joints, env_ids=env_ids)
            stiffness = robot.data.joint_stiffness[env_ids][:, joints]
            if not bool((stiffness > 0).all()):
                raise RuntimeError("Initial placement preparation requires position PD stiffness")
            self.sim.forward()
            gravity = robot.root_physx_view.get_gravity_compensation_forces()[env_ids][:, joints]
            robot.set_joint_position_target(q + gravity / stiffness, joint_ids=joints, env_ids=env_ids)
            for _ in range(30):
                prep_step()
                gravity = robot.root_physx_view.get_gravity_compensation_forces()[env_ids][:, joints]
                robot.set_joint_position_target(q + gravity / stiffness, joint_ids=joints, env_ids=env_ids)
            self._restore_cube_before_closure(env_ids)
            for _ in range(30):
                prep_step()
                gravity = robot.root_physx_view.get_gravity_compensation_forces()[env_ids][:, joints]
                robot.set_joint_position_target(q + gravity / stiffness, joint_ids=joints, env_ids=env_ids)
            self._grasp_guard(env_ids)
            if not getattr(self, "_placement_open_diagnostic_printed", False):
                print("placement_open_settled=" + str({
                    key: value[env_ids[:2]].detach().cpu().tolist()
                    for key, value in self._placement_grasp_diagnostics.items()}), flush=True)
                self._placement_open_diagnostic_printed = True
            close_q = gripper._close_command.expand(len(env_ids), -1)
            closing_target = open_q.clone()
            consecutive = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
            certified = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            maximum_consecutive = consecutive.clone()
            table_absent = consecutive.clone()
            pose_unstable = consecutive.clone()
            # Bias the existing PD target by G/Kp only during preparation. Unlike
            # accumulating IK error into the target, this has no integral windup.
            # Policy steps still use the unmodified legacy action/controller.
            for _ in range(self.cfg.placement_prep_steps):
                pending = ~certified
                closing_target[pending] = torch.maximum(close_q[pending], closing_target[pending] - 0.02 * self.physics_dt)
                robot.set_joint_position_target(closing_target[pending], joint_ids=fingers, env_ids=env_ids[pending])
                prep_step()
                valid = self._grasp_guard(env_ids) & (closing_target <= close_q + 1e-7).all(dim=-1)
                fully_closed = (closing_target <= close_q + 1e-7).all(dim=-1)
                table_absent += fully_closed & ~self._placement_grasp_diagnostics['table'][env_ids]
                pose_unstable += fully_closed & ((self._placement_pose_window.position_span[env_ids] > .001)
                    | (self._placement_pose_window.orientation_span[env_ids] > math.radians(1)))
                consecutive[pending] = torch.where(valid[pending], consecutive[pending] + 1, 0)
                maximum_consecutive = torch.maximum(maximum_consecutive, consecutive)
                newly_certified = pending & (consecutive >= self.cfg.placement_grasp_guard_steps)
                prep_step.freeze_certified(env_ids[newly_certified])
                certified |= newly_certified
                if bool(certified.all()):
                    return
                pending = ~certified
                gravity = robot.root_physx_view.get_gravity_compensation_forces()[env_ids[pending]][:, joints]
                robot.set_joint_position_target(q[pending] + gravity / stiffness[pending], joint_ids=joints, env_ids=env_ids[pending])
        failed = env_ids[~certified]
        diagnostics = {name: value[failed[:5]].detach().cpu().tolist()
                       for name, value in self._placement_grasp_diagnostics.items()}
        diagnostics['maximum_consecutive_valid'] = maximum_consecutive[:10].tolist()
        diagnostics['minimum_historical_consecutive_valid'] = int(maximum_consecutive.min())
        diagnostics['fully_closed_table_absent_counts'] = table_absent[:10].tolist()
        diagnostics['fully_closed_pose_unstable_counts'] = pose_unstable[:10].tolist()
        raise RuntimeError(f"Initial physical placement grasp invalid in envs {failed.tolist()}; "
                           "requires stable bilateral cube contact, table contact, vertical hand, "
                           "and fingers clear of table. No policy episode was started. "
                           f"Diagnostics(first5)={diagnostics}")

    def _set_measured_grasp(self, env_ids):
        """Write the captured pose, not the robot's policy action offset/defaults."""
        robot = self.scene["robot"]
        q = torch.tensor(self.cfg.placement_initial_arm_positions, device=self.device,
                         dtype=torch.float32).expand(len(env_ids), -1).clone()
        if q.shape[1] != 7 or not bool(torch.isfinite(q).all()):
            raise ValueError("Measured placement pose requires seven finite arm joints")
        width = self.cfg.placement_initial_gripper_width
        if not math.isfinite(width) or not 0 < width <= .08:
            raise ValueError("Measured placement width must be in (0,0.08]m")
        fingers = torch.full((len(env_ids), 2), width / 2, device=self.device)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q),
                                       joint_ids=self._placement_arm_ids, env_ids=env_ids)
        robot.write_joint_state_to_sim(fingers, torch.zeros_like(fingers),
                                       joint_ids=self._placement_finger_ids, env_ids=env_ids)
        robot.set_joint_position_target(q, joint_ids=self._placement_arm_ids, env_ids=env_ids)
        robot.set_joint_position_target(fingers, joint_ids=self._placement_finger_ids, env_ids=env_ids)
        self._restore_cube_before_closure(env_ids)
        return q

    def _prepare_measured_grasp(self, env_ids):
        robot = self.scene["robot"]
        q = self._set_measured_grasp(env_ids)
        joints = self._placement_arm_ids
        stiffness = robot.data.joint_stiffness[env_ids][:, joints]
        if not bool((stiffness > 0).all()):
            raise RuntimeError("Measured placement preparation requires position PD stiffness")
        self._placement_pose_window = PoseWindowStability(
            self.cfg.placement_grasp_guard_steps + 1, self.num_envs, self.device)
        self._placement_table_window = torch.zeros(
            self.cfg.placement_grasp_guard_steps + 1, self.num_envs,
            dtype=torch.bool, device=self.device)
        self.sim.forward()
        self.scene.update(dt=0.)
        print("placement_measured_initial=" + str({
            "arm_joint_names": [robot.joint_names[i] for i in joints],
            "arm_q": robot.data.joint_pos[env_ids[:2]][:, joints].tolist(),
            "hand_xyz_base": (robot.data.body_link_pos_w[env_ids[:2], self._placement_hand_id]
                              - robot.data.root_pos_w[env_ids[:2]]).tolist(),
            "cube_xyz_base": (self.scene["object"].data.root_pos_w[env_ids[:2]]
                              - robot.data.root_pos_w[env_ids[:2]]).tolist(),
            "finger_width": robot.data.joint_pos[env_ids[:2]][:, self._placement_finger_ids].sum(-1).tolist(),
        }), flush=True)
        consecutive = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        certified = torch.zeros_like(consecutive, dtype=torch.bool)
        close = self.action_manager.get_term("gripper_action")._close_command
        with self._isolate_preparation(env_ids) as prep_step:
            for _ in range(self.cfg.placement_prep_steps):
                pending = ~certified
                ids = env_ids[pending]
                gravity = robot.root_physx_view.get_gravity_compensation_forces()[ids][:, joints]
                robot.set_joint_position_target(q[pending] + gravity / stiffness[pending],
                                                joint_ids=joints, env_ids=ids)
                robot.set_joint_position_target(close.expand(len(ids), -1),
                                                joint_ids=self._placement_finger_ids, env_ids=ids)
                prep_step()
                valid = self._grasp_guard(env_ids)
                consecutive[pending] = torch.where(valid[pending], consecutive[pending] + 1, 0)
                newly_certified = pending & (consecutive >= self.cfg.placement_grasp_guard_steps)
                prep_step.freeze_certified(env_ids[newly_certified])
                certified |= newly_certified
                if bool(certified.all()):
                    return
        failed = env_ids[~certified]
        diagnostics = {key: value[failed[:5]].detach().cpu().tolist()
                       for key, value in self._placement_grasp_diagnostics.items()}
        raise RuntimeError(f"Initial physical placement grasp invalid in envs {failed.tolist()}; "
                           f"measured FR3 pose, no IK adjustment; Diagnostics(first5)={diagnostics}")
