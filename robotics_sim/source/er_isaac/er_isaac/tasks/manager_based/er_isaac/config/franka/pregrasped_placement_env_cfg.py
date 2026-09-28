"""Opt-in placement siblings. Registration and shared camera sizing live outside this file."""
from copy import deepcopy

from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ...mdp.placement import (PlacementGoalCommandCfg, PlacementGripperActionCfg,
                             placement_dropped, placement_reward)
from .vertical15_table55_env_cfg import Vertical15Table55EnvCfg, Vertical15Table55EvalEnvCfg


@configclass
class PlacementRewardsCfg:
    # RewardManager multiplies by step_dt. At 30 Hz this emits the specified
    # [0, 1] stage maximum per policy step, rather than [0, 1/30].
    placement = RewardTermCfg(func=placement_reward, weight=30.0)


class _PlacementMixin:
    placement_contact_threshold_n: float = 0.1
    placement_hand_offset_m: float = 0.1034
    placement_ik_iterations: int = 100
    placement_prep_steps: int = 300
    placement_grasp_guard_steps: int = 12
    # Initialize from a captured grasp pose with estimated cube position
    # and identity cube rotation.
    placement_initial_arm_positions: tuple | None = (
        0.05866348694477573, 0.3587113019815921, -0.05949372965396844,
        -2.5234578334544184, -0.0015859592232857196, 2.898845289280666,
        0.7835674301525485)
    placement_initial_gripper_width: float = 0.04864389821887016

    def _configure_placement(self):
        for name in ("scene", "actions", "commands", "events", "terminations", "curriculum"):
            setattr(self, name, deepcopy(getattr(self, name)))
        if tuple(self.scene.robot.init_state.pos) != (0., 0., 0.):
            raise ValueError("Placement Table55 requires the robot base at the environment origin")
        if tuple(self.scene.robot.init_state.rot) != (1., 0., 0., 0.):
            raise ValueError("Placement requires an unrotated robot base")
        self.scene.object.init_state.pos = (
            (0.4563215994839803, -0.002490832284263561, 0.0505)
            if self.placement_initial_arm_positions is not None else (0.45, 0., 0.0505))
        self.scene.object.init_state.rot = (1., 0., 0., 0.)
        self.scene.object.spawn.size = (0.049,) * 3
        self.scene.object.spawn.activate_contact_sensors = True
        self.scene.robot.spawn.activate_contact_sensors = True
        # Resolve the opposing finger contacts without changing controller gains.
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 32
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 8
        self.scene.object.spawn.rigid_props.solver_position_iteration_count = 32
        self.scene.object.spawn.rigid_props.solver_velocity_iteration_count = 8

        cube = self.scene.object.prim_path
        robot = self.scene.robot.prim_path
        # The entire radius-15 circle is on the main protective board. Include
        # the underlying main slab too; height alone never proves table contact.
        # These static AssetBase cuboids have no rigid-body root. PhysX filters
        # must name their actual collision mesh (shapes.py), not the Xform.
        table_filters = [self.scene.protective_board.prim_path + "/geometry/mesh",
                         self.scene.table.prim_path + "/geometry/mesh"]
        for name, source, targets in (
            ("placement_cube_left", cube, [robot + "/panda_leftfinger"]),
            ("placement_cube_right", cube, [robot + "/panda_rightfinger"]),
            ("placement_cube_table", cube, table_filters),
            ("placement_left_table", robot + "/panda_leftfinger", table_filters),
            ("placement_right_table", robot + "/panda_rightfinger", table_filters),
        ):
            setattr(self.scene, name, ContactSensorCfg(
                prim_path=source, filter_prim_paths_expr=targets,
                update_period=0.0, history_length=0))

        old_goal = self.commands.object_pose
        self.commands.object_pose = PlacementGoalCommandCfg(
            asset_name="robot", body_name="panda_hand", debug_vis=False,
            resampling_time_range=(1e9, 1e9), ranges=deepcopy(old_goal.ranges))
        for axis, value in (("pos_x", 0.45), ("pos_y", 0.), ("pos_z", 0.0505)):
            setattr(self.commands.object_pose.ranges, axis, (value, value))
        old_gripper = self.actions.gripper_action
        self.actions.gripper_action = PlacementGripperActionCfg(
            asset_name=old_gripper.asset_name, joint_names=deepcopy(old_gripper.joint_names),
            open_command_expr=deepcopy(old_gripper.open_command_expr),
            close_command_expr=deepcopy(old_gripper.close_command_expr))
        self.rewards = PlacementRewardsCfg()
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None
        self.terminations.object_dropping = TerminationTermCfg(func=placement_dropped)
        self.episode_length_s = 3.0
        self.sim.dt = 1 / 60
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.rerender_on_reset = True
        # Keep inherited image_proprio untouched: RGB84 + q7/dq7/prev8/handz1/goalxyz3.
        # No reward state, contact, width, lift flag, or success enters observations.


@configclass
class PregraspedPlacementEnvCfg(_PlacementMixin, Vertical15Table55EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_placement()


@configclass
class PregraspedPlacementEvalEnvCfg(_PlacementMixin, Vertical15Table55EvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_placement()
