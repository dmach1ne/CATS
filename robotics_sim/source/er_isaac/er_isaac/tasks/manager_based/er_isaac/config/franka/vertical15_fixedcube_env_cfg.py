"""GoalFixedScale with a fixed cube and a stationary target 15 cm above it."""

from copy import deepcopy

from isaaclab.utils import configclass

from .goal_fixed_scale_env_cfg import (
    FrankaVisualLiftOverlay2DProprioLegacyControllerGoalFixedScaleEnvCfg,
    FrankaVisualLiftOverlay2DProprioLegacyControllerEvalGoalFixedScaleEnvCfg,
    FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerGoalFixedScaleEnvCfg,
    FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerEvalGoalFixedScaleEnvCfg,
)


class _Vertical15FixedCubeMixin:
    def _configure_vertical15_fixed_cube(self):
        for name in ("scene", "commands", "events", "rewards"):
            setattr(self, name, deepcopy(getattr(self, name)))
        # Pose commands are robot-base-relative, while object spawn is env-local.
        if tuple(self.scene.robot.init_state.rot) != (1.0, 0.0, 0.0, 0.0):
            raise ValueError("Vertical15FixedCube requires an unrotated robot base")
        table_z = self.scene.table.init_state.pos[2]
        cube_pos = (0.45, 0.0, table_z + 0.049 / 2)
        self.scene.object.spawn.size = (0.049,) * 3
        self.scene.object.spawn.visual_material.diffuse_color = (237 / 255, 214 / 255, 53 / 255)
        self.scene.object.init_state.pos = cube_pos
        self.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.object.init_state.lin_vel = (0.0, 0.0, 0.0)
        self.scene.object.init_state.ang_vel = (0.0, 0.0, 0.0)
        reset = self.events.reset_object_position.params
        reset["pose_range"] = {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
        reset["velocity_range"] = {}
        target = tuple(p - b + (0.15 if i == 2 else 0.0)
                       for i, (p, b) in enumerate(zip(cube_pos, self.scene.robot.init_state.pos)))
        for axis, value in zip(("pos_x", "pos_y", "pos_z"), target):
            setattr(self.commands.object_pose.ranges, axis, (value, value))
        # Re-sampling returns the same target; it never follows the moving cube.
        self.commands.object_pose.resampling_time_range = (6.0, 6.0)
        self.rewards.lifting_object.weight = 15.0
        self.rewards.lifting_object.params["minimum_height"] = cube_pos[2]
        self.rewards.lifting_object.params["maximum_height"] = table_z + 0.10


@configclass
class Vertical15FixedCubeEnvCfg(_Vertical15FixedCubeMixin, FrankaVisualLiftOverlay2DProprioLegacyControllerGoalFixedScaleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_vertical15_fixed_cube()


@configclass
class Vertical15FixedCubeEvalEnvCfg(_Vertical15FixedCubeMixin, FrankaVisualLiftOverlay2DProprioLegacyControllerEvalGoalFixedScaleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_vertical15_fixed_cube()


@configclass
class Vertical15FixedCubeRewardPenaltyEnvCfg(_Vertical15FixedCubeMixin, FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerGoalFixedScaleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_vertical15_fixed_cube()


@configclass
class Vertical15FixedCubeRewardPenaltyEvalEnvCfg(_Vertical15FixedCubeMixin, FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerEvalGoalFixedScaleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_vertical15_fixed_cube()
