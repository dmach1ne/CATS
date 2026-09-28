from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import (
    EventCfg as LiftEventCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import ObjectTableSceneCfg

from ... import mdp
from .ik_video_env_cfg import FULL_FOCAL, FULL_H, FULL_W
from .real_cam_env_cfg import (
    REAL_APERTURE,
    REAL_CAM_POS,
    REAL_CAM_ROT_ASIS,
    REAL_CAM_ROT_XYZ_DEG,
    REAL_FOCAL,
    POLICY_CAM_CX_PX,
    POLICY_CAM_CY_PX,
    POLICY_CAM_FX_PX,
    POLICY_CAM_FY_PX,
    TARGET_MARKER_RADIUS_M,
    LIFT_TARGET_Z_RANGE,
    _make_camera,
    _make_floor,
    _make_wall,
)
from .joint_pos_camera_env_cfg import apply_legacy_panda_joint_position_controller


RESTING_OBJECT_ROOT_Z_M = 0.025
LIFT_REWARD_FULL_HEIGHT_M = 0.10
LIFT_OBJECT_SIZE_M = 0.05
LIFT_OBJECT_COLOR = (237 / 255, 214 / 255, 53 / 255)  # #EDD635


def _make_target_marker() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetViz",
        spawn=sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.3)),
    )


@configclass
class VisualLiftSceneCfg(ObjectTableSceneCfg):
    """Standard Lift scene with the Visual Reach camera and visual-only target."""

    tiled_camera: TiledCameraCfg = _make_camera(REAL_CAM_ROT_ASIS)
    target_viz: RigidObjectCfg = _make_target_marker()
    black_wall: RigidObjectCfg = _make_wall()
    color_floor: RigidObjectCfg = _make_floor()


@configclass
class VisualLiftEvalSceneCfg(VisualLiftSceneCfg):
    """Training scene plus a recording-only camera with the full real-camera field of view."""

    full_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/FullCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=REAL_CAM_POS, rot=REAL_CAM_ROT_ASIS, convention="opengl"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=FULL_FOCAL,
            focus_distance=400.0,
            horizontal_aperture=REAL_APERTURE,
            clipping_range=(0.05, 20.0),
        ),
        width=FULL_W,
        height=FULL_H,
    )


@configclass
class VisualLiftOverlay2DSceneCfg(VisualLiftSceneCfg):
    """Training scene without the rendered 3-D target marker."""

    target_viz: RigidObjectCfg | None = None


@configclass
class VisualLiftOverlay2DEvalSceneCfg(VisualLiftEvalSceneCfg):
    """Evaluation scene without the rendered 3-D target marker."""

    target_viz: RigidObjectCfg | None = None


@configclass
class VisualLiftObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.image_stack,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "stack_num": 3,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class VisualLiftProprioObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        image_proprio = ObsTerm(
            func=mdp.visual_lift_image_proprio,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
                "ee_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
                "data_type": "rgb",
                "command_name": "object_pose",
                "camera_translation": REAL_CAM_POS,
                "camera_rotation_xyz_deg": REAL_CAM_ROT_XYZ_DEG,
                "camera_fx_px": POLICY_CAM_FX_PX,
                "camera_fy_px": POLICY_CAM_FY_PX,
                "camera_cx_px": POLICY_CAM_CX_PX,
                "camera_cy_px": POLICY_CAM_CY_PX,
                "clipping_range": (0.05, 20.0),
                "target_marker_radius_m": TARGET_MARKER_RADIUS_M,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class VisualLiftEventCfg(LiftEventCfg):
    sync_target_viz_to_object_pose = EventTerm(
        func=mdp.set_marker_pose_from_command,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        params={"command_name": "object_pose"},
    )


@configclass
class FrankaVisualLiftEnvCfg(FrankaCubeLiftEnvCfg):
    """Standard-reward Visual Lift configuration used by the reference variant."""

    scene: VisualLiftSceneCfg = VisualLiftSceneCfg(num_envs=128, env_spacing=10.0)
    observations: VisualLiftObservationsCfg = VisualLiftObservationsCfg()
    events: VisualLiftEventCfg = VisualLiftEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # UsdFileCfg has no size field; spawn matching visual/collision geometry.
        object_spawn = self.scene.object.spawn
        self.scene.object.spawn = sim_utils.CuboidCfg(
            size=(LIFT_OBJECT_SIZE_M,) * 3,
            rigid_props=object_spawn.rigid_props,
            mass_props=object_spawn.mass_props,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=LIFT_OBJECT_COLOR),
        )
        # Use the legacy Panda joint-position controller for the default Lift
        # task, matching the controller used by the current training protocol.
        apply_legacy_panda_joint_position_controller(self)
        self.commands.object_pose.debug_vis = False
        # Match the Visual Reach policy and camera target rate: 60 Hz physics / 2 = 30 Hz.
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 6.0
        # Use the lower Lift-specific range appropriate for grasp-and-lift training.
        self.commands.object_pose.ranges.pos_x = (0.45, 0.45)
        self.commands.object_pose.ranges.pos_z = LIFT_TARGET_Z_RANGE
        self.commands.object_pose.resampling_time_range = (6.0, 6.0)
        # Place the 5 cm cube bottom at the table plane (z=0).
        object_init_pos = self.scene.object.init_state.pos
        self.scene.object.init_state.pos = (
            object_init_pos[0],
            object_init_pos[1],
            RESTING_OBJECT_ROOT_Z_M,
        )
        # Dense lift credit: 0 at table contact and the inherited full reward
        # at 10 cm, so reset cannot receive lift reward.
        self.rewards.lifting_object = RewTerm(
            func=mdp.object_lift_height_linear,
            weight=self.rewards.lifting_object.weight,
            params={
                "minimum_height": RESTING_OBJECT_ROOT_Z_M,
                "maximum_height": LIFT_REWARD_FULL_HEIGHT_M,
            },
        )
        self.rewards.track_min_object_goal_distance = RewTerm(
            func=mdp.MinimumObjectGoalDistanceLogger,
            weight=1.0,
        )


@configclass
class FrankaVisualLiftRewardPenaltyEnvCfg(FrankaVisualLiftEnvCfg):
    """Reference Visual Lift retaining Isaac Lab's action/velocity penalties."""


@configclass
class FrankaVisualLiftBaseEnvCfg(FrankaVisualLiftEnvCfg):
    """Primary Base Visual Lift configuration without reward regularizers."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate = None
        self.rewards.joint_vel = None
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


@configclass
class FrankaVisualLiftOverlay2DEnvCfg(FrankaVisualLiftEnvCfg):
    """Visual Lift with the target composited into each policy frame."""

    scene: VisualLiftOverlay2DSceneCfg = VisualLiftOverlay2DSceneCfg(
        num_envs=128, env_spacing=10.0
    )

    def __post_init__(self):
        super().__post_init__()
        self.events.sync_target_viz_to_object_pose = None
        image_term = getattr(self.observations.policy, "image", None)
        if image_term is not None:
            self.observations.policy.image.func = mdp.image_stack_with_target_overlay
            self.observations.policy.image.params = {
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "stack_num": 3,
                "command_name": "object_pose",
                "camera_translation": REAL_CAM_POS,
                "camera_rotation_xyz_deg": REAL_CAM_ROT_XYZ_DEG,
                "camera_fx_px": POLICY_CAM_FX_PX,
                "camera_fy_px": POLICY_CAM_FY_PX,
                "camera_cx_px": POLICY_CAM_CX_PX,
                "camera_cy_px": POLICY_CAM_CY_PX,
                "clipping_range": (0.05, 20.0),
                "target_marker_radius_m": TARGET_MARKER_RADIUS_M,
            }


@configclass
class FrankaVisualLiftOverlay2DRewardPenaltyEnvCfg(FrankaVisualLiftOverlay2DEnvCfg):
    """Overlay2D Lift retaining Isaac Lab's action and velocity penalties."""


@configclass
class FrankaVisualLiftOverlay2DBaseEnvCfg(FrankaVisualLiftOverlay2DEnvCfg):
    """Primary Overlay2D Lift without reward regularizers."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate = None
        self.rewards.joint_vel = None
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


@configclass
class FrankaVisualLiftOverlay2DProprioEnvCfg(FrankaVisualLiftOverlay2DBaseEnvCfg):
    """Overlay2D Base with a single RGB frame and 23-D proprioception."""

    observations: VisualLiftProprioObservationsCfg = VisualLiftProprioObservationsCfg()


@configclass
class FrankaVisualLiftOverlay2DProprioRewardPenaltyEnvCfg(
    FrankaVisualLiftOverlay2DRewardPenaltyEnvCfg
):
    """Overlay2D reward-penalty reference with raw proprioception."""

    observations: VisualLiftProprioObservationsCfg = VisualLiftProprioObservationsCfg()


@configclass
class FrankaVisualLiftOverlay2DProprioLegacyControllerEnvCfg(
    FrankaVisualLiftOverlay2DProprioEnvCfg
):
    """Current visual Lift task with the pre-parity Panda arm controller."""

    def __post_init__(self):
        super().__post_init__()
        apply_legacy_panda_joint_position_controller(self)


@configclass
class FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerEnvCfg(
    FrankaVisualLiftOverlay2DProprioRewardPenaltyEnvCfg
):
    """Reward-penalty visual Lift task with the pre-parity arm controller."""

    def __post_init__(self):
        super().__post_init__()
        apply_legacy_panda_joint_position_controller(self)


@configclass
class FrankaVisualLiftEvalEnvCfg(FrankaVisualLiftRewardPenaltyEnvCfg):
    scene: VisualLiftEvalSceneCfg = VisualLiftEvalSceneCfg(num_envs=50, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaVisualLiftRewardPenaltyEvalEnvCfg(FrankaVisualLiftEvalEnvCfg):
    """Evaluation configuration for the reward-penalty reference."""


@configclass
class FrankaVisualLiftBaseEvalEnvCfg(FrankaVisualLiftBaseEnvCfg):
    """Evaluation configuration for the primary no-penalty Base."""

    scene: VisualLiftEvalSceneCfg = VisualLiftEvalSceneCfg(num_envs=50, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaVisualLiftOverlay2DRewardPenaltyEvalEnvCfg(
    FrankaVisualLiftOverlay2DRewardPenaltyEnvCfg
):
    scene: VisualLiftOverlay2DEvalSceneCfg = VisualLiftOverlay2DEvalSceneCfg(
        num_envs=50, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaVisualLiftOverlay2DBaseEvalEnvCfg(FrankaVisualLiftOverlay2DBaseEnvCfg):
    scene: VisualLiftOverlay2DEvalSceneCfg = VisualLiftOverlay2DEvalSceneCfg(
        num_envs=50, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaVisualLiftOverlay2DProprioEvalEnvCfg(
    FrankaVisualLiftOverlay2DProprioEnvCfg
):
    """Evaluation configuration for the Overlay2D proprio policy."""

    scene: VisualLiftOverlay2DEvalSceneCfg = VisualLiftOverlay2DEvalSceneCfg(
        num_envs=50, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaVisualLiftOverlay2DProprioRewardPenaltyEvalEnvCfg(
    FrankaVisualLiftOverlay2DProprioRewardPenaltyEnvCfg
):
    """Evaluation configuration for the proprio reward-penalty reference."""

    scene: VisualLiftOverlay2DEvalSceneCfg = VisualLiftOverlay2DEvalSceneCfg(
        num_envs=50, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaVisualLiftOverlay2DProprioLegacyControllerEvalEnvCfg(
    FrankaVisualLiftOverlay2DProprioLegacyControllerEnvCfg
):
    """Evaluation configuration for the legacy-controller proprio policy."""

    scene: VisualLiftOverlay2DEvalSceneCfg = VisualLiftOverlay2DEvalSceneCfg(
        num_envs=50, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerEvalEnvCfg(
    FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerEnvCfg
):
    """Evaluation configuration for the legacy reward-penalty proprio policy."""

    scene: VisualLiftOverlay2DEvalSceneCfg = VisualLiftOverlay2DEvalSceneCfg(
        num_envs=50, env_spacing=2.5
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
