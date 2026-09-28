# reverse 브랜치 태스크(ER-Reach-Franka-CNN-v0)를 그대로 두고
#   ① 카메라를 우리 robotic_manipulation setting3 실물 렌즈로 교체
#   ② 배경(벽·바닥)에 색을 입힘  ← 흰 배경이면 DART 증강 시 팔이 사라지는 문제
# 만 바꾼 변형. 보상·행동·커맨드·종료조건·관측구조는 일절 손대지 않는다.
#
# 카메라 내부파라미터(고정, '실물 세팅'의 본체):
#   L515 f1182366 RGB 640×480 EEPROM calibration
#   → center-crop 480×480 → resize 84×84 로 변환한 OpenCV K를 사용한다.
#
# Isaac Lab 2.2.1의 PinholeCameraCfg.from_intrinsic_matrix()는 fx/fy를 평균내고
# cx/cy offset을 무시한다. Policy input과 2D overlay는 이 유효 pinhole intrinsic에
# 맞추며 distortion은 simulation-to-real 계약에서 사용하지 않는다.
#
# 카메라 외부파라미터 두 안:
#   Aimed — 위치·렌즈 그대로, 시선만 reverse 워크스페이스 중심(0.45,0,0.325)으로.
#           타깃 범위 9개 모서리 전부 최악 16.29° (반화각 21.85°) → 전부 화면 안.
#   AsIs  — setting3 자세를 문자 그대로 이식. reverse 타깃 z=0.5 구간이 25.2~28.5°로
#           화각을 벗어난다. 대조·검증용.
#
# 주의: reverse 의 sync_target_viz_to_ee_pose 는 카메라→타깃 직선의 30% 지점에
# 빨간 마커를 놓는다. 따라서 camera_pos_raw 를 카메라 위치와 **반드시** 일치시켜야
# 마커가 화면상 올바른 자리에 찍힌다.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from ... import mdp
from ...er_isaac_env_cfg import ReachRGBCameraSceneCfg
from ...reach_cfg import RewardPenaltyCurriculumCfg, RewardPenaltyRewardsCfg
from .joint_pos_camera_env_cfg import (
    FrankaReachCNNEnvCfg,
    apply_legacy_panda_joint_position_controller,
)

# ── 실물 렌즈(고정) ──────────────────────────────────────────────────────────
REAL_CAM_POS = (1.29457219, -0.00127079938, 0.89823765)
REAL_CAM_ROT_ASIS = (
    0.657173792825192,
    0.280839883312581,
    0.263454202181990,
    0.647953277109838,
)
REAL_CAM_ROT_ROS_OPTICAL = (
    0.280839883312581,
    -0.657173792825192,
    -0.647953277109838,
    0.263454202181990,
)
REAL_CAM_ROT_AIMED = (0.620017, 0.317306, 0.326904, 0.638772)
REAL_FOCAL = 26.1270
REAL_APERTURE = 20.955
REAL_CAM_ROT_XYZ_DEG = (2.25576561, 45.25221894, 88.25015104)
REAL_CAM_IMAGE_SIZE = (84, 84)
REAL_CAM_FX_PX = 104.55361328125
REAL_CAM_FY_PX = 104.574462890625
REAL_CAM_CX_PX = 44.3632637023926
REAL_CAM_CY_PX = 42.7241527557373
REAL_CAM_INTRINSIC = (
    REAL_CAM_FX_PX,
    0.0,
    REAL_CAM_CX_PX,
    0.0,
    REAL_CAM_FY_PX,
    REAL_CAM_CY_PX,
    0.0,
    0.0,
    1.0,
)
# Isaac/Omniverse uses the mean focal length and a centered principal point.
POLICY_CAM_FX_PX = 104.5640380859375
POLICY_CAM_FY_PX = 104.5640380859375
POLICY_CAM_CX_PX = 42.0
POLICY_CAM_CY_PX = 42.0
REAL_CAM_CLIPPING_RANGE = (0.05, 20.0)
TARGET_MARKER_RADIUS_M = 0.025

# ── 배경색 ───────────────────────────────────────────────────────────────────
# 흰색이면 흰 Franka 가 배경에 묻힌다(DART 증강 시 팔이 사라지는 원인).
WALL_COLOR = (0.31, 0.40, 0.52)    # 중간 톤 청회색
FLOOR_COLOR = (0.46, 0.42, 0.36)   # 따뜻한 중간 톤

# Low-Reach is the default Reach target-height protocol. Lift has a separate,
# lower range because it must first grasp the cube from the table.
REACH_TARGET_Z_RANGE = (0.25, 0.35)
REACH_LOW_TARGET_Z_RANGE = REACH_TARGET_Z_RANGE
LIFT_TARGET_Z_RANGE = (0.10, 0.20)

_KINEMATIC = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
)


def _make_wall() -> RigidObjectCfg:
    """원본 black_wall 과 형상·위치 동일, 색만 입힌 것."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlackWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 6.0, 6.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=WALL_COLOR, roughness=0.8, metallic=0.0
            ),
            rigid_props=_KINEMATIC,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _make_floor() -> RigidObjectCfg:
    """내려다보는 카메라의 프레임을 채우는 시각 전용 바닥.

    reverse 는 ground 를 제거했다(수평 카메라의 시야를 가려서). 우리 카메라는
    내려다보므로 바닥이 없으면 프레임 상당 부분이 빈 배경이 된다. 충돌은 끈다.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ColorFloor",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=FLOOR_COLOR, roughness=0.85, metallic=0.0
            ),
            rigid_props=_KINEMATIC,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.02), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _make_camera(rot) -> TiledCameraCfg:
    width, height = REAL_CAM_IMAGE_SIZE
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=REAL_CAM_POS, rot=rot, convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=list(REAL_CAM_INTRINSIC),
            width=width,
            height=height,
            focal_length=REAL_FOCAL,
            focus_distance=400.0,
            clipping_range=REAL_CAM_CLIPPING_RANGE,
        ),
        width=width,
        height=height,
    )


@configclass
class RealCamSceneCfg(ReachRGBCameraSceneCfg):
    """원본 씬 + 실물 카메라 + 색 배경(벽 덮어쓰기, 바닥 추가)."""

    tiled_camera: TiledCameraCfg = _make_camera(REAL_CAM_ROT_AIMED)
    black_wall: RigidObjectCfg = _make_wall()
    color_floor: RigidObjectCfg = _make_floor()


@configclass
class FrankaReachCNNRealEnvCfg(FrankaReachCNNEnvCfg):
    """실물 렌즈 + 색 배경. 시선은 reverse 워크스페이스를 겨냥(권장)."""

    scene: RealCamSceneCfg = RealCamSceneCfg(num_envs=128, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        # 마커 투영 기준점을 카메라 위치와 일치시킨다(안 맞추면 마커가 엉뚱한 곳에 찍힘)
        self.events.sync_target_viz_to_ee_pose.params["camera_pos_raw"] = REAL_CAM_POS

        # Preserve the established Overlay2D Reach timing independently of Lift.
        self.episode_length_s = 18.0
        self.commands.ee_pose.resampling_time_range = (6.0, 6.0)

        # Keep the simulation and real evaluation target distributions identical.
        self.commands.ee_pose.ranges.pos_z = REACH_TARGET_Z_RANGE


@configclass
class FrankaReachCNNRealAsIsEnvCfg(FrankaReachCNNRealEnvCfg):
    """setting3 자세를 문자 그대로 이식한 대조군(타깃 상단이 화각을 벗어남)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.tiled_camera.offset = TiledCameraCfg.OffsetCfg(
            pos=REAL_CAM_POS, rot=REAL_CAM_ROT_ASIS, convention="opengl"
        )


@configclass
class RealCamOverlay2DSceneCfg(RealCamSceneCfg):
    """RealAsIs scene without a rendered target object."""

    target_viz: RigidObjectCfg | None = None


@configclass
class VisualReachProprioObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        image_proprio = ObsTerm(
            func=mdp.visual_reach_image_proprio,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "robot_cfg": SceneEntityCfg(
                    "robot", joint_names=["panda_joint.*"]
                ),
                "ee_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
                "data_type": "rgb",
                "command_name": "ee_pose",
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
class FrankaReachCNNRealAsIsOverlay2DEnvCfg(FrankaReachCNNRealAsIsEnvCfg):
    """Reach observation with the ASAP_real 2D target overlay."""

    scene: RealCamOverlay2DSceneCfg = RealCamOverlay2DSceneCfg(num_envs=128, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.events.sync_target_viz_to_ee_pose = None
        image_term = getattr(self.observations.policy, "image", None)
        if image_term is not None:
            image_term.func = mdp.image_with_target_overlay
            image_term.params = {
                "sensor_cfg": image_term.params["sensor_cfg"],
                "data_type": "rgb",
                "command_name": "ee_pose",
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
class FrankaReachCNNRealAsIsOverlay2DLowZNoHeightPenaltyEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DEnvCfg
):
    """Visual-only Overlay2D Reach with a lower target range and no height term."""

    def __post_init__(self):
        super().__post_init__()
        apply_legacy_panda_joint_position_controller(self)
        self.commands.ee_pose.ranges.pos_z = REACH_LOW_TARGET_Z_RANGE
        self.rewards.low_end_effector_height = None


@configclass
class FrankaReachCNNRealAsIsOverlay2DLowZRewardPenaltyNoHeightPenaltyEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DLowZNoHeightPenaltyEnvCfg
):
    """Low-Z visual-only Reach with the original action regularizers."""

    rewards: RewardPenaltyRewardsCfg = RewardPenaltyRewardsCfg()
    curriculum: RewardPenaltyCurriculumCfg = RewardPenaltyCurriculumCfg()


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DEnvCfg
):
    """Overlay2D Reach with one RGB frame and 22-D raw proprioception."""

    observations: VisualReachProprioObservationsCfg = VisualReachProprioObservationsCfg()


@configclass
class FrankaReachCNNRealAsIsOverlay2DRewardPenaltyEnvCfg(FrankaReachCNNRealAsIsOverlay2DEnvCfg):
    """Overlay2D Reach with the original reward regularizers."""

    rewards: RewardPenaltyRewardsCfg = RewardPenaltyRewardsCfg()
    curriculum: RewardPenaltyCurriculumCfg = RewardPenaltyCurriculumCfg()


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DRewardPenaltyEnvCfg
):
    """Reward-penalty Reach with the 22-D proprioception policy input."""

    observations: VisualReachProprioObservationsCfg = VisualReachProprioObservationsCfg()


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioEnvCfg
):
    """Current visual Reach task with the pre-parity Panda arm controller."""

    def __post_init__(self):
        super().__post_init__()
        apply_legacy_panda_joint_position_controller(self)


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyLegacyControllerEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyEnvCfg
):
    """Reward-penalty visual Reach task with the pre-parity arm controller."""

    def __post_init__(self):
        super().__post_init__()
        apply_legacy_panda_joint_position_controller(self)


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerEnvCfg
):
    """Legacy-controller visual Reach without the low end-effector height term."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.low_end_effector_height = None


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyLegacyControllerNoHeightPenaltyEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyLegacyControllerEnvCfg
):
    """Legacy-controller Reward Penalty Reach without the height term."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.low_end_effector_height = None


@configclass
class FrankaReachCNNRealEnvCfg_PLAY(FrankaReachCNNRealEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
