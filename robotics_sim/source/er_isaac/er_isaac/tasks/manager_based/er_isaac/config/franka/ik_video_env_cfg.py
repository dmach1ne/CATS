# 실물 캘리브레이션(setting3) 카메라 그대로 두고, 목표점을 IK 로 따라가는 모습을
# 영상으로 뽑기 위한 진단용 환경.
#
# 학습용이 아니다. 학습 대비 바뀐 것은 아래 넷뿐이고 씬 기하·카메라·보상은 그대로다.
#   ① 행동항  JointPosition → DifferentialInverseKinematics (절대 pose)
#   ② 로봇    FRANKA_PANDA_CFG → FRANKA_PANDA_HIGH_PD_CFG  (IK 추종용 강한 PD)
#   ③ 카메라  실물 1280×720 전체 화각을 재현한 관찰용 카메라를 하나 더 붙임
#   ④ 마커    진짜 목표 위치를 보여줄 초록 구를 하나 더 붙임
#
# ③ 보충 — 정책이 보는 84×84 는 실물 1280×720 의 **중앙 정사각 크롭**에 해당한다.
#   실물 fy=897.707, H=720  → 세로 화각 43.70°
#   실물 fx=897.029, W=1280 → 가로 화각 71.01°
#   960×540 (16:9) 로 같은 화각을 내려면 focal_length = 14.694, aperture = 20.955.
#   검산: fx = 960·14.694/20.955 = 673.1 → 1280 환산 897.4 (실물 897.03)
#         fy = 540·14.694/11.787 = 673.2 → 720 환산 897.6 (실물 897.71)
#
# ④ 보충 — 원래 있는 빨간 target_viz 는 목표가 아니다. reverse 는 카메라→목표 직선의
#   30% 지점에 그것을 놓는다(화면상 위치는 같고 크게 보이게 하는 장치). 그래서
#   진짜 목표가 어디인지 보려면 별도 마커가 필요하다.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from ...er_isaac_env_cfg import ReachRGBCameraSceneCfg
from .joint_pos_camera_env_cfg import FrankaReachCNNEnvCfg
from .real_cam_env_cfg import (
    REAL_APERTURE,
    REAL_CAM_POS,
    REAL_CAM_ROT_ASIS,
    FrankaReachCNNRealAsIsEnvCfg,
    RealCamSceneCfg,
)

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

# 실물 1280×720 전체 화각을 960×540 으로 재현
FULL_W, FULL_H = 960, 540
FULL_FOCAL = 14.694

_KINEMATIC = RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    disable_gravity=True,
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
)


@configclass
class IkVideoSceneCfg(RealCamSceneCfg):
    """실물 카메라 씬 + 관찰용 전체화각 카메라 + 진짜 목표 마커."""

    full_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/RealFullCamera",
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

    true_target_viz: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TrueTargetViz",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.25)),
            rigid_props=_KINEMATIC,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.325), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class FrankaReachIkVideoEnvCfg(FrankaReachCNNRealAsIsEnvCfg):
    """실물 캘리브레이션 카메라 + IK 구동. 영상 촬영 전용."""

    scene: IkVideoSceneCfg = IkVideoSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()

        # IK 추종을 위해 더 단단한 PD 로 교체(isaaclab 의 ik_abs 예시와 동일한 처방)
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 절대 pose 를 그대로 받는 IK 행동항.
        # body_offset 은 두지 않는다 — 태스크의 커맨드/보상이 panda_hand 기준이라
        # 그 프레임을 그대로 목표로 삼아야 태스크와 같은 것을 보게 된다.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=1.0,
        )

        self.observations.policy.enable_corruption = False


# ── 대조군: 친구분 원본 세팅 그대로 (카메라·배경·목표범위 전부 원본) ──────────
#   카메라 (2.0, 0, 0.4) ros · focal 24.0 → 화각 47.42° (정사각)
#   배경   검정 벽, 색 바닥 없음
#   목표   pos_z (0.15, 0.50) — 원본 그대로
# 관찰용 full_camera 는 같은 화각을 540×540 으로 키운 것뿐이다(잘라내는 것 없음).

ORIG_CAM_POS = (2.0, 0.0, 0.4)
ORIG_CAM_ROT = (-0.5, 0.5, 0.5, -0.5)
ORIG_FOCAL = 24.0
ORIG_APERTURE = 20.955


@configclass
class OrigIkVideoSceneCfg(ReachRGBCameraSceneCfg):
    """원본 씬 그대로 + 관찰용 고해상 카메라 + 진짜 목표 마커."""

    full_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/FullCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=ORIG_CAM_POS, rot=ORIG_CAM_ROT, convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=ORIG_FOCAL,
            focus_distance=400.0,
            horizontal_aperture=ORIG_APERTURE,
            clipping_range=(0.1, 20.0),
        ),
        width=540,
        height=540,
    )

    true_target_viz: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TrueTargetViz",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.25)),
            rigid_props=_KINEMATIC,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.325), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class FrankaReachOrigIkVideoEnvCfg(FrankaReachCNNEnvCfg):
    """친구분 원본 세팅 + IK 구동. 영상 촬영 전용."""

    scene: OrigIkVideoSceneCfg = OrigIkVideoSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=1.0,
        )
        self.observations.policy.enable_corruption = False
