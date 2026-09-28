# 학습된 정책 평가용 환경.
#
# 학습 환경(ER-Reach-Franka-CNN-RealAsIs-v0)과 **정책이 보는 것이 완전히 동일**해야 한다.
# 그래서 씬에 물체를 추가하지 않는다 — 초록 목표 마커 같은 걸 넣으면 84×84 관측이
# 오염되어 학습 때와 다른 입력이 된다. 목표 위치는 영상에서 사후 투영 십자선으로만 그린다.
#
# 추가하는 것은 관찰용 카메라 하나뿐이다. 카메라는 센서라서 다른 카메라 이미지에
# 찍히지 않으므로 정책 관측에 영향이 없다.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .ik_video_env_cfg import FULL_FOCAL, FULL_H, FULL_W
from .real_cam_env_cfg import (
    REAL_APERTURE,
    REAL_CAM_POS,
    REAL_CAM_ROT_ASIS,
    FrankaReachCNNRealAsIsEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DLowZNoHeightPenaltyEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DLowZRewardPenaltyNoHeightPenaltyEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DProprioEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerNoHeightPenaltyEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DRewardPenaltyEnvCfg,
    RealCamSceneCfg,
)


@configclass
class EvalSceneCfg(RealCamSceneCfg):
    """학습 씬 그대로 + 관찰용 전체화각 카메라(실물 1280×720 화각을 960×540 으로)."""

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
class FrankaReachCNNRealAsIsEvalEnvCfg(FrankaReachCNNRealAsIsEnvCfg):
    scene: EvalSceneCfg = EvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class Overlay2DEvalSceneCfg(EvalSceneCfg):
    target_viz: RigidObjectCfg | None = None


@configclass
class FrankaReachCNNRealAsIsOverlay2DEvalEnvCfg(FrankaReachCNNRealAsIsOverlay2DEnvCfg):
    scene: Overlay2DEvalSceneCfg = Overlay2DEvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class FrankaReachCNNRealAsIsOverlay2DLowZNoHeightPenaltyEvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DLowZNoHeightPenaltyEnvCfg
):
    scene: Overlay2DEvalSceneCfg = Overlay2DEvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class FrankaReachCNNRealAsIsOverlay2DLowZRewardPenaltyNoHeightPenaltyEvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DLowZRewardPenaltyNoHeightPenaltyEnvCfg
):
    scene: Overlay2DEvalSceneCfg = Overlay2DEvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class FrankaReachCNNRealAsIsOverlay2DRewardPenaltyEvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DRewardPenaltyEnvCfg
):
    scene: Overlay2DEvalSceneCfg = Overlay2DEvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioEvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioEnvCfg
):
    scene: Overlay2DEvalSceneCfg = Overlay2DEvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyEvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyEnvCfg
):
    scene: Overlay2DEvalSceneCfg = Overlay2DEvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerEvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerEnvCfg
):
    scene: Overlay2DEvalSceneCfg = Overlay2DEvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerEvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerEnvCfg
):
    scene: Overlay2DEvalSceneCfg = Overlay2DEvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyEvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyEnvCfg
):
    scene: Overlay2DEvalSceneCfg = Overlay2DEvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerNoHeightPenaltyEvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerNoHeightPenaltyEnvCfg
):
    scene: Overlay2DEvalSceneCfg = Overlay2DEvalSceneCfg(num_envs=1, env_spacing=10)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
