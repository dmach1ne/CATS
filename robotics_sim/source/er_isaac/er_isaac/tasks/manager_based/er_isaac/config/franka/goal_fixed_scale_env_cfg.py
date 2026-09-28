"""Opt-in goal observations; all controller, reward and eval settings are inherited."""

from copy import deepcopy

from isaaclab.utils import configclass
from .real_cam_env_cfg import FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyEnvCfg
from .eval_env_cfg import FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyEvalEnvCfg
from .real_cam_env_cfg import FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyLegacyControllerNoHeightPenaltyEnvCfg
from .eval_env_cfg import FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyLegacyControllerNoHeightPenaltyEvalEnvCfg
from .visual_lift_env_cfg import FrankaVisualLiftOverlay2DProprioLegacyControllerEnvCfg
from .visual_lift_env_cfg import FrankaVisualLiftOverlay2DProprioLegacyControllerEvalEnvCfg
from .visual_lift_env_cfg import FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerEnvCfg
from .visual_lift_env_cfg import FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerEvalEnvCfg


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyGoalFixedScaleEnvCfg(FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyEvalGoalFixedScaleEnvCfg(FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyLegacyControllerNoHeightPenaltyGoalFixedScaleEnvCfg(FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyLegacyControllerNoHeightPenaltyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyLegacyControllerNoHeightPenaltyEvalGoalFixedScaleEnvCfg(FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyLegacyControllerNoHeightPenaltyEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaVisualLiftOverlay2DProprioLegacyControllerGoalFixedScaleEnvCfg(FrankaVisualLiftOverlay2DProprioLegacyControllerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaVisualLiftOverlay2DProprioLegacyControllerEvalGoalFixedScaleEnvCfg(FrankaVisualLiftOverlay2DProprioLegacyControllerEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerGoalFixedScaleEnvCfg(FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerEvalGoalFixedScaleEnvCfg(FrankaVisualLiftOverlay2DProprioRewardPenaltyLegacyControllerEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True
