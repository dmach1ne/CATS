"""Opt-in goal observations; all controller, reward and eval settings are inherited."""

from copy import deepcopy

from isaaclab.utils import configclass
from .real_cam_env_cfg import FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyEnvCfg
from .eval_env_cfg import FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyEvalEnvCfg
from .real_cam_env_cfg import FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerNoHeightPenaltyEnvCfg
from .eval_env_cfg import FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerNoHeightPenaltyEvalEnvCfg
from .cube_manipulation_env_cfg import FrankaCubeManipulationOverlay2DProprioJointPositionControllerEnvCfg
from .cube_manipulation_env_cfg import FrankaCubeManipulationOverlay2DProprioJointPositionControllerEvalEnvCfg
from .cube_manipulation_env_cfg import FrankaCubeManipulationOverlay2DProprioRewardPenaltyJointPositionControllerEnvCfg
from .cube_manipulation_env_cfg import FrankaCubeManipulationOverlay2DProprioRewardPenaltyJointPositionControllerEvalEnvCfg


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyGoalFixedScaleEnvCfg(FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyEvalGoalFixedScaleEnvCfg(FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerNoHeightPenaltyGoalFixedScaleEnvCfg(FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerNoHeightPenaltyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerNoHeightPenaltyEvalGoalFixedScaleEnvCfg(FrankaReachCNNRealAsIsOverlay2DProprioRewardPenaltyJointPositionControllerNoHeightPenaltyEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaCubeManipulationOverlay2DProprioJointPositionControllerGoalFixedScaleEnvCfg(FrankaCubeManipulationOverlay2DProprioJointPositionControllerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaCubeManipulationOverlay2DProprioJointPositionControllerEvalGoalFixedScaleEnvCfg(FrankaCubeManipulationOverlay2DProprioJointPositionControllerEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaCubeManipulationOverlay2DProprioRewardPenaltyJointPositionControllerGoalFixedScaleEnvCfg(FrankaCubeManipulationOverlay2DProprioRewardPenaltyJointPositionControllerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True


@configclass
class FrankaCubeManipulationOverlay2DProprioRewardPenaltyJointPositionControllerEvalGoalFixedScaleEnvCfg(FrankaCubeManipulationOverlay2DProprioRewardPenaltyJointPositionControllerEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations = deepcopy(self.observations)
        self.observations.policy.image_proprio.params["include_goal"] = True
