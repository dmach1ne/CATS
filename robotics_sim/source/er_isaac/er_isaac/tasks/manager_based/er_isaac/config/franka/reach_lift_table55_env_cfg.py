"""Reach training and evaluation with the Place Cube table geometry."""

from copy import deepcopy

from isaaclab.utils import configclass

from .goal_fixed_scale_env_cfg import (
    FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyGoalFixedScaleEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyEvalGoalFixedScaleEnvCfg,
)
from .vertical15_table55_env_cfg import install_table55


@configclass
class FrankaReachGoalFixedScaleLiftTable55EnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyGoalFixedScaleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene = deepcopy(self.scene)
        install_table55(self.scene)


@configclass
class FrankaReachGoalFixedScaleLiftTable55EvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioLegacyControllerNoHeightPenaltyEvalGoalFixedScaleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene = deepcopy(self.scene)
        install_table55(self.scene)
