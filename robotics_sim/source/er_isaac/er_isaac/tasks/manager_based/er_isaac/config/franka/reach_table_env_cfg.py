"""Reach training and evaluation with the Place Cube table geometry."""

from copy import deepcopy

from isaaclab.utils import configclass

from .goal_fixed_scale_env_cfg import (
    FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyGoalFixedScaleEnvCfg,
    FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyEvalGoalFixedScaleEnvCfg,
)
from .cube_table_env_cfg import install_table55


@configclass
class FrankaReachGoalFixedScaleTableEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyGoalFixedScaleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene = deepcopy(self.scene)
        install_table55(self.scene)


@configclass
class FrankaReachGoalFixedScaleTableEvalEnvCfg(
    FrankaReachCNNRealAsIsOverlay2DProprioJointPositionControllerNoHeightPenaltyEvalGoalFixedScaleEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene = deepcopy(self.scene)
        install_table55(self.scene)
