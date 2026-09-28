"""SAC entry point for direct policy-output sensitivity evaluation."""

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "sac"))
sys.path.insert(0, str(_REPO_ROOT / "sac" / "tests"))

from sac.models.custom_sac import CustomSAC
from sac.models.l2c2_sac import L2C2SAC
from sac.tests.modules.envs import make_ant_env, make_hopper_env, make_lunar_env, make_pendulum_env, make_walker_env
from tools.policy_output_sensitivity import run_evaluation


ENV_FACTORIES = {'ant': make_ant_env, 'hopper': make_hopper_env, 'lunar': make_lunar_env, 'pendulum': make_pendulum_env, 'walker': make_walker_env}


if __name__ == "__main__":
    run_evaluation(
        "sac", CustomSAC, ENV_FACTORIES,
        method_model_classes={"l2c2_sac": L2C2SAC},
    )
