"""PPO L2C2 entry point for compact resumable Lsp rollout collection."""

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "ppo"))
sys.path.insert(0, str(_REPO_ROOT / "ppo" / "tests"))

from ppo.models.custom_ppo import L2C2PPO
from ppo.tests.modules.envs import make_ant_env, make_hopper_env, make_lunar_env, make_pendulum_env, make_walker_env
from tools.lsp_compact_collector import run_compact_collection


ENV_FACTORIES = {'ant': make_ant_env, 'hopper': make_hopper_env, 'lunar': make_lunar_env, 'pendulum': make_pendulum_env, 'walker': make_walker_env}


if __name__ == "__main__":
    run_compact_collection("ppo", L2C2PPO, ENV_FACTORIES)
