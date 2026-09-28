# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents

for _placement_id, _placement_cfg in (
    ("Franka-PlaceCube-v0", "PregraspedPlacementEnvCfg"),
    ("Franka-PlaceCube-Eval-v0", "PregraspedPlacementEvalEnvCfg"),
):
    gym.register(
        id=_placement_id,
        entry_point="er_isaac.tasks.manager_based.er_isaac.mdp.placement:PregraspedPlacementEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.pregrasped_placement_env_cfg:{_placement_cfg}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:place_ppo.yaml",
            "rl_games_goal_fixed_scale_cfg_entry_point": f"{agents.__name__}:place_ppo.yaml",
        },
    )

gym.register(
    id="Franka-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_table_env_cfg:FrankaReachGoalFixedScaleTableEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:reach_ppo.yaml",
        "rl_games_goal_fixed_scale_cfg_entry_point": f"{agents.__name__}:reach_ppo.yaml",
    },
)

gym.register(
    id="Franka-Reach-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_table_env_cfg:FrankaReachGoalFixedScaleTableEvalEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:reach_ppo.yaml",
        "rl_games_goal_fixed_scale_cfg_entry_point": f"{agents.__name__}:reach_ppo.yaml",
    },
)
