# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents

for _placement_id, _placement_cfg in (
    ("ER-Place-Franka-Pregrasped-Table55-GoalFixedScale-RGB84-Front25Transport10-v0", "PregraspedPlacementEnvCfg"),
    ("ER-Place-Franka-Pregrasped-Table55-GoalFixedScale-RGB84-Front25Transport10-Eval-v0", "PregraspedPlacementEvalEnvCfg"),
):
    gym.register(
        id=_placement_id,
        entry_point="er_isaac.tasks.manager_based.er_isaac.mdp.placement:PregraspedPlacementEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.pregrasped_placement_env_cfg:{_placement_cfg}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_visual_lift_compact_proprio_goal_fixed_scale_ppo_cfg.yaml",
            "rl_games_goal_fixed_scale_cfg_entry_point": f"{agents.__name__}:rl_games_visual_lift_compact_proprio_goal_fixed_scale_ppo_cfg.yaml",
        },
    )

gym.register(
    id="ER-Reach-Franka-CNN-RealAsIs-Overlay2D-Proprio-LegacyController-NoHeightPenalty-GoalFixedScale-LiftTable55-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_lift_table55_env_cfg:FrankaReachGoalFixedScaleLiftTable55EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_visual_reach_compact_proprio_goal_fixed_scale_ppo_cfg.yaml",
        "rl_games_goal_fixed_scale_cfg_entry_point": f"{agents.__name__}:rl_games_visual_reach_compact_proprio_goal_fixed_scale_ppo_cfg.yaml",
    },
)

gym.register(
    id="ER-Reach-Franka-CNN-RealAsIs-Overlay2D-Proprio-Eval-LegacyController-NoHeightPenalty-GoalFixedScale-LiftTable55-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_lift_table55_env_cfg:FrankaReachGoalFixedScaleLiftTable55EvalEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_visual_reach_compact_proprio_goal_fixed_scale_ppo_cfg.yaml",
        "rl_games_goal_fixed_scale_cfg_entry_point": f"{agents.__name__}:rl_games_visual_reach_compact_proprio_goal_fixed_scale_ppo_cfg.yaml",
    },
)
