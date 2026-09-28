# Direct simulator recipes

Use the Isaac Lab Python environment, accept its terms separately, and run from `robotics_sim/`.

```bash
export PYTHONPATH="$PWD/source/er_isaac:$PWD/scripts/rl_games:$PWD/scripts:$PYTHONPATH"
```

The commands below preserve the recorded task/config arguments. They are individual
original train/evaluation invocations, NOT a replacement batch launcher. Run the
training cohorts through your existing GPU scheduler. No serial-only workflow or
new seed restriction is imposed. Checkpoints and their saved agent.yaml are external.

Reach training seeds: 178132, 410580, 922852, 787576, 660993.
Place training seeds: 178132, 410580, 922852.
Evaluation seeds: 652165, 709797, 242820, 579787, 70701.

## Reach: BASE

```bash
python scripts/rl_games/train.py --task ER-Reach-Franka-CNN-RealAsIs-Overlay2D-Proprio-LegacyController-NoHeightPenalty-GoalFixedScale-v0 --agent rl_games_goal_fixed_scale_cfg_entry_point --mode base --seed "$SEED" --num_envs 128 --max_iterations 3000 --headless --enable_cameras
```

## Reach: CAPS

```bash
python scripts/rl_games/train.py --task ER-Reach-Franka-CNN-RealAsIs-Overlay2D-Proprio-LegacyController-NoHeightPenalty-GoalFixedScale-v0 --agent rl_games_goal_fixed_scale_cfg_entry_point --mode caps --seed "$SEED" --num_envs 128 --max_iterations 3000 --headless --enable_cameras --caps-lambda-t 0.5 --caps-lambda-s 0.1 --caps-sigma 0.2 --caps-preserve-target-overlay
```

## Reach: CATS

```bash
python scripts/rl_games/train.py --task ER-Reach-Franka-CNN-RealAsIs-Overlay2D-Proprio-LegacyController-NoHeightPenalty-GoalFixedScale-v0 --agent rl_games_goal_fixed_scale_cfg_entry_point --mode cats --seed "$SEED" --num_envs 128 --max_iterations 3000 --headless --enable_cameras --alpha-max 5.0
```

## Reach: evaluation

```bash
python scripts/eval_success.py --checkpoint "$CHECKPOINT" --agent-cfg "$AGENT_CFG" --task ER-Reach-Franka-CNN-RealAsIs-Overlay2D-Proprio-Eval-LegacyController-NoHeightPenalty-GoalFixedScale-v0 --headless --enable_cameras --targets 30 --eval-seeds 652165 709797 242820 579787 70701 --out evaluation.mp4 --summary summary.json
```

## Place: BASE

```bash
python scripts/rl_games/train.py --task ER-Place-Franka-Pregrasped-Table55-GoalFixedScale-RGB84-Front25Transport10-v0 --agent rl_games_goal_fixed_scale_cfg_entry_point --mode base --seed "$SEED" --num_envs 512 --max_iterations 1500 --headless --enable_cameras agent.params.config.horizon_length=50 agent.params.config.save_frequency=100 agent.params.config.entropy_coef=0.0 agent.params.config.lr_schedule=null agent.params.config.learning_rate=0.0003 agent.params.config.mini_epochs=4 agent.params.network.separate=false agent.params.config.gamma=0.8 agent.params.config.tau=0.9 agent.params.config.e_clip=0.2 agent.params.config.critic_coef=0.5 agent.params.config.grad_norm=0.5 agent.params.config.minibatch_size=800 'agent.params.network.image_shape=[84,84,3]'
```

## Place: CAPS

```bash
python scripts/rl_games/train.py --task ER-Place-Franka-Pregrasped-Table55-GoalFixedScale-RGB84-Front25Transport10-v0 --agent rl_games_goal_fixed_scale_cfg_entry_point --mode caps --seed "$SEED" --num_envs 512 --max_iterations 1500 --headless --enable_cameras --caps-lambda-t 0.5 --caps-lambda-s 0.1 --caps-sigma 0.2 agent.params.config.horizon_length=50 agent.params.config.save_frequency=100 agent.params.config.entropy_coef=0.0 agent.params.config.lr_schedule=null agent.params.config.learning_rate=0.0003 agent.params.config.mini_epochs=4 agent.params.network.separate=false agent.params.config.gamma=0.8 agent.params.config.tau=0.9 agent.params.config.e_clip=0.2 agent.params.config.critic_coef=0.5 agent.params.config.grad_norm=0.5 agent.params.config.minibatch_size=800 'agent.params.network.image_shape=[84,84,3]'
```

## Place: CATS

```bash
python scripts/rl_games/train.py --task ER-Place-Franka-Pregrasped-Table55-GoalFixedScale-RGB84-Front25Transport10-v0 --agent rl_games_goal_fixed_scale_cfg_entry_point --mode cats --seed "$SEED" --num_envs 512 --max_iterations 1500 --headless --enable_cameras --alpha-max 1.0 agent.params.config.horizon_length=50 agent.params.config.save_frequency=100 agent.params.config.entropy_coef=0.0 agent.params.config.lr_schedule=null agent.params.config.learning_rate=0.0003 agent.params.config.mini_epochs=4 agent.params.network.separate=false agent.params.config.gamma=0.8 agent.params.config.tau=0.9 agent.params.config.e_clip=0.2 agent.params.config.critic_coef=0.5 agent.params.config.grad_norm=0.5 agent.params.config.minibatch_size=800 'agent.params.network.image_shape=[84,84,3]'
```

## Place: evaluation

```bash
python scripts/evaluation/eval_transport10_parallel.py --checkpoint "$CHECKPOINT" --agent-cfg "$AGENT_CFG" --task ER-Place-Franka-Pregrasped-Table55-GoalFixedScale-RGB84-Front25Transport10-Eval-v0 --headless --enable_cameras --trials 50 --eval-seeds 652165 709797 242820 579787 70701 --goal-schedule goals.json --out summary.json
```
