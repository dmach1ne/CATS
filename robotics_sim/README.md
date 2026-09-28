# Robot Simulation

Training and evaluation for Reach and Place Cube in Isaac Lab, comparing
Base PPO, CAPS and CATS.

- **Reach:** move the end effector to a target in free space.
- **Place Cube:** start with a grasped cube and place it at a target on the table.

Policies use images, target coordinates and proprioceptive inputs.

## Setup

Use an Isaac Lab environment with Isaac Sim, `isaaclab_assets`, `isaaclab_rl`,
`isaaclab_tasks`, `rl_games`, PyYAML, Pillow, NumPy and imageio-ffmpeg.
The experiments used Isaac Lab 2.2.1. Install the simulator and assets separately.

From the repository root:

```bash
cd robotics_sim
export PYTHONPATH="$PWD/source/er_isaac:$PWD/scripts/rl_games:$PWD/scripts:$PYTHONPATH"
SEED=178132
```

Use the following training seeds for each method:

| Task | Training seeds | Environments | Iterations |
| --- | --- | ---: | ---: |
| Reach | 178132, 410580, 922852, 787576, 660993 | 128 | 3000 |
| Place Cube | 178132, 410580, 922852 | 512 | 1500 |

Run each seed with the corresponding command below. Jobs may be scheduled
independently on available GPUs. Place commands include training overrides
that are not fully captured by the YAML configuration alone.

For evaluation, set `CHECKPOINT` to the trained checkpoint and `AGENT_CFG`
to its saved `agent.yaml`. Each command evaluates five seeds; Reach uses
30 targets per seed and Place Cube uses 50 trials per seed. Keep `goals.json`
for the paired Place Cube target schedule.

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

## Place Cube Success

After Place evaluation, compute the reported success metric:

```bash
python score_place.py summary.json --output reported_metrics.json
```

Success requires XY error at most 5 cm, cube height within 0.0505 m +/- 0.005 m,
table contact, gripper width at least 0.06 m and no finger contact, with no
hold-time requirement. This differs from the training success metric.
Report training-seed means with untruncated 95% t-confidence intervals.

Only simulation code is included. The impedance action module controls the
simulated robot; it is not a physical-robot driver.
