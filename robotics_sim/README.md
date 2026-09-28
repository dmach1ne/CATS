# Robot Simulation

Training and evaluation for Reach and Place Cube in Isaac Lab, comparing
Base PPO, CAPS and CATS.

- **Reach:** move the end effector to a target in free space.
- **Place Cube:** start with a grasped cube and place it at a target on the table.

## Policy and Observations

Base, CAPS and CATS use the same visual actor-critic architecture within each
task. Each observation contains one 84 x 84 RGB image from a fixed external
camera and a vector of robot measurements and target coordinates. The target
is projected onto the image as a marker and is also supplied as XYZ coordinates
in the robot base frame.

The vector contains seven joint positions, seven joint velocities, the previous
action, end-effector height relative to the environment origin, and target XYZ.
It has 25 entries for Reach and 26 for Place Cube because Place includes a
gripper action. Cube pose, contact flags and reward-stage variables are not
additional policy inputs.

The network divides RGB values by 255. Joint positions are scaled by 1/pi,
joint velocities by 0.5, and height and target coordinates by 2; previous
actions are unchanged. These are fixed scales, not running observation
statistics. See the [observation construction][observations] and
[network implementation][network].

The image encoder has four ReLU convolutional layers with 16, 32, 64 and 64
channels, kernel sizes 6, 4, 3 and 3, and strides 2, 2, 1 and 1. Its output is
average-pooled to a 4 x 4 grid and projected to 256 features. A separate MLP
maps the measurement vector through 64 and 128 units. The two feature vectors
are concatenated and passed through a 256/128-unit ReLU MLP shared by the
policy-mean and value heads. Training uses a Gaussian policy with a learned,
state-independent log standard deviation; evaluation uses the policy mean.

## Actions and Control

Both tasks run physics at 60 Hz and apply a policy action every two physics
steps, giving a 30 Hz control rate. The seven arm outputs specify joint-position
targets, not torques or increments from the current joint position:
`joint target = default joint position + 0.5 * policy output`.
The simulator uses the Franka Panda asset and its joint-position actuators,
configured by the [shared controller setup][controller].

Place Cube adds an eighth output for the gripper. A nonnegative command starts
opening; opening remains latched until the trial resets. Finger targets advance
gradually rather than jumping immediately to the open position. See
[OpeningState][placement-math]. All three methods use the same controller.

## Tasks and Rewards

**Reach** tracks successive targets in free space. Training episodes last
18 seconds, with a new target every 6 seconds. The reward combines position
tracking, fine-grained position tracking and orientation tracking. The selected
configuration disables action-rate, joint-velocity and low-height penalties.
Target ranges, camera geometry and observation settings are defined in the
[Reach configuration][reach-config] and its [base task configuration][reach-base].

**Place Cube** starts from a prepared grasp, so learning to grasp is not part of
the task. Each trial lasts up to 3 seconds and uses one target on the table.
Targets are sampled on a forward semicircle of radius 0.25 m centered at
XY (0.45, 0), rather than uniformly inside a disk. The staged reward encourages
transport, lowering and release; it also rewards object and robot stillness
near the goal. The training-stage logic requires an elevated arrival before
lowering and uses a continuous hold for success. The reported success metric
is computed separately below. See the [Place configuration][place-config],
[reset and goal sampling][placement], and [reward and stage logic][placement-math].

## Regularization

Regularizers act on the seven arm policy means before controller scaling;
the gripper output is excluded. CATS penalizes consecutive mean-action
differences and increases the coefficient linearly with training progress to
`--alpha-max`. Transitions ending in an environment reset are masked out.
The temporal loss averages half the squared difference over valid transitions
and arm coordinates. See [the loss][cats-loss] and [the PPO integration][cats-agent].

CAPS combines this temporal loss with a spatial loss at fixed coefficients.
Its spatial perturbation adds Gaussian noise to RGB, joint positions, joint
velocities and end-effector height, scaled to give the requested standard
deviation after network preprocessing. Previous actions and target coordinates
are left unchanged. The Reach command also preserves target-marker pixels;
the Place command does not enable that mask. See [the spatial perturbation][caps-loss].

Network and PPO defaults are in the [Reach YAML][reach-agent] and
[Place YAML][place-agent]. The Place commands below override several PPO values;
use the saved `agent.yaml` from each run when evaluating its checkpoint.

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
independently on available GPUs. The Place Cube commands below override the
YAML defaults with the experiment settings.

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

## Evaluation Metrics

Evaluation uses deterministic policy means. For Reach, use `success_rate`
(equivalently, `hold_success_1s`), which requires staying within the distance
threshold continuously for one second. The reported thresholds are 5 and
10 cm; `minimum_distance_success_rate` is a different metric and does not
require holding the target.

Smoothness is the frequency-weighted action score computed separately for
each of the seven arm outputs at 30 Hz, then averaged across coordinates and
target windows or trials. It uses policy outputs before joint-position
scaling, not measured joint motion, and excludes the gripper. Lower is smoother.
See [the metric implementation][metrics].

### Place Cube Success

After Place evaluation, compute the reported success metric:

```bash
python score_place.py summary.json --output reported_metrics.json
```

Success requires XY error at most 5 cm, cube height within 0.0505 m +/- 0.005 m,
table contact, gripper width at least 0.06 m and no finger contact, with no
hold-time requirement. This differs from the training success metric.
Report training-seed means with untruncated 95% t-confidence intervals.
Evaluation seeds and trials are averaged within each trained policy first;
training seeds, not individual trials, are the independent units for these
intervals. Keep the per-training-seed evaluation outputs for this aggregation.

[observations]: source/er_isaac/er_isaac/tasks/manager_based/er_isaac/mdp/observation.py
[network]: scripts/rl_games/models/visual_proprio_network.py
[controller]: source/er_isaac/er_isaac/tasks/manager_based/er_isaac/config/franka/joint_pos_camera_env_cfg.py
[reach-config]: source/er_isaac/er_isaac/tasks/manager_based/er_isaac/config/franka/real_cam_env_cfg.py
[reach-base]: source/er_isaac/er_isaac/tasks/manager_based/er_isaac/reach_cfg.py
[place-config]: source/er_isaac/er_isaac/tasks/manager_based/er_isaac/config/franka/pregrasped_placement_env_cfg.py
[placement]: source/er_isaac/er_isaac/tasks/manager_based/er_isaac/mdp/placement.py
[placement-math]: source/er_isaac/er_isaac/tasks/manager_based/er_isaac/mdp/placement_math.py
[cats-loss]: scripts/rl_games/models/cats.py
[cats-agent]: scripts/rl_games/models/a2c_continuous_cats.py
[caps-loss]: scripts/rl_games/models/caps.py
[reach-agent]: source/er_isaac/er_isaac/tasks/manager_based/er_isaac/config/franka/agents/rl_games_visual_reach_compact_proprio_goal_fixed_scale_ppo_cfg.yaml
[place-agent]: source/er_isaac/er_isaac/tasks/manager_based/er_isaac/config/franka/agents/rl_games_visual_lift_compact_proprio_goal_fixed_scale_ppo_cfg.yaml
[metrics]: scripts/evaluation/lift_metrics.py
