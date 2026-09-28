# Revisiting Temporal Regularization for Smooth Control in Deep Reinforcement Learning

Code for **Revisiting Temporal Regularization for Smooth Control in Deep
Reinforcement Learning**. We introduce **CATS**, a policy regularizer that
combines a temporal penalty with a linearly increasing coefficient. CATS
encourages smooth control without an explicit spatial penalty.

The repository includes the PPO and SAC benchmark experiments, their
hyperparameters, and the Reach and Place Cube simulation tasks. The sections
below follow the experiments in the paper.

## Setup

The benchmark uses Python 3.10, PyTorch, Stable-Baselines3 2.5.0, Gymnasium
1.0.0 and MuJoCo 3.3.3. LunarLander additionally requires Box2D. Analysis scripts
use NumPy, SciPy, h5py, pandas and matplotlib; training uses tensorboard.

[environment-original.yml](environment-original.yml) records the training
environment, including PyTorch 2.6.0. Robot simulation requires a separate
Isaac Lab environment; see [robotics_sim](robotics_sim/README.md).

Run benchmark commands from `benchmark/`. Set the import paths for the algorithm
you want to use:

```bash
cd benchmark
ALG=ppo  # ppo or sac
export PYTHONPATH="$PWD:$PWD/$ALG:$PWD/$ALG/tests:$PWD/$ALG/models"
```

Training scripts read `$ALG/tests/seeds.txt`; evaluation uses
`$ALG/tests/validation_seeds.txt`. In sweep commands, `--seeds` specifies the
number of runs. `--workers` controls concurrent training jobs.

## Experiments

### 1. Main Results

The benchmark covers Ant, Hopper, Walker, continuous LunarLander and Pendulum,
with Base, CATS, CAPS, L2C2, Grad-CAPS (GRAD), ASAP and LipsNet++.

Train CATS on PPO Hopper:

```bash
python ppo/tests/caps_linear_sweep.py \
  --env hopper --alphas 5 --seeds 10 --workers 5 --device auto
```

Train the comparison methods:

```bash
python ppo/tests/test_all.py \
  --envs hopper --algs vanilla caps l2c2 grad asap --workers 5 --device auto
```

These scripts run full-budget training and then evaluate the saved policies.
Task budgets and method parameters are in `$ALG/tests/modules/params.py`.
The selected CATS coefficients and LipsNet++ profiles are:

| Task | PPO CATS | SAC CATS | PPO LipsNet++ | SAC LipsNet++ |
| --- | ---: | ---: | :---: | :---: |
| Ant | 100 | 100 | F | E |
| Hopper | 5 | 80 | E | F |
| LunarLander | 0.25 | 30 | H | F |
| Pendulum | 10 | 40 | C | D |
| Walker | 1 | 40 | H | G |

These settings are also available in [selected_settings.json](benchmark/selected_settings.json).
LipsNet++ profiles are defined in `afqr_common/lipsnetpp_experiment.py`.
For PPO Hopper:

```bash
python ppo/tests/lipsnetpp_sweep.py --env hopper --profiles E --seeds 10 --workers 5
python ppo/tests/evaluate_lipsnetpp.py \
  --envs hopper --profiles E --training-seeds 10 --evaluation-episodes 10
```

For SAC, switch the import paths and script paths to `sac` and use the SAC
settings above. Pass the selected settings explicitly; sweep defaults may differ.

**Observation noise.** Evaluate the trained policies on Hopper or Walker:

```bash
python ppo/tests/eval_noise_robustness.py \
  --env hopper --algs vanilla caps_linear_ppo_alpha5.0 \
  --obs_scale_alg vanilla --sigmas 0,0.1,0.2,0.3,0.4,0.5
```

Add other trained method prefixes to `--algs` for the full comparison.
LipsNet++ uses `tools/evaluate_lipsnetpp_noise.py`.

**Training cost.** Use `ppo/tests/benchmark_training_cost.py` to measure
wall-clock training time. Match the hardware and training configuration when
comparing overhead.

### 2. Temporal vs. Spatial Regularization

Compare return at matched smoothness levels on Hopper and Walker:

| Comparison | Script |
| --- | --- |
| Temporal-only CAPS | `tools/train_four_target_temporal.py` |
| Spatial-only CAPS and LipsNet++ | `tools/train_smoothness_matched.py` |
| Spatial-only CAPS training | `tools/train_caps_spatial_only.py` |

Use each script's `--help` for coefficient, seed and evaluation options.

### 3. Transition-Induced Spatial Effect

Compare the transition-induced spatial effect of Base and CATS:

1. Collect stochastic transitions with `$ALG/tests/collect_lsp_compact.py`.
2. Build the pooled normalizer with `tools/build_lsp_compact_normalizer.py`.
3. Run `tools/analyze_lsp_compact_cell.py` on the collected transitions.

Collection seeds are in `lsp_validation_seeds.txt`. The analyzer uses 20,000
queries by default and supports CPU/CUDA neighbor search. Use the same
normalizer for both policies so their neighborhood distances use the same scale.

### 4. Policy Sensitivity

Measure changes in policy outputs under observation noise:

```bash
python ppo/tests/eval_policy_output_sensitivity.py \
  --env hopper --algs vanilla caps_linear_ppo_alpha5.0 \
  --obs_scale_alg vanilla --own_only --sigmas 0,0.05,0.1,0.15,0.2,0.25,0.3
```

Each policy is evaluated on its own clean visited observations. The metric is
half the mean squared output difference over ten noise draws per observation.
LipsNet++ uses `tools/evaluate_lipsnetpp_policy_sensitivity.py`.

For the next-state-noise extension on LunarLander and Pendulum, train with
`$ALG/tests/caps_linear_nn_sweep.py`, the selected coefficient and
`--next_noise_sigma 0.05`, then run the same sensitivity evaluation.

### 5. Linear Ramp-Up

Compare CATS with Fixed-middle and Fixed-final on Ant, Hopper and Walker.
For PPO Hopper, whose final CATS coefficient is 5:

```bash
python ppo/tests/caps_fixed_sweep.py \
  --env hopper --lamT 2.5,5 --lamS 0 --seeds 10 --workers 5
```

Fixed-middle uses half the final CATS coefficient; Fixed-final uses the full
coefficient throughout training. For SAC, keep `--ps 0.0`. Evaluate the PPO
policies with the sensitivity script above for the ramp-up sensitivity comparison.

### 6. Robot Simulation

See [robotics_sim/README.md](robotics_sim/README.md) for Reach and Place Cube
training, evaluation and task settings. This release includes simulation code
only, not checkpoints or physical-robot deployment software.

## Reading the Results

Clean-evaluation CSV fields ending in `_std` contain standard deviations, not
95% confidence intervals.

## Acknowledgments

The benchmark builds on the ASAP codebase and Stable-Baselines3. Robot
simulation uses Isaac Lab and rl_games.
