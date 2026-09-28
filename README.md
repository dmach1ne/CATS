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
`$ALG/tests/validation_seeds.txt`. In training commands, `--seeds` specifies the
number of runs. `--workers` controls concurrent training jobs.

## Experiments

### 1. Main Results

The benchmark covers Ant, Hopper, Walker, continuous LunarLander and Pendulum,
with Base, CATS, CAPS, L2C2, Grad-CAPS (GRAD), ASAP and LipsNet++.

Train CATS on PPO Hopper:

```bash
python ppo/tests/train_cats.py --env hopper --workers 5 --device auto
```

Train the comparison methods:

```bash
python ppo/tests/test_all.py \
  --envs hopper --algs vanilla caps l2c2 grad asap --workers 5 --device auto
```

These scripts run full-budget training and then evaluate the saved policies.
Task budgets and method parameters are in `$ALG/tests/modules/params.py`.
CATS and LipsNet++ load their selected hyperparameters directly from
[selected_settings.json](benchmark/selected_settings.json).
To train and evaluate LipsNet++ on PPO Hopper:

```bash
python ppo/tests/train_lipsnetpp.py --env hopper --workers 5
python ppo/tests/evaluate_lipsnetpp.py --env hopper
```

For SAC, switch the import paths and script paths to `sac`. Each command trains
the selected configuration across ten seeds, not a hyperparameter search.

**Observation noise (Table II).** Evaluate the trained policies on Hopper and
Walker under PPO and SAC, with noise magnitudes from 0 to 0.5 in steps of 0.1:

```bash
for TASK in hopper walker; do
  python "$ALG/tests/eval_noise_robustness.py" --env "$TASK"
done
```

This evaluates Base, CAPS, L2C2, GRAD, ASAP and CATS. LipsNet++ uses
`tools/evaluate_lipsnetpp_noise.py` with the same noise magnitudes.

**Training cost.** Use `ppo/tests/benchmark_training_cost.py` to measure
wall-clock training time. Match the hardware and training configuration when
comparing overhead.

### 2. Temporal vs. Spatial Regularization

Compare return at matched smoothness levels on Hopper and Walker:

| Comparison | Script |
| --- | --- |
| Temporal-only CAPS | `tools/train_four_target_temporal.py` |
| Spatial-only CAPS and LipsNet++ | `tools/train_smoothness_matched.py` |

Use `--seeds 10` for the reported comparisons. Each script trains and evaluates
one coefficient across seeds; see `--help` for coefficient and evaluation options.

### 3. Transition-Induced Spatial Effect

Compare the transition-induced spatial effect of Base and CATS:

1. Collect stochastic transitions with `$ALG/tests/collect_lsp_compact.py`.
2. Build the pooled normalizer with `tools/build_lsp_compact_normalizer.py`.
3. Run `tools/analyze_lsp_compact_cell.py` on the collected transitions.

Collection seeds are in `lsp_validation_seeds.txt`. The analyzer uses 20,000
queries by default and supports CPU/CUDA neighbor search. Use the same
normalizer for both policies so their neighborhood distances use the same scale.

### 4. Policy Sensitivity

Figure 3 compares Base, CAPS, L2C2, ASAP, LipsNet++ and CATS on all five tasks
under PPO and SAC. Evaluate Base, CAPS, L2C2, ASAP and CATS with:

```bash
for TASK in ant hopper lunar pendulum walker; do
  python "$ALG/tests/eval_policy_output_sensitivity.py" --env "$TASK"
done
```

Each policy is evaluated on its own clean visited observations, using ten noise
draws per observation. The reported metric is half the mean squared difference
between clean and perturbed policy outputs. The noise magnitudes are
0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90 and 1.00.
LipsNet++ uses `tools/evaluate_lipsnetpp_policy_sensitivity.py` with the same
noise magnitudes and its selected configuration.

Figure 4 compares Base, CATS and CATS with next-state noise on LunarLander and
Pendulum. The extension keeps the selected CATS coefficient and adds Gaussian
next-state noise with standard deviation 0.05:

```bash
for TASK in lunar pendulum; do
  python "$ALG/tests/train_cats_next_noise.py" --env "$TASK" --workers 5
  python "$ALG/tests/eval_policy_output_sensitivity.py" \
    --env "$TASK" --experiment next-state-noise
done
```

### 5. Linear Ramp-Up

Compare CATS with Fixed-middle and Fixed-final on Ant, Hopper and Walker.
For PPO Hopper:

```bash
python ppo/tests/train_fixed_controls.py --env hopper --workers 5
```

Fixed-middle uses half the final CATS coefficient; Fixed-final uses the full
coefficient throughout training. Both coefficients are derived from the selected
CATS setting. Use `--control middle` or `--control final` to run only one control.

Table VI compares **Fixed-middle and CATS**, on **PPO Ant, Hopper and Walker**,
at noise magnitudes 0.05 through 0.30 in increments of 0.05:

```bash
for TASK in ant hopper walker; do
  python ppo/tests/eval_policy_output_sensitivity.py \
    --env "$TASK" --experiment ramp-up
done
```

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
