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
number of runs. `--workers` controls concurrent training jobs and defaults to
the number of training seeds.

## Experiments

Start by training the benchmark policies in Section 1. We reuse these checkpoints
to evaluate observation-noise robustness, the transition-induced spatial effect,
and policy sensitivity. The regularization comparison, next-state-noise extension,
fixed-coefficient controls, and robot experiments require additional training,
as described below.

### 1. Main Results

The benchmark covers Ant, Hopper, Walker, continuous LunarLander and Pendulum,
with Base, CATS, CAPS, L2C2, Grad-CAPS (GRAD), ASAP and LipsNet++.

#### Training and Clean Evaluation (Table I)

With `ALG` and the import paths set as above, train and evaluate the methods
on Hopper:

```bash
python "$ALG/tests/train_cats.py" --env hopper --device auto

python "$ALG/tests/test_all.py" \
  --envs hopper --algs vanilla caps l2c2 grad asap --device auto

python "$ALG/tests/train_lipsnetpp.py" --env hopper --device auto
python "$ALG/tests/evaluate_lipsnetpp.py" --env hopper
```

Replace `hopper` with `ant`, `walker`, `lunar` or `pendulum` for the other tasks.
For SAC, set `ALG=sac` and update the import paths as shown in Setup.

Training uses ten seeds and each task's full training budget. CATS and the
comparison-method launcher evaluate their saved policies after training;
LipsNet++ uses the separate evaluation command above. Keep these checkpoints
for the observation-noise, spatial-effect and policy-sensitivity evaluations.

Task budgets and method parameters are in `$ALG/tests/modules/params.py`.
CATS and LipsNet++ load their selected hyperparameters from
[selected_settings.json](benchmark/selected_settings.json).

#### Observation-Noise Evaluation (Table II)

Evaluate the checkpoints trained above on Hopper and Walker under PPO and SAC,
with noise magnitudes from 0 to 0.5 in steps of 0.1. No additional training is
required:

```bash
for TASK in hopper walker; do
  python "$ALG/tests/eval_noise_robustness.py" --env "$TASK"
done
```

This evaluates Base, CAPS, L2C2, GRAD, ASAP and CATS. LipsNet++ uses
`tools/evaluate_lipsnetpp_noise.py` with the same noise magnitudes.

#### Training Overhead

Use `ppo/tests/benchmark_training_cost.py` to measure wall-clock training time
on PPO Walker. This runs training separately rather than evaluating an existing
checkpoint. Use the same hardware and training configuration for each method
when comparing overhead.

### 2. Temporal vs. Spatial Regularization

Train and evaluate temporal-only CAPS, spatial-only CAPS and LipsNet++ at the
four smoothness targets on Hopper and Walker:

```bash
for TASK in hopper walker; do
  python tools/train_smoothness_matched.py --algorithm "$ALG" --env "$TASK"
done
```

The coefficients for each method and target are in
[selected_settings.json](benchmark/selected_settings.json). Training uses ten
seeds. Use `--method caps_fixed`, `--method caps_spatial` or `--method lipsnetpp`
to run one method, and `--target-smoothness 0.2` to run one target.

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
Pendulum. Train the extension with Gaussian noise of standard deviation 0.05
on the next states used by the temporal penalty, then evaluate it alongside
the Base and CATS policies trained in Section 1:

```bash
for TASK in lunar pendulum; do
  python "$ALG/tests/train_cats_next_noise.py" --env "$TASK"
  python "$ALG/tests/eval_policy_output_sensitivity.py" \
    --env "$TASK" --experiment next-state-noise
done
```

### 5. Linear Ramp-Up

Compare CATS with Fixed-middle and Fixed-final on Ant, Hopper and Walker.
For PPO Hopper:

```bash
python ppo/tests/train_fixed_controls.py --env hopper
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
training, evaluation and task settings.

## Reading the Results

Clean-evaluation CSV fields ending in `_std` contain standard deviations, not
95% confidence intervals.

## Acknowledgments

The benchmark builds on the ASAP codebase and Stable-Baselines3. Robot
simulation uses Isaac Lab and rl_games.
