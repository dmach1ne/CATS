# Benchmark Experiments

[Back to the project README](../README.md)

Complete the [installation](../README.md#installation) first. Run all commands
below from `benchmark/`, with the algorithm-specific import paths set:

```bash
cd benchmark
ALG=ppo  # ppo or sac
export PYTHONPATH="$PWD:$PWD/$ALG:$PWD/$ALG/tests:$PWD/$ALG/models"
```

When switching algorithms, rerun the export command.

Training scripts read `$ALG/tests/seeds.txt`; evaluation uses
`$ALG/tests/validation_seeds.txt`. In training commands, `--seeds` specifies the
number of runs, not a seed value. `--workers` controls concurrent training jobs
and defaults to the number of training seeds. Lower it if GPU memory is limited.

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
For SAC, set `ALG=sac` and update the import paths as shown at the top of this guide.

Training uses ten seeds and each task's full training budget. CATS and the
comparison-method launcher evaluate their saved policies after training;
LipsNet++ uses the separate evaluation command above. Keep these checkpoints
for the observation-noise, spatial-effect and policy-sensitivity evaluations.

Task budgets and method parameters are in `$ALG/tests/modules/params.py`.
CATS and LipsNet++ load their selected hyperparameters from
[selected_settings.json](selected_settings.json).

#### Observation-Noise Evaluation (Table II)

Evaluate the checkpoints trained above on Hopper and Walker under PPO and SAC,
with noise magnitudes from 0 to 0.5 in steps of 0.1. No additional training is
required:

```bash
for TASK in hopper walker; do
  python "$ALG/tests/eval_noise_robustness.py" --env "$TASK"
done
```

This evaluates Base, CAPS, L2C2, GRAD, ASAP and CATS. Evaluate LipsNet++
at the same noise magnitudes with:

```bash
for TASK in hopper walker; do
  python tools/evaluate_lipsnetpp_noise.py \
    --algorithm "$ALG" --environment "$TASK" \
    --output "$ALG/results/lipsnetpp_noise/$TASK"
done
```

#### Training Overhead

Use `ppo/tests/benchmark_training_cost.py` to measure wall-clock training time
on PPO Walker. This runs training separately rather than evaluating an existing
checkpoint. Use the same hardware and training configuration for each method
when comparing overhead.

With `ALG=ppo` and the PPO import paths set, run the timing measurements
sequentially so methods do not compete for the same GPU:

```bash
for METHOD in base caps l2c2 grad asap lpp cats; do
  while read -r SEED; do
    python ppo/tests/benchmark_training_cost.py \
      --method "$METHOD" --seed "$SEED" --device cuda \
      --output-root ppo/results/training_cost
  done < <(head -n 10 ppo/tests/seeds.txt)
done
```

### 2. Temporal vs. Spatial Regularization

Train and evaluate temporal-only CAPS, spatial-only CAPS and LipsNet++ at the
four smoothness targets on Hopper and Walker:

```bash
for TASK in hopper walker; do
  python tools/train_smoothness_matched.py --algorithm "$ALG" --env "$TASK"
done
```

The coefficients for each method and target are in
[selected_settings.json](selected_settings.json). Training uses ten
seeds. Use `--method caps_fixed`, `--method caps_spatial` or `--method lipsnetpp`
to run one method, and `--target-smoothness 0.2` to run one target.

### 3. Transition-Induced Spatial Effect

Compare the transition-induced spatial effect of Base and CATS. The following
example collects transitions for Hopper, pools the observation normalizer
across both methods, and analyzes each method:

```bash
TASK=hopper
ALPHA=$(python -c "from cats_common.experiment_settings import selected_settings; print(float(selected_settings('$ALG', '$TASK')['cats']['cats_alpha']))")
BASE=vanilla
if [ "$ALG" = ppo ]; then BASE=vanilla_ppo; fi
CATS="cats_${ALG}_alpha${ALPHA}"
OUT="$ALG/results/spatial_effect/$TASK"

for METHOD in "$BASE" "$CATS"; do
  python "$ALG/tests/collect_lsp_compact.py" \
    --root-dir "$ALG/results/pths" --alg "$METHOD" --env "$TASK" \
    --seed-file lsp_validation_seeds.txt \
    --scratch-dir "$OUT/scratch/$METHOD" \
    --publish-dir "$OUT/transitions/$METHOD"
done

python tools/build_lsp_compact_normalizer.py \
  --cell "Base=$OUT/transitions/$BASE" \
  --cell "CATS=$OUT/transitions/$CATS" \
  --output "$OUT/normalizer.json"

for METHOD in "$BASE" "$CATS"; do
  python tools/analyze_lsp_compact_cell.py \
    --cell-dir "$OUT/transitions/$METHOD" \
    --normalizer "$OUT/normalizer.json" \
    --method-label "$METHOD" --neighbor-backend auto \
    --output-dir "$OUT/analysis/$METHOD"
done
```

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
Evaluate LipsNet++ at the same noise magnitudes with:

```bash
for TASK in ant hopper lunar pendulum walker; do
  python tools/evaluate_lipsnetpp_policy_sensitivity.py \
    --algorithm "$ALG" --environment "$TASK" \
    --output "$ALG/results/lipsnetpp_sensitivity/$TASK"
done
```

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

See [robotics_sim/README.md](../robotics_sim/README.md) for Reach and Place Cube
training, evaluation and task settings.

## Reading the Results

Training writes checkpoints to `$ALG/results/pths/` and TensorBoard logs to
`$ALG/results/tensorboard_logs/`. Clean evaluation prints its output directory;
LipsNet++ clean evaluation writes `summary.csv` and `episodes.csv` under
`$ALG/results/pths/lipsnetpp_evaluation/<task>/`. The commands above give explicit
output directories for transition analysis and LipsNet++ noise evaluations.

Clean-evaluation CSV fields ending in `_std` contain standard deviations, not
95% confidence intervals.
