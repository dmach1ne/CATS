# CATS

Code for the CATS paper, organized in the order of Section V. Benchmark scripts
retain the original seed files, parallel training and evaluation workflows.
The robotics release covers simulation only.

## Environment

[environment-original.yml](environment-original.yml) preserves the archived
environment, including PyTorch 2.6.0, Stable-Baselines3 2.5.0, Gymnasium 1.0.0
and MuJoCo 3.3.3. It is not a fully pinned lockfile or a tested portable install.
Other dependencies include NumPy, SciPy, h5py, pandas, matplotlib, tensorboard,
imageio and Box2D support. Isaac Lab uses a separate environment.

Local checks used PyTorch 2.9.1 on CPU, not the archived runtime. See
[release status](RELEASE_STATUS.md) for verification limits.

## V-A. Experimental Setup

Methods: PPO/SAC Base, CATS, CAPS, L2C2, Grad-CAPS (GRAD), ASAP and LipsNet++.
Tasks: Ant, Hopper, Walker, continuous LunarLander and Pendulum.

- Budgets and parameters: `benchmark/{ppo,sac}/tests/modules/params.py`.
- Original seed files: `benchmark/{ppo,sac}/tests/{seeds,validation_seeds}.txt`.
- Selected coefficients/profiles: [selected_settings.json](benchmark/selected_settings.json).
- Detailed commands and settings: [benchmark guide](benchmark/README.md).

The sweep scripts read the seed files and build parallel jobs. Their `--seeds`
option is a count, not an individual seed. Generic sweep defaults are not
necessarily the selected paper configurations.

## V-B. Main Results

### Clean Observations (Table I)

From `benchmark/`, select the algorithm's import roots. These example commands
run full-budget training, not smoke tests or the complete table:

```bash
cd benchmark
export PYTHONPATH="$PWD:$PWD/ppo:$PWD/ppo/tests:$PWD/ppo/models"
python ppo/tests/test_all.py --envs hopper walker --algs vanilla caps l2c2 grad asap --workers 5 --device auto
python ppo/tests/caps_linear_sweep.py --env hopper --alphas 5 --seeds 10 --workers 5 --device auto
python ppo/tests/lipsnetpp_sweep.py --env hopper --profiles E --seeds 10 --workers 5
python ppo/tests/evaluate_lipsnetpp.py --envs hopper --profiles E --training-seeds 10 --evaluation-episodes 10
```

For SAC, use its import roots, scripts and selected coefficients/profiles.
Baseline and CATS batches invoke the original clean evaluator after training;
LipsNet++ uses a separate evaluator. Evaluation reports return and FFT smoothness.

### Observation Noise (Table II)

Evaluate trained policies on Hopper and Walker with
`benchmark/{ppo,sac}/tests/eval_noise_robustness.py`. LipsNet++ uses
`benchmark/tools/evaluate_lipsnetpp_noise.py`. The original Base-derived feature
scale and noise-evaluation implementation are retained.

### Computational Cost (Table III)

`benchmark/ppo/tests/benchmark_training_cost.py` is the original wall-clock
timing entry point. Matching the historical optimized snapshot and timing-run
configuration remains necessary before claiming reproduction of the table.

## V-C. Temporal and Spatial Regularization

For the matched-smoothness comparison on Hopper and Walker (Fig. 2):

- Temporal-only CAPS: `benchmark/tools/train_four_target_temporal.py`.
- Spatial-only CAPS and LipsNet++: `benchmark/tools/train_smoothness_matched.py`.
- Original spatial-only entry point: `benchmark/tools/train_caps_spatial_only.py`.

Original coefficient/seed options remain. Historical selected coefficient grids
are not reconstructed by guessing from the figure.

## V-D. Empirical Validation of the Spatial Effect

The Table IV pipeline consists of:

1. `benchmark/{ppo,sac}/tests/collect_lsp_compact.py`: stochastic transitions.
2. `benchmark/tools/build_lsp_compact_normalizer.py`: external pooled normalizer.
3. `benchmark/tools/analyze_lsp_compact_cell.py`: spatial-effect estimates.

The analyzer retains 20,000 queries, cross-episode pairing, collection checks
and CPU/CUDA neighbor search. L2C2/LipsNet++ collectors are also included for
the normalization pipeline. The exact historical normalizer cohort still needs
verification; do not substitute an arbitrary policy subset.

## V-E. Policy Sensitivity to Observation Noise

For Fig. 3, use `benchmark/{ppo,sac}/tests/eval_policy_output_sensitivity.py`
and `benchmark/tools/evaluate_lipsnetpp_policy_sensitivity.py`. Evaluation uses
each policy's own clean visited observations and half the mean squared output
difference over ten noise draws. Shared-state evaluation is excluded.

For the next-state-noise extension (Fig. 4), use
`benchmark/{ppo,sac}/tests/caps_linear_nn_sweep.py` on LunarLander/Pendulum with
the selected coefficient and `--next_noise_sigma 0.05`, followed by own-state
sensitivity evaluation.

## V-F. Linear Ramp-Up Ablation

For Table V, compare CATS against Fixed-middle and Fixed-final on Ant, Hopper
and Walker with `benchmark/{ppo,sac}/tests/caps_fixed_sweep.py`. Set `--lamS 0`
and `--lamT` to half/full the selected CATS coefficient; keep SAC `--ps 0.0`.
CATS uses the linear sweep from V-B.

For Table VI, evaluate PPO policies with the own-state sensitivity scripts from
V-E. No separate evaluation bank or spatial penalty is introduced.

## V-G. Sim-to-Real Experiments: Simulation Code

The [robotics guide](robotics_sim/README.md) covers Reach/Place Cube environments,
visual/proprioceptive networks and Base/CAPS/CATS training and evaluation.
[RECIPES.md](robotics_sim/RECIPES.md) records seeds, task arguments and overrides.
`robotics_sim/score_place.py` computes reported Place success from evaluation traces.

Physical-robot drivers, deployment binaries and checkpoints are not included.
Isaac runtime validation remains outstanding.

## Scope and Provenance

Internal result CSVs, raw research archives, checkpoints, cluster infrastructure,
paper/animation generators and agent logs are not distributed. No new policy
selection or aggregation CLI is included. Reproduction settings remain with
the code; source hashes are recorded in `source_provenance.json` and `MANIFEST.sha256`.

See [validation.json](validation.json), [release status](RELEASE_STATUS.md) and
[third-party notices](THIRD_PARTY_NOTICES.md). Local checks do not establish full
paper-result reproduction. License review is required before public upload.
