# Original benchmark workflow

Run commands from this directory. Select the algorithm's import roots before
running its scripts; PPO and SAC each have their own `models` and `tests` modules.

```bash
cd benchmark
export PYTHONPATH="$PWD:$PWD/ppo:$PWD/ppo/tests:$PWD/ppo/models"
python ppo/tests/test_all.py --envs hopper walker --algs vanilla caps l2c2 grad asap --workers 5 --device auto
python ppo/tests/caps_linear_sweep.py --env hopper --alphas 5 --seeds 10 --workers 5 --device auto
```

For SAC, replace `ppo` with `sac` in the import roots and entry-point path.
These commands run real, paper-budget training. They are not smoke examples.
They load `tests/seeds.txt`; `--seeds` in the sweep scripts means a COUNT, not
an individual seed. Original worker defaults, completion checks, checkpoint
layout and post-training evaluation are retained. `test_all.py` uses the first
ten training seeds. Held-out evaluation uses `tests/validation_seeds.txt`.

## Settings

The original task budgets and Base/CAPS/L2C2/GRAD/ASAP parameters are in
`{ppo,sac}/tests/modules/params.py`. Do not treat generic sweep defaults as the
selected paper settings. Pass the relevant CATS coefficient explicitly:

| Task | PPO CATS alpha | SAC CATS alpha | PPO LipsNet++ profile | SAC LipsNet++ profile |
|---|---:|---:|---|---|
| Ant | 100 | 100 | F | E |
| Hopper | 5 | 80 | E | F |
| LunarLander | 0.25 | 30 | H | F |
| Pendulum | 10 | 40 | C | D |
| Walker | 1 | 40 | H | G |

Source: `selected_settings.json`, which records the archived table-source filename and hash.
These are table-source settings, not an independent reconstruction of every
historical tuning job. PPO LunarLander CATS raw-seed provenance remains open.
Profile definitions are in `afqr_common/lipsnetpp_experiment.py`.

```bash
python ppo/tests/lipsnetpp_sweep.py --env hopper --profiles E --seeds 10 --workers 5
python ppo/tests/evaluate_lipsnetpp.py --envs hopper --profiles E --training-seeds 10 --evaluation-episodes 10
```

The LipsNet++ original sweep/evaluator defaults use three policies, so the
final-result commands explicitly request ten. Its eight profiles are A through
H. No replacement tuning/selection algorithm is supplied.

## Reported ablations

- Fixed-middle/final: `caps_fixed_sweep.py`, with `--lamS 0` and `--lamT`
  set to half/full selected CATS alpha. Keep SAC `--ps 0.0` for these results.
- Temporal-only CAPS: `tools/train_four_target_temporal.py`.
- Spatial-only CAPS and LipsNet++ matched smoothing:
  `tools/train_smoothness_matched.py`; original coefficient/seed options remain.
- Spatial-only CAPS training: `tools/train_caps_spatial_only.py`.
- Next-state-noise extension: `caps_linear_nn_sweep.py`, explicitly passing
  `--next_noise_sigma 0.05` and the selected coefficient for LunarLander/Pendulum.
- Overhead: `ppo/tests/benchmark_training_cost.py` is the original timing entry
  point, not the discarded wrapper's combined setup/train/save timer. Exact
  snapshot and cohort matching remain necessary before reproducing the table.

Use each script's `--help`. Historical matched-smoothness selection grids are
not reconstructed by guessing from the final table.

## Clean and noisy evaluation

Original clean evaluation is called by the batch scripts through
`tests/modules/action_extractor.py`, retaining deterministic evaluation,
saved VecNormalize, FFT scores, original CSV fields and action archives.
Those CSV `*_std` columns are standard deviations, not 95% CI half-widths.

```bash
python ppo/tests/eval_noise_robustness.py --env hopper --algs vanilla caps_linear_ppo_alpha5.0 --obs_scale_alg vanilla --sigmas 0,0.1,0.2,0.3,0.4,0.5
python ppo/tests/eval_policy_output_sensitivity.py --env hopper --algs vanilla caps_linear_ppo_alpha5.0 --obs_scale_alg vanilla --own_only --sigmas 0,0.05,0.1,0.15,0.2,0.25,0.3
```

These are Base/CATS examples; include the other trained method prefixes when
reproducing comparisons. Noise levels must match the relevant reported table
or figure rather than the legacy script's defaults. Sensitivity measures half
the mean squared deterministic policy-output difference with ten draws per
visited observation. The original own-state RNG and batched-prediction checks
are retained. Shared-state evaluation is not exposed in this release; legacy
null shared-output fields remain for schema compatibility. A shared Base-derived
feature scale is still used; this is not a shared-state evaluation bank.

LipsNet++ uses the separate original `tools/evaluate_lipsnetpp_noise.py` and
`tools/evaluate_lipsnetpp_policy_sensitivity.py`, with checkpoint manifests and
only the current history slot perturbed. Their own-state routines are retained.

## Transition-induced spatial effect

The original pipeline is retained instead of the reconstructed `analyze.py`:

1. `{ppo,sac}/tests/collect_lsp_compact.py` collects stochastic transitions in
   validated chunks. L2C2/LipsNet++ have their corresponding collectors.
2. `tools/build_lsp_compact_normalizer.py` builds the external pooled normalizer
   from complete source collections, not an arbitrary Base/CATS-only subset.
3. `tools/analyze_lsp_compact_cell.py` consumes that normalizer, defaults to
   20,000 queries and checks 10 policies x 10,000 episodes with 500-episode chunks.
   Original CPU/CUDA neighbor options and cross-episode pair statistics remain.

`lsp_validation_seeds.txt` contains the preserved collection seeds. The exact
normalizer source cohort for each published cell still needs to be tied to its
historical manifest; no fabricated end-to-end table-reproduction command is
provided while that provenance is unresolved.

## What changed for release

Only paper-task registration/import filtering, dependency extraction and removal
of shared-state sensitivity and noise plot output were applied to original
benchmark entry points. Trainer/environment function ASTs and copied numerical
sources are checked against the archive. No replacement training CLI, automatic
selection tool, exact-version evaluation gate or forced CPU execution remains.
