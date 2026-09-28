# Reach and Place Cube: simulation only

Simulation tasks, visual/proprioceptive networks, Base/CAPS/CATS training and
evaluation. No real-robot driver, checkpoint, ONNX or simulator/assets bundled.
The impedance module is a simulation action term, not physical-robot control.

## Runtime

Install and accept Isaac Sim/Isaac Lab terms separately. Use its Python with
isaaclab, isaaclab_assets, isaaclab_rl, isaaclab_tasks, rl_games, PyYAML, Pillow,
NumPy and imageio-ffmpeg. The recorded Isaac Lab version is 2.2.1; full simulator
dependency pinning remains unverified. Syntax and assembly checks passed;
Isaac runtime has NOT been tested here.

Use the original `scripts/rl_games/train.py` and evaluation entry points.
[RECIPES.md](RECIPES.md) lists the recorded task arguments, seeds and overrides.
The added single-policy `launch.py` wrapper has been removed. The original
cluster scheduler is not distributed, and the direct recipes do not prescribe
serial training. No simulator terms are automatically accepted.

Reach: 128 environments, 3000 iterations, five training seeds. YAML horizon 48.
CATS final coefficient 5; CAPS temporal 0.5, spatial 0.1, sigma 0.2 with target
overlay preserved. Place: 512 environments, 1500 iterations, horizon 50, three
training seeds; CATS coefficient 1; CAPS 0.5/0.1/0.2 without overlay preservation.
All explicit Place training overrides are in RECIPES.md; YAML alone is incomplete.

## Evaluate

Use the evaluation commands in RECIPES.md and the agent.yaml saved with the checkpoint.
Five recorded evaluation seeds, 30 Reach targets or 50 Place trials per seed.
Place reuses the recorded paired goals.json. Original evaluators also record
diagnostic simulation videos, so their video dependencies are required; no
manuscript figure/animation generators are included.

Recompute Place's reported zero-hold success from the 30 Hz traces:

```bash
python robotics_sim/score_place.py /tmp/place-eval/summary.json --output /tmp/place-eval/reported_metrics.json
```

This metric-only interface was extracted for release from the existing report
predicate; it is not an original batch launcher or a policy-selection tool.
Success: XY error <=5 cm, cube height 0.0505 m +/-0.005 m, table contact,
gripper width >=0.06 m, and no finger contact in one frame. Training reward and
success include transport-height conditions; do not substitute them for the
reported metric. Aggregate training-seed means with untruncated 95% t-CIs.

## Provenance

The recorded Place payload plus training/evaluation overlays supplies the shared
source. Only Reach/Place registrations and required local modules/configs remain.
Automatic EULA acceptance and machine-specific defaults are removed; explicit
saved agent config is required. Numerical implementations are otherwise retained.
The shared snapshot contains Reach, but identity to each earlier Reach job is
not yet verified. See the parent RELEASE_STATUS.md before making release claims.
