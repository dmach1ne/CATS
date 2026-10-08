#!/usr/bin/env python3
"""Evaluate Base versus selected LipsNet++ direct policy-output sensitivity."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cats_common.lipsnetpp_experiment import selected_profile
from cats_common.lipsnetpp_noise_evaluation import (
    PolicyCheckpoint,
    compute_base_observation_scale,
    discover_selected_checkpoints,
)
from cats_common.lipsnetpp_sensitivity_evaluation import (
    evaluate_lipsnetpp_policy_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    parser.add_argument(
        "--environment",
        choices=("ant", "hopper", "lunar", "pendulum", "walker"),
        required=True,
    )
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument(
        "--sigmas",
        default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--warmup-episodes", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--noise-draws", type=int, default=10)
    parser.add_argument("--noise-seed", type=int, default=20260717)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = selected_profile(args.algorithm, args.environment)

    if args.checkpoint_manifest is None:
        base_checkpoints, lipsnetpp_checkpoints = discover_selected_checkpoints(
            args.algorithm, args.environment
        )
    else:
        manifest = json.loads(
            args.checkpoint_manifest.read_text(encoding="utf-8")
        )
        for field, expected in (
            ("algorithm", args.algorithm),
            ("environment", args.environment),
            ("profile", profile.name),
        ):
            if str(manifest[field]).lower() != str(expected).lower():
                raise ValueError(
                    f"checkpoint manifest {field}={manifest[field]!r}, "
                    f"expected {expected!r}"
                )
        base_checkpoints = [
            PolicyCheckpoint(int(item["training_seed"]), Path(item["path"]))
            for item in manifest["base_checkpoints"]
        ]
        lipsnetpp_checkpoints = [
            PolicyCheckpoint(int(item["training_seed"]), Path(item["path"]))
            for item in manifest["lipsnetpp_checkpoints"]
        ]

    algorithm_root = ROOT / args.algorithm
    for import_root in (algorithm_root, algorithm_root / "tests"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    if args.algorithm == "ppo":
        from ppo.models.custom_ppo import CustomPPO as BaseModel
        from ppo.models.lipsnetpp_ppo import (
            LipsNetPlusPlusPPO as LipsNetModel,
        )

        seed_path = Path("ppo/tests/validation_seeds.txt")
    else:
        from sac.models.custom_sac import CustomSAC as BaseModel
        from sac.models.lipsnetpp_sac import (
            LipsNetPlusPlusSAC as LipsNetModel,
        )

        seed_path = Path("sac/tests/validation_seeds.txt")

    validation_seeds = [
        int(line)
        for line in seed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = max(args.warmup_episodes, args.evaluation_episodes)
    if len(validation_seeds) < required:
        raise ValueError(f"{seed_path} has fewer than {required} seeds")

    observation_scale, provenance = compute_base_observation_scale(
        BaseModel,
        base_checkpoints,
        args.environment,
        validation_seeds,
        args.warmup_episodes,
        args.device,
    )
    evaluate_lipsnetpp_policy_sensitivity(
        args.algorithm,
        BaseModel,
        LipsNetModel,
        base_checkpoints,
        lipsnetpp_checkpoints,
        args.environment,
        profile,
        observation_scale,
        validation_seeds[: args.evaluation_episodes],
        [float(value) for value in args.sigmas.split(",")],
        args.noise_draws,
        args.noise_seed,
        args.batch_size,
        args.output,
        provenance,
        args.device,
    )


if __name__ == "__main__":
    main()
