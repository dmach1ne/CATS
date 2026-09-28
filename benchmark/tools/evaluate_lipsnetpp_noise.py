#!/usr/bin/env python3
"""Evaluate selected LipsNet++ checkpoints with the AFQR paired-noise protocol."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afqr_common.lipsnetpp_experiment import selected_profile
from afqr_common.lipsnetpp_noise_evaluation import (
    PolicyCheckpoint,
    compute_base_observation_scale,
    evaluate_lipsnetpp_noise,
)


def checkpoint_spec(value: str) -> PolicyCheckpoint:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "checkpoint must use TRAINING_SEED=/path/to/final.zip"
        ) from error
    return PolicyCheckpoint(seed, Path(path_text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    parser.add_argument(
        "--environment",
        choices=("hopper", "walker"),
        required=True,
    )
    parser.add_argument(
        "--base-checkpoint",
        action="append",
        type=checkpoint_spec,
    )
    parser.add_argument(
        "--lipsnetpp-checkpoint",
        action="append",
        type=checkpoint_spec,
    )
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument(
        "--sigmas",
        default="0.0,0.1,0.2,0.3,0.4,0.5",
    )
    parser.add_argument("--warmup-episodes", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--noise-seed", type=int, default=20260717)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = selected_profile(args.algorithm, args.environment)

    if args.checkpoint_manifest is not None:
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
    else:
        if not args.base_checkpoint or not args.lipsnetpp_checkpoint:
            parser.error(
                "provide --checkpoint-manifest or both repeated checkpoint options"
            )
        base_checkpoints = args.base_checkpoint
        lipsnetpp_checkpoints = args.lipsnetpp_checkpoint

    algorithm_root = ROOT / args.algorithm
    for import_root in (algorithm_root, algorithm_root / "tests"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    if args.algorithm == "ppo":
        from ppo.models.custom_ppo import CustomPPO as BaseModel
        from ppo.models.lipsnetpp_ppo import LipsNetPlusPlusPPO as LipsNetModel

        seed_path = Path("ppo/tests/validation_seeds.txt")
    else:
        from sac.models.custom_sac import CustomSAC as BaseModel
        from sac.models.lipsnetpp_sac import LipsNetPlusPlusSAC as LipsNetModel

        seed_path = Path("sac/tests/validation_seeds.txt")

    validation_seeds = [
        int(line)
        for line in seed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required_seeds = max(args.warmup_episodes, args.evaluation_episodes)
    if len(validation_seeds) < required_seeds:
        raise ValueError(
            f"{seed_path} has {len(validation_seeds)} seeds; "
            f"{required_seeds} are required"
        )

    base_seeds = {item.training_seed for item in base_checkpoints}
    lipsnet_seeds = {
        item.training_seed for item in lipsnetpp_checkpoints
    }
    if base_seeds != lipsnet_seeds:
        raise ValueError(
            "Base and LipsNet++ checkpoints must use the same training seeds"
        )

    observation_scale, provenance = compute_base_observation_scale(
        BaseModel,
        base_checkpoints,
        args.environment,
        validation_seeds,
        args.warmup_episodes,
        args.device,
    )
    evaluate_lipsnetpp_noise(
        args.algorithm,
        LipsNetModel,
        lipsnetpp_checkpoints,
        args.environment,
        profile,
        observation_scale,
        [float(value) for value in args.sigmas.split(",")],
        validation_seeds[: args.evaluation_episodes],
        args.noise_seed,
        args.output,
        provenance,
        args.device,
    )


if __name__ == "__main__":
    main()
