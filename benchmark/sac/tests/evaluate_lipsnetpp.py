"""Evaluate the selected LipsNet++ SAC configuration."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAC_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, SAC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.append(str(import_root))

from afqr_common.lipsnetpp_evaluation import evaluate_profiles
from afqr_common.lipsnetpp_experiment import selected_profile
from sac.models.lipsnetpp_sac import LipsNetPlusPlusSAC


DEFAULT_ENVS = ["ant", "hopper", "lunar", "pendulum", "walker"]


def load_seeds(path: Path, limit: int, offset: int = 0) -> list[int]:
    seeds = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return seeds[offset : offset + limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=DEFAULT_ENVS, required=True)
    parser.add_argument("--training-seeds", type=int, default=10)
    parser.add_argument("--training-seed-offset", type=int, default=0)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sac/results/pths/lipsnetpp_evaluation"),
    )
    args = parser.parse_args()

    training_seeds = load_seeds(
        Path("sac/tests/seeds.txt"),
        args.training_seeds,
        args.training_seed_offset,
    )
    evaluation_seeds = load_seeds(
        Path("sac/tests/validation_seeds.txt"),
        args.evaluation_episodes,
    )
    evaluate_profiles(
        "sac",
        LipsNetPlusPlusSAC,
        Path("sac/results/pths"),
        [args.env],
        [selected_profile("sac", args.env)],
        training_seeds,
        evaluation_seeds,
        args.output / args.env,
    )


if __name__ == "__main__":
    main()
