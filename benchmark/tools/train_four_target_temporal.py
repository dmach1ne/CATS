#!/usr/bin/env python3
"""Train one CAPS temporal-only coefficient for a fixed smoothness target."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ("hopper", "walker")


def scalar_label(value: float) -> str:
    return f"{value:g}"


def load_seeds(path: Path, count: int, offset: int) -> list[int]:
    seeds = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = seeds[offset : offset + count]
    if len(selected) != count:
        raise ValueError("requested seed range exceeds seeds.txt")
    return selected


def run_one(name: str, function) -> None:
    print(f"[START] {name}", flush=True)
    function()
    print(f"[DONE] {name}", flush=True)


def remove_intermediate_checkpoints(run_dirs: list[Path]) -> None:
    for run_dir in run_dirs:
        for checkpoint in run_dir.glob("mid_*.zip"):
            checkpoint.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    parser.add_argument("--env", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--coefficient", type=float, required=True)
    parser.add_argument("--target-smoothness", type=float, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.coefficient <= 0 or args.target_smoothness <= 0:
        parser.error("coefficient and target smoothness must be positive")
    if args.seeds <= 0 or args.workers <= 0 or args.seed_offset < 0:
        parser.error("seed and worker counts must be positive; offset nonnegative")

    algorithm_root = ROOT / args.algorithm
    for import_root in (ROOT, algorithm_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    from tests.modules.controller import train_caps_fixed
    from tests.modules.envs import make_hopper_env, make_walker_env
    from tests.modules.params import alg_args, env_args, env_timestep

    factories = {"hopper": make_hopper_env, "walker": make_walker_env}
    seeds = load_seeds(
        algorithm_root / "tests" / "seeds.txt",
        args.seeds,
        args.seed_offset,
    )
    evaluation_seeds = load_seeds(
        algorithm_root / "tests" / "validation_seeds.txt",
        args.evaluation_episodes,
        0,
    )
    result_root = algorithm_root / "results" / "pths"
    save_dir = result_root / args.env
    log_dir = algorithm_root / "results" / "tensorboard_logs" / args.env
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    method_args = alg_args["caps"][args.env].copy()
    method_args["caps_lamT"] = args.coefficient
    method_args["caps_lamS"] = 0.0
    prefix = f"caps_fixed_{args.algorithm}_lamT{args.coefficient}_lamS0.0"
    run_dirs = [save_dir / f"{prefix}_{seed}" for seed in seeds]

    jobs = []
    for seed, run_dir in zip(seeds, run_dirs):
        final_path = run_dir / "final.zip"
        if final_path.is_file():
            print(f"[SKIP] {final_path}", flush=True)
            continue
        jobs.append(
            (
                f"{args.algorithm}/{args.env}/caps_fixed/{args.coefficient:g}/{seed}",
                partial(
                    train_caps_fixed,
                    seed,
                    env_timestep[args.env],
                    str(save_dir) + os.sep,
                    str(log_dir) + os.sep,
                    factories[args.env],
                    env_args[args.env],
                    method_args,
                    args.device,
                    False,
                ),
            )
        )

    print(
        f"algorithm={args.algorithm} env={args.env} method=caps_fixed "
        f"coefficient={args.coefficient} target={args.target_smoothness} "
        f"seeds={seeds} workers={args.workers} pending={len(jobs)}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, name, function): name
            for name, function in jobs
        }
        for future in as_completed(futures):
            future.result()

    missing = [str(path / "final.zip") for path in run_dirs if not (path / "final.zip").is_file()]
    if missing:
        raise RuntimeError(f"missing final checkpoints: {missing}")
    remove_intermediate_checkpoints(run_dirs)

    target_label = scalar_label(args.target_smoothness)
    coefficient_label = scalar_label(args.coefficient)
    output_dir = (
        algorithm_root
        / "results"
        / "four_target_temporal"
        / args.env
        / f"target_{target_label}"
        / f"coef_{coefficient_label}"
        / f"seeds{args.seed_offset}-{args.seed_offset + args.seeds - 1}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    from afqr_common.policy_evaluation import (
        EVALUATION_PROTOCOL,
        evaluate_flat_policy_checkpoints,
    )
    if args.algorithm == "ppo":
        from ppo.models.custom_ppo import CustomPPO as model_class
    else:
        from sac.models.custom_sac import CustomSAC as model_class
    evaluate_flat_policy_checkpoints(
        args.algorithm,
        model_class,
        args.env,
        "caps_fixed",
        args.coefficient,
        list(zip(seeds, [run_dir / "final.zip" for run_dir in run_dirs])),
        evaluation_seeds,
        output_dir,
    )

    provenance = {
        "algorithm": args.algorithm,
        "method": "caps_fixed",
        "environment": args.env,
        "target_smoothness": args.target_smoothness,
        "matching_tolerance_relative": 0.10,
        "coefficient": args.coefficient,
        "lambda_t": args.coefficient,
        "lambda_s": 0.0,
        "training_seeds": seeds,
        "evaluation_seeds": evaluation_seeds,
        "workers": args.workers,
        "deterministic_evaluation": True,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "source_bundle_sha256": os.environ.get("AFQR_SOURCE_SHA256"),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[RESULT] {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
