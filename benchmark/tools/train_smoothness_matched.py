#!/usr/bin/env python3
"""Train and evaluate temporal-only CAPS, spatial-only CAPS and LipsNet++."""

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
METHODS = ("caps_fixed", "caps_spatial", "lipsnetpp")


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
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--env", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--target-smoothness", type=float, nargs="+",
                        choices=(0.2, 0.4, 0.6, 0.8), default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent training jobs (default: number of training seeds)")
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.workers is None:
        args.workers = args.seeds

    if args.seeds <= 0 or args.workers <= 0 or args.seed_offset < 0:
        parser.error("seed and worker counts must be positive; offset nonnegative")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from afqr_common.experiment_settings import matched_coefficient

    methods = METHODS if args.method == "all" else (args.method,)
    for method in methods:
        for target in args.target_smoothness:
            config = argparse.Namespace(**vars(args))
            config.method = method
            config.target_smoothness = target
            config.coefficient = matched_coefficient(args.algorithm, args.env, method, target)
            train_configuration(config)


def train_configuration(args) -> None:

    algorithm_root = ROOT / args.algorithm
    for import_root in (ROOT, algorithm_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    from tests.modules.controller import train_caps, train_caps_fixed, train_lipsnetpp
    from tests.modules.envs import make_hopper_env, make_walker_env
    from tests.modules.params import alg_args, env_args, env_timestep
    from afqr_common.experiment_settings import selected_settings

    lips_settings = selected_settings(args.algorithm, args.env)["lipsnetpp"]
    args.lambda_h = lips_settings["lipsnetpp_lambda_h"]
    args.history_len = lips_settings["history_len"]

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

    coefficient_label = scalar_label(args.coefficient)
    if args.method in {"caps_fixed", "caps_spatial"}:
        method_args = alg_args["caps"][args.env].copy()
        temporal = args.method == "caps_fixed"
        method_args["caps_lamT"] = args.coefficient if temporal else 0.0
        method_args["caps_lamS"] = 0.0 if temporal else args.coefficient
        if temporal:
            prefix = f"caps_fixed_{args.algorithm}_lamT{args.coefficient}_lamS0.0"
            train_function = train_caps_fixed
        else:
            prefix = f"caps_{args.algorithm}_lamT0.0_lamS{args.coefficient}"
            train_function = train_caps
        train_args = method_args
        run_dirs = [save_dir / f"{prefix}_{seed}" for seed in seeds]
    else:
        from afqr_common.lipsnetpp_experiment import (
            LipsNetPlusPlusProfile,
            run_name,
        )

        profile = LipsNetPlusPlusProfile(
            name=f"matched_lk{coefficient_label}",
            lambda_k=args.coefficient,
            lambda_h=args.lambda_h,
            history_len=args.history_len,
        )
        train_args = {
            "profile_name": profile.name,
            "lipsnetpp_lambda_k": profile.lambda_k,
            "lipsnetpp_lambda_h": profile.lambda_h,
            "history_len": profile.history_len,
        }
        train_function = train_lipsnetpp
        run_dirs = [save_dir / run_name(args.algorithm, profile, seed) for seed in seeds]

    jobs = []
    for seed, run_dir in zip(seeds, run_dirs):
        final_path = run_dir / "final.zip"
        if final_path.is_file():
            print(f"[SKIP] {final_path}", flush=True)
            continue
        name = f"{args.algorithm}/{args.env}/{args.method}/{coefficient_label}/{seed}"
        jobs.append(
            (
                name,
                partial(
                    train_function,
                    seed,
                    env_timestep[args.env],
                    str(save_dir) + os.sep,
                    str(log_dir) + os.sep,
                    factories[args.env],
                    env_args[args.env],
                    train_args,
                    args.device,
                    False,
                ),
            )
        )

    print(
        f"algorithm={args.algorithm} method={args.method} env={args.env} "
        f"coefficient={args.coefficient} seeds={seeds} workers={args.workers} "
        f"pending={len(jobs)}",
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

    output_dir = (
        algorithm_root
        / "results"
        / "smoothness_matched"
        / args.method
        / args.env
        / f"coef_{coefficient_label}"
        / f"seeds{args.seed_offset}-{args.seed_offset + args.seeds - 1}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "lipsnetpp":
        from afqr_common.lipsnetpp_evaluation import evaluate_profiles
        if args.algorithm == "ppo":
            from ppo.models.lipsnetpp_ppo import LipsNetPlusPlusPPO as model_class
        else:
            from sac.models.lipsnetpp_sac import LipsNetPlusPlusSAC as model_class
        evaluate_profiles(
            args.algorithm,
            model_class,
            result_root,
            [args.env],
            [profile],
            seeds,
            evaluation_seeds,
            output_dir,
        )
    else:
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
            args.method,
            args.coefficient,
            list(zip(seeds, [run_dir / "final.zip" for run_dir in run_dirs])),
            evaluation_seeds,
            output_dir,
        )

    provenance = {
        "algorithm": args.algorithm,
        "method": args.method,
        "environment": args.env,
        "target_smoothness": args.target_smoothness,
        "coefficient": args.coefficient,
        "lambda_t": train_args.get("caps_lamT"),
        "lambda_s": train_args.get("caps_lamS"),
        "sigma": train_args.get("caps_sigma") if args.method == "caps_spatial" else None,
        "lambda_h": train_args.get("lipsnetpp_lambda_h"),
        "history_len": train_args.get("history_len"),
        "training_seeds": seeds,
        "evaluation_seeds": evaluation_seeds,
        "workers": args.workers,
        "deterministic_evaluation": True,
        "evaluation_protocol": (
            EVALUATION_PROTOCOL
            if args.method != "lipsnetpp"
            else "explicit_validation_seed_raw_env_with_history_v1"
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[RESULT] {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
