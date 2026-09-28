"""Train the original CAPS implementation with only its spatial penalty."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path


LAM_T = 0.0
LAM_S = 0.5


def run_job(job_name: str, train_func, *args, **kwargs) -> None:
    started = datetime.now()
    print(f"[START] {job_name} - {started:%Y-%m-%d %H:%M:%S}", flush=True)
    train_func(*args, **kwargs)
    finished = datetime.now()
    print(
        f"[DONE] {job_name} - {finished:%Y-%m-%d %H:%M:%S} "
        f"(elapsed: {finished - started})",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    parser.add_argument("--env", choices=("ant", "hopper", "walker"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only and args.eval_only:
        parser.error("--plan-only and --eval-only cannot be used together")

    repo_root = Path(__file__).resolve().parents[1]
    algorithm_root = repo_root / args.algorithm
    sys.path.insert(0, str(algorithm_root))

    from tests.modules import action_extractor
    from tests.modules.controller import train_caps
    from tests.modules.envs import make_ant_env, make_hopper_env, make_walker_env
    from tests.modules.params import alg_args, env_args, env_timestep

    env_factories = {
        "ant": make_ant_env,
        "hopper": make_hopper_env,
        "walker": make_walker_env,
    }
    seed_path = algorithm_root / "tests/seeds.txt"
    action_extractor.seed_file_path = str(
        algorithm_root / "tests/validation_seeds.txt"
    )
    seeds = [
        int(line.strip())
        for line in seed_path.read_text().splitlines()
        if line.strip()
    ][: args.seeds]
    if len(seeds) != args.seeds:
        raise ValueError(f"Expected {args.seeds} seeds, found {len(seeds)}")

    save_root = Path(args.algorithm) / "results/pths"
    log_root = Path(args.algorithm) / "results/tensorboard_logs"
    save_dir = save_root / args.env
    log_dir = log_root / args.env
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    spatial_args = alg_args["caps"][args.env].copy()
    spatial_args["caps_lamT"] = LAM_T
    spatial_args["caps_lamS"] = LAM_S
    print(
        f"algorithm={args.algorithm} env={args.env} seeds={seeds} "
        f"workers={args.workers} lamT={LAM_T} lamS={LAM_S} "
        f"sigma={spatial_args.get('caps_sigma')}",
        flush=True,
    )
    if args.plan_only:
        return

    prefix = f"caps_{args.algorithm}_lamT{LAM_T}_lamS{LAM_S}"
    jobs = []
    if args.eval_only:
        eval_missing = [
            seed
            for seed in seeds
            if not (save_dir / f"{prefix}_{seed}" / "final.zip").is_file()
        ]
        if eval_missing:
            raise RuntimeError(
                f"Evaluation-only mode is missing checkpoints for seeds: {eval_missing}"
            )
        print(f"[EVAL ONLY] found all {len(seeds)} checkpoints", flush=True)
    else:
        for seed in seeds:
            final_path = save_dir / f"{prefix}_{seed}" / "final.zip"
            if final_path.is_file():
                print(f"[SKIP] {final_path}", flush=True)
                continue
            job_name = f"{args.algorithm}/{args.env}/{prefix}_{seed}"
            jobs.append(
                (
                    job_name,
                    partial(
                        run_job,
                        job_name,
                        train_caps,
                        seed,
                        env_timestep[args.env],
                        str(save_dir) + os.sep,
                        str(log_dir) + os.sep,
                        env_factories[args.env],
                        env_args[args.env],
                        spatial_args,
                        device=args.device,
                    ),
                )
            )

    failures = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_name = {
                executor.submit(job): name for name, job in jobs
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001
                    print(f"[FAIL] {name}: {error}", flush=True)
                    failures.append(name)
    if failures:
        raise RuntimeError(f"Training failed for {len(failures)} seeds: {failures}")

    missing = [
        seed
        for seed in seeds
        if not (save_dir / f"{prefix}_{seed}" / "final.zip").is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing final checkpoints for seeds: {missing}")

    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_group = f"caps_spatial_only/{args.env}/{timestamp}_lamT{LAM_T}_lamS{LAM_S}"
    action_extractor.test_some_path(
        str(save_root) + os.sep,
        True,
        [prefix],
        output_group,
        target_envs=[args.env],
    )
    print(f"Results saved under {save_root / output_group}", flush=True)


if __name__ == "__main__":
    main()
