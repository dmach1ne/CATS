"""Train the selected LipsNet++ PPO configuration across seeds."""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.extend([REPO_ROOT, PPO_ROOT])

from cats_common.lipsnetpp_experiment import selected_profile, run_name
from tests.modules.controller import train_lipsnetpp
from tests.modules.envs import (
    make_ant_env,
    make_hopper_env,
    make_lunar_env,
    make_pendulum_env,
    make_walker_env,
)
from tests.modules.params import env_args, env_timestep


TRAIN_ENVS = {
    "ant": make_ant_env,
    "hopper": make_hopper_env,
    "lunar": make_lunar_env,
    "pendulum": make_pendulum_env,
    "walker": make_walker_env,
}
SAVE_ROOT = "./ppo/results/pths"
LOG_ROOT = "./ppo/results/tensorboard_logs"
SEED_FILE = "./ppo/tests/seeds.txt"


def _load_seeds(limit: int, offset: int = 0) -> list[int]:
    with open(SEED_FILE, encoding="utf-8") as seed_file:
        seeds = [
            int(line.strip())
            for line in seed_file
            if line.strip()
        ]
    return seeds[offset : offset + limit]


def _run(job_name: str, function) -> None:
    print(f"[START] {job_name}", flush=True)
    function()
    print(f"[DONE] {job_name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=TRAIN_ENVS, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent training jobs (default: number of training seeds)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.workers is None:
        args.workers = args.seeds
    if args.workers <= 0:
        parser.error("--workers must be positive")

    profile = selected_profile("ppo", args.env)
    profile_names = [profile.name]
    if args.seeds <= 0 or args.seed_offset < 0:
        raise ValueError("--seeds must be positive and --seed-offset nonnegative")
    seeds = _load_seeds(args.seeds, args.seed_offset)
    if len(seeds) != args.seeds:
        raise ValueError("requested seed range exceeds seeds.txt")
    target_envs = [args.env] if args.env else list(TRAIN_ENVS)

    jobs = []
    skipped = 0
    for env_name in target_envs:
        save_dir = os.path.join(SAVE_ROOT, env_name)
        log_dir = os.path.join(LOG_ROOT, env_name)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        for profile_name in profile_names:
            profile_args = {
                "profile_name": profile.name,
                "lipsnetpp_lambda_k": profile.lambda_k,
                "lipsnetpp_lambda_h": profile.lambda_h,
                "history_len": profile.history_len,
            }
            for seed in seeds:
                name = run_name("ppo", profile, seed)
                final_path = os.path.join(save_dir, name, "final.zip")
                if os.path.isfile(final_path):
                    print(f"[SKIP] {env_name}/{name}", flush=True)
                    skipped += 1
                    continue
                job_name = f"{env_name}/{name}"
                jobs.append(
                    (
                        job_name,
                        partial(
                            train_lipsnetpp,
                            seed,
                            env_timestep[env_name],
                            save_dir,
                            log_dir,
                            TRAIN_ENVS[env_name],
                            env_args[env_name],
                            profile_args,
                            args.device,
                            False,
                        ),
                    )
                )

    print(
        f"envs={target_envs} profiles={profile_names} seeds={seeds} "
        f"pending={len(jobs)} skipped={skipped}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_name = {
            executor.submit(_run, job_name, function): job_name
            for job_name, function in jobs
        }
        for future in as_completed(future_to_name):
            job_name = future_to_name[future]
            try:
                future.result()
            except Exception as error:
                print(f"[FAIL] {job_name}: {error}", flush=True)
                raise


if __name__ == "__main__":
    main()
