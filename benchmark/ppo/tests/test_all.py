import sys
import gymnasium as gym
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse
from datetime import datetime

from stable_baselines3.common.env_util import make_vec_env

from modules.action_extractor import test_some_path

from modules.envs import make_ant_env, make_hopper_env, make_lunar_env, make_pendulum_env, make_walker_env

from modules.params import env_timestep, env_args, alg_args
from modules.controller import train_vanilla, train_caps, train_l2c2, train_grad, train_asap

train_envs_dict = dict({'ant': make_ant_env, 'hopper': make_hopper_env, 'lunar': make_lunar_env, 'pendulum': make_pendulum_env, 'walker': make_walker_env})

alg_cnts = dict({
    "vanilla": train_vanilla,
    "caps" : train_caps,
    "l2c2" : train_l2c2,
    "grad" : train_grad,
    "asap" : train_asap,
})

alg_file_patterns = {
    "vanilla": "vanilla_ppo_{seed}",
    "caps": "caps_ppo_{seed}",
    "l2c2": "l2c2_ppo_{seed}",
    "grad": "grad_ppo_{seed}",
    "asap": "asap_ppo_{seed}",
}

save_dir_root = f"./ppo/results/pths/"
logs_dir_root = f"./ppo/results/tensorboard_logs/"
seed_file_path = "./ppo/tests/seeds.txt"
result_dir_root = f"./ppo/results/"
max_num = 10

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def run_job_with_logging(job_name, train_func, *args, **kwargs):
    start_time = datetime.now()
    print(f"[START] {job_name} - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        result = train_func(*args, **kwargs)
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"[DONE] {job_name} - {end_time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {elapsed})")
        return result
    except Exception as e:
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"[FAIL] {job_name} - {end_time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {elapsed}) - Error: {e}")
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        nargs="+",
        choices=['ant', 'hopper', 'lunar', 'pendulum', 'walker'],
        default=['ant', 'hopper', 'lunar', 'pendulum', 'walker'],
        help="List of environments to train on."
    )
    parser.add_argument(
        "--algs",
        nargs="+",
        choices=list(alg_cnts.keys()),
        help="List of algorithms to train."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training: 'cpu', 'cuda', or 'cuda:<id>'."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers."
    )
    args = parser.parse_args()
    print(f"Selected envs: {args.envs}")
    print(f"Selected algs: {args.algs}")
    print(f"Selected device: {args.device}")
    print(f"Selected workers: {args.workers}")

    train_algs = args.algs
    train_envs = args.envs

    seeds = load_seeds(seed_file_path)
    if len(seeds) < max_num:
        raise ValueError(f"Not enough seeds in {seed_file_path}: required {max_num}, found {len(seeds)}")

    max_concurrent_num = args.workers

    jobs = []
    for env_name in train_envs:

        save_dir = os.path.join(save_dir_root, env_name) + '/'
        log_dir = os.path.join(logs_dir_root, env_name) + '/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            time.sleep(1)

        for num in range(max_num):
            seed = seeds[num]
            for alg_name in train_algs:
                expected_folder = alg_file_patterns[alg_name].format(seed=seed)
                expected_path = os.path.join(save_dir, expected_folder, "final.zip")

                if os.path.isfile(expected_path):
                    print(f"[SKIP] {env_name}/{alg_name}_{seed} - final.zip already exists")
                    continue

                job_name = f"{env_name}/{alg_name}_{seed}"
                jobs.append((
                    job_name,
                    partial(run_job_with_logging, job_name, alg_cnts[alg_name], seed, env_timestep[env_name], save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], alg_args[alg_name][env_name], device=args.device),
                ))

    print(f"\n=== Starting {len(jobs)} jobs with {max_concurrent_num} workers ===")
    with ProcessPoolExecutor(max_workers=max_concurrent_num) as executor:
        future_to_job = {executor.submit(job_func): job_name for job_name, job_func in jobs}

        for future in as_completed(future_to_job):
            job_name = future_to_job[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[ERROR] {job_name} generated an exception: {exc}")

    print(f"=== All jobs completed ===")

    eval_pth_names = []
    for alg in args.algs:
        if alg in alg_file_patterns:
            pattern = alg_file_patterns[alg]
            if "_{seed}" in pattern:
                prefix = pattern.split("_{seed}")[0]
                eval_pth_names.append(prefix)
            else:
                eval_pth_names.append(alg)
        else:
            print(f"[WARNING] No file pattern found for algorithm: {alg}")

    timestamp = datetime.now().strftime("%m%d_%H%M")
    alg_str = "_".join(args.algs) if len(args.algs) <= 4 else f"{len(args.algs)}algs"
    env_str = "_".join(args.envs) if len(args.envs) <= 3 else f"{len(args.envs)}envs"
    sub_dir = f"test_all/{timestamp}_{env_str}_{alg_str}"

    print(f"Evaluating envs: {args.envs}")
    print(f"Evaluating algs: {eval_pth_names}")

    test_some_path(save_dir_root, True, eval_pth_names, sub_dir, target_envs=args.envs)
    print(f"Results saved to: {save_dir_root}{sub_dir}/")
