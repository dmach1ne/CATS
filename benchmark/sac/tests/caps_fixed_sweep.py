import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.modules.envs import make_ant_env, make_hopper_env, make_walker_env, make_pendulum_env, make_lunar_env

try:
    from tests.modules.action_extractor import test_some_path
except ImportError:
    # Fallback: try adding TD3 path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../td3")))
    from tests.modules.action_extractor import test_some_path

from tests.modules.params import env_timestep, env_args, alg_args
from tests.modules.controller import train_caps_fixed

from datetime import datetime

train_envs_dict = {'ant': make_ant_env, 'hopper': make_hopper_env, 'lunar': make_lunar_env, 'pendulum': make_pendulum_env, 'walker': make_walker_env}

save_dir_root = "./sac/results/pths/"
logs_dir_root = "./sac/results/tensorboard_logs/"
seed_file_path = "./sac/tests/seeds.txt"


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None, 
                        choices=list(train_envs_dict.keys()),
                        help="Environment to test")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--workers", type=int, default=5, help="Max concurrent workers")
    parser.add_argument("--lamT", type=str, default=None,
                        help="Comma-separated lamT values (default: 0.1,1.0,5.0,10.0)")
    parser.add_argument("--lamS", type=str, default=None,
                        help="Comma-separated lamS values (default: 0.0)")
    parser.add_argument("--ps", type=str, default="0.0",
                        help="Comma-separated penalty start ratios (0.0=start, 0.5=midpoint, 1.0=never)")
    args = parser.parse_args()

    seeds = load_seeds(seed_file_path)[:args.seeds]

    lamT_values = [0.1, 1.0, 5.0, 10.0]
    if args.lamT:
        lamT_values = [float(l.strip()) for l in args.lamT.split(",")]

    lamS_values = [0.0]
    if args.lamS:
        lamS_values = [float(l.strip()) for l in args.lamS.split(",")]

    ps_values = [float(p.strip()) for p in args.ps.split(",")]

    # (lamT, lamS) grid
    pairs = [(t, s) for t in lamT_values for s in lamS_values]

    if args.env:
        target_envs = [args.env]
    else:
        target_envs = ['ant', 'hopper', 'lunar', 'pendulum', 'walker']

    print(f"=== CAPS FIXED SAC Grid Sweep ===")
    print(f"Target Environments: {target_envs}")
    print(f"LamT values: {lamT_values}")
    print(f"LamS values: {lamS_values}")
    print(f"Grid pairs ({len(pairs)}): {pairs}")
    print(f"Penalty start ratios: {ps_values}")
    print(f"Seeds: {seeds}")
    
    jobs = []
    skipped_count = 0
    
    for env_name in target_envs:
        save_dir = os.path.join(save_dir_root, env_name) + '/'
        log_dir = os.path.join(logs_dir_root, env_name) + '/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            time.sleep(0.1)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            time.sleep(0.1)

        for lamT, lamS in pairs:
          for ps_val in ps_values:
            if env_name in alg_args.get("caps", {}):
                base_args = alg_args["caps"][env_name].copy()
            else:
                base_args = {}
            current_alg_args = base_args.copy()
            current_alg_args["caps_lamT"] = lamT
            current_alg_args["caps_lamS"] = lamS
            current_alg_args["penalty_start_ratio"] = ps_val

            ps_suffix = f"_ps{ps_val}" if ps_val > 0 else ""
            for seed in seeds:
                folder_name = f"caps_fixed_sac_lamT{lamT}_lamS{lamS}{ps_suffix}_{seed}"
                final_zip_path = os.path.join(save_dir, folder_name, "final.zip")
                if os.path.isfile(final_zip_path):
                    print(f"[SKIP] {env_name}/{folder_name} - final.zip already exists")
                    skipped_count += 1
                    continue

                job_name = f"{env_name}/caps_fixed_sac_lamT{lamT}_lamS{lamS}{ps_suffix}_{seed}"
                jobs.append((
                    job_name,
                    partial(
                        run_job_with_logging, job_name, train_caps_fixed,
                        seed, env_timestep[env_name], save_dir, log_dir,
                        train_envs_dict[env_name], env_args[env_name], current_alg_args,
                        args.device
                    )
                ))
    
    print(f"Skipped {skipped_count} jobs, {len(jobs)} jobs remaining")
    
    if len(jobs) > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(job_func): job_name for job_name, job_func in jobs}
            for future in as_completed(future_to_job):
                job_name = future_to_job[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[ERROR] {job_name} generated an exception: {exc}")
    
    # Free training memory before eval
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== Evaluation ===")
    pth_names = []
    for lamT, lamS in pairs:
        for ps_val in ps_values:
            ps_suffix = f"_ps{ps_val}" if ps_val > 0 else ""
            pth_names.append(f"caps_fixed_sac_lamT{lamT}_lamS{lamS}{ps_suffix}")

    timestamp = datetime.now().strftime("%m%d_%H%M")
    pair_str = "_".join([f"T{t}S{s}" for t, s in pairs])
    ps_str = "_".join([str(p) for p in ps_values])

    env_folder = args.env if args.env else "all"
    sub_dir = f"caps_fixed_sweep/{env_folder}/{timestamp}_{pair_str}_ps{ps_str}"

    test_some_path(save_dir_root, True, pth_names, sub_dir, target_envs=target_envs)
    print("Results saved to:", os.path.join(save_dir_root, sub_dir))
