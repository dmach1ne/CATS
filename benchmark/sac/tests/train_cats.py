import sys
from pathlib import Path
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from afqr_common.experiment_settings import selected_settings

from tests.modules.envs import make_ant_env, make_hopper_env, make_walker_env, make_pendulum_env, make_lunar_env

from tests.modules.action_extractor import test_some_path

from tests.modules.params import env_timestep, env_args, alg_args
from tests.modules.controller import train_caps_linear

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
    args = parser.parse_args()
    
    seeds = load_seeds(seed_file_path)[:args.seeds]
    
    if args.env is None:
        parser.error("--env is required")
    alpha_values = [float(selected_settings("sac", args.env)["cats"]["cats_alpha"])]
    
    if args.env:
        target_envs = [args.env]
    else:
        target_envs = ['ant', 'hopper', 'lunar', 'pendulum', 'walker']

    print("=== CATS SAC ===")
    print(f"Target Environments: {target_envs}")
    print(f"Alpha values: {alpha_values}")
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

        for alpha_val in alpha_values:
            if env_name in alg_args.get("caps_linear", {}):
                base_args = alg_args["caps_linear"][env_name].copy()
            else:
                base_args = {}
            current_alg_args = base_args.copy()
            current_alg_args["cats_alpha"] = alpha_val
            
            for seed in seeds:
                folder_name = f"caps_linear_sac_alpha{alpha_val}_{seed}"
                final_zip_path = os.path.join(save_dir, folder_name, "final.zip")
                if os.path.isfile(final_zip_path):
                    print(f"[SKIP] {env_name}/{folder_name} - final.zip already exists")
                    skipped_count += 1
                    continue
                
                job_name = f"{env_name}/caps_linear_sac_alpha{alpha_val}_{seed}"
                jobs.append((
                    job_name,
                    partial(
                        run_job_with_logging, job_name, train_caps_linear, 
                        seed, env_timestep[env_name], save_dir, log_dir, 
                        train_envs_dict[env_name], env_args.get(env_name, {}), current_alg_args, 
                        args.device, detail_info=True
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
    pth_names = [f"caps_linear_sac_alpha{alpha_val}" for alpha_val in alpha_values]
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    alpha_str = "_".join([str(a) for a in alpha_values])
    
    env_folder = args.env if args.env else "all"
    sub_dir = f"cats/{env_folder}/{timestamp}_alpha{alpha_str}"
    
    test_some_path(save_dir_root, True, pth_names, sub_dir, target_envs=target_envs)
    print("Results saved to:", os.path.join(save_dir_root, sub_dir))
