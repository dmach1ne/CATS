import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.modules.envs import make_ant_env, make_hopper_env, make_lunar_env, make_pendulum_env, make_walker_env

try:
    from tests.modules.action_extractor import test_some_path
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../td3")))
    from tests.modules.action_extractor import test_some_path

from tests.modules.controller import train_caps_linear_nn
from tests.modules.params import alg_args, env_args, env_timestep


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
        print(f"[DONE] {job_name} - {end_time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {end_time - start_time})")
        return result
    except Exception as e:
        end_time = datetime.now()
        print(f"[FAIL] {job_name} - {end_time.strftime('%Y-%m-%d %H:%M:%S')} (elapsed: {end_time - start_time}) - Error: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None, choices=list(train_envs_dict.keys()))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--alphas", type=str, default=None,
                        help="Comma-separated alpha values (default: 5.0,17.5,30.0,50.0)")
    parser.add_argument("--next_noise_sigma", type=float, default=0.1)
    args = parser.parse_args()

    seeds = load_seeds(seed_file_path)[:args.seeds]
    alpha_values = [5.0, 17.5, 30.0, 50.0]
    if args.alphas:
        alpha_values = [float(v.strip()) for v in args.alphas.split(",")]

    target_envs = [args.env] if args.env else ['ant', 'hopper', 'lunar', 'pendulum', 'walker']

    print("=== CAPS LINEAR NN SAC Alpha Sweep ===")
    print(f"Target Environments: {target_envs}")
    print(f"Alpha values: {alpha_values}")
    print(f"Next noise sigma: {args.next_noise_sigma}")
    print(f"Seeds: {seeds}")

    jobs = []
    skipped_count = 0
    for env_name in target_envs:
        save_dir = os.path.join(save_dir_root, env_name) + "/"
        log_dir = os.path.join(logs_dir_root, env_name) + "/"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        time.sleep(0.1)

        for alpha_val in alpha_values:
            base_args = alg_args.get("caps_linear", {}).get(env_name, {}).copy()
            current_alg_args = base_args.copy()
            current_alg_args["cats_alpha"] = alpha_val
            current_alg_args["next_noise_sigma"] = args.next_noise_sigma

            for seed in seeds:
                folder_name = f"caps_linear_nn_sac_alpha{alpha_val}_sigma{args.next_noise_sigma}_{seed}"
                final_zip_path = os.path.join(save_dir, folder_name, "final.zip")
                if os.path.isfile(final_zip_path):
                    print(f"[SKIP] {env_name}/{folder_name} - final.zip already exists")
                    skipped_count += 1
                    continue

                job_name = f"{env_name}/caps_linear_nn_sac_alpha{alpha_val}_sigma{args.next_noise_sigma}_{seed}"
                jobs.append((
                    job_name,
                    partial(
                        run_job_with_logging, job_name, train_caps_linear_nn,
                        seed, env_timestep[env_name], save_dir, log_dir,
                        train_envs_dict[env_name], env_args.get(env_name, {}),
                        current_alg_args, args.device, detail_info=True,
                    ),
                ))

    print(f"Skipped {skipped_count} jobs, {len(jobs)} jobs remaining")
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(job_func): job_name for job_name, job_func in jobs}
            for future in as_completed(future_to_job):
                job_name = future_to_job[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[ERROR] {job_name} generated an exception: {exc}")

    print("\n=== Evaluation ===")
    pth_names = [f"caps_linear_nn_sac_alpha{alpha}_sigma{args.next_noise_sigma}" for alpha in alpha_values]
    timestamp = datetime.now().strftime("%m%d_%H%M")
    alpha_str = "_".join([str(a) for a in alpha_values])
    env_folder = args.env if args.env else "all"
    sub_dir = f"caps_linear_nn_sweep/{env_folder}/{timestamp}_alpha{alpha_str}_sigma{args.next_noise_sigma}"

    test_some_path(save_dir_root, True, pth_names, sub_dir, target_envs=target_envs)
    print("Results saved to:", os.path.join(save_dir_root, sub_dir))
