"""
Evaluate policy robustness under noisy observations.
- Warm-up rollout to compute per-feature obs std
- Add uniform noise scaled by per-feature std to obs before policy prediction
- Measure reward degradation and smoothness across noise levels

usage: python -m sac.tests.eval_noise_robustness --env walker
"""
import sys
import os
import re
import argparse
import hashlib
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from sac.tests.modules.envs import make_ant_env, make_hopper_env, make_lunar_env, make_pendulum_env, make_walker_env
from sac.tests.modules.action_extractor import calculate_smoothness_np
from sac.models.custom_sac import CustomSAC
from sac.tests.modules.params import alg_args
from cats_common.experiment_settings import selected_settings
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

seed_file_path = "./sac/tests/validation_seeds.txt"

train_envs_dict = {'ant': make_ant_env, 'hopper': make_hopper_env, 'lunar': make_lunar_env, 'pendulum': make_pendulum_env, 'walker': make_walker_env}


def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def find_matching_files(save_dir, al_name):
    """Find model files matching algorithm name. Handles both subdirectory and flat .zip formats."""
    if not os.path.isdir(save_dir):
        return []
    search_pattern = al_name if al_name.endswith('_') else al_name + '_'
    matched_files = []
    for fname in os.listdir(save_dir):
        name = fname.replace(".zip", "")
        if name.startswith(search_pattern) and name[len(search_pattern):].split('_')[0].isdigit():
            full_path = os.path.join(save_dir, fname)
            if os.path.isdir(full_path) and os.path.isfile(os.path.join(full_path, "final.zip")):
                matched_files.append(os.path.join(full_path, "final.zip"))
            elif fname.endswith(".zip"):
                matched_files.append(full_path)
    return sorted(matched_files)


def extract_train_seed(filename):
    model_name = filename.replace("/final.zip", "").replace(".zip", "")
    match = re.search(r'_(\d+)$', model_name)
    return match.group(1) if match else model_name


def make_eval_env(env_name, model_dir):
    """Create eval environment, with VecNormalize if available."""
    vec_norm_path = os.path.join(model_dir, 'vec_normalize.pkl')
    if os.path.isfile(vec_norm_path):
        eval_env = DummyVecEnv([train_envs_dict[env_name](None)])
        eval_env = VecNormalize.load(vec_norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        return eval_env, True
    else:
        env = train_envs_dict[env_name](None)()
        return env, False


def reset_eval_env(env, use_vec_env, seed):
    """Reset an environment to a specified evaluation seed."""
    if use_vec_env:
        env.seed(seed)
        return env.reset()
    obs, _ = env.reset(seed=seed)
    return obs


def make_noise_rng(base_seed, env_name, train_seed, eval_seed):
    """Create a stable common-random-number stream for one policy episode."""
    payload = f"{base_seed}|{env_name}|{train_seed}|{eval_seed}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def warmup_rollout(model, env, use_vec_env, n_episodes, seeds):
    """Run clean episodes to collect observations for per-feature std."""
    all_obs = []
    for episode_idx in range(n_episodes):
        eval_seed = seeds[episode_idx % len(seeds)]
        obs = reset_eval_env(env, use_vec_env, eval_seed)
        while True:
            all_obs.append(obs.flatten() if use_vec_env else obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            if use_vec_env:
                obs, _, done, _ = env.step(action)
                if done[0]:
                    break
            else:
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

    all_obs = np.array(all_obs)
    obs_std = np.std(all_obs, axis=0) + 1e-8
    return obs_std


def eval_with_noise(model, env, use_vec_env, sigma, obs_std, n_episodes, seeds,
                    noise_seed, env_name, train_seed):
    """Run episodes with noisy observations. Returns per-episode rewards and smoothness."""
    episode_rewards = []
    episode_smoothness = []

    for episode_idx in range(n_episodes):
        eval_seed = seeds[episode_idx % len(seeds)]
        obs = reset_eval_env(env, use_vec_env, eval_seed)
        rng = make_noise_rng(noise_seed, env_name, train_seed, eval_seed)

        actions = []
        total_reward = 0.0

        while True:
            # Add uniform noise scaled by per-feature std
            if sigma > 0:
                noise = sigma * obs_std * rng.uniform(-1, 1, size=obs.shape)
                obs_noisy = obs + noise
            else:
                obs_noisy = obs

            action, _ = model.predict(obs_noisy, deterministic=True)

            if use_vec_env:
                actions.append(action[0])
                obs, reward, done, _ = env.step(action)
                total_reward += reward[0]
                if done[0]:
                    break
            else:
                actions.append(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

        episode_rewards.append(total_reward)
        if len(actions) > 1:
            episode_smoothness.append(calculate_smoothness_np(np.array(actions)))

    return episode_rewards, episode_smoothness


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy robustness under noisy observations")
    parser.add_argument("--root_dir", type=str, default="./sac/results/pths/")
    parser.add_argument("--env", required=True, choices=("hopper", "walker"))
    parser.add_argument("--sigmas", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5",
                        help="Comma-separated noise levels")
    parser.add_argument("--output_dir", type=str, default="./sac/results/analysis/noise_robustness/")
    parser.add_argument("--warmup_episodes", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--max_seeds", type=int, default=None)
    parser.add_argument("--noise_seed", type=int, default=20260717,
                        help="Base seed for common observation-noise streams")
    args = parser.parse_args()
    alpha = float(selected_settings("sac", args.env)["cats"]["cats_alpha"])
    caps = alg_args["caps"][args.env]
    args.obs_scale_alg = "vanilla"
    args.algs = [
        args.obs_scale_alg,
        f"caps_sac_lamT{caps['caps_lamT']}_lamS{caps['caps_lamS']}",
        "l2c2_sac", "grad_sac", "asap_sac",
        f"cats_sac_alpha{alpha}",
    ]

    sigmas = [float(s.strip()) for s in args.sigmas.split(",")]
    seeds = load_seeds(seed_file_path)
    save_dir = os.path.join(args.root_dir, args.env)

    # Discover models
    all_alg_files = {}
    for al_name in args.algs:
        files = find_matching_files(save_dir, al_name)
        if not files and not al_name.endswith('_sac'):
            files = find_matching_files(save_dir, al_name + '_sac')
        if not files:
            raise FileNotFoundError(f"No models found for '{al_name}' in '{save_dir}'")
        if args.max_seeds:
            files = files[:args.max_seeds]
        all_alg_files[al_name] = files
        print(f"  {al_name}: {len(files)} models")

    if not all_alg_files:
        print("No models found. Exiting.")
        return

    # Compute shared obs_std from a reference policy warm-up.
    vanilla_name = args.obs_scale_alg
    scale_files = None
    if vanilla_name:
        scale_files = find_matching_files(save_dir, vanilla_name)
        if not scale_files and not vanilla_name.endswith('_sac'):
            scale_files = find_matching_files(save_dir, vanilla_name + '_sac')
        if args.max_seeds:
            scale_files = scale_files[:args.max_seeds]
        if not scale_files:
            raise FileNotFoundError(f"No observation-scale models found for '{vanilla_name}' in '{save_dir}'")
    else:
        for name in all_alg_files:
            if 'vanilla' in name:
                vanilla_name = name
                scale_files = all_alg_files[name]
                break

    shared_obs_std = None
    if vanilla_name:
        print(f"\n{'='*60}")
        print(f"Computing shared obs_std from vanilla warm-up")
        print(f"{'='*60}")
        all_warmup_obs = []
        for filename in scale_files:
            model_path = filename
            model_dir = os.path.dirname(model_path)
            env, use_vec_env = make_eval_env(args.env, model_dir)
            model = CustomSAC.load(model_path, env=env)
            obs_std = warmup_rollout(model, env, use_vec_env, args.warmup_episodes, seeds)
            all_warmup_obs.append(obs_std)
            if use_vec_env:
                env.close()
            else:
                env.close()
            del model
        shared_obs_std = np.mean(all_warmup_obs, axis=0)
        print(f"  Shared obs_std range [{shared_obs_std.min():.4f}, {shared_obs_std.max():.4f}] (from {len(all_warmup_obs)} vanilla models)")
    else:
        print("WARNING: No vanilla algorithm found. Each algorithm will use its own obs_std.")

    # Results: {alg: {sigma: {rewards: [...], smoothness: [...]}}}
    results = {}
    raw_records = []

    for al_name, files in all_alg_files.items():
        print(f"\n{'='*60}")
        print(f"Algorithm: {al_name} ({len(files)} models)")
        print(f"{'='*60}")

        alg_results = {s: {"rewards": [], "smoothness": []} for s in sigmas}

        for filename in files:
            train_seed = extract_train_seed(filename)
            model_path = filename  # find_matching_files returns full paths
            model_dir = os.path.dirname(model_path)

            env, use_vec_env = make_eval_env(args.env, model_dir)
            model = CustomSAC.load(model_path, env=env)

            if shared_obs_std is not None:
                obs_std = shared_obs_std
            else:
                obs_std = warmup_rollout(model, env, use_vec_env, args.warmup_episodes, seeds)
            print(f"  seed {train_seed}: obs_std range [{obs_std.min():.4f}, {obs_std.max():.4f}]")

            for sigma in sigmas:
                # Re-create env for clean state
                if use_vec_env:
                    env.close()
                    env, _ = make_eval_env(args.env, model_dir)
                    model = CustomSAC.load(model_path, env=env)

                ep_rewards, ep_smooth = eval_with_noise(
                    model, env, use_vec_env, sigma, obs_std,
                    args.eval_episodes, seeds, args.noise_seed, args.env,
                    train_seed
                )
                mean_r = np.mean(ep_rewards)
                mean_s = np.mean(ep_smooth) if ep_smooth else 0.0
                alg_results[sigma]["rewards"].append(mean_r)
                alg_results[sigma]["smoothness"].append(mean_s)
                raw_records.append({
                    "algorithm": al_name,
                    "train_seed": int(train_seed) if train_seed.isdigit() else train_seed,
                    "sigma": sigma,
                    "evaluation_seeds": seeds[:args.eval_episodes],
                    "episode_rewards": [float(value) for value in ep_rewards],
                    "episode_smoothness": [float(value) for value in ep_smooth],
                    "reward_mean": float(mean_r),
                    "smoothness_mean": float(mean_s),
                })
                print(f"    σ={sigma:.3f}: reward={mean_r:.1f}, smoothness={mean_s:.4f}")

            if use_vec_env:
                env.close()
            else:
                env.close()
            del model

        results[al_name] = alg_results

    # Output
    env_output_dir = os.path.join(args.output_dir, args.env)
    os.makedirs(env_output_dir, exist_ok=True)

    # Save JSON
    stats = {
        "env": args.env,
        "sigmas": sigmas,
        "warmup_episodes": args.warmup_episodes,
        "eval_episodes": args.eval_episodes,
        "obs_scale_alg": vanilla_name,
        "noise_seed": args.noise_seed,
        "common_noise_key": "environment/train_seed/evaluation_seed",
        "algorithms": {}
    }
    for al_name, alg_results in results.items():
        alg_stats = {}
        for sigma, data in alg_results.items():
            alg_stats[str(sigma)] = {
                "reward_mean": float(np.mean(data["rewards"])),
                "reward_std": float(np.std(data["rewards"], ddof=1)) if len(data["rewards"]) > 1 else 0.0,
                "smoothness_mean": float(np.mean(data["smoothness"])),
                "smoothness_std": float(np.std(data["smoothness"], ddof=1)) if len(data["smoothness"]) > 1 else 0.0,
                "n_seeds": len(data["rewards"]),
            }
        stats["algorithms"][al_name] = alg_stats

    with open(os.path.join(env_output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    raw_output = {
        "env": args.env,
        "sigmas": sigmas,
        "warmup_episodes": args.warmup_episodes,
        "eval_episodes": args.eval_episodes,
        "obs_scale_alg": vanilla_name,
        "noise_seed": args.noise_seed,
        "common_noise_key": "environment/train_seed/evaluation_seed",
        "records": raw_records,
    }
    with open(os.path.join(env_output_dir, "raw_results.json"), "w") as f:
        json.dump(raw_output, f, indent=2)

    print(f"\nResults saved to {env_output_dir}")


if __name__ == "__main__":
    main()
