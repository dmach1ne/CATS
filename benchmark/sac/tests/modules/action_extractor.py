import os
from pathlib import Path

from cats_common.checkpoints import select_checkpoints

import numpy as np

import csv

from .envs import make_ant_env, make_hopper_env, make_lunar_env, make_pendulum_env, make_walker_env

from models.custom_sac import CustomSAC

max_loop = 10

seed_file_path = "./sac/tests/validation_seeds.txt"

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

train_envs_dict = dict({'ant': make_ant_env, 'hopper': make_hopper_env, 'lunar': make_lunar_env, 'pendulum': make_pendulum_env, 'walker': make_walker_env})

def calculate_smoothness_np(actions: np.ndarray, fs: float = 1.0) -> float:
    """
    NumPy 로 구현한 smoothness 지표.
    actions: shape (T,) 또는 (T, d)  (T = timestep 수, d = 액션 차원)
    fs: 샘플링 주파수 (기본 1.0)
    """
    # 1차원일 때 (T,) -> (T,1)
    a = np.array(actions, dtype=float)
    if a.ndim == 1:
        a = a[:, None]

    n = a.shape[0]
    if n < 2:
        return 0.0

    # FFT
    # axis=0 방향으로 fft, 양의 주파수 절반만 취함
    yf = np.fft.fft(a, axis=0)
    yf = np.abs(yf[: n // 2, :])    # shape (n//2, d)

    # 주파수 벡터 생성 (n//2 길이)
    freqs = np.fft.fftfreq(n, d=1/fs)[: n // 2]  # shape (n//2,)
    freqs = freqs.reshape(-1, 1)                  # (n//2,1)

    # 식: Sm = 2/(n*fs) * sum_i (M_i * f_i)
    smooth_per_dim = (2.0 / (n * fs)) * np.sum(freqs * yf, axis=0)  # shape (d,)

    # 다차원 액션이면 차원별 평균, 1차원 액션이면 그냥 원소 반환
    return float(np.mean(smooth_per_dim))

import traceback

def find_matching_files(save_dir: str, al_name: str) -> list[str]:
    if not os.path.isdir(save_dir):
        return []
    # al_name 뒤에 _가 없으면 추가해서 정확한 매칭 (alpha1 vs alpha10 구분)
    search_pattern = al_name if al_name.endswith('_') else al_name + '_'
    
    matched_files = []
    for fname in os.listdir(save_dir):
        full_path = os.path.join(save_dir, fname)
        # Check if fname starts with search pattern
        if fname.startswith(search_pattern):
            if os.path.isfile(full_path) and fname.endswith(".zip"):
                 # It's a file (e.g. .zip)
                 # We need only full path so load function can treat it
                 pass # Actually the original logic was returning just fname, but let's see how load uses it.
                 # The original used: model = LipsSAC.load(f"{save_dir}{filename}", ...)
                 matched_files.append(fname)
            elif os.path.isdir(full_path):
                # It's a directory, look for final.zip
                final_zip = os.path.join(full_path, "final.zip")
                if os.path.isfile(final_zip):
                    # We return directory/final.zip relative path
                    matched_files.append(os.path.join(fname, "final.zip"))
    return matched_files

def test_vanilla(root_dir, al_name, env_name, deterministic=True, mode="rgb_array", expected_seeds=None):
    try:
        seeds = load_seeds(seed_file_path)
        counter = 0
        if env_name not in train_envs_dict:
            raise ValueError(f"Unknown environment: {env_name}")
        save_dir = root_dir + env_name + "/"
        
        # Sort files to ensure deterministic order
        if expected_seeds is None:
            training_seed_file = Path(__file__).resolve().parents[1] / "seeds.txt"
            expected_seeds = load_seeds(training_seed_file)[:10]
        files = select_checkpoints(sorted(find_matching_files(save_dir, al_name)), expected_seeds)
        env = train_envs_dict[env_name](mode)()
        
        # Store mean values for each model (seed)
        seed_mean_smoothness = []
        seed_mean_rewards = []
        all_actions = []

        for filename in files:
            counter = 0  # 각 모델(학습 시드)마다 평가용 시드 인덱스를 0부터 시작하도록 초기화
            # os.path.join handles the path correctly whether filename implies a subdir or not
            model = CustomSAC.load(os.path.join(save_dir, filename), env=env)
            obs, info = env.reset(seed=seeds[counter % len(seeds)])
            counter += 1
            
            # Temporary lists for this specific model
            model_actions_list = []
            model_rewards_list = []
            
            for _ in range(max_loop):
                actions = []
                total_reward = 0.0
                while True:
                    action, _states = model.predict(obs, deterministic=deterministic)
                    actions.append(action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        obs, info = env.reset(seed=seeds[counter % len(seeds)])
                        counter += 1
                        break
                
                model_actions_list.append(np.array(actions))
                model_rewards_list.append(total_reward)
                all_actions.append(np.array(actions))
            
            # Calculate smoothness for each episode of this model
            model_smoothness_list = [calculate_smoothness_np(acts) for acts in model_actions_list]
            
            # Aggregate per-model statistics (Mean over 10 episodes)
            if len(model_smoothness_list) > 0:
                seed_mean_smoothness.append(np.mean(model_smoothness_list))
                seed_mean_rewards.append(np.mean(model_rewards_list))

        if len(seed_mean_smoothness) == 0:
            raise RuntimeError(f"No evaluation results for {al_name} on {env_name}")
        else:
            # Final statistics across seeds (N=len(files))
            smoothness_avg = np.mean(seed_mean_smoothness)
            smoothness_std = np.std(seed_mean_smoothness, ddof=1)
            mean_reward = np.mean(seed_mean_rewards)
            std_reward = np.std(seed_mean_rewards, ddof=1)
            print(f"Evaluated {len(files)} models (Seed-level aggregated): {[os.path.basename(os.path.dirname(f)) if 'final.zip' in f else f for f in files]}")
            return float(smoothness_avg), float(smoothness_std), float(mean_reward), float(std_reward), np.array(all_actions, dtype=object), len(files)
    except Exception as e:
        print(f"Error in test_vanilla: {e}")
        traceback.print_exc()
        raise

def test_some_path(root_dir, deterministic=True, add_al_names : list[str] = [], sub_dir="", target_envs=None, expected_seeds=None):
    basic_al = []
    basic_envs = target_envs if target_envs is not None else list(train_envs_dict.keys())

    all_al = basic_al + add_al_names

    sm_mean_rows = [["al_name"]+basic_envs]
    sm_std_rows = [["al_name"]+basic_envs]
    re_mean_rows = [["al_name"]+basic_envs]
    re_std_rows = [["al_name"]+basic_envs]

    combined_path = os.path.join(root_dir, sub_dir)
    os.makedirs(combined_path, exist_ok=True)
    
    for al_name in all_al:
        sm_mean_row = [al_name]
        sm_std_row = [al_name]
        re_mean_row = [al_name]
        re_std_row = [al_name]
        for env_name in basic_envs:
            smoothness_mean, smoothness_std, reward_mean, reward_std, action_np, _ = test_vanilla(root_dir, al_name, env_name, deterministic, expected_seeds=expected_seeds)
            
            if isinstance(action_np, np.ndarray) and action_np.size != 0:
                comb_env_path = os.path.join(combined_path, env_name)
                os.makedirs(comb_env_path, exist_ok=True)
                np.savez(
                    os.path.join(comb_env_path, f"{al_name}_action_data.npz"),
                    action_list=action_np
                )

            sm_mean_row.append(smoothness_mean)
            sm_std_row.append(smoothness_std)
            re_mean_row.append(reward_mean)
            re_std_row.append(reward_std)
        sm_mean_rows.append(sm_mean_row)
        sm_std_rows.append(sm_std_row)
        re_mean_rows.append(re_mean_row)
        re_std_rows.append(re_std_row)

    
    with open(os.path.join(combined_path, "sm_mean.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in sm_mean_rows:
            writer.writerow(row)
    
    with open(os.path.join(combined_path, "sm_std.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in sm_std_rows:
            writer.writerow(row)
    
    with open(os.path.join(combined_path, "re_mean.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in re_mean_rows:
            writer.writerow(row)
    
    with open(os.path.join(combined_path, "re_std.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in re_std_rows:
            writer.writerow(row)
