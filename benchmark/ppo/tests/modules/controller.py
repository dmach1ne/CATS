import json

import os

import time

import torch

from typing import Callable

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from stable_baselines3.common.callbacks import CheckpointCallback

from cats_common.lipsnetpp_experiment import LipsNetPlusPlusProfile, run_name

from cats_common.observation_history import ObservationHistoryVecEnv

from models.custom_ppo import CustomPPO, L2C2PPO, CAPSPPO

from models.grad_ppo import GRADPPO

from models.asap_only_s_ppo import ASAPPolicy_share

from models.asap_ppo import ASAPPPO

from models.cats_ppo import CATSPPO

from models.cats_next_noise_ppo import CATSNextNoisePPO

from models.lipsnetpp_ppo import LipsNetPlusPlusPPO, LipsNetPlusPlusPolicy

_ENV_ONLY_KEYS = {'n_envs', 'normalize'}

def _make_vec_env(mkenv_func, env_args, local_save_dir=None):
    """공통 환경 생성 헬퍼. DummyVecEnv + (VecNormalize) + VecMonitor를 반환."""
    n_envs = env_args.get('n_envs', 1)
    use_normalize = env_args.get('normalize', False)
    gamma = env_args.get('gamma', 0.99)
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    if use_normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=gamma)
    vec_env = VecMonitor(vec_env)
    return vec_env

def _save_vec_normalize(vec_env, local_save_dir):
    """Save VecNormalize statistics even when additional wrappers are present."""
    inner_env = vec_env
    while inner_env is not None:
        if isinstance(inner_env, VecNormalize):
            inner_env.save(os.path.join(local_save_dir, 'vec_normalize.pkl'))
            return
        inner_env = getattr(inner_env, 'venv', None)

def _filter_ppo_args(env_args, alg_args):
    """env_args에서 환경 전용 키를 제거하고 alg_args와 합쳐서 PPO 모델 인자를 반환."""
    ppo_args = {k: v for k, v in env_args.items() if k not in _ENV_ONLY_KEYS}
    ppo_args.update(alg_args)
    return ppo_args

def _make_lipsnetpp_vec_env(mkenv_func, env_args, history_len):
    """Apply baseline normalization before constructing observation histories."""
    n_envs = env_args.get('n_envs', 1)
    use_normalize = env_args.get('normalize', False)
    gamma = env_args.get('gamma', 0.99)
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    if use_normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            gamma=gamma,
        )
    vec_env = ObservationHistoryVecEnv(vec_env, history_len=history_len)
    return VecMonitor(vec_env)

def _json_default(value):
    return getattr(value, '__name__', repr(value))

def train_lipsnetpp(
    seed: int,
    total_time_steps: int,
    save_dir: str,
    log_dir: str,
    mkenv_func: Callable,
    env_args: dict,
    alg_args: dict,
    device: str = 'auto',
    detail_info: bool = True,
):
    """Train LipsNet++ while preserving the environment's baseline PPO config."""
    profile = LipsNetPlusPlusProfile(
        name=str(alg_args.get('profile_name', 'custom')),
        lambda_k=float(alg_args['lipsnetpp_lambda_k']),
        lambda_h=float(alg_args['lipsnetpp_lambda_h']),
        history_len=int(alg_args['history_len']),
    )
    pth_name = run_name('ppo', profile, seed)
    log_name = pth_name.upper()
    local_save_dir = os.path.join(save_dir, pth_name)
    os.makedirs(local_save_dir, exist_ok=True)

    vec_env = _make_lipsnetpp_vec_env(
        mkenv_func,
        env_args,
        profile.history_len,
    )
    ppo_args = _filter_ppo_args(env_args, {})
    policy_kwargs = dict(ppo_args.pop('policy_kwargs', {}))
    policy_kwargs.update(
        history_len=profile.history_len,
        lipsnetpp_lambda_k=profile.lambda_k,
        lipsnetpp_lambda_h=profile.lambda_h,
        lipsnetpp_kernel_scale=float(
            alg_args.get('lipsnetpp_kernel_scale', 0.02)
        ),
    )
    ppo_args['policy_kwargs'] = policy_kwargs

    with open(
        os.path.join(local_save_dir, 'experiment_config.json'),
        'w',
        encoding='utf-8',
    ) as config_file:
        json.dump(
            {
                'algorithm': 'PPO',
                'seed': seed,
                'total_timesteps': total_time_steps,
                'profile': profile.as_dict(),
                'environment_args': env_args,
                'model_args': ppo_args,
            },
            config_file,
            indent=2,
            default=_json_default,
        )

    model = LipsNetPlusPlusPPO(
        LipsNetPlusPlusPolicy,
        vec_env,
        verbose=0,
        tensorboard_log=log_dir,
        seed=seed,
        device=device,
        **ppo_args,
    )
    callback = None
    if detail_info:
        model.save(os.path.join(local_save_dir, 'mid_00000_steps'))
        callback = CheckpointCallback(
            save_freq=max(total_time_steps // 5, 1),
            save_path=local_save_dir,
            name_prefix='mid',
        )
    if torch.cuda.is_available() and model.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(model.device)
    training_started = time.perf_counter()
    model.learn(
        total_timesteps=total_time_steps,
        tb_log_name=log_name,
        callback=callback,
    )
    elapsed_seconds = time.perf_counter() - training_started
    peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(model.device))
        if torch.cuda.is_available() and model.device.type == 'cuda'
        else None
    )
    model.save(os.path.join(local_save_dir, 'final'))
    _save_vec_normalize(vec_env, local_save_dir)
    with open(
        os.path.join(local_save_dir, 'runtime_metrics.json'),
        'w',
        encoding='utf-8',
    ) as runtime_file:
        json.dump(
            {
                'elapsed_seconds': elapsed_seconds,
                'completed_timesteps': model.num_timesteps,
                'peak_cuda_memory_bytes': peak_cuda_memory_bytes,
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
                'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
                'node': os.environ.get('SLURMD_NODENAME'),
            },
            runtime_file,
            indent=2,
        )
    vec_env.close()
    del model

def train_vanilla(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable, env_args: dict, alg_args: dict, device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pth_name = f'vanilla_ppo_{seed}'
    log_name = f'PPO_{seed}'
    local_save_dir = os.path.join(save_dir, pth_name)
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    checkpoint_callback = CheckpointCallback(save_freq=total_time_steps // 5, save_path=local_save_dir, name_prefix='mid')
    vec_env = _make_vec_env(mkenv_func, env_args, local_save_dir)
    ppo_args = _filter_ppo_args(env_args, alg_args)
    model = CustomPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed, device=device, **ppo_args)
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))
    model.learn(total_timesteps=total_time_steps, tb_log_name=log_name, callback=checkpoint_callback)
    save_name = os.path.join(local_save_dir, 'final')
    model.save(save_name)
    _save_vec_normalize(vec_env, local_save_dir)
    vec_env.close()
    del model

def train_caps(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable, env_args: dict, alg_args: dict, device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if ('caps_lamT' in alg_args) and ('caps_lamS' in alg_args):
        lam_t = alg_args['caps_lamT']
        lam_s = alg_args['caps_lamS']
        pth_name = f'caps_ppo_lamT{lam_t}_lamS{lam_s}_{seed}'
        log_name = f'CAPS_PPO_lamT{lam_t}_lamS{lam_s}_{seed}'
    else:
        pth_name = f'caps_ppo_{seed}'
        log_name = f'CAPS_PPO_{seed}'
    local_save_dir = os.path.join(save_dir, pth_name)
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    checkpoint_callback = CheckpointCallback(save_freq=total_time_steps // 5, save_path=local_save_dir, name_prefix='mid')
    vec_env = _make_vec_env(mkenv_func, env_args, local_save_dir)
    ppo_args = _filter_ppo_args(env_args, alg_args)
    model = CAPSPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed, device=device, **ppo_args)
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))
    model.learn(total_timesteps=total_time_steps, tb_log_name=log_name, callback=checkpoint_callback)
    save_name = os.path.join(local_save_dir, 'final')
    model.save(save_name)
    _save_vec_normalize(vec_env, local_save_dir)
    vec_env.close()
    del model

def train_caps_fixed(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable, env_args: dict, alg_args: dict, device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if ('caps_lamT' in alg_args) and ('caps_lamS' in alg_args):
        lam_t = alg_args['caps_lamT']
        lam_s = alg_args['caps_lamS']
        pth_name = f'caps_fixed_ppo_lamT{lam_t}_lamS{lam_s}_{seed}'
        log_name = f'CAPS_FIXED_PPO_lamT{lam_t}_lamS{lam_s}_{seed}'
    else:
        pth_name = f'caps_fixed_ppo_{seed}'
        log_name = f'CAPS_FIXED_PPO_{seed}'
    local_save_dir = os.path.join(save_dir, pth_name)
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    checkpoint_callback = CheckpointCallback(save_freq=total_time_steps // 5, save_path=local_save_dir, name_prefix='mid')
    vec_env = _make_vec_env(mkenv_func, env_args, local_save_dir)
    ppo_args = _filter_ppo_args(env_args, alg_args)
    model = CAPSPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed, device=device, **ppo_args)
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))
    model.learn(total_timesteps=total_time_steps, tb_log_name=log_name, callback=checkpoint_callback)
    save_name = os.path.join(local_save_dir, 'final')
    model.save(save_name)
    _save_vec_normalize(vec_env, local_save_dir)
    vec_env.close()
    del model

def train_cats(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable, env_args: dict, alg_args: dict, device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if 'cats_alpha' in alg_args:
        alpha_val = alg_args['cats_alpha']
        pth_name = f'cats_ppo_alpha{alpha_val}_{seed}'
        log_name = f'CATS_PPO_alpha{alpha_val}_{seed}'
    else:
        pth_name = f'cats_ppo_{seed}'
        log_name = f'CATS_PPO_{seed}'
    local_save_dir = os.path.join(save_dir, pth_name)
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    checkpoint_callback = CheckpointCallback(save_freq=total_time_steps // 5, save_path=local_save_dir, name_prefix='mid')
    vec_env = _make_vec_env(mkenv_func, env_args, local_save_dir)
    ppo_args = _filter_ppo_args(env_args, alg_args)
    model = CATSPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed, device=device, **ppo_args)
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))
    model.learn(total_timesteps=total_time_steps, tb_log_name=log_name, callback=checkpoint_callback)
    save_name = os.path.join(local_save_dir, 'final')
    model.save(save_name)
    _save_vec_normalize(vec_env, local_save_dir)
    vec_env.close()
    del model

def train_cats_next_noise(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable, env_args: dict, alg_args: dict, device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if 'cats_alpha' in alg_args:
        alpha_val = alg_args['cats_alpha']
        sigma_val = alg_args.get('next_noise_sigma', 0.1)
        pth_name = f'cats_next_noise_ppo_alpha{alpha_val}_sigma{sigma_val}_{seed}'
        log_name = f'CATS_NEXT_NOISE_PPO_alpha{alpha_val}_sigma{sigma_val}_{seed}'
    else:
        pth_name = f'cats_next_noise_ppo_{seed}'
        log_name = f'CATS_NEXT_NOISE_PPO_{seed}'
    local_save_dir = os.path.join(save_dir, pth_name)
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    checkpoint_callback = CheckpointCallback(save_freq=total_time_steps // 5, save_path=local_save_dir, name_prefix='mid')
    vec_env = _make_vec_env(mkenv_func, env_args, local_save_dir)
    ppo_args = _filter_ppo_args(env_args, alg_args)
    model = CATSNextNoisePPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed, device=device, **ppo_args)
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))
    model.learn(total_timesteps=total_time_steps, tb_log_name=log_name, callback=checkpoint_callback)
    save_name = os.path.join(local_save_dir, 'final')
    model.save(save_name)
    _save_vec_normalize(vec_env, local_save_dir)
    vec_env.close()
    del model

def train_l2c2(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable, env_args: dict, alg_args: dict, device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pth_name = f'l2c2_ppo_{seed}'
    log_name = f'L2C2PPO_{seed}'
    local_save_dir = os.path.join(save_dir, pth_name)
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    checkpoint_callback = CheckpointCallback(save_freq=total_time_steps // 5, save_path=local_save_dir, name_prefix='mid')
    vec_env = _make_vec_env(mkenv_func, env_args, local_save_dir)
    ppo_args = _filter_ppo_args(env_args, alg_args)
    model = L2C2PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed, device=device, **ppo_args)
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))
    model.learn(total_timesteps=total_time_steps, tb_log_name=log_name, callback=checkpoint_callback)
    save_name = os.path.join(local_save_dir, 'final')
    model.save(save_name)
    _save_vec_normalize(vec_env, local_save_dir)
    vec_env.close()
    del model

def train_grad(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable, env_args: dict, alg_args: dict, device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if 'grad_lamT' in alg_args:
        lam_val = alg_args['grad_lamT']
        pth_name = f'grad_ppo_lamT{lam_val}_{seed}'
        log_name = f'GRAD_PPO_lamT{lam_val}_{seed}'
    else:
        pth_name = f'grad_ppo_{seed}'
        log_name = f'GRAD_PPO_{seed}'
    local_save_dir = os.path.join(save_dir, pth_name)
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    checkpoint_callback = CheckpointCallback(save_freq=total_time_steps // 5, save_path=local_save_dir, name_prefix='mid')
    vec_env = _make_vec_env(mkenv_func, env_args, local_save_dir)
    ppo_args = _filter_ppo_args(env_args, alg_args)
    model = GRADPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed, device=device, **ppo_args)
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))
    model.learn(total_timesteps=total_time_steps, tb_log_name=log_name, callback=checkpoint_callback)
    save_name = os.path.join(local_save_dir, 'final')
    model.save(save_name)
    _save_vec_normalize(vec_env, local_save_dir)
    vec_env.close()
    del model

def train_asap(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable, env_args: dict, alg_args: dict, device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pth_name = f'asap_ppo_{seed}'
    log_name = f'ASAP_PPO_{seed}'
    local_save_dir = os.path.join(save_dir, pth_name)
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)
    checkpoint_callback = CheckpointCallback(save_freq=total_time_steps // 5, save_path=local_save_dir, name_prefix='mid')
    vec_env = _make_vec_env(mkenv_func, env_args, local_save_dir)
    ppo_args = _filter_ppo_args(env_args, alg_args)
    model = ASAPPPO(ASAPPolicy_share, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed, device=device, **ppo_args)
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))
    model.learn(total_timesteps=total_time_steps, tb_log_name=log_name, callback=checkpoint_callback)
    save_name = os.path.join(local_save_dir, 'final')
    model.save(save_name)
    _save_vec_normalize(vec_env, local_save_dir)
    vec_env.close()
    del model
