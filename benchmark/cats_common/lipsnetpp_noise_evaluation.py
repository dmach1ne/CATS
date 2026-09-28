"""Paired observation-noise evaluation for selected LipsNet++ policies."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from cats_common.lipsnetpp_evaluation import (
    _actor_observation,
    _load_normalizer,
    calculate_smoothness,
    make_environment,
)
from cats_common.lipsnetpp_experiment import LipsNetPlusPlusProfile


@dataclass(frozen=True)
class PolicyCheckpoint:
    training_seed: int
    path: Path


def make_noise_rng(
    base_seed: int,
    environment: str,
    training_seed: int,
    evaluation_seed: int,
) -> np.random.Generator:
    """Match the existing CATS paired-noise common-random-number stream."""
    payload = (
        f"{base_seed}|{environment}|{training_seed}|{evaluation_seed}"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def noisy_observation(
    observation: np.ndarray,
    observation_scale: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    scale = np.asarray(observation_scale, dtype=np.float32)
    if observation.shape != scale.shape:
        raise ValueError(
            f"observation/scale shape mismatch: {observation.shape} != {scale.shape}"
        )
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0:
        return observation.copy()
    noise = sigma * scale * rng.uniform(-1.0, 1.0, size=observation.shape)
    return (observation + noise).astype(np.float32, copy=False)


def advance_history(
    history: np.ndarray,
    current_observation: np.ndarray,
) -> np.ndarray:
    """Insert the current policy observation at row zero."""
    next_history = np.zeros_like(history)
    next_history[0] = current_observation
    next_history[1:] = history[:-1]
    return next_history


def _validate_checkpoints(checkpoints: Sequence[PolicyCheckpoint]) -> None:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    seeds = [checkpoint.training_seed for checkpoint in checkpoints]
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    missing = [
        str(checkpoint.path)
        for checkpoint in checkpoints
        if not checkpoint.path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing checkpoints: " + ", ".join(missing))


def compute_base_observation_scale(
    model_class,
    checkpoints: Sequence[PolicyCheckpoint],
    environment: str,
    evaluation_seeds: Sequence[int],
    warmup_episodes: int,
    device: str = "cpu",
) -> tuple[np.ndarray, list[dict]]:
    """Average per-policy observation stds for the CATS evaluation protocol."""
    _validate_checkpoints(checkpoints)
    if warmup_episodes <= 0 or warmup_episodes > len(evaluation_seeds):
        raise ValueError("warmup_episodes must fit within evaluation_seeds")

    policy_scales = []
    provenance = []
    for checkpoint in checkpoints:
        model = model_class.load(checkpoint.path, device=device)
        normalizer, normalizer_env = _load_normalizer(
            checkpoint.path.parent,
            environment,
        )
        env = make_environment(environment)
        observations = []
        try:
            for evaluation_seed in evaluation_seeds[:warmup_episodes]:
                raw_observation, _ = env.reset(seed=evaluation_seed)
                while True:
                    actor_observation = _actor_observation(
                        raw_observation,
                        normalizer,
                    )
                    observations.append(actor_observation.copy())
                    action, _ = model.predict(
                        actor_observation,
                        deterministic=True,
                    )
                    (
                        raw_observation,
                        _,
                        terminated,
                        truncated,
                        _,
                    ) = env.step(action)
                    if terminated or truncated:
                        break
        finally:
            env.close()
            if normalizer_env is not None:
                normalizer_env.close()

        policy_scale = np.std(np.asarray(observations), axis=0) + 1e-8
        policy_scales.append(policy_scale)
        provenance.append(
            {
                "training_seed": checkpoint.training_seed,
                "checkpoint": str(checkpoint.path.resolve()),
                "observation_count": len(observations),
                "scale_min": float(policy_scale.min()),
                "scale_max": float(policy_scale.max()),
            }
        )
    return np.mean(policy_scales, axis=0), provenance


def _evaluate_episode(
    model,
    normalizer,
    environment: str,
    history_len: int,
    observation_scale: np.ndarray,
    sigma: float,
    noise_seed: int,
    training_seed: int,
    evaluation_seed: int,
) -> tuple[float, float]:
    env = make_environment(environment)
    rng = make_noise_rng(
        noise_seed,
        environment,
        training_seed,
        evaluation_seed,
    )
    actions = []
    reward_sum = 0.0
    try:
        raw_observation, _ = env.reset(seed=evaluation_seed)
        actor_observation = _actor_observation(raw_observation, normalizer)
        history = np.zeros(
            (history_len, actor_observation.shape[0]),
            dtype=np.float32,
        )
        while True:
            policy_observation = noisy_observation(
                actor_observation,
                observation_scale,
                sigma,
                rng,
            )
            history = advance_history(history, policy_observation)
            action, _ = model.predict(history, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            actions.append(action)
            (
                raw_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)
            reward_sum += float(reward)
            if terminated or truncated:
                break
            actor_observation = _actor_observation(raw_observation, normalizer)
    finally:
        env.close()
    return reward_sum, calculate_smoothness(np.asarray(actions))


def evaluate_lipsnetpp_noise(
    algorithm: str,
    model_class,
    checkpoints: Sequence[PolicyCheckpoint],
    environment: str,
    profile: LipsNetPlusPlusProfile,
    observation_scale: np.ndarray,
    sigmas: Sequence[float],
    evaluation_seeds: Sequence[int],
    noise_seed: int,
    output_dir: Path,
    base_scale_provenance: Sequence[dict],
    device: str = "cpu",
) -> dict:
    """Evaluate exact return and smoothness under paired observation noise."""
    _validate_checkpoints(checkpoints)
    if not sigmas or any(sigma < 0 for sigma in sigmas):
        raise ValueError("sigmas must be a non-empty sequence of nonnegative values")
    if not evaluation_seeds:
        raise ValueError("evaluation_seeds must not be empty")

    raw_records = []
    per_sigma = {
        float(sigma): {"rewards": [], "smoothness": []}
        for sigma in sigmas
    }
    for checkpoint in checkpoints:
        model = model_class.load(checkpoint.path, device=device)
        normalizer, normalizer_env = _load_normalizer(
            checkpoint.path.parent,
            environment,
        )
        try:
            for sigma in sigmas:
                episode_rewards = []
                episode_smoothness = []
                for evaluation_seed in evaluation_seeds:
                    reward, smoothness = _evaluate_episode(
                        model,
                        normalizer,
                        environment,
                        profile.history_len,
                        observation_scale,
                        float(sigma),
                        noise_seed,
                        checkpoint.training_seed,
                        evaluation_seed,
                    )
                    episode_rewards.append(reward)
                    episode_smoothness.append(smoothness)

                reward_mean = float(np.mean(episode_rewards))
                smoothness_mean = float(np.mean(episode_smoothness))
                per_sigma[float(sigma)]["rewards"].append(reward_mean)
                per_sigma[float(sigma)]["smoothness"].append(smoothness_mean)
                raw_records.append(
                    {
                        "training_seed": checkpoint.training_seed,
                        "checkpoint": str(checkpoint.path.resolve()),
                        "sigma": float(sigma),
                        "evaluation_seeds": list(evaluation_seeds),
                        "episode_rewards": episode_rewards,
                        "episode_smoothness": episode_smoothness,
                        "reward_mean": reward_mean,
                        "smoothness_mean": smoothness_mean,
                    }
                )
        finally:
            if normalizer_env is not None:
                normalizer_env.close()

    stats = {}
    for sigma, values in per_sigma.items():
        count = len(values["rewards"])
        ddof = 1 if count > 1 else 0
        stats[str(sigma)] = {
            "reward_mean": float(np.mean(values["rewards"])),
            "reward_std": float(np.std(values["rewards"], ddof=ddof)),
            "smoothness_mean": float(np.mean(values["smoothness"])),
            "smoothness_std": float(np.std(values["smoothness"], ddof=ddof)),
            "n_seeds": count,
        }

    output = {
        "algorithm": algorithm.upper(),
        "environment": environment,
        "method": "LipsNet++",
        "profile": profile.as_dict(),
        "sigmas": [float(sigma) for sigma in sigmas],
        "evaluation_seeds": list(evaluation_seeds),
        "noise_seed": noise_seed,
        "noise_distribution": "uniform(-1, 1)",
        "noise_formula": "sigma * shared_base_observation_std * U(-1, 1)",
        "common_noise_key": "environment/training_seed/evaluation_seed",
        "history_noise_semantics": (
            "the noisy current observation is inserted into the policy history "
            "at every environment step"
        ),
        "device": device,
        "observation_scale": np.asarray(observation_scale).tolist(),
        "base_scale_provenance": list(base_scale_provenance),
        "stats": stats,
        "records": raw_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_results.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    (output_dir / "stats.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in output.items()
                if key != "records"
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
