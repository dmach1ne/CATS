"""Deterministic clean-environment evaluation for LipsNet++ checkpoints."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from afqr_common.lipsnetpp_experiment import LipsNetPlusPlusProfile, run_name


ENVIRONMENT_IDS = {
    "ant": "Ant-v5",
    "hopper": "Hopper-v5",
    "lunar": "LunarLander-v3",
    "pendulum": "Pendulum-v1",
    "walker": "Walker2d-v5",
}


@dataclass(frozen=True)
class EvaluationSummary:
    algorithm: str
    environment: str
    profile: str
    lambda_k: float
    lambda_h: float
    history_len: int
    training_seeds: int
    evaluation_episodes_per_seed: int
    reward_mean: float
    reward_std: float
    smoothness_mean: float
    smoothness_std: float


def calculate_smoothness(actions: np.ndarray, sampling_frequency: float = 1.0) -> float:
    action_array = np.asarray(actions, dtype=float)
    if action_array.ndim == 1:
        action_array = action_array[:, None]
    sample_count = action_array.shape[0]
    if sample_count < 2:
        return 0.0
    magnitudes = np.abs(np.fft.fft(action_array, axis=0)[: sample_count // 2])
    frequencies = np.fft.fftfreq(
        sample_count,
        d=1 / sampling_frequency,
    )[: sample_count // 2, None]
    per_dimension = (
        2.0
        / (sample_count * sampling_frequency)
        * np.sum(frequencies * magnitudes, axis=0)
    )
    return float(np.mean(per_dimension))


def make_environment(environment: str):
    environment_id = ENVIRONMENT_IDS[environment]
    if environment == "lunar":
        return gym.make(environment_id, continuous=True)
    return gym.make(environment_id)


def _load_normalizer(
    model_dir: Path,
    environment: str,
) -> tuple[VecNormalize | None, DummyVecEnv | None]:
    normalizer_path = model_dir / "vec_normalize.pkl"
    if not normalizer_path.is_file():
        return None, None
    normalizer_env = DummyVecEnv([lambda: make_environment(environment)])
    normalizer = VecNormalize.load(normalizer_path, normalizer_env)
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer, normalizer_env


def _actor_observation(
    raw_observation: np.ndarray,
    normalizer: VecNormalize | None,
) -> np.ndarray:
    observation = np.asarray(raw_observation, dtype=np.float32)
    if normalizer is not None:
        observation = normalizer.normalize_obs(observation[None, :])[0]
    return observation


def _evaluate_checkpoint(
    model_class,
    checkpoint: Path,
    environment: str,
    history_len: int,
    evaluation_seeds: Sequence[int],
) -> tuple[float, float, list[dict[str, float | int]]]:
    model = model_class.load(checkpoint, device="cpu")
    normalizer, normalizer_env = _load_normalizer(checkpoint.parent, environment)
    env = make_environment(environment)
    episode_rows: list[dict[str, float | int]] = []
    try:
        for evaluation_seed in evaluation_seeds:
            raw_observation, _ = env.reset(seed=evaluation_seed)
            current = _actor_observation(raw_observation, normalizer)
            history = np.zeros(
                (history_len, current.shape[0]),
                dtype=np.float32,
            )
            history[0] = current
            actions = []
            reward_sum = 0.0

            while True:
                action, _ = model.predict(history, deterministic=True)
                actions.append(np.asarray(action, dtype=np.float32))
                raw_observation, reward, terminated, truncated, _ = env.step(action)
                reward_sum += float(reward)
                if terminated or truncated:
                    break
                current = _actor_observation(raw_observation, normalizer)
                history[1:] = history[:-1].copy()
                history[0] = current

            smoothness = calculate_smoothness(np.asarray(actions))
            episode_rows.append(
                {
                    "evaluation_seed": evaluation_seed,
                    "reward": reward_sum,
                    "smoothness": smoothness,
                }
            )
    finally:
        env.close()
        if normalizer_env is not None:
            normalizer_env.close()

    return (
        float(np.mean([row["reward"] for row in episode_rows])),
        float(np.mean([row["smoothness"] for row in episode_rows])),
        episode_rows,
    )


def evaluate_profiles(
    algorithm: str,
    model_class,
    result_root: Path,
    environments: Sequence[str],
    profiles: Sequence[LipsNetPlusPlusProfile],
    training_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
    output_dir: Path,
) -> list[EvaluationSummary]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    raw_rows = []

    for environment in environments:
        for profile in profiles:
            seed_rewards = []
            seed_smoothness = []
            for training_seed in training_seeds:
                name = run_name(algorithm, profile, training_seed)
                checkpoint = result_root / environment / name / "final.zip"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                reward, smoothness, episode_rows = _evaluate_checkpoint(
                    model_class,
                    checkpoint,
                    environment,
                    profile.history_len,
                    evaluation_seeds,
                )
                seed_rewards.append(reward)
                seed_smoothness.append(smoothness)
                for row in episode_rows:
                    raw_rows.append(
                        {
                            "algorithm": algorithm.upper(),
                            "environment": environment,
                            "profile": profile.name,
                            "training_seed": training_seed,
                            **row,
                        }
                    )

            ddof = 1 if len(training_seeds) > 1 else 0
            summaries.append(
                EvaluationSummary(
                    algorithm=algorithm.upper(),
                    environment=environment,
                    profile=profile.name,
                    lambda_k=profile.lambda_k,
                    lambda_h=profile.lambda_h,
                    history_len=profile.history_len,
                    training_seeds=len(training_seeds),
                    evaluation_episodes_per_seed=len(evaluation_seeds),
                    reward_mean=float(np.mean(seed_rewards)),
                    reward_std=float(np.std(seed_rewards, ddof=ddof)),
                    smoothness_mean=float(np.mean(seed_smoothness)),
                    smoothness_std=float(np.std(seed_smoothness, ddof=ddof)),
                )
            )

    summary_rows = [asdict(summary) for summary in summaries]
    _write_rows(output_dir / "summary.csv", summary_rows)
    _write_rows(output_dir / "episodes.csv", raw_rows)
    with (output_dir / "provenance.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "algorithm": algorithm.upper(),
                "result_root": str(result_root.resolve()),
                "environments": list(environments),
                "profiles": [profile.as_dict() for profile in profiles],
                "training_seeds": list(training_seeds),
                "evaluation_seeds": list(evaluation_seeds),
                "deterministic": True,
                "aggregation": (
                    "episode mean per training seed, then mean/sample std "
                    "across training seeds"
                ),
            },
            file,
            indent=2,
        )
    return summaries


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty evaluation output: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
