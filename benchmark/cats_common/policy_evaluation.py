"""Seed-controlled deterministic evaluation for flat-observation policies."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cats_common.lipsnetpp_evaluation import calculate_smoothness


EVALUATION_PROTOCOL = "explicit_validation_seed_raw_env_cpu_v1"
ENVIRONMENT_IDS = {
    "ant": "Ant-v5",
    "hopper": "Hopper-v5",
    "lunar": "LunarLander-v3",
    "pendulum": "Pendulum-v1",
    "walker": "Walker2d-v5",
}


@dataclass(frozen=True)
class FlatPolicyEvaluationSummary:
    algorithm: str
    environment: str
    method: str
    coefficient: float
    training_seeds: int
    evaluation_episodes_per_seed: int
    reward_mean: float
    reward_std: float
    smoothness_mean: float
    smoothness_std: float
    evaluation_protocol: str = EVALUATION_PROTOCOL


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


def evaluate_flat_checkpoint(
    model_class,
    checkpoint: Path,
    environment: str,
    evaluation_seeds: Sequence[int],
) -> tuple[float, float, list[dict[str, float | int]]]:
    """Evaluate one checkpoint on the exact requested raw-environment seeds."""
    model = model_class.load(checkpoint, device="cpu")
    normalizer, normalizer_env = _load_normalizer(checkpoint.parent, environment)
    env = make_environment(environment)
    episode_rows: list[dict[str, float | int]] = []
    try:
        for evaluation_seed in evaluation_seeds:
            raw_observation, _ = env.reset(seed=evaluation_seed)
            observation = _actor_observation(raw_observation, normalizer)
            actions = []
            reward_sum = 0.0

            while True:
                action, _ = model.predict(observation, deterministic=True)
                actions.append(np.asarray(action, dtype=np.float32))
                raw_observation, reward, terminated, truncated, _ = env.step(action)
                reward_sum += float(reward)
                if terminated or truncated:
                    break
                observation = _actor_observation(raw_observation, normalizer)

            episode_rows.append(
                {
                    "evaluation_seed": evaluation_seed,
                    "reward": reward_sum,
                    "smoothness": calculate_smoothness(np.asarray(actions)),
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


def evaluate_flat_policy_checkpoints(
    algorithm: str,
    model_class,
    environment: str,
    method: str,
    coefficient: float,
    checkpoints: Sequence[tuple[int, Path]],
    evaluation_seeds: Sequence[int],
    output_dir: Path,
) -> FlatPolicyEvaluationSummary:
    """Aggregate episode means per training seed, then across training seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rewards = []
    seed_smoothness = []
    episode_rows = []

    for training_seed, checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        reward, smoothness, rows = evaluate_flat_checkpoint(
            model_class,
            checkpoint,
            environment,
            evaluation_seeds,
        )
        seed_rewards.append(reward)
        seed_smoothness.append(smoothness)
        for row in rows:
            episode_rows.append(
                {
                    "algorithm": algorithm.upper(),
                    "environment": environment,
                    "method": method,
                    "coefficient": coefficient,
                    "training_seed": training_seed,
                    **row,
                }
            )

    ddof = 1 if len(checkpoints) > 1 else 0
    summary = FlatPolicyEvaluationSummary(
        algorithm=algorithm.upper(),
        environment=environment,
        method=method,
        coefficient=coefficient,
        training_seeds=len(checkpoints),
        evaluation_episodes_per_seed=len(evaluation_seeds),
        reward_mean=float(np.mean(seed_rewards)),
        reward_std=float(np.std(seed_rewards, ddof=ddof)),
        smoothness_mean=float(np.mean(seed_smoothness)),
        smoothness_std=float(np.std(seed_smoothness, ddof=ddof)),
    )
    _write_rows(output_dir / "summary.csv", [asdict(summary)])
    _write_rows(output_dir / "episodes.csv", episode_rows)
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "algorithm": algorithm.upper(),
                "environment": environment,
                "method": method,
                "coefficient": coefficient,
                "checkpoints": [
                    {"training_seed": seed, "path": str(path.resolve())}
                    for seed, path in checkpoints
                ],
                "evaluation_seeds": list(evaluation_seeds),
                "deterministic": True,
                "normalization": (
                    "Raw observations are normalized with the checkpoint's frozen "
                    "VecNormalize statistics before policy inference."
                ),
                "aggregation": (
                    "episode mean per training seed, then mean/sample std across "
                    "training seeds"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty evaluation output: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
