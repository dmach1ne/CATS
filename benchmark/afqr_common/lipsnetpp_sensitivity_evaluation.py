"""Direct policy-output sensitivity for Base and selected LipsNet++ policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from afqr_common.lipsnetpp_evaluation import (
    _actor_observation,
    _load_normalizer,
    make_environment,
)
from afqr_common.lipsnetpp_experiment import LipsNetPlusPlusProfile
from afqr_common.lipsnetpp_noise_evaluation import PolicyCheckpoint


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _batched_policy_outputs(model, inputs: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    outputs = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        tensor, _ = model.policy.obs_to_tensor(batch)
        with torch.no_grad():
            action = model.policy._predict(tensor, deterministic=True)
        values = np.asarray(action.detach().cpu().numpy(), dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if len(values) != len(batch):
            raise ValueError(
                f"policy output rows {len(values)} != input rows {len(batch)}"
            )
        outputs.append(values)
    return np.concatenate(outputs, axis=0)


def _batched_environment_outputs(
    model,
    inputs: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        action, _ = model.predict(batch, deterministic=True)
        values = np.asarray(action, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if len(values) != len(batch):
            raise ValueError(
                f"environment output rows {len(values)} != input rows {len(batch)}"
            )
        outputs.append(values)
    return np.concatenate(outputs, axis=0)


def collect_clean_rollout(
    model,
    normalizer,
    environment: str,
    evaluation_seeds: Sequence[int],
    history_len: int | None,
) -> dict[str, np.ndarray]:
    """Collect deterministic trajectories and policy inputs."""
    raw_rows = []
    actor_rows = []
    policy_inputs = []
    episode_ids = []
    timesteps = []
    state_evaluation_seeds = []
    episode_lengths = []
    env = make_environment(environment)
    try:
        for episode_id, evaluation_seed in enumerate(evaluation_seeds):
            raw_observation, _ = env.reset(seed=int(evaluation_seed))
            history = None
            timestep = 0
            while True:
                actor_observation = _actor_observation(
                    raw_observation,
                    normalizer,
                )
                if history_len is None:
                    policy_input = actor_observation
                else:
                    if history is None:
                        history = np.zeros(
                            (history_len, actor_observation.shape[0]),
                            dtype=np.float32,
                        )
                    history[1:] = history[:-1].copy()
                    history[0] = actor_observation
                    policy_input = history.copy()

                raw_rows.append(
                    np.asarray(raw_observation, dtype=np.float32).copy()
                )
                actor_rows.append(actor_observation.copy())
                policy_inputs.append(policy_input.copy())
                episode_ids.append(episode_id)
                timesteps.append(timestep)
                state_evaluation_seeds.append(int(evaluation_seed))

                action, _ = model.predict(policy_input, deterministic=True)
                (
                    raw_observation,
                    _,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)
                timestep += 1
                if terminated or truncated:
                    break
            episode_lengths.append(timestep)
    finally:
        env.close()

    return {
        "raw_observation": np.stack(raw_rows),
        "actor_observation": np.stack(actor_rows),
        "policy_input": np.stack(policy_inputs),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "timestep": np.asarray(timesteps, dtype=np.int32),
        "evaluation_seed": np.asarray(
            state_evaluation_seeds,
            dtype=np.int64,
        ),
        "episode_length": np.asarray(episode_lengths, dtype=np.int32),
    }





def make_noise_directions(
    state_count: int,
    noise_draws: int,
    observation_dim: int,
    *seed_parts: object,
) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(*seed_parts))
    return rng.uniform(
        -1.0,
        1.0,
        size=(state_count, noise_draws, observation_dim),
    ).astype(np.float32)


def make_own_noise_directions(
    rollout: dict[str, np.ndarray],
    noise_draws: int,
    observation_dim: int,
    noise_seed: int,
    algorithm: str,
    environment: str,
    training_seed: int,
) -> np.ndarray:
    """Match the frozen Base/CAPS/L2C2/CATS per-episode noise stream."""
    directions = np.empty(
        (len(rollout["policy_input"]), noise_draws, observation_dim),
        dtype=np.float32,
    )
    for episode_id in np.unique(rollout["episode_id"]):
        indices = np.flatnonzero(rollout["episode_id"] == episode_id)
        indices = indices[np.argsort(rollout["timestep"][indices])]
        evaluation_seed = int(rollout["evaluation_seed"][indices[0]])
        rng = np.random.default_rng(
            stable_seed(
                noise_seed,
                algorithm,
                environment,
                training_seed,
                evaluation_seed,
                "own",
            )
        )
        directions[indices] = rng.uniform(
            -1.0,
            1.0,
            size=(len(indices), noise_draws, observation_dim),
        ).astype(np.float32)
    return directions


def evaluate_policy_inputs(
    model,
    policy_inputs: np.ndarray,
    observation_scale: np.ndarray,
    directions: np.ndarray,
    sigmas: Sequence[float],
    batch_size: int,
) -> dict[str, dict]:
    """Perturb the current observation and measure deterministic action change."""
    if policy_inputs.ndim not in (2, 3):
        raise ValueError(f"unsupported policy input shape: {policy_inputs.shape}")
    observation_dim = policy_inputs.shape[-1]
    if observation_scale.shape != (observation_dim,):
        raise ValueError("observation scale shape mismatch")
    if directions.shape != (
        len(policy_inputs),
        directions.shape[1],
        observation_dim,
    ):
        raise ValueError("noise direction shape mismatch")

    clean_policy = _batched_policy_outputs(model, policy_inputs, batch_size)
    clean_environment = _batched_environment_outputs(
        model,
        policy_inputs,
        batch_size,
    )
    noise_draws = directions.shape[1]
    results = {}
    for sigma in sigmas:
        if sigma == 0.0:
            policy_per_draw = np.zeros(noise_draws, dtype=np.float64)
            environment_per_draw = np.zeros(noise_draws, dtype=np.float64)
            normalized_input_mse = 0.0
            raw_input_mse = 0.0
        else:
            normalized_noise = float(sigma) * directions
            noise = (
                normalized_noise * observation_scale[None, None, :]
            ).astype(np.float32)
            if policy_inputs.ndim == 2:
                noisy = policy_inputs[:, None, :] + noise
            else:
                noisy = np.repeat(
                    policy_inputs[:, None, :, :],
                    noise_draws,
                    axis=1,
                )
                noisy[:, :, 0, :] += noise
            flattened = noisy.reshape(
                len(policy_inputs) * noise_draws,
                *policy_inputs.shape[1:],
            )
            noisy_policy = _batched_policy_outputs(
                model,
                flattened,
                batch_size,
            ).reshape(len(policy_inputs), noise_draws, -1)
            noisy_environment = _batched_environment_outputs(
                model,
                flattened,
                batch_size,
            ).reshape(len(policy_inputs), noise_draws, -1)
            policy_difference = noisy_policy - clean_policy[:, None, :]
            environment_difference = (
                noisy_environment - clean_environment[:, None, :]
            )
            policy_per_draw = 0.5 * np.mean(
                np.square(policy_difference, dtype=np.float64),
                axis=(0, 2),
            )
            environment_per_draw = 0.5 * np.mean(
                np.square(environment_difference, dtype=np.float64),
                axis=(0, 2),
            )
            normalized_input_mse = float(
                np.mean(np.square(normalized_noise, dtype=np.float64))
            )
            raw_input_mse = float(
                np.mean(np.square(noise, dtype=np.float64))
            )

        policy_half_mse = float(np.mean(policy_per_draw))
        environment_half_mse = float(np.mean(environment_per_draw))
        results[str(float(sigma))] = {
            "policy_half_mse": policy_half_mse,
            "environment_half_mse": environment_half_mse,
            "policy_half_mse_per_draw": policy_per_draw.tolist(),
            "environment_half_mse_per_draw": environment_per_draw.tolist(),
            "normalized_input_mse": normalized_input_mse,
            "raw_input_mse": raw_input_mse,
            "policy_gain": (
                policy_half_mse / normalized_input_mse
                if normalized_input_mse > 0.0
                else None
            ),
            "environment_gain": (
                environment_half_mse / normalized_input_mse
                if normalized_input_mse > 0.0
                else None
            ),
        }
    return results


def evaluate_lipsnetpp_policy_sensitivity(
    algorithm: str,
    base_model_class,
    lipsnetpp_model_class,
    base_checkpoints: Sequence[PolicyCheckpoint],
    lipsnetpp_checkpoints: Sequence[PolicyCheckpoint],
    environment: str,
    profile: LipsNetPlusPlusProfile,
    observation_scale: np.ndarray,
    evaluation_seeds: Sequence[int],
    sigmas: Sequence[float],
    noise_draws: int,
    noise_seed: int,
    batch_size: int,
    output_dir: Path,
    scale_provenance: Sequence[dict],
    device: str = "cpu",
) -> dict:
    base_by_seed = {item.training_seed: item for item in base_checkpoints}
    lipsnetpp_by_seed = {
        item.training_seed: item for item in lipsnetpp_checkpoints
    }
    if set(base_by_seed) != set(lipsnetpp_by_seed):
        raise ValueError("Base and LipsNet++ training seeds must match")
    if noise_draws != 10:
        raise ValueError("the frozen protocol requires 10 noise draws")

    records = []
    for training_seed in sorted(base_by_seed):
        base_checkpoint = base_by_seed[training_seed].path
        lipsnetpp_checkpoint = lipsnetpp_by_seed[training_seed].path
        base_model = base_model_class.load(base_checkpoint, device=device)
        lipsnetpp_model = lipsnetpp_model_class.load(
            lipsnetpp_checkpoint,
            device=device,
        )
        base_normalizer, base_normalizer_env = _load_normalizer(
            base_checkpoint.parent,
            environment,
        )
        lipsnetpp_normalizer, lipsnetpp_normalizer_env = _load_normalizer(
            lipsnetpp_checkpoint.parent,
            environment,
        )
        try:
            base_rollout = collect_clean_rollout(
                base_model,
                base_normalizer,
                environment,
                evaluation_seeds,
                history_len=None,
            )
            lipsnetpp_rollout = collect_clean_rollout(
                lipsnetpp_model,
                lipsnetpp_normalizer,
                environment,
                evaluation_seeds,
                history_len=profile.history_len,
            )
            pass
            pass

            method_specs = (("Base", base_model, base_rollout), ("LipsNet++", lipsnetpp_model, lipsnetpp_rollout))
            for method, model, own_rollout in method_specs:
                own_directions = make_own_noise_directions(
                    own_rollout,
                    noise_draws,
                    observation_scale.shape[0],
                    noise_seed,
                    algorithm,
                    environment,
                    training_seed,
                )
                records.append(
                    {'method': method, 'training_seed': training_seed, 'checkpoint': str((base_checkpoint if method == 'Base' else lipsnetpp_checkpoint).resolve()), 'own_state_count': len(own_rollout['policy_input']), 'episode_lengths': own_rollout['episode_length'].tolist(), 'own': evaluate_policy_inputs(model, own_rollout['policy_input'], observation_scale, own_directions, sigmas, batch_size)}
                )
        finally:
            if base_normalizer_env is not None:
                base_normalizer_env.close()
            if lipsnetpp_normalizer_env is not None:
                lipsnetpp_normalizer_env.close()

    output = {'format': 'afqr_lipsnetpp_policy_output_sensitivity_v1', 'status': 'completed', 'algorithm': algorithm, 'environment': environment, 'method_pair': ['Base', 'LipsNet++'], 'profile': profile.as_dict(), 'training_seeds': sorted(base_by_seed), 'evaluation_seeds': list(evaluation_seeds), 'sigmas': [float(value) for value in sigmas], 'noise_distribution': 'Uniform(-1, 1)', 'noise_formula': 'current_actor_observation + sigma * shared_base_observation_std * epsilon', 'history_noise_semantics': 'only history[0], the current observation, is perturbed; earlier history rows remain clean', 'primary_scope': 'own', 'primary_state_distribution': "each policy's own clean deterministic visitation", 'primary_state_weighting': 'uniform over all visited timesteps within each policy', 'own_noise_key': 'noise_seed/algorithm/environment/training_seed/evaluation_seed/own', 'noise_reuse_across_sigmas': True, 'metric': '0.5 * mean squared deterministic action-output difference', 'noise_draws_per_state': noise_draws, 'noise_seed': noise_seed, 'batch_size': batch_size, 'device': device, 'observation_scale': np.asarray(observation_scale).tolist(), 'base_scale_provenance': list(scale_provenance), 'records': records}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "policy_output_sensitivity.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    return output
