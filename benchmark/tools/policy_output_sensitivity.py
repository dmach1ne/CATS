"""Evaluate direct policy-output sensitivity on clean policy visitations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import numpy as np


ENV_NAMES = {'ant': 'Ant-v5', 'hopper': 'Hopper-v5', 'lunar': 'LunarLander-v3', 'pendulum': 'Pendulum-v1', 'walker': 'Walker2d-v5'}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_eval_seeds(path: Path, count: int) -> list[int]:
    seeds = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(seeds) < count:
        raise ValueError(f"{path} has {len(seeds)} seeds, expected at least {count}")
    selected = seeds[:count]
    if len(set(selected)) != len(selected):
        raise ValueError("Evaluation seeds must be unique")
    return selected


def load_saved_observation_scale(
    path: Path,
    algorithm: str,
    environment: str,
    eval_seeds: Sequence[int],
    noise_seed: int,
) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"Saved sensitivity result is incomplete: {path}")
    if payload.get("algorithm") != algorithm:
        raise ValueError(f"Saved sensitivity algorithm mismatch: {path}")
    if payload.get("environment") != ENV_NAMES[environment]:
        raise ValueError(f"Saved sensitivity environment mismatch: {path}")
    if payload.get("evaluation_seeds") != list(eval_seeds):
        raise ValueError(f"Saved sensitivity evaluation seeds mismatch: {path}")
    if payload.get("noise_seed") != noise_seed:
        raise ValueError(f"Saved sensitivity noise seed mismatch: {path}")
    scale = np.asarray(payload.get("shared_observation_scale"), dtype=np.float32)
    if scale.ndim != 1 or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(f"Invalid saved observation scale: {path}")
    return scale


def find_matching_models(save_dir: Path, algorithm_name: str) -> list[Path]:
    if not save_dir.is_dir():
        return []
    search_prefix = algorithm_name if algorithm_name.endswith("_") else algorithm_name + "_"
    matched: list[Path] = []
    for entry in save_dir.iterdir():
        name = entry.name.removesuffix(".zip")
        suffix = name[len(search_prefix):] if name.startswith(search_prefix) else ""
        if not suffix or not suffix.split("_", 1)[0].isdigit():
            continue
        if entry.is_dir() and (entry / "final.zip").is_file():
            matched.append(entry / "final.zip")
        elif entry.is_file() and entry.suffix == ".zip":
            matched.append(entry)
    return sorted(matched, key=lambda path: str(path))


def extract_train_seed(model_path: Path) -> str:
    model_name = model_path.parent.name if model_path.name == "final.zip" else model_path.stem
    match = re.search(r"_(\d+)$", model_name)
    return match.group(1) if match else model_name


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def parse_sigmas(value: str) -> list[float]:
    sigmas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not sigmas or sigmas[0] != 0.0:
        raise ValueError("Sigma grid must start at 0.0")
    if any(sigma < 0.0 for sigma in sigmas):
        raise ValueError("Sigma values must be nonnegative")
    if sigmas != sorted(set(sigmas)):
        raise ValueError("Sigma values must be strictly increasing and unique")
    return sigmas


def make_eval_env(
    env_factory: Callable[[str | None], Callable[[], Any]], model_path: Path
) -> tuple[Any, bool]:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec_normalize_path = model_path.parent / "vec_normalize.pkl"
    if vec_normalize_path.is_file():
        env = DummyVecEnv([env_factory(None)])
        env = VecNormalize.load(str(vec_normalize_path), env)
        env.training = False
        env.norm_reward = False
        return env, True
    return env_factory(None)(), False


def reset_eval_env(env: Any, use_vec_env: bool, seed: int) -> np.ndarray:
    if use_vec_env:
        env.seed(seed)
        return np.asarray(env.reset(), dtype=np.float32)
    obs, _ = env.reset(seed=seed)
    return np.asarray(obs, dtype=np.float32)


def collect_clean_observations(
    model: Any,
    env: Any,
    use_vec_env: bool,
    eval_seeds: Sequence[int],
    max_episode_steps: int | None,
) -> dict[str, np.ndarray]:
    if hasattr(model.policy, "set_training_mode"):
        model.policy.set_training_mode(False)

    observations: list[np.ndarray] = []
    raw_observations: list[np.ndarray] = []
    episode_ids: list[int] = []
    timesteps: list[int] = []
    state_eval_seeds: list[int] = []
    episode_lengths: list[int] = []

    for episode_id, eval_seed in enumerate(eval_seeds):
        obs = reset_eval_env(env, use_vec_env, int(eval_seed))
        timestep = 0
        while True:
            observations.append(obs.reshape(-1).copy())
            raw_obs = env.get_original_obs() if use_vec_env else obs
            raw_observations.append(np.asarray(raw_obs).reshape(-1).copy())
            episode_ids.append(episode_id)
            timesteps.append(timestep)
            state_eval_seeds.append(int(eval_seed))
            action, _ = model.predict(obs, deterministic=True)

            if use_vec_env:
                next_obs, _, done, _ = env.step(action)
                terminated = bool(done[0])
                truncated = False
            else:
                next_obs, _, terminated, truncated, _ = env.step(action)
                terminated = bool(terminated)
                truncated = bool(truncated)

            timestep += 1
            obs = np.asarray(next_obs, dtype=np.float32)
            reached_cap = max_episode_steps is not None and timestep >= max_episode_steps
            if terminated or truncated or reached_cap:
                break
        episode_lengths.append(timestep)

    if not observations:
        raise RuntimeError("Clean rollout produced no observations")
    arrays = {
        "obs": np.stack(observations).astype(np.float32, copy=False),
        "raw_obs": np.stack(raw_observations).astype(np.float32, copy=False),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "timestep": np.asarray(timesteps, dtype=np.int32),
        "eval_seed": np.asarray(state_eval_seeds, dtype=np.int64),
        "episode_length": np.asarray(episode_lengths, dtype=np.int32),
    }
    if not np.all(np.isfinite(arrays["obs"])):
        raise ValueError("Clean rollout observations contain non-finite values")
    if not np.all(np.isfinite(arrays["raw_obs"])):
        raise ValueError("Clean rollout raw observations contain non-finite values")
    return arrays





def make_own_noise_directions(
    rollout: dict[str, np.ndarray],
    observation_dim: int,
    noise_draws: int,
    noise_seed: int,
    algorithm: str,
    environment: str,
    train_seed: str,
) -> np.ndarray:
    directions = np.empty(
        (len(rollout["obs"]), noise_draws, observation_dim), dtype=np.float32
    )
    for episode_id in np.unique(rollout["episode_id"]):
        indices = np.flatnonzero(rollout["episode_id"] == episode_id)
        order = np.argsort(rollout["timestep"][indices])
        indices = indices[order]
        eval_seed = int(rollout["eval_seed"][indices[0]])
        rng = np.random.default_rng(
            stable_seed(noise_seed, algorithm, environment, train_seed, eval_seed, "own")
        )
        directions[indices] = rng.uniform(
            -1.0, 1.0, size=(len(indices), noise_draws, observation_dim)
        ).astype(np.float32)
    return directions








def _ensure_batch_output(values: Any, expected_rows: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if len(array) != expected_rows:
        raise ValueError(f"Expected {expected_rows} action rows, received {array.shape}")
    return array


def batched_policy_outputs(model: Any, observations: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    outputs: list[np.ndarray] = []
    for start in range(0, len(observations), batch_size):
        batch = observations[start : start + batch_size]
        obs_tensor, _ = model.policy.obs_to_tensor(batch)
        with torch.no_grad():
            action = model.policy._predict(obs_tensor, deterministic=True)
        outputs.append(_ensure_batch_output(action.detach().cpu().numpy(), len(batch)))
    return np.concatenate(outputs, axis=0)


def batched_environment_outputs(
    model: Any, observations: np.ndarray, batch_size: int
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(observations), batch_size):
        batch = observations[start : start + batch_size]
        action, _ = model.predict(batch, deterministic=True)
        outputs.append(_ensure_batch_output(action, len(batch)))
    return np.concatenate(outputs, axis=0)


def validate_batch_parity(model: Any, observations: np.ndarray) -> None:
    sample = observations[: min(3, len(observations))]
    policy_batch = batched_policy_outputs(model, sample, len(sample))
    env_batch = batched_environment_outputs(model, sample, len(sample))
    policy_single = np.concatenate(
        [batched_policy_outputs(model, row[None, :], 1) for row in sample], axis=0
    )
    env_single = np.concatenate(
        [batched_environment_outputs(model, row[None, :], 1) for row in sample], axis=0
    )
    np.testing.assert_allclose(policy_batch, policy_single, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(env_batch, env_single, rtol=1e-5, atol=1e-6)


def evaluate_sensitivity(
    model: Any,
    observations: np.ndarray,
    observation_scale: np.ndarray,
    directions: np.ndarray,
    sigmas: Sequence[float],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    if observations.ndim != 2:
        raise ValueError(f"Expected 2-D observations, received {observations.shape}")
    if directions.ndim != 3 or directions.shape[0] != len(observations):
        raise ValueError("Noise-direction state count does not match observations")
    if directions.shape[2] != observations.shape[1]:
        raise ValueError("Noise-direction observation dimension does not match")
    if observation_scale.shape != (observations.shape[1],):
        raise ValueError("Observation scale shape does not match observations")

    clean_policy = batched_policy_outputs(model, observations, batch_size)
    clean_environment = batched_environment_outputs(model, observations, batch_size)
    noise_draws = directions.shape[1]
    metrics: dict[str, dict[str, Any]] = {}

    for sigma in sigmas:
        if sigma == 0.0:
            policy_per_draw = np.zeros(noise_draws, dtype=np.float64)
            environment_per_draw = np.zeros(noise_draws, dtype=np.float64)
            normalized_input_mse = 0.0
            raw_input_mse = 0.0
        else:
            normalized_noise = float(sigma) * directions
            raw_noise = normalized_noise * observation_scale[None, None, :]
            noisy_observations = (
                observations[:, None, :] + raw_noise
            ).reshape(-1, observations.shape[1])
            noisy_policy = batched_policy_outputs(
                model, noisy_observations, batch_size
            ).reshape(len(observations), noise_draws, -1)
            noisy_environment = batched_environment_outputs(
                model, noisy_observations, batch_size
            ).reshape(len(observations), noise_draws, -1)
            policy_difference = noisy_policy - clean_policy[:, None, :]
            environment_difference = noisy_environment - clean_environment[:, None, :]
            policy_per_draw = 0.5 * np.mean(
                np.square(policy_difference, dtype=np.float64), axis=(0, 2)
            )
            environment_per_draw = 0.5 * np.mean(
                np.square(environment_difference, dtype=np.float64), axis=(0, 2)
            )
            normalized_input_mse = float(
                np.mean(np.square(normalized_noise, dtype=np.float64))
            )
            raw_input_mse = float(np.mean(np.square(raw_noise, dtype=np.float64)))

        policy_half_mse = float(np.mean(policy_per_draw))
        environment_half_mse = float(np.mean(environment_per_draw))
        metrics[str(float(sigma))] = {
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
    return metrics


def build_parser(algorithm: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Evaluate {algorithm.upper()} policy-output sensitivity"
    )
    parser.add_argument("--root_dir", default=f"./{algorithm}/results/pths/")
    parser.add_argument("--algs", nargs="+", required=True)
    parser.add_argument("--obs_scale_alg", required=True)
    parser.add_argument("--observation_scale_file", default=None)
    parser.add_argument("--env", choices=sorted(ENV_NAMES), required=True)
    parser.add_argument(
        "--sigmas", default="0.0,0.05,0.1,0.15,0.2,0.25,0.3"
    )
    parser.add_argument(
        "--output_dir", default=f"./{algorithm}/results/analysis/policy_output_sensitivity/"
    )
    parser.add_argument("--seed_file", default=f"./{algorithm}/tests/validation_seeds.txt")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--warmup_episodes", type=int, default=10)
    parser.add_argument("--expected_models", type=int, default=10)
    parser.add_argument("--max_models", type=int, default=None)
    parser.add_argument("--noise_draws", type=int, default=10)
    parser.add_argument("--noise_seed", type=int, default=20260717)
    parser.add_argument("--bank_seed", type=int, default=20260721)
    parser.add_argument("--shared_bank_per_method", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--own_only", action="store_true", default=True)
    return parser


def _resolve_model_paths(
    save_dir: Path,
    names: Sequence[str],
    algorithm: str,
    expected_models: int,
    max_models: int | None = None,
) -> dict[str, dict[str, Path]]:
    by_method: dict[str, dict[str, Path]] = {}
    for name in names:
        paths = find_matching_models(save_dir, name)
        suffix = f"_{algorithm}"
        if not paths and not name.endswith(suffix):
            paths = find_matching_models(save_dir, name + suffix)
        if max_models is not None:
            paths = paths[:max_models]
        if len(paths) != expected_models:
            raise RuntimeError(
                f"Expected {expected_models} models for {name}, found {len(paths)}"
            )
        seed_map = {extract_train_seed(path): path for path in paths}
        if len(seed_map) != len(paths):
            raise RuntimeError(f"Duplicate training seed found for {name}")
        by_method[name] = seed_map

    expected_seeds = set(next(iter(by_method.values())))
    for name, seed_map in by_method.items():
        if set(seed_map) != expected_seeds:
            raise RuntimeError(f"Training seeds for {name} do not match other methods")
    return by_method


def resolve_model_class(
    method: str, default_model_class: Any, method_model_classes: dict[str, Any] | None
) -> Any:
    if not method_model_classes:
        return default_model_class
    return method_model_classes.get(method, default_model_class)


def run_evaluation(
    algorithm: str,
    model_class: Any,
    env_factories: dict[str, Callable[[str | None], Callable[[], Any]]],
    argv: Sequence[str] | None = None,
    method_model_classes: dict[str, Any] | None = None,
) -> None:
    args = build_parser(algorithm).parse_args(argv)
    if args.warmup_episodes > args.eval_episodes:
        raise ValueError("warmup_episodes cannot exceed eval_episodes")
    if args.noise_draws != 10:
        raise ValueError("This frozen protocol requires exactly 10 noise draws")
    if args.shared_bank_per_method <= 0 or args.batch_size <= 0:
        raise ValueError("Shared-bank size and batch size must be positive")

    sigmas = parse_sigmas(args.sigmas)
    eval_seeds = load_eval_seeds(Path(args.seed_file), args.eval_episodes)
    save_dir = Path(args.root_dir) / args.env
    output_dir = Path(args.output_dir) / args.env
    output_path = output_dir / "policy_output_sensitivity.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    model_paths = _resolve_model_paths(
        save_dir, args.algs, algorithm, args.expected_models, args.max_models
    )
    scale_paths = None
    if args.observation_scale_file is None and args.obs_scale_alg not in model_paths:
        scale_paths = _resolve_model_paths(
            save_dir, [args.obs_scale_alg], algorithm, args.expected_models, args.max_models
        )[args.obs_scale_alg]
    elif args.observation_scale_file is None:
        scale_paths = model_paths[args.obs_scale_alg]

    train_seeds = sorted(
        next(iter(model_paths.values())),
        key=lambda value: int(value) if value.isdigit() else value,
    )
    if scale_paths is not None and set(scale_paths) != set(train_seeds):
        raise RuntimeError("Observation-scale policy seeds do not match evaluated policy seeds")
    clean_rollouts: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for method, seed_map in model_paths.items():
        for train_seed in train_seeds:
            model_path = seed_map[train_seed]
            env, use_vec_env = make_eval_env(env_factories[args.env], model_path)
            try:
                selected_model_class = resolve_model_class(
                    method, model_class, method_model_classes
                )
                model = selected_model_class.load(
                    str(model_path), env=env, device=args.device
                )
                rollout = collect_clean_observations(
                    model, env, use_vec_env, eval_seeds, args.max_episode_steps
                )
                clean_rollouts[(method, train_seed)] = rollout
                print(
                    f"Collected {method} seed {train_seed}: "
                    f"{len(rollout['obs'])} own-visit states",
                    flush=True,
                )
            finally:
                env.close()
                if "model" in locals():
                    del model

    if args.observation_scale_file is not None:
        shared_observation_scale = load_saved_observation_scale(
            Path(args.observation_scale_file),
            algorithm,
            args.env,
            eval_seeds,
            args.noise_seed,
        )
        observation_scale_source = str(Path(args.observation_scale_file).resolve())
    else:
        # Per-Base-policy std (+1e-8), followed by an average across policies.
        scale_vectors: list[np.ndarray] = []
        assert scale_paths is not None
        for train_seed in train_seeds:
            if (args.obs_scale_alg, train_seed) in clean_rollouts:
                rollout = clean_rollouts[(args.obs_scale_alg, train_seed)]
            else:
                model_path = scale_paths[train_seed]
                env, use_vec_env = make_eval_env(env_factories[args.env], model_path)
                try:
                    selected_model_class = resolve_model_class(
                        args.obs_scale_alg, model_class, method_model_classes
                    )
                    model = selected_model_class.load(
                        str(model_path), env=env, device=args.device
                    )
                    rollout = collect_clean_observations(
                        model, env, use_vec_env, eval_seeds, args.max_episode_steps
                    )
                finally:
                    env.close()
                    if "model" in locals():
                        del model
            warmup_mask = rollout["episode_id"] < args.warmup_episodes
            scale_vectors.append(
                np.std(rollout["obs"][warmup_mask], axis=0).astype(np.float32) + 1e-8
            )
        shared_observation_scale = np.mean(scale_vectors, axis=0).astype(np.float32)
        observation_scale_source = "recomputed from Base clean rollouts"
    if not np.all(np.isfinite(shared_observation_scale)):
        raise ValueError("Shared observation scale contains non-finite values")

    observation_shapes = {rollout["obs"].shape[1:] for rollout in clean_rollouts.values()}
    if len(observation_shapes) != 1:
        raise ValueError(f"Observation shapes differ across methods: {observation_shapes}")
    shared_banks: dict[str, dict[str, Any]] = {}
    pass  # Paper scope: own-state sensitivity only.

    records: list[dict[str, Any]] = []
    for method, seed_map in model_paths.items():
        for train_seed in train_seeds:
            model_path = seed_map[train_seed]
            rollout = clean_rollouts[(method, train_seed)]
            env, use_vec_env = make_eval_env(env_factories[args.env], model_path)
            try:
                selected_model_class = resolve_model_class(
                    method, model_class, method_model_classes
                )
                model = selected_model_class.load(
                    str(model_path), env=env, device=args.device
                )
                if hasattr(model.policy, "set_training_mode"):
                    model.policy.set_training_mode(False)
                validate_batch_parity(model, rollout["obs"])
                own_directions = make_own_noise_directions(
                    rollout, rollout["obs"].shape[1], args.noise_draws,
                    args.noise_seed, algorithm, args.env, train_seed,
                )
                own_metrics = evaluate_sensitivity(
                    model, rollout["obs"], shared_observation_scale,
                    own_directions, sigmas, args.batch_size,
                )
                shared_metrics = None
                shared_state_count = 0
                pass  # Shared-state evaluation is not reported.
                action_shape = list(
                    batched_policy_outputs(model, rollout["obs"][:1], 1).shape[1:]
                )
                records.append({
                    "method": method,
                    "train_seed": int(train_seed) if train_seed.isdigit() else train_seed,
                    "checkpoint": str(model_path),
                    "uses_vec_normalize": use_vec_env,
                    "shared_states_normalized_by_target_wrapper": use_vec_env,
                    "n_own_states": int(len(rollout["obs"])),
                    "episode_lengths": rollout["episode_length"].tolist(),
                    "observation_shape": list(rollout["obs"].shape[1:]),
                    "policy_action_shape": action_shape,
                    "own": own_metrics,
                    "shared": shared_metrics,
                })
                print(
                    f"Evaluated {method} seed {train_seed}: own={len(rollout['obs'])}, "
                    f"shared={shared_state_count}",
                    flush=True,
                )
            finally:
                env.close()
                if "model" in locals():
                    del model

    payload = {
        "format": "afqr_policy_output_sensitivity_v1",
        "status": "completed",
        "algorithm": algorithm,
        "environment": ENV_NAMES[args.env],
        "environment_key": args.env,
        "methods": args.algs,
        "obs_scale_method": args.obs_scale_alg,
        "obs_scale_protocol": "mean of per-Base-policy clean-rollout feature std vectors",
        "obs_scale_source": observation_scale_source,
        "shared_observation_scale": shared_observation_scale.tolist(),
        "sigmas": sigmas,
        "noise_distribution": "Uniform(-1, 1)",
        "noise_formula": "obs + sigma * shared_observation_scale * epsilon",
        "noise_draws_per_state": args.noise_draws,
        "noise_reuse_across_sigmas": True,
        "noise_seed": args.noise_seed,
        "bank_seed": args.bank_seed,
        "eval_episodes": args.eval_episodes,
        "warmup_episodes": args.warmup_episodes,
        "evaluation_seeds": eval_seeds,
        "models_per_method": args.expected_models,
        "primary_state_distribution": "each policy's own clean deterministic visitation",
        "primary_state_weighting": "uniform over all visited timesteps within each policy",
        "shared_bank_per_source_method": args.shared_bank_per_method,
        "shared_bank_total_states": 0 if args.own_only else args.shared_bank_per_method * len(args.algs),
        "shared_bank_sources": [] if args.own_only else args.algs,
        "own_only": args.own_only,
        "shared_bank_state_space": "raw environment observations",
        "shared_bank_actor_input_transform": (
            "each target policy's own VecNormalize wrapper, when present"
        ),
        "metric": "0.5 * mean squared deterministic action-output difference",
        "policy_output": "policy._predict(..., deterministic=True)",
        "environment_output": "model.predict(..., deterministic=True)",
        "batch_size": args.batch_size,
        "records": records,
        "shared_bank_metadata": {
            train_seed: {
                "source_counts": bank["source_counts"],
                "selected_indices": bank["selected_indices"],
            }
            for train_seed, bank in shared_banks.items()
        },
    }
    atomic_write_json(output_path, payload)
    print(f"Saved {len(records)} policy records to {output_path}", flush=True)
