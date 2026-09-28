"""Collect raw transitions for later transition-conditioned L_sp analyses."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Callable, Sequence

import numpy as np


ENV_NAMES = {'ant': 'Ant-v5', 'hopper': 'Hopper-v5', 'lunar': 'LunarLander-v3', 'pendulum': 'Pendulum-v1', 'walker': 'Walker2d-v5'}


def load_eval_seeds(path: Path, count: int) -> list[int]:
    seeds = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(seeds) < count:
        raise ValueError(f"{path} contains {len(seeds)} seeds, expected at least {count}")
    selected = seeds[:count]
    if len(set(selected)) != count:
        raise ValueError(f"First {count} seeds in {path} are not unique")
    return selected


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


def derive_action_seed(eval_seed: int, episode_index: int) -> int:
    # Stable across methods and training seeds, independent of Python hash randomization.
    return int((eval_seed * 1_664_525 + 1_013_904_223 + episode_index) % (2**31 - 1))


def seed_action_rng(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).copy()


def collect_policy_transitions(
    model: Any,
    env: Any,
    eval_seeds: Sequence[int],
    rollout_mode: str,
    max_episode_steps: int | None = None,
    observation_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if rollout_mode not in {"stochastic", "deterministic"}:
        raise ValueError(f"Unsupported rollout mode: {rollout_mode}")

    if hasattr(model, "policy") and hasattr(model.policy, "set_training_mode"):
        model.policy.set_training_mode(False)

    rows: dict[str, list[Any]] = {
        "obs": [],
        "next_obs": [],
        "actor_obs": [],
        "next_actor_obs": [],
        "mean_action": [],
        "next_mean_action": [],
        "sampled_action": [],
        "reward": [],
        "terminated": [],
        "truncated": [],
        "episode_id": [],
        "timestep": [],
        "eval_seed": [],
        "action_seed": [],
    }
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_action_seeds: list[int] = []
    use_sde = bool(getattr(model, "use_sde", False))
    sde_sample_freq = int(getattr(model, "sde_sample_freq", -1))
    sde_noise_reset_count = 0
    if use_sde and not callable(getattr(getattr(model, "policy", None), "reset_noise", None)):
        raise RuntimeError("gSDE model policy does not expose reset_noise")

    for episode_id, eval_seed in enumerate(eval_seeds):
        obs, _ = env.reset(seed=int(eval_seed))
        action_seed = derive_action_seed(int(eval_seed), episode_id)
        seed_action_rng(action_seed)
        episode_return = 0.0
        timestep = 0

        while True:
            actor_obs = (
                observation_transform(np.asarray(obs, dtype=np.float32))
                if observation_transform is not None
                else np.asarray(obs, dtype=np.float32)
            )
            mean_action, _ = model.predict(actor_obs, deterministic=True)
            if rollout_mode == "stochastic":
                if use_sde and (
                    timestep == 0
                    or (sde_sample_freq > 0 and timestep % sde_sample_freq == 0)
                ):
                    model.policy.reset_noise(1)
                    sde_noise_reset_count += 1
                sampled_action, _ = model.predict(actor_obs, deterministic=False)
            else:
                sampled_action = np.asarray(mean_action).copy()

            next_obs, reward, terminated, truncated, _ = env.step(sampled_action)
            next_actor_obs = (
                observation_transform(np.asarray(next_obs, dtype=np.float32))
                if observation_transform is not None
                else np.asarray(next_obs, dtype=np.float32)
            )
            next_mean_action, _ = model.predict(next_actor_obs, deterministic=True)

            rows["obs"].append(_float_array(obs))
            rows["next_obs"].append(_float_array(next_obs))
            rows["actor_obs"].append(_float_array(actor_obs))
            rows["next_actor_obs"].append(_float_array(next_actor_obs))
            rows["mean_action"].append(_float_array(mean_action))
            rows["next_mean_action"].append(_float_array(next_mean_action))
            rows["sampled_action"].append(_float_array(sampled_action))
            rows["reward"].append(float(reward))
            rows["terminated"].append(bool(terminated))
            rows["truncated"].append(bool(truncated))
            rows["episode_id"].append(episode_id)
            rows["timestep"].append(timestep)
            rows["eval_seed"].append(int(eval_seed))
            rows["action_seed"].append(action_seed)

            episode_return += float(reward)
            timestep += 1
            obs = next_obs

            reached_cap = max_episode_steps is not None and timestep >= max_episode_steps
            if terminated or truncated or reached_cap:
                if reached_cap and not (terminated or truncated):
                    rows["truncated"][-1] = True
                break

        episode_returns.append(episode_return)
        episode_lengths.append(timestep)
        episode_action_seeds.append(action_seed)

    if not rows["obs"]:
        raise RuntimeError("No transitions were collected")

    arrays = {
        "obs": np.stack(rows["obs"]).astype(np.float32, copy=False),
        "next_obs": np.stack(rows["next_obs"]).astype(np.float32, copy=False),
        "actor_obs": np.stack(rows["actor_obs"]).astype(np.float32, copy=False),
        "next_actor_obs": np.stack(rows["next_actor_obs"]).astype(np.float32, copy=False),
        "mean_action": np.stack(rows["mean_action"]).astype(np.float32, copy=False),
        "next_mean_action": np.stack(rows["next_mean_action"]).astype(np.float32, copy=False),
        "sampled_action": np.stack(rows["sampled_action"]).astype(np.float32, copy=False),
        "reward": np.asarray(rows["reward"], dtype=np.float32),
        "terminated": np.asarray(rows["terminated"], dtype=np.bool_),
        "truncated": np.asarray(rows["truncated"], dtype=np.bool_),
        "episode_id": np.asarray(rows["episode_id"], dtype=np.int32),
        "timestep": np.asarray(rows["timestep"], dtype=np.int32),
        "eval_seed": np.asarray(rows["eval_seed"], dtype=np.int64),
        "action_seed": np.asarray(rows["action_seed"], dtype=np.int64),
        "episode_env_seed": np.asarray(eval_seeds, dtype=np.int64),
        "episode_action_seed": np.asarray(episode_action_seeds, dtype=np.int64),
        "episode_return": np.asarray(episode_returns, dtype=np.float64),
        "episode_length": np.asarray(episode_lengths, dtype=np.int32),
    }
    summary = {
        "n_transitions": int(len(arrays["obs"])),
        "n_episodes": int(len(eval_seeds)),
        "episode_return_mean": float(np.mean(arrays["episode_return"])),
        "episode_length_mean": float(np.mean(arrays["episode_length"])),
        "mean_sample_action_mse": float(
            np.mean(np.square(arrays["sampled_action"] - arrays["mean_action"]))
        ),
        "sde_noise_reset_count": sde_noise_reset_count,
    }
    return arrays, summary


def validate_arrays(
    arrays: dict[str, np.ndarray],
    expected_episodes: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    rollout_mode: str,
) -> None:
    transition_keys = [
        "obs", "next_obs", "actor_obs", "next_actor_obs", "mean_action", "next_mean_action", "sampled_action",
        "reward", "terminated", "truncated", "episode_id", "timestep",
        "eval_seed", "action_seed",
    ]
    lengths = {key: len(arrays[key]) for key in transition_keys}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Transition array lengths differ: {lengths}")
    if len(np.unique(arrays["episode_id"])) != expected_episodes:
        raise ValueError("Collected episode count does not match the requested seed count")
    for key in ("obs", "next_obs", "actor_obs", "next_actor_obs", "mean_action", "next_mean_action", "sampled_action", "reward"):
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"Non-finite values found in {key}")
    tolerance = 1e-5
    if np.any(arrays["sampled_action"] < action_low - tolerance) or np.any(
        arrays["sampled_action"] > action_high + tolerance
    ):
        raise ValueError("Sampled actions fall outside the environment action bounds")
    action_mse = float(np.mean(np.square(arrays["sampled_action"] - arrays["mean_action"])))
    if rollout_mode == "stochastic" and action_mse == 0.0:
        raise ValueError("Stochastic rollout produced no difference from mean actions")
    if rollout_mode == "deterministic" and action_mse != 0.0:
        raise ValueError("Deterministic rollout actions differ from mean actions")


def build_parser(algorithm: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Collect raw {algorithm.upper()} transitions for later L_sp analysis"
    )
    parser.add_argument("--root_dir", default=f"./{algorithm}/results/pths/")
    parser.add_argument("--algs", nargs="+", required=True)
    parser.add_argument("--env", choices=sorted(ENV_NAMES), required=True)
    parser.add_argument(
        "--output_dir", default=f"./{algorithm}/results/raw/lsp_transitions/"
    )
    parser.add_argument("--seed_file", default=f"./{algorithm}/tests/validation_seeds.txt")
    parser.add_argument("--n_eval_seeds", type=int, default=30)
    parser.add_argument(
        "--rollout_mode", choices=["stochastic", "deterministic"], default="stochastic"
    )
    parser.add_argument("--expected_models", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_collection(
    algorithm: str,
    model_class: Any,
    env_factories: dict[str, Callable[[str | None], Callable[[], Any]]],
    argv: Sequence[str] | None = None,
) -> None:
    args = build_parser(algorithm).parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    root_dir = Path(args.root_dir)
    save_dir = root_dir / args.env
    output_root = Path(args.output_dir) / args.env
    eval_seeds = load_eval_seeds(Path(args.seed_file), args.n_eval_seeds)

    model_paths: dict[str, list[Path]] = {}
    for algorithm_name in args.algs:
        paths = find_matching_models(save_dir, algorithm_name)
        suffix = f"_{algorithm}"
        if not paths and not algorithm_name.endswith(suffix):
            paths = find_matching_models(save_dir, algorithm_name + suffix)
        if len(paths) != args.expected_models:
            raise RuntimeError(
                f"Expected {args.expected_models} models for {algorithm_name}, found {len(paths)}"
            )
        model_paths[algorithm_name] = paths

    collection_records: list[dict[str, Any]] = []
    for algorithm_name, paths in model_paths.items():
        for model_path in paths:
            train_seed = extract_train_seed(model_path)
            destination = output_root / algorithm_name / f"train_seed_{train_seed}"
            npz_path = destination / f"{args.rollout_mode}.npz"
            json_path = destination / f"{args.rollout_mode}.json"
            if not args.overwrite and (npz_path.exists() or json_path.exists()):
                raise FileExistsError(f"Output already exists: {destination}")

            env = env_factories[args.env](None)()
            vec_normalize = None
            try:
                model = model_class.load(str(model_path), env=env, device=args.device)
                vec_normalize_path = model_path.parent / "vec_normalize.pkl"
                observation_transform = None
                if vec_normalize_path.is_file():
                    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

                    vec_normalize = VecNormalize.load(
                        str(vec_normalize_path), DummyVecEnv([env_factories[args.env](None)])
                    )
                    vec_normalize.training = False
                    vec_normalize.norm_reward = False

                    def observation_transform(observation: np.ndarray) -> np.ndarray:
                        return np.asarray(
                            vec_normalize.normalize_obs(observation), dtype=np.float32
                        )

                use_sde = bool(getattr(model, "use_sde", False))
                sde_sample_freq = int(getattr(model, "sde_sample_freq", -1))
                arrays, summary = collect_policy_transitions(
                    model,
                    env,
                    eval_seeds,
                    args.rollout_mode,
                    max_episode_steps=args.max_episode_steps,
                    observation_transform=observation_transform,
                )
                action_low = np.asarray(env.action_space.low, dtype=np.float32)
                action_high = np.asarray(env.action_space.high, dtype=np.float32)
                validate_arrays(
                    arrays, args.n_eval_seeds, action_low, action_high, args.rollout_mode
                )
                manifest = {
                    "format": "afqr_lsp_raw_v2",
                    "algorithm": algorithm,
                    "method": algorithm_name,
                    "environment": ENV_NAMES[args.env],
                    "environment_key": args.env,
                    "train_seed": train_seed,
                    "rollout_mode": args.rollout_mode,
                    "transition_action": (
                        "sampled_from_policy" if args.rollout_mode == "stochastic" else "mean_action"
                    ),
                    "discrepancy_action": "deterministic_mean_action",
                    "checkpoint_path": str(model_path),
                    "checkpoint_sha256": sha256_file(model_path),
                    "uses_vec_normalize": vec_normalize is not None,
                    "vec_normalize_path": (
                        str(vec_normalize_path) if vec_normalize is not None else None
                    ),
                    "vec_normalize_sha256": (
                        sha256_file(vec_normalize_path) if vec_normalize is not None else None
                    ),
                    "stored_observation_coordinates": "raw_environment_observation",
                    "actor_observation_coordinates": (
                        "checkpoint_vec_normalize" if vec_normalize is not None else "raw_environment_observation"
                    ),
                    "model_class": type(model).__name__,
                    "inference_device": str(model.device),
                    "use_sde": use_sde,
                    "sde_sample_freq": sde_sample_freq,
                    "sde_noise_reset_protocol": (
                        "episode_start_and_every_sde_sample_freq_steps"
                        if use_sde and args.rollout_mode == "stochastic"
                        else None
                    ),
                    "eval_seeds": eval_seeds,
                    "action_seeds": arrays["episode_action_seed"].tolist(),
                    "action_space_low": action_low.tolist(),
                    "action_space_high": action_high.tolist(),
                    "observation_shape": list(arrays["obs"].shape[1:]),
                    "action_shape": list(arrays["mean_action"].shape[1:]),
                    "max_episode_steps_override": args.max_episode_steps,
                    "code_commit": git_commit(repo_root),
                    "python_version": platform.python_version(),
                    "numpy_version": np.__version__,
                    "torch_version": package_version("torch"),
                    "gymnasium_version": package_version("gymnasium"),
                    "stable_baselines3_version": package_version("stable-baselines3"),
                    **summary,
                }
                arrays["metadata_json"] = np.asarray(
                    json.dumps(manifest, sort_keys=True), dtype=np.str_
                )
                atomic_save_npz(npz_path, arrays)
                atomic_write_json(json_path, manifest)
                collection_records.append({
                    "method": algorithm_name,
                    "train_seed": train_seed,
                    "npz": str(npz_path.relative_to(output_root)),
                    "manifest": str(json_path.relative_to(output_root)),
                    **summary,
                })
                print(
                    f"Saved {algorithm_name} seed {train_seed}: "
                    f"{summary['n_transitions']} transitions -> {npz_path}",
                    flush=True,
                )
            finally:
                if vec_normalize is not None:
                    vec_normalize.close()
                env.close()
                if "model" in locals():
                    del model

    root_manifest = {
        "format": "afqr_lsp_collection_v2",
        "algorithm": algorithm,
        "environment": ENV_NAMES[args.env],
        "environment_key": args.env,
        "methods": args.algs,
        "models_per_method": args.expected_models,
        "n_eval_seeds": args.n_eval_seeds,
        "eval_seeds": eval_seeds,
        "rollout_mode": args.rollout_mode,
        "records": collection_records,
        "status": "completed",
    }
    atomic_write_json(output_root / "collection_manifest.json", root_manifest)
    print(f"Completed {len(collection_records)} policies under {output_root}", flush=True)
