"""Streaming, resumable compact rollout collection for 10k Lsp analysis."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Callable, Iterator, Sequence

import h5py
import numpy as np

from tools.lsp_transition_collector import (
    ENV_NAMES,
    derive_action_seed,
    extract_train_seed,
    find_matching_models,
    load_eval_seeds,
    seed_action_rng,
    sha256_file,
)


FORMAT = "afqr_lsp_compact_chunk_v1"
COLLECTION_FORMAT = "afqr_lsp_compact_collection_v1"
BUFFER_SIZE = 8_192
REQUIRED_DATASETS = ("obs", "next_obs", "mean_action", "episode_id")


def enforce_scratch_headroom(scratch_dir: Path, local_output: Path | None = None) -> None:
    safety_bytes = int(os.environ.get("AFQR_SCRATCH_SAFETY_BYTES", "0"))
    work_bytes = int(os.environ.get("AFQR_SCRATCH_WORK_BYTES", "0"))
    free_bytes = shutil.disk_usage(scratch_dir).free
    if safety_bytes and free_bytes < safety_bytes:
        print(
            f"Scratch safety floor violated: free={free_bytes} safety={safety_bytes}",
            flush=True,
        )
        raise SystemExit(75)
    if (
        local_output is not None
        and work_bytes
        and local_output.is_file()
        and local_output.stat().st_size > work_bytes
    ):
        print(
            f"Chunk exceeds reserved work budget: "
            f"size={local_output.stat().st_size} budget={work_bytes}",
            flush=True,
        )
        raise SystemExit(75)


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


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class CompactChunkWriter:
    def __init__(self, path: Path, compression_level: int = 4) -> None:
        self.path = path
        self.handle = h5py.File(path, "w")
        self.compression_level = compression_level
        self.datasets: dict[str, h5py.Dataset] = {}
        self.buffers: dict[str, list[np.ndarray | int]] = {
            key: [] for key in REQUIRED_DATASETS
        }
        self.transition_count = 0
        self.normalization_count = 0
        self.normalization_sum: np.ndarray | None = None
        self.normalization_sumsq: np.ndarray | None = None
        self.action_mse_sum = 0.0
        self.action_element_count = 0
        self.episode_ids: list[int] = []
        self.episode_env_seeds: list[int] = []
        self.episode_action_seeds: list[int] = []
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []

    def _create_datasets(self, obs: np.ndarray, action: np.ndarray) -> None:
        specs = {
            "obs": (np.float32, obs.shape),
            "next_obs": (np.float32, obs.shape),
            "mean_action": (np.float32, action.shape),
            "episode_id": (np.int32, ()),
        }
        for key, (dtype, trailing_shape) in specs.items():
            chunk_rows = BUFFER_SIZE
            chunks = (chunk_rows, *trailing_shape)
            self.datasets[key] = self.handle.create_dataset(
                key,
                shape=(0, *trailing_shape),
                maxshape=(None, *trailing_shape),
                chunks=chunks,
                dtype=dtype,
                compression="gzip",
                compression_opts=self.compression_level,
                shuffle=True,
            )

    def append(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        mean_action: np.ndarray,
        sampled_action: np.ndarray,
        episode_id: int,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        mean_action = np.asarray(mean_action, dtype=np.float32)
        sampled_action = np.asarray(sampled_action, dtype=np.float32)
        for name, value in (
            ("obs", obs),
            ("next_obs", next_obs),
            ("mean_action", mean_action),
            ("sampled_action", sampled_action),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Non-finite values in {name}")
        if not self.datasets:
            self._create_datasets(obs, mean_action)
            self.normalization_sum = np.zeros_like(obs, dtype=np.float64)
            self.normalization_sumsq = np.zeros_like(obs, dtype=np.float64)
        if (
            obs.shape != self.datasets["obs"].shape[1:]
            or next_obs.shape != self.datasets["next_obs"].shape[1:]
            or mean_action.shape != self.datasets["mean_action"].shape[1:]
        ):
            raise ValueError("Transition shape changed within a chunk")

        self.buffers["obs"].append(obs)
        self.buffers["next_obs"].append(next_obs)
        self.buffers["mean_action"].append(mean_action)
        self.buffers["episode_id"].append(int(episode_id))
        self.normalization_count += 2
        assert self.normalization_sum is not None
        assert self.normalization_sumsq is not None
        self.normalization_sum += obs.astype(np.float64) + next_obs.astype(np.float64)
        self.normalization_sumsq += np.square(obs, dtype=np.float64) + np.square(
            next_obs, dtype=np.float64
        )
        difference = sampled_action.astype(np.float64) - mean_action.astype(np.float64)
        self.action_mse_sum += float(np.sum(difference * difference))
        self.action_element_count += int(difference.size)
        if len(self.buffers["obs"]) >= BUFFER_SIZE:
            self.flush()

    def finish_episode(
        self,
        episode_id: int,
        env_seed: int,
        action_seed: int,
        episode_return: float,
        episode_length: int,
    ) -> None:
        self.episode_ids.append(int(episode_id))
        self.episode_env_seeds.append(int(env_seed))
        self.episode_action_seeds.append(int(action_seed))
        self.episode_returns.append(float(episode_return))
        self.episode_lengths.append(int(episode_length))

    def flush(self) -> None:
        row_count = len(self.buffers["obs"])
        if row_count == 0:
            return
        start = self.transition_count
        stop = start + row_count
        for key, dataset in self.datasets.items():
            dataset.resize(stop, axis=0)
            if key == "episode_id":
                values = np.asarray(self.buffers[key], dtype=np.int32)
            else:
                values = np.stack(self.buffers[key])
            dataset[start:stop] = values
            self.buffers[key].clear()
        self.transition_count = stop

    def close(self) -> dict[str, Any]:
        self.flush()
        if self.transition_count == 0:
            raise RuntimeError("No transitions were collected")
        if not self.episode_ids:
            raise RuntimeError("No episodes were completed")
        episode_payload = {
            "episode_id_values": np.asarray(self.episode_ids, dtype=np.int32),
            "episode_env_seed": np.asarray(self.episode_env_seeds, dtype=np.int64),
            "episode_action_seed": np.asarray(self.episode_action_seeds, dtype=np.int64),
            "episode_return": np.asarray(self.episode_returns, dtype=np.float64),
            "episode_length": np.asarray(self.episode_lengths, dtype=np.int32),
            "normalization_sum": np.asarray(self.normalization_sum, dtype=np.float64),
            "normalization_sumsq": np.asarray(
                self.normalization_sumsq, dtype=np.float64
            ),
        }
        for key, values in episode_payload.items():
            self.handle.create_dataset(key, data=values)
        self.handle.attrs["format"] = FORMAT
        self.handle.attrs["transition_count"] = self.transition_count
        self.handle.attrs["episode_count"] = len(self.episode_ids)
        self.handle.attrs["normalization_count"] = self.normalization_count
        self.handle.flush()
        self.handle.close()
        return {
            "transition_count": self.transition_count,
            "episode_count": len(self.episode_ids),
            "episode_return_mean": float(np.mean(self.episode_returns)),
            "episode_length_mean": float(np.mean(self.episode_lengths)),
            "mean_sample_action_mse": (
                self.action_mse_sum / self.action_element_count
                if self.action_element_count
                else 0.0
            ),
            "normalization_count": self.normalization_count,
            "normalization_sum": np.asarray(
                self.normalization_sum, dtype=np.float64
            ).tolist(),
            "normalization_sumsq": np.asarray(
                self.normalization_sumsq, dtype=np.float64
            ).tolist(),
        }


def validate_compact_chunk(
    h5_path: Path,
    manifest: dict[str, Any],
    expected_episode_count: int,
) -> None:
    if manifest.get("format") != FORMAT or manifest.get("status") != "completed":
        raise ValueError(f"Invalid compact manifest: {h5_path}")
    if manifest.get("episode_count") != expected_episode_count:
        raise ValueError(f"Unexpected episode count: {h5_path}")
    if manifest.get("h5_size_bytes") != h5_path.stat().st_size:
        raise ValueError(f"Unexpected HDF5 size: {h5_path}")
    with h5py.File(h5_path, "r") as handle:
        if handle.attrs.get("format") != FORMAT:
            raise ValueError(f"Unexpected HDF5 format: {h5_path}")
        lengths = {key: len(handle[key]) for key in REQUIRED_DATASETS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Compact dataset lengths differ: {lengths}")
        if lengths["obs"] != manifest["transition_count"]:
            raise ValueError(f"Transition count mismatch: {h5_path}")
        if len(handle["episode_id_values"]) != expected_episode_count:
            raise ValueError(f"Episode summary count mismatch: {h5_path}")
        if int(handle.attrs["normalization_count"]) != 2 * lengths["obs"]:
            raise ValueError(f"Normalization count mismatch: {h5_path}")


def existing_chunk_is_complete(h5_path: Path, json_path: Path) -> bool:
    if not h5_path.is_file() or not json_path.is_file():
        return False
    manifest = json.loads(json_path.read_text())
    expected = int(manifest["episode_stop"]) - int(manifest["episode_start"])
    validate_compact_chunk(h5_path, manifest, expected)
    if sha256_file(h5_path) != manifest.get("h5_sha256"):
        raise ValueError(f"Existing compact HDF5 checksum mismatch: {h5_path}")
    return True


@contextmanager
def publish_lock(lock_root: Path, lane: int) -> Iterator[None]:
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"lane_{lane}.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def publish_chunk(
    local_h5: Path,
    local_json: Path,
    target_h5: Path,
    target_json: Path,
    lane_count: int,
    lane_key: str,
) -> None:
    lane = int(hashlib.sha256(lane_key.encode()).hexdigest(), 16) % lane_count
    lock_root = target_h5.parents[2] / ".publish_locks"
    with publish_lock(lock_root, lane):
        if existing_chunk_is_complete(target_h5, target_json):
            return
        if target_h5.exists() or target_json.exists():
            raise FileExistsError(f"Incomplete target chunk exists: {target_h5}")
        target_h5.parent.mkdir(parents=True, exist_ok=True)
        partial_h5 = target_h5.with_suffix(
            target_h5.suffix + f".partial_{os.environ.get('SLURM_JOB_ID', 'local')}"
        )
        partial_json = target_json.with_suffix(
            target_json.suffix + f".partial_{os.environ.get('SLURM_JOB_ID', 'local')}"
        )
        shutil.copy2(local_h5, partial_h5)
        expected_hash = json.loads(local_json.read_text())["h5_sha256"]
        if sha256_file(partial_h5) != expected_hash:
            partial_h5.unlink(missing_ok=True)
            raise IOError(f"Published HDF5 hash mismatch: {target_h5}")
        shutil.copy2(local_json, partial_json)
        os.replace(partial_h5, target_h5)
        os.replace(partial_json, target_json)


def collect_chunk(
    model: Any,
    env: Any,
    observation_transform: Callable[[np.ndarray], np.ndarray] | None,
    eval_seeds: Sequence[int],
    episode_start: int,
    h5_path: Path,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[dict[str, Any], int]:
    writer = CompactChunkWriter(h5_path)
    use_sde = bool(getattr(model, "use_sde", False))
    sde_sample_freq = int(getattr(model, "sde_sample_freq", -1))
    sde_reset_count = 0
    tolerance = 1e-5
    for local_index, eval_seed in enumerate(eval_seeds):
        global_episode_id = episode_start + local_index
        obs, _ = env.reset(seed=int(eval_seed))
        action_seed = derive_action_seed(int(eval_seed), global_episode_id)
        seed_action_rng(action_seed)
        episode_return = 0.0
        timestep = 0
        while True:
            actor_obs = (
                observation_transform(np.asarray(obs, dtype=np.float32))
                if observation_transform is not None
                else np.asarray(obs, dtype=np.float32)
            )
            policy_input = actor_obs
            mean_action, _ = model.predict(policy_input, deterministic=True)
            if use_sde and (
                timestep == 0
                or (sde_sample_freq > 0 and timestep % sde_sample_freq == 0)
            ):
                model.policy.reset_noise(1)
                sde_reset_count += 1
            sampled_action, _ = model.predict(policy_input, deterministic=False)
            sampled_action = np.asarray(sampled_action, dtype=np.float32)
            if np.any(sampled_action < action_low - tolerance) or np.any(
                sampled_action > action_high + tolerance
            ):
                raise ValueError("Sampled action outside environment bounds")
            next_obs, reward, terminated, truncated, _ = env.step(sampled_action)
            writer.append(
                np.asarray(obs, dtype=np.float32),
                np.asarray(next_obs, dtype=np.float32),
                np.asarray(mean_action, dtype=np.float32),
                sampled_action,
                global_episode_id,
            )
            episode_return += float(reward)
            timestep += 1
            obs = next_obs
            if terminated or truncated:
                break
        writer.finish_episode(
            global_episode_id,
            int(eval_seed),
            action_seed,
            episode_return,
            timestep,
        )
    summary = writer.close()
    return summary, sde_reset_count


def build_parser(algorithm: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Collect compact resumable {algorithm.upper()} Lsp rollouts"
    )
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--alg", required=True)
    parser.add_argument("--env", choices=sorted(ENV_NAMES), required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path, required=True)
    parser.add_argument("--expected-models", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--publish-lanes", type=int, default=2)
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--max-chunks-per-model", type=int)
    return parser


def run_compact_collection(
    algorithm: str,
    model_class: Any,
    env_factories: dict[str, Callable[[str | None], Callable[[], Any]]],
    argv: Sequence[str] | None = None,
) -> None:
    args = build_parser(algorithm).parse_args(argv)
    if args.episode_count <= 0 or args.chunk_size <= 0:
        raise SystemExit("episode-count and chunk-size must be positive")
    if args.episode_count % args.chunk_size != 0:
        raise SystemExit("episode-count must be divisible by chunk-size")
    if args.publish_lanes <= 0:
        raise SystemExit("publish-lanes must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    collector_source_hash = sha256_file(Path(__file__))
    eval_seeds = load_eval_seeds(args.seed_file, args.episode_count)
    model_dir = args.root_dir / args.env
    model_paths = find_matching_models(model_dir, args.alg)
    algorithm_suffix = f"_{algorithm}"
    if not model_paths and not args.alg.endswith(algorithm_suffix):
        model_paths = find_matching_models(model_dir, args.alg + algorithm_suffix)
    if len(model_paths) != args.expected_models:
        raise RuntimeError(
            f"Expected {args.expected_models} models for {args.alg}, found {len(model_paths)}"
        )
    if args.max_models is not None:
        if args.max_models <= 0:
            raise SystemExit("max-models must be positive")
        model_paths = model_paths[: args.max_models]
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    args.publish_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for model_path in model_paths:
        train_seed = extract_train_seed(model_path)
        env = env_factories[args.env](None)()
        vec_normalize = None
        try:
            model = model_class.load(str(model_path), env=env, device=args.device)
            if hasattr(model, "policy") and hasattr(model.policy, "set_training_mode"):
                model.policy.set_training_mode(False)
            vec_normalize_path = model_path.parent / "vec_normalize.pkl"
            observation_transform = None
            if vec_normalize_path.is_file():
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

                vec_normalize = VecNormalize.load(
                    str(vec_normalize_path),
                    DummyVecEnv([env_factories[args.env](None)]),
                )
                vec_normalize.training = False
                vec_normalize.norm_reward = False

                def observation_transform(observation: np.ndarray) -> np.ndarray:
                    return np.asarray(
                        vec_normalize.normalize_obs(observation), dtype=np.float32
                    )

            checkpoint_hash = sha256_file(model_path)
            model_chunk_count = 0
            for episode_start in range(0, args.episode_count, args.chunk_size):
                if (
                    args.max_chunks_per_model is not None
                    and model_chunk_count >= args.max_chunks_per_model
                ):
                    break
                episode_stop = episode_start + args.chunk_size
                relative = (
                    Path(args.alg)
                    / f"train_seed_{train_seed}"
                    / f"episodes_{episode_start:05d}_{episode_stop:05d}"
                )
                target_h5 = args.publish_dir / relative.with_suffix(".h5")
                target_json = args.publish_dir / relative.with_suffix(".json")
                if existing_chunk_is_complete(target_h5, target_json):
                    records.append(
                        {
                            "train_seed": train_seed,
                            "episode_start": episode_start,
                            "episode_stop": episode_stop,
                            "manifest": str(target_json.relative_to(args.publish_dir)),
                            "h5": str(target_h5.relative_to(args.publish_dir)),
                            "status": "reused",
                        }
                    )
                    model_chunk_count += 1
                    continue

                local_stem = (
                    args.scratch_dir
                    / f"{args.alg}_{train_seed}_{episode_start:05d}_{episode_stop:05d}"
                )
                local_h5 = local_stem.with_suffix(".h5")
                local_json = local_stem.with_suffix(".json")
                enforce_scratch_headroom(args.scratch_dir)
                summary, sde_reset_count = collect_chunk(
                    model,
                    env,
                    observation_transform,
                    eval_seeds[episode_start:episode_stop],
                    episode_start,
                    local_h5,
                    np.asarray(env.action_space.low, dtype=np.float32),
                    np.asarray(env.action_space.high, dtype=np.float32),
                )
                if summary["mean_sample_action_mse"] == 0.0:
                    raise RuntimeError("Stochastic rollout matched mean actions exactly")
                h5_hash = sha256_file(local_h5)
                manifest = {
                    "format": FORMAT,
                    "status": "completed",
                    "algorithm": algorithm,
                    "environment": ENV_NAMES[args.env],
                    "environment_key": args.env,
                    "method": args.alg,
                    "train_seed": train_seed,
                    "episode_start": episode_start,
                    "episode_stop": episode_stop,
                    "eval_seeds": eval_seeds[episode_start:episode_stop],
                    "rollout_mode": "stochastic",
                    "transition_action": "sampled_from_policy",
                    "discrepancy_action": "deterministic_mean_action",
                    "stored_arrays": list(REQUIRED_DATASETS),
                    "checkpoint_path": str(model_path),
                    "checkpoint_sha256": checkpoint_hash,
                    "uses_vec_normalize": vec_normalize is not None,
                    "vec_normalize_path": (
                        str(vec_normalize_path) if vec_normalize is not None else None
                    ),
                    "vec_normalize_sha256": (
                        sha256_file(vec_normalize_path)
                        if vec_normalize is not None
                        else None
                    ),
                    "stored_observation_coordinates": "raw_environment_observation",
                    "actor_observation_coordinates": (
                        "checkpoint_vec_normalize"
                        if vec_normalize is not None
                        else "raw_environment_observation"
                    ),
                    "policy_input": "current_observation",
                    "action_seed_rule": (
                        "(eval_seed * 1664525 + 1013904223 + global_episode_id) "
                        "% (2**31 - 1)"
                    ),
                    "sde_noise_reset_count": sde_reset_count,
                    "h5_sha256": h5_hash,
                    "h5_size_bytes": local_h5.stat().st_size,
                    "code_commit": git_commit(repo_root),
                    "collector_source_sha256": collector_source_hash,
                    "python_version": platform.python_version(),
                    "numpy_version": np.__version__,
                    "h5py_version": h5py.__version__,
                    "torch_version": package_version("torch"),
                    "gymnasium_version": package_version("gymnasium"),
                    "stable_baselines3_version": package_version(
                        "stable-baselines3"
                    ),
                    **summary,
                }
                atomic_write_json(local_json, manifest)
                validate_compact_chunk(local_h5, manifest, args.chunk_size)
                enforce_scratch_headroom(args.scratch_dir, local_h5)
                publish_chunk(
                    local_h5,
                    local_json,
                    target_h5,
                    target_json,
                    args.publish_lanes,
                    f"{algorithm}:{args.env}:{args.alg}:{train_seed}:{episode_start}",
                )
                local_h5.unlink()
                local_json.unlink()
                enforce_scratch_headroom(args.scratch_dir)
                records.append(
                    {
                        "train_seed": train_seed,
                        "episode_start": episode_start,
                        "episode_stop": episode_stop,
                        "manifest": str(target_json.relative_to(args.publish_dir)),
                        "h5": str(target_h5.relative_to(args.publish_dir)),
                        "status": "collected",
                    }
                )
                model_chunk_count += 1
                print(
                    f"Published {args.alg} seed={train_seed} "
                    f"episodes={episode_start}:{episode_stop}",
                    flush=True,
                )
        finally:
            if vec_normalize is not None:
                vec_normalize.close()
            env.close()
            if "model" in locals():
                del model

    expected_record_count = len(model_paths) * (
        args.episode_count // args.chunk_size
    )
    limited_run = args.max_models is not None or args.max_chunks_per_model is not None
    collection = {
        "format": COLLECTION_FORMAT,
        "status": "pilot_completed" if limited_run else "completed",
        "algorithm": algorithm,
        "environment": ENV_NAMES[args.env],
        "environment_key": args.env,
        "method": args.alg,
        "models": len(model_paths),
        "episode_count_per_model": args.episode_count,
        "chunk_size": args.chunk_size,
        "policy_input": "current_observation",
        "expected_record_count": expected_record_count,
        "observed_record_count": len(records),
        "seed_file": str(args.seed_file),
        "seed_file_sha256": sha256_file(args.seed_file),
        "records": records,
    }
    if not limited_run and len(records) != expected_record_count:
        raise RuntimeError(
            f"Expected {expected_record_count} records, found {len(records)}"
        )
    manifest_name = (
        "pilot_manifest.json" if limited_run else "collection_manifest.json"
    )
    atomic_write_json(args.publish_dir / manifest_name, collection)
    print(
        f"Compact collection {collection['status']}: "
        f"{len(records)} chunk records -> {args.publish_dir}",
        flush=True,
    )
