"""Analyze one compact 10k-rollout Lsp method cell."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import t

try:
    import torch
except ImportError:
    torch = None


COLLECTION_FORMAT = "afqr_lsp_compact_collection_v1"
CHUNK_FORMAT = "afqr_lsp_compact_chunk_v1"
ANALYSIS_FORMAT = "afqr_lsp_compact_cell_analysis_v1"
K_GRID = (4, 32, 128)
PAIR_K = 4
QUERY_SEED_OFFSET = 20_260_721


def external_neighbors(
    tree: cKDTree,
    next_obs: np.ndarray,
    episode_id: np.ndarray,
    query_idx: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact nearest transitions outside each query transition's episode."""
    query = next_obs[query_idx]
    search_k = min(len(next_obs), max(2 * max_k, max_k + 1))
    while True:
        distances, neighbors = tree.query(query, k=search_k, workers=-1)
        if search_k == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        valid = episode_id[neighbors] != episode_id[query_idx, None]
        if np.all(np.sum(valid, axis=1) >= max_k):
            break
        if search_k == len(next_obs):
            raise ValueError("Not enough cross-episode neighbors")
        search_k = min(len(next_obs), 2 * search_k)

    positions = np.broadcast_to(np.arange(search_k), valid.shape)
    selected = np.where(valid, positions, search_k)
    selected = np.partition(selected, max_k - 1, axis=1)[:, :max_k]
    selected.sort(axis=1)
    return (
        np.take_along_axis(neighbors, selected, axis=1),
        np.take_along_axis(distances, selected, axis=1),
    )


def cuda_external_neighbors(
    next_obs: np.ndarray,
    episode_id: np.ndarray,
    query_idx: np.ndarray,
    max_k: int,
    query_chunk_size: int = 1_024,
    candidate_chunk_size: int = 32_768,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact cross-episode k-NN using chunked CUDA matrix products."""
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is not available")
    device = torch.device("cuda")
    states = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    episodes = torch.as_tensor(episode_id, dtype=torch.int64, device=device)
    queries = torch.as_tensor(query_idx, dtype=torch.int64, device=device)
    output_indices = []
    output_distances = []
    with torch.inference_mode():
        state_norm = torch.sum(states * states, dim=1)
        for query_start in range(0, len(queries), query_chunk_size):
            query_ids = queries[query_start : query_start + query_chunk_size]
            query_states = states[query_ids]
            query_norm = torch.sum(query_states * query_states, dim=1, keepdim=True)
            query_episodes = episodes[query_ids]
            best_distance = torch.full(
                (len(query_ids), max_k), float("inf"), device=device
            )
            best_index = torch.full(
                (len(query_ids), max_k), -1, dtype=torch.int64, device=device
            )
            for candidate_start in range(0, len(states), candidate_chunk_size):
                candidate_stop = min(len(states), candidate_start + candidate_chunk_size)
                candidates = states[candidate_start:candidate_stop]
                distance = (
                    query_norm
                    + state_norm[candidate_start:candidate_stop][None, :]
                    - 2.0 * (query_states @ candidates.T)
                )
                same_episode = (
                    query_episodes[:, None]
                    == episodes[candidate_start:candidate_stop][None, :]
                )
                distance.masked_fill_(same_episode, float("inf"))
                local_k = min(max_k, len(candidates))
                local_distance, local_position = torch.topk(
                    distance, k=local_k, dim=1, largest=False, sorted=True
                )
                local_index = local_position + candidate_start
                merged_distance = torch.cat((best_distance, local_distance), dim=1)
                merged_index = torch.cat((best_index, local_index), dim=1)
                best_distance, selected = torch.topk(
                    merged_distance, k=max_k, dim=1, largest=False, sorted=True
                )
                best_index = torch.gather(merged_index, 1, selected)
            if torch.any(best_index < 0) or torch.any(~torch.isfinite(best_distance)):
                raise ValueError("Not enough cross-episode neighbors")
            output_indices.append(best_index.cpu().numpy())
            output_distances.append(
                torch.sqrt(torch.clamp_min(best_distance, 0.0)).cpu().numpy()
            )
    return np.concatenate(output_indices), np.concatenate(output_distances)


def local_u_statistic(
    query_idx: np.ndarray,
    neighbor_idx: np.ndarray,
    actions: np.ndarray,
    episode_id: np.ndarray,
) -> np.ndarray:
    """Average squared action distance over cross-episode candidate pairs."""
    batch_size, k = neighbor_idx.shape
    candidate_idx = np.concatenate((query_idx[:, None], neighbor_idx), axis=1)
    candidate_actions = actions[candidate_idx]
    candidate_episodes = episode_id[candidate_idx]
    candidate_norms = np.sum(candidate_actions * candidate_actions, axis=2)
    candidate_count = k + 1

    total_distance = (
        candidate_count * np.sum(candidate_norms, axis=1)
        - np.sum(np.sum(candidate_actions, axis=1) ** 2, axis=1)
    )
    total_pairs = candidate_count * (candidate_count - 1) // 2

    episode_count = int(np.max(episode_id)) + 1
    row_offset = np.arange(batch_size, dtype=np.int64)[:, None] * episode_count
    group_id = (row_offset + candidate_episodes).ravel()
    minlength = batch_size * episode_count
    counts = np.bincount(group_id, minlength=minlength).reshape(
        batch_size, episode_count
    )
    norm_sums = np.bincount(
        group_id, weights=candidate_norms.ravel(), minlength=minlength
    ).reshape(batch_size, episode_count)
    action_sums = np.empty(
        (batch_size, episode_count, actions.shape[1]), dtype=np.float64
    )
    for action_dim in range(actions.shape[1]):
        action_sums[:, :, action_dim] = np.bincount(
            group_id,
            weights=candidate_actions[:, :, action_dim].ravel(),
            minlength=minlength,
        ).reshape(batch_size, episode_count)

    same_episode_distance = np.sum(
        counts * norm_sums - np.sum(action_sums * action_sums, axis=2), axis=1
    )
    same_episode_pairs = np.sum(counts * (counts - 1) // 2, axis=1)
    valid_pairs = total_pairs - same_episode_pairs
    if np.any(valid_pairs <= 0):
        raise ValueError("A query candidate set contains no cross-episode pair")
    return np.maximum(0.0, total_distance - same_episode_distance) / valid_pairs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def mean_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) < 2:
        return {"mean": mean, "ci95": 0.0}
    ci = float(
        t.ppf(0.975, len(array) - 1)
        * np.std(array, ddof=1)
        / np.sqrt(len(array))
    )
    return {"mean": mean, "ci95": ci}


def validate_collection(
    cell_dir: Path,
    expected_models: int,
    expected_episodes: int,
    expected_chunk_size: int,
    manifest_name: str = "collection_manifest.json",
    expected_status: str = "completed",
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = cell_dir / manifest_name
    manifest = json.loads(manifest_path.read_text())
    expected_records = expected_models * (expected_episodes // expected_chunk_size)
    if (
        manifest.get("format") != COLLECTION_FORMAT
        or manifest.get("status") != expected_status
        or manifest.get("models") != expected_models
        or manifest.get("episode_count_per_model") != expected_episodes
        or manifest.get("chunk_size") != expected_chunk_size
        or manifest.get("observed_record_count") != expected_records
    ):
        raise ValueError(f"Incomplete or incompatible collection: {manifest_path}")
    by_seed: dict[str, list[dict[str, Any]]] = {}
    for record in manifest["records"]:
        by_seed.setdefault(str(record["train_seed"]), []).append(record)
    if len(by_seed) != expected_models:
        raise ValueError(f"Expected {expected_models} training policies: {cell_dir}")
    expected_starts = list(range(0, expected_episodes, expected_chunk_size))
    for train_seed, records in by_seed.items():
        records.sort(key=lambda row: int(row["episode_start"]))
        starts = [int(row["episode_start"]) for row in records]
        stops = [int(row["episode_stop"]) for row in records]
        if starts != expected_starts or stops != [
            start + expected_chunk_size for start in expected_starts
        ]:
            raise ValueError(f"Non-contiguous chunks for seed {train_seed}")
    return manifest, by_seed


def load_policy(
    cell_dir: Path,
    records: list[dict[str, Any]],
    verify_hashes: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays: dict[str, list[np.ndarray]] = {
        "obs": [],
        "next_obs": [],
        "mean_action": [],
        "episode_id": [],
    }
    chunk_provenance = []
    for record in records:
        chunk_manifest_path = cell_dir / record["manifest"]
        chunk_manifest = json.loads(chunk_manifest_path.read_text())
        h5_path = cell_dir / record["h5"]
        if (
            chunk_manifest.get("format") != CHUNK_FORMAT
            or chunk_manifest.get("status") != "completed"
            or chunk_manifest.get("h5_size_bytes") != h5_path.stat().st_size
        ):
            raise ValueError(f"Invalid chunk: {chunk_manifest_path}")
        if verify_hashes and sha256_file(h5_path) != chunk_manifest["h5_sha256"]:
            raise ValueError(f"HDF5 checksum mismatch: {h5_path}")
        with h5py.File(h5_path, "r") as handle:
            if handle.attrs.get("format") != CHUNK_FORMAT:
                raise ValueError(f"Unexpected HDF5 format: {h5_path}")
            lengths = []
            for key in arrays:
                value = np.asarray(handle[key])
                arrays[key].append(value)
                lengths.append(len(value))
            if len(set(lengths)) != 1:
                raise ValueError(f"Transition array length mismatch: {h5_path}")
            if int(handle.attrs["transition_count"]) != lengths[0]:
                raise ValueError(f"Transition count mismatch: {h5_path}")
        chunk_provenance.append(
            {
                "manifest": record["manifest"],
                "manifest_sha256": sha256_file(chunk_manifest_path),
                "h5": record["h5"],
                "h5_sha256": chunk_manifest["h5_sha256"],
                "transition_count": chunk_manifest["transition_count"],
            }
        )
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    lengths = {key: len(value) for key, value in merged.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Merged transition array mismatch: {lengths}")
    for key, value in merged.items():
        if key != "episode_id" and not np.all(np.isfinite(value)):
            raise ValueError(f"Non-finite merged array: {key}")
    return merged, chunk_provenance


def nearest_neighbors(
    next_obs: np.ndarray,
    episode_id: np.ndarray,
    query_idx: np.ndarray,
    max_k: int,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    use_cuda = backend == "cuda" or (
        backend == "auto"
        and torch is not None
        and torch.cuda.is_available()
    )
    if use_cuda:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA neighbor backend requested but unavailable")
        return cuda_external_neighbors(
            next_obs,
            episode_id,
            query_idx,
            max_k=max_k,
            query_chunk_size=512,
            candidate_chunk_size=32_768,
        )
    tree = cKDTree(next_obs)
    output_idx = []
    output_distance = []
    for start in range(0, len(query_idx), 2_000):
        indices, distances = external_neighbors(
            tree,
            next_obs,
            episode_id,
            query_idx[start : start + 2_000],
            max_k=max_k,
        )
        output_idx.append(indices)
        output_distance.append(distances)
    return np.concatenate(output_idx), np.concatenate(output_distance)


def calculate_lsp(
    query_idx: np.ndarray,
    neighbor_idx: np.ndarray,
    actions: np.ndarray,
    episode_id: np.ndarray,
    k_grid: tuple[int, ...],
) -> dict[str, float]:
    sums = {k: 0.0 for k in k_grid}
    counts = {k: 0 for k in k_grid}
    for start in range(0, len(query_idx), 128):
        stop = min(len(query_idx), start + 128)
        query_chunk = query_idx[start:stop]
        neighbor_chunk = neighbor_idx[start:stop]
        for k in k_grid:
            values = local_u_statistic(
                query_chunk, neighbor_chunk[:, :k], actions, episode_id
            )
            sums[k] += float(np.sum(values))
            counts[k] += len(values)
    return {str(k): sums[k] / counts[k] for k in k_grid}


def build_pair_cache(
    query_idx: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_distance: np.ndarray,
    obs: np.ndarray,
    actions: np.ndarray,
    episode_id: np.ndarray,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    candidates = np.concatenate(
        (query_idx[:, None], neighbor_idx[:, :PAIR_K]), axis=1
    )
    pair_left_position, pair_right_position = np.triu_indices(PAIR_K + 1, k=1)
    left = candidates[:, pair_left_position].reshape(-1)
    right = candidates[:, pair_right_position].reshape(-1)
    valid = episode_id[left] != episode_id[right]
    left = left[valid]
    right = right[valid]
    scaled_difference = (
        obs[left].astype(np.float64) - obs[right].astype(np.float64)
    ) / scale
    current_distance = np.linalg.norm(scaled_difference, axis=1)
    action_difference = np.sum(
        (
            actions[left].astype(np.float64)
            - actions[right].astype(np.float64)
        )
        ** 2,
        axis=1,
    )
    return {
        "query_idx": query_idx.astype(np.int64),
        "neighbor_idx_k4": neighbor_idx[:, :PAIR_K].astype(np.int64),
        "neighbor_distance_k4": neighbor_distance[:, :PAIR_K].astype(np.float32),
        "pair_left": left.astype(np.int64),
        "pair_right": right.astype(np.int64),
        "current_distance": current_distance.astype(np.float32),
        "action_difference": action_difference.astype(np.float32),
    }


def analyze_policy(
    train_seed: str,
    arrays: dict[str, np.ndarray],
    center: np.ndarray,
    scale: np.ndarray,
    query_count: int,
    backend: str,
    output_dir: Path,
) -> dict[str, Any]:
    transition_count = len(arrays["next_obs"])
    if transition_count < query_count:
        raise ValueError(
            f"Seed {train_seed} has {transition_count} transitions, "
            f"fewer than {query_count} queries"
        )
    query_seed = int(train_seed) + QUERY_SEED_OFFSET
    query_idx = np.sort(
        np.random.default_rng(query_seed).choice(
            transition_count, size=query_count, replace=False
        )
    )
    next_obs_norm = (
        arrays["next_obs"].astype(np.float64) - center
    ) / scale
    next_obs_norm = next_obs_norm.astype(np.float32)
    neighbor_idx, neighbor_distance = nearest_neighbors(
        next_obs_norm,
        arrays["episode_id"],
        query_idx,
        max(K_GRID),
        backend,
    )
    lsp = calculate_lsp(
        query_idx,
        neighbor_idx,
        arrays["mean_action"],
        arrays["episode_id"],
        K_GRID,
    )
    radius = {
        str(k): float(np.median(neighbor_distance[:, k - 1])) for k in K_GRID
    }
    pair_cache = build_pair_cache(
        query_idx,
        neighbor_idx,
        neighbor_distance,
        arrays["obs"],
        arrays["mean_action"],
        arrays["episode_id"],
        scale,
    )
    pair_dir = output_dir / "pair_cache"
    pair_dir.mkdir(parents=True, exist_ok=True)
    pair_path = pair_dir / f"train_seed_{train_seed}_k4.npz"
    np.savez_compressed(pair_path, **pair_cache)
    return {
        "train_seed": train_seed,
        "transition_count": transition_count,
        "episode_count": int(len(np.unique(arrays["episode_id"]))),
        "query_count": query_count,
        "query_seed": query_seed,
        "l_sp": lsp,
        "radius_median": radius,
        "pair_cache": str(pair_path.relative_to(output_dir)),
        "pair_cache_sha256": sha256_file(pair_path),
        "eligible_k4_pair_count": int(len(pair_cache["pair_left"])),
        "current_distance_mean": float(np.mean(pair_cache["current_distance"])),
        "action_difference_mean": float(np.mean(pair_cache["action_difference"])),
    }


def aggregate(policies: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_training_policies": len(policies),
        "k": {
            str(k): {
                "l_sp": mean_ci([row["l_sp"][str(k)] for row in policies]),
                "radius_median": mean_ci(
                    [row["radius_median"][str(k)] for row in policies]
                ),
            }
            for k in K_GRID
        },
        "eligible_k4_pair_count": int(
            sum(row["eligible_k4_pair_count"] for row in policies)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--queries", type=int, default=20_000)
    parser.add_argument("--expected-models", type=int, default=10)
    parser.add_argument("--expected-episodes", type=int, default=10_000)
    parser.add_argument("--expected-chunk-size", type=int, default=500)
    parser.add_argument(
        "--manifest-name",
        choices=("collection_manifest.json", "pilot_manifest.json"),
        default="collection_manifest.json",
    )
    parser.add_argument(
        "--expected-status",
        choices=("completed", "pilot_completed"),
        default="completed",
    )
    parser.add_argument("--neighbor-backend", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-h5-hash-verification", action="store_true")
    args = parser.parse_args()
    if args.queries <= 0:
        raise SystemExit("queries must be positive")
    collection, by_seed = validate_collection(
        args.cell_dir,
        args.expected_models,
        args.expected_episodes,
        args.expected_chunk_size,
        args.manifest_name,
        args.expected_status,
    )
    normalizer = json.loads(args.normalizer.read_text())
    if (
        normalizer.get("format") != "afqr_lsp_compact_normalizer_v1"
        or normalizer.get("status") != "completed"
        or normalizer.get("algorithm") != collection["algorithm"]
        or normalizer.get("environment") != collection["environment"]
    ):
        raise ValueError("Normalizer does not match collection condition")
    center = np.asarray(normalizer["center"], dtype=np.float64)
    scale = np.asarray(normalizer["scale"], dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = []
    provenance = {}
    for index, (train_seed, records) in enumerate(sorted(by_seed.items()), start=1):
        print(
            f"[{index:02d}/{len(by_seed)}] {args.method_label} seed={train_seed}",
            flush=True,
        )
        arrays, chunk_provenance = load_policy(
            args.cell_dir,
            records,
            verify_hashes=not args.skip_h5_hash_verification,
        )
        result = analyze_policy(
            train_seed,
            arrays,
            center,
            scale,
            args.queries,
            args.neighbor_backend,
            args.output_dir,
        )
        policies.append(result)
        provenance[train_seed] = chunk_provenance
        del arrays
    payload = {
        "format": ANALYSIS_FORMAT,
        "status": "completed",
        "algorithm": collection["algorithm"],
        "environment": collection["environment"],
        "environment_key": collection["environment_key"],
        "method": args.method_label,
        "collection_method": collection["method"],
        "collection_manifest_name": args.manifest_name,
        "collection_status": collection["status"],
        "collection_manifest_sha256": sha256_file(
            args.cell_dir / args.manifest_name
        ),
        "normalizer_sha256": sha256_file(args.normalizer),
        "normalization": {
            "type": normalizer["type"],
            "statistics_source": normalizer["statistics_source"],
            "normalization_count": normalizer["normalization_count"],
        },
        "rollout_mode": "stochastic",
        "discrepancy_action": "deterministic_mean_action",
        "query_count_per_policy": args.queries,
        "query_seed_rule": "training_seed + 20260721",
        "neighbor_exclusion": "same rollout episode",
        "k_grid": list(K_GRID),
        "pair_cache_k": PAIR_K,
        "pair_cache_reuse": (
            "current-distance extension must use these exact query and neighbor indices"
        ),
        "neighbor_backend": args.neighbor_backend,
        "neighbor_algorithm": (
            "exact chunked CUDA squared-Euclidean search"
            if args.neighbor_backend == "cuda"
            else (
                "exact SciPy cKDTree search"
                if args.neighbor_backend == "cpu"
                else "auto-selected exact backend"
            )
        ),
        "h5_checksum_verification": not args.skip_h5_hash_verification,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "scipy_version": package_version("scipy"),
        "torch_version": package_version("torch"),
        "policies": policies,
        "summary": aggregate(policies),
        "chunk_provenance": provenance,
    }
    atomic_write_json(args.output_dir / "analysis_manifest.json", payload)
    print(
        f"Completed {collection['algorithm']} {collection['environment']} "
        f"{args.method_label}: {len(policies)} policies",
        flush=True,
    )


if __name__ == "__main__":
    main()
