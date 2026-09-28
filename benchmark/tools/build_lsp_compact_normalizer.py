"""Build one pooled observation normalizer from compact Lsp collections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


COLLECTION_FORMAT = "afqr_lsp_compact_collection_v1"
CHUNK_FORMAT = "afqr_lsp_compact_chunk_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_cell(value: str) -> tuple[str, Path]:
    method, separator, path = value.partition("=")
    if not separator or not method or not path:
        raise argparse.ArgumentTypeError("cell must be METHOD=PATH")
    return method, Path(path)


def load_chunk_statistics(
    cell_dir: Path, record: dict[str, Any]
) -> tuple[int, np.ndarray, np.ndarray, bool, str]:
    manifest_path = cell_dir / record["manifest"]
    chunk = json.loads(manifest_path.read_text())
    if chunk.get("format") != CHUNK_FORMAT or chunk.get("status") != "completed":
        raise ValueError(f"Invalid chunk manifest: {manifest_path}")
    count = int(chunk["normalization_count"])
    used_h5_fallback = (
        "normalization_sum" not in chunk or "normalization_sumsq" not in chunk
    )
    if used_h5_fallback:
        h5_path = cell_dir / record["h5"]
        with h5py.File(h5_path, "r") as handle:
            if handle.attrs.get("format") != CHUNK_FORMAT:
                raise ValueError(f"Invalid compact HDF5: {h5_path}")
            observed_count = int(handle.attrs["normalization_count"])
            values_sum = np.asarray(handle["normalization_sum"], dtype=np.float64)
            values_sumsq = np.asarray(handle["normalization_sumsq"], dtype=np.float64)
        if observed_count != count:
            raise ValueError(f"Normalization count mismatch: {h5_path}")
    else:
        values_sum = np.asarray(chunk["normalization_sum"], dtype=np.float64)
        values_sumsq = np.asarray(chunk["normalization_sumsq"], dtype=np.float64)
    if values_sum.ndim != 1 or values_sum.shape != values_sumsq.shape:
        raise ValueError(f"Invalid normalization vectors: {manifest_path}")
    if count <= 0 or not np.all(np.isfinite(values_sum)) or not np.all(
        np.isfinite(values_sumsq)
    ):
        raise ValueError(f"Invalid normalization statistics: {manifest_path}")
    return (
        count,
        values_sum,
        values_sumsq,
        used_h5_fallback,
        sha256_file(manifest_path),
    )


def build_normalizer(
    cells: list[tuple[str, Path]],
    expected_models: int,
    expected_episodes: int,
    expected_chunk_size: int,
    manifest_name: str = "collection_manifest.json",
    expected_status: str = "completed",
) -> dict[str, Any]:
    if len({method for method, _ in cells}) != len(cells):
        raise ValueError("Cell methods must be unique")
    total_count = 0
    total_sum: np.ndarray | None = None
    total_sumsq: np.ndarray | None = None
    fallback_count = 0
    sources: list[dict[str, Any]] = []
    algorithm = None
    environment = None
    expected_records = expected_models * (expected_episodes // expected_chunk_size)

    for method, cell_dir in cells:
        collection_path = cell_dir / manifest_name
        collection = json.loads(collection_path.read_text())
        if (
            collection.get("format") != COLLECTION_FORMAT
            or collection.get("status") != expected_status
            or collection.get("models") != expected_models
            or collection.get("episode_count_per_model") != expected_episodes
            or collection.get("chunk_size") != expected_chunk_size
            or collection.get("observed_record_count") != expected_records
            or len(collection.get("records", [])) != expected_records
        ):
            raise ValueError(f"Incomplete or incompatible collection: {collection_path}")
        records_by_seed: dict[str, list[dict[str, Any]]] = {}
        for record in collection["records"]:
            records_by_seed.setdefault(str(record["train_seed"]), []).append(record)
        expected_starts = list(
            range(0, expected_episodes, expected_chunk_size)
        )
        if len(records_by_seed) != expected_models:
            raise ValueError(f"Unexpected policy grid: {collection_path}")
        for train_seed, records in records_by_seed.items():
            spans = sorted(
                (
                    int(record["episode_start"]),
                    int(record["episode_stop"]),
                )
                for record in records
            )
            expected_spans = [
                (start, start + expected_chunk_size)
                for start in expected_starts
            ]
            if spans != expected_spans:
                raise ValueError(
                    f"Non-contiguous chunk grid for seed {train_seed}: "
                    f"{collection_path}"
                )
        if algorithm is None:
            algorithm = collection["algorithm"]
            environment = collection["environment"]
        elif (
            collection["algorithm"] != algorithm
            or collection["environment"] != environment
        ):
            raise ValueError("Normalizer cells must share algorithm and environment")

        source_count = 0
        source_sum: np.ndarray | None = None
        source_sumsq: np.ndarray | None = None
        source_fallback = 0
        chunk_hashes = []
        for record in collection["records"]:
            count, values_sum, values_sumsq, used_fallback, manifest_hash = (
                load_chunk_statistics(cell_dir, record)
            )
            if source_sum is None:
                source_sum = np.zeros_like(values_sum)
                source_sumsq = np.zeros_like(values_sumsq)
            if values_sum.shape != source_sum.shape:
                raise ValueError(f"Observation shape changed in {cell_dir}")
            source_count += count
            source_sum += values_sum
            assert source_sumsq is not None
            source_sumsq += values_sumsq
            source_fallback += int(used_fallback)
            chunk_hashes.append(manifest_hash)
        assert source_sum is not None and source_sumsq is not None
        if total_sum is None:
            total_sum = np.zeros_like(source_sum)
            total_sumsq = np.zeros_like(source_sumsq)
        if source_sum.shape != total_sum.shape:
            raise ValueError("Observation shape differs across methods")
        total_count += source_count
        total_sum += source_sum
        assert total_sumsq is not None
        total_sumsq += source_sumsq
        fallback_count += source_fallback
        sources.append(
            {
                "method": method,
                "cell_dir": str(cell_dir),
                "collection_manifest_name": manifest_name,
                "collection_status": collection["status"],
                "collection_manifest_sha256": sha256_file(collection_path),
                "chunk_manifest_digest_sha256": hashlib.sha256(
                    "\n".join(chunk_hashes).encode()
                ).hexdigest(),
                "normalization_count": source_count,
                "h5_fallback_chunks": source_fallback,
            }
        )

    assert total_sum is not None and total_sumsq is not None
    center = total_sum / total_count
    variance = np.maximum(total_sumsq / total_count - center * center, 0.0)
    scale = np.sqrt(variance)
    small = scale < 1e-8
    scale[small] = 1.0
    return {
        "format": "afqr_lsp_compact_normalizer_v1",
        "status": "completed",
        "algorithm": algorithm,
        "environment": environment,
        "methods": [method for method, _ in cells],
        "type": "feature-wise pooled z-score",
        "statistics_source": (
            "all obs and next_obs across all five methods"
            if len(cells) == 5
            else f"all obs and next_obs across {len(cells)} supplied method cell(s)"
        ),
        "collection_manifest_name": manifest_name,
        "collection_status": expected_status,
        "population_standard_deviation": True,
        "normalization_count": total_count,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "small_scale_threshold": 1e-8,
        "small_scale_replacement": 1.0,
        "small_scale_dimensions": np.flatnonzero(small).tolist(),
        "h5_fallback_chunks": fallback_count,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", type=parse_cell, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    args = parser.parse_args()
    payload = build_normalizer(
        args.cell,
        args.expected_models,
        args.expected_episodes,
        args.expected_chunk_size,
        args.manifest_name,
        args.expected_status,
    )
    atomic_write_json(args.output, payload)
    print(
        f"Built {payload['algorithm']} {payload['environment']} normalizer from "
        f"{payload['normalization_count']} observations; "
        f"HDF5 fallbacks={payload['h5_fallback_chunks']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
