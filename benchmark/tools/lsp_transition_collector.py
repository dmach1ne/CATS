"""Seed and checkpoint helpers for compact transition collection."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

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
