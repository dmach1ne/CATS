from pathlib import Path


def select_checkpoints(files, expected_seeds):
    """Select exactly one completed checkpoint for each requested training seed."""
    seeds = list(expected_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be nonempty and unique")
    by_seed = {}
    for filename in files:
        path = Path(filename)
        name = path.parent.name if path.name == "final.zip" else path.stem
        seed = int(name.rsplit("_", 1)[-1])
        if seed not in seeds:
            continue
        if seed in by_seed:
            raise ValueError(f"Multiple checkpoints for training seed {seed}")
        by_seed[seed] = filename
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise FileNotFoundError(f"Missing completed checkpoints for training seeds: {missing}")
    return [by_seed[seed] for seed in seeds]
