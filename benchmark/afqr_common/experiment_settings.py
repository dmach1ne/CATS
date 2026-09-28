"""Load the selected configurations used in the benchmark experiments."""

import json
from pathlib import Path


def selected_settings(algorithm: str, environment: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "selected_settings.json"
    return json.loads(path.read_text())["settings"][f"{algorithm}/{environment}"]


def matched_coefficient(algorithm: str, environment: str, method: str, target: float) -> float:
    path = Path(__file__).resolve().parents[1] / "selected_settings.json"
    settings = json.loads(path.read_text())["smoothness_matched"]
    return float(settings[f"{algorithm}/{environment}"][method][f"{target:g}"])
