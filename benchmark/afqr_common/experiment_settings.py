"""Load the selected configurations used in the benchmark experiments."""

import json
from pathlib import Path


def selected_settings(algorithm: str, environment: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "selected_settings.json"
    return json.loads(path.read_text())["settings"][f"{algorithm}/{environment}"]
