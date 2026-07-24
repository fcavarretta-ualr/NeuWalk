import json
from pathlib import Path

import numpy as np


def _jsonable(value):
    """Convert NumPy values and nested containers to JSON-compatible data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_parameters(parameters, path):
    """Save fitted preset parameters as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(parameters), indent=2), encoding="utf-8")
    return path


def load_parameters(path):
    """Load fitted preset parameters from JSON."""
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)
