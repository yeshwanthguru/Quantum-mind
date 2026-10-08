"""Reproducibility metadata for experiments."""
from __future__ import annotations

import json
import platform
from pathlib import Path
from datetime import datetime, timezone

from .. import __version__


def create_manifest(
    experiment,
    dataset,
    models,
    metrics,
    seed,
    split=None,
    dataset_version=None,
    dataset_hash=None,
    notes=None,
):
    """Create a JSON-serialisable reproducibility manifest."""
    if not experiment or not dataset:
        raise ValueError("experiment and dataset are required")
    if not models:
        raise ValueError("models must not be empty")
    if not metrics:
        raise ValueError("metrics must not be empty")
    return {
        "experiment": str(experiment),
        "dataset": str(dataset),
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "models": [str(x) for x in models],
        "metrics": [str(x) for x in metrics],
        "split": split,
        "seed": int(seed),
        "quantum_mind_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }


def write_manifest(manifest, path):
    """Write a manifest as formatted JSON and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output
