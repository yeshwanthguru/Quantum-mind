"""Reproducibility metadata for experiments."""
from __future__ import annotations
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from .. import __version__

def sha256_file(path):
    """Return the SHA-256 digest of a dataset or result file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def create_manifest(experiment, dataset, models, metrics, seed, split=None,
                    dataset_version=None, dataset_hash=None, data_source=None,
                    model_provenance=None, parameters=None, notes=None):
    """Create a JSON-serialisable reproducibility manifest."""
    if not experiment or not dataset:
        raise ValueError("experiment and dataset are required")
    if not models or not metrics:
        raise ValueError("models and metrics must not be empty")
    return {
        "schema_version": "1.0", "experiment": str(experiment),
        "dataset": str(dataset), "dataset_version": dataset_version,
        "dataset_hash": dataset_hash, "data_source": data_source,
        "models": [str(x) for x in models], "model_provenance": model_provenance,
        "metrics": [str(x) for x in metrics], "split": split, "seed": int(seed),
        "parameters": parameters or {}, "quantum_mind_version": __version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "notes": notes,
    }

def write_manifest(manifest, path):
    """Write a manifest as formatted JSON and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output
