#!/usr/bin/env python3
"""Reproducible HRI benchmark protocol.

The harness evaluates all models on identical held-out records. Synthetic data
are useful for debugging, but are never treated as human validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from quantum_mind.evaluation import (
    brier_score, classification_report, compare_models, create_manifest,
    bootstrap_ci, log_loss, write_manifest,
)

def _decision_costs(y, p, ask_cost, error_cost, ask_resolution):
    """Return realised costs under a binary act/ask policy."""
    if not 0 <= ask_resolution <= 1:
        raise ValueError("ask_resolution must be in [0, 1]")
    act = p >= 0.5
    no_ask_cost = error_cost * (act != y)
    uncertainty = np.minimum(p, 1.0 - p)
    ask = error_cost * uncertainty > ask_cost + (1.0 - ask_resolution) * error_cost * uncertainty
    ask_costs = ask_cost + (1.0 - ask_resolution) * error_cost * uncertainty
    costs = np.where(ask, ask_costs, no_ask_cost)
    return costs, ask

def evaluate(records, prediction_fields, ask_cost=1.0, error_cost=5.0,
             ask_resolution=1.0, seed=0, n_boot=2000, n_permutations=2000):
    """Score probability models, decision costs, and paired model differences."""
    if not records:
        raise ValueError("records must not be empty")
    y = np.asarray([r["y"] for r in records], float)
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("each record must contain binary 'y'")
    if ask_cost < 0 or error_cost <= 0:
        raise ValueError("ask_cost must be >= 0 and error_cost must be positive")

    predictions = {}
    out = {}
    rng = np.random.default_rng(seed)
    for name, field in prediction_fields.items():
        p = np.asarray([r[field] for r in records], float)
        report = classification_report(y, p)
        costs, ask = _decision_costs(y, p, ask_cost, error_cost, ask_resolution)
        estimate, low, high = bootstrap_ci(costs, rng=rng, n_boot=n_boot)
        report.update(
            mean_decision_cost=estimate,
            decision_cost_ci95=[low, high],
            asks=int(ask.sum()),
            ask_rate=float(np.mean(ask)),
            ask_resolution=float(ask_resolution),
        )
        predictions[name] = p
        out[name] = report

    if len(predictions) > 1:
        out["_comparisons"] = {
            "log_loss": compare_models(
                y, predictions, log_loss, n_boot=n_boot,
                n_permutations=n_permutations, seed=seed,
            ),
            "brier": compare_models(
                y, predictions, brier_score, n_boot=n_boot,
                n_permutations=n_permutations, seed=seed + 1,
            ),
        }
    return out

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-name", default="unknown")
    parser.add_argument("--dataset-version")
    parser.add_argument("--data-source", default="synthetic")
    parser.add_argument("--ask-cost", type=float, default=1.0)
    parser.add_argument("--error-cost", type=float, default=5.0)
    parser.add_argument("--ask-resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    args = parser.parse_args(argv)

    records = json.loads(args.input.read_text())
    fields = {k: k for k in records[0] if k.startswith("p_")}
    if not fields:
        raise SystemExit("input must contain at least one p_<model> probability field")

    result = evaluate(
        records, fields, args.ask_cost, args.error_cost, args.ask_resolution,
        args.seed, args.n_boot, args.n_permutations,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    else:
        print(payload)

    if args.manifest:
        manifest = create_manifest(
            experiment="hri_decision_benchmark",
            dataset=args.dataset_name,
            dataset_version=args.dataset_version,
            data_source=args.data_source,
            models=list(fields),
            metrics=["log_loss", "brier", "ece", "accuracy", "decision_cost"],
            split=None,
            seed=args.seed,
            parameters={
                "ask_cost": args.ask_cost,
                "error_cost": args.error_cost,
                "ask_resolution": args.ask_resolution,
                "n_boot": args.n_boot,
                "n_permutations": args.n_permutations,
            },
        )
        write_manifest(manifest, args.manifest)

if __name__ == "__main__":
    main()
