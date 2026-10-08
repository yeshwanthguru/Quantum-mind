#!/usr/bin/env python3
"""Reproducible HRI benchmark protocol.

The harness evaluates predictions from any model family on the same held-out
records. Synthetic data are useful for debugging, but are not human validation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from quantum_mind.evaluation import classification_report, bootstrap_ci

def evaluate(records, prediction_fields, ask_cost=1.0, error_cost=5.0, seed=0, n_boot=2000):
    """Score named probability columns against the same binary observations."""
    if not records:
        raise ValueError("records must not be empty")
    y = np.asarray([r["y"] for r in records], float)
    if not np.all(np.isin(y,(0,1))):
        raise ValueError("each record must contain binary 'y'")
    out, rng = {}, np.random.default_rng(seed)
    for name, field in prediction_fields.items():
        p = np.asarray([r[field] for r in records], float)
        report = classification_report(y,p)
        ask = error_cost*(1-p) > ask_cost
        costs = np.where(ask, ask_cost, error_cost*(1-y))
        estimate, low, high = bootstrap_ci(costs, rng=rng, n_boot=n_boot)
        report.update(mean_decision_cost=estimate, decision_cost_ci95=[low,high], asks=int(ask.sum()))
        out[name] = report
    return out

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ask-cost", type=float, default=1.0)
    parser.add_argument("--error-cost", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    records = json.loads(args.input.read_text())
    fields = {k:k for k in records[0] if k.startswith("p_")}
    if not fields:
        raise SystemExit("input must contain at least one p_<model> probability field")
    print(json.dumps(evaluate(records, fields, args.ask_cost, args.error_cost, args.seed, args.n_boot),
                     indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
