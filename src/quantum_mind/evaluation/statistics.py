"""Paired statistical tests for model-comparison experiments."""
from __future__ import annotations

from itertools import combinations
import numpy as np

from .metrics import _validate_binary


def _paired_metric(y_true, p_a, p_b, metric):
    y, a = _validate_binary(y_true, p_a)
    _, b = _validate_binary(y_true, p_b)
    return metric(y, a) - metric(y, b)


def paired_bootstrap_difference(
    y_true,
    p_a,
    p_b,
    metric,
    n_boot=2000,
    confidence=0.95,
    rng=None,
):
    """Estimate a paired bootstrap confidence interval for metric(A) - metric(B)."""
    y, a = _validate_binary(y_true, p_a)
    _, b = _validate_binary(y_true, p_b)
    if not callable(metric):
        raise TypeError("metric must be callable")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    generator = np.random.default_rng(rng)
    n = len(y)
    indices = generator.integers(0, n, size=(n_boot, n))
    estimates = np.asarray([_paired_metric(y[idx], a[idx], b[idx], metric) for idx in indices])
    alpha = (1.0 - confidence) / 2.0
    return {
        "difference": float(_paired_metric(y, a, b, metric)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": float(confidence),
        "n_boot": int(n_boot),
    }


def paired_permutation_test(y_true, p_a, p_b, metric, n_permutations=5000, rng=None):
    """Test the null that two paired models have equal metric performance.

    The paired differences are sign-flipped under the null. The returned
    two-sided p-value is suitable for comparing models on the same records.
    """
    y, a = _validate_binary(y_true, p_a)
    _, b = _validate_binary(y_true, p_b)
    if not callable(metric):
        raise TypeError("metric must be callable")
    if n_permutations < 100:
        raise ValueError("n_permutations must be at least 100")
    observed = float(_paired_metric(y, a, b, metric))
    generator = np.random.default_rng(rng)
    differences = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = generator.choice((-1.0, 1.0), size=len(y))
        differences[i] = float(np.mean(signs * np.asarray([
            metric(np.asarray([y[j]]), np.asarray([a[j]]))
            - metric(np.asarray([y[j]]), np.asarray([b[j]]))
            for j in range(len(y))
        ])))
    p_value = (1.0 + np.count_nonzero(np.abs(differences) >= abs(observed))) / (n_permutations + 1.0)
    return {
        "difference": observed,
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
    }


def compare_models(y_true, predictions, metric, n_boot=2000, n_permutations=5000, seed=0):
    """Compare every pair of named probability predictions with paired tests."""
    if not predictions:
        raise ValueError("predictions must not be empty")
    names = list(predictions)
    rng = np.random.default_rng(seed)
    results = []
    for a, b in combinations(names, 2):
        bootstrap = paired_bootstrap_difference(
            y_true, predictions[a], predictions[b], metric, n_boot=n_boot, rng=rng
        )
        permutation = paired_permutation_test(
            y_true, predictions[a], predictions[b], metric,
            n_permutations=n_permutations, rng=rng
        )
        results.append({"model_a": a, "model_b": b, **bootstrap, **permutation})
    return results
