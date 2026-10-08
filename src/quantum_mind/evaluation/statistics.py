"""Paired statistical tests for model-comparison experiments."""
from __future__ import annotations

from itertools import combinations
import numpy as np

from .metrics import _validate_binary, brier_score, log_loss


def _paired_metric(y_true, p_a, p_b, metric):
    y, a = _validate_binary(y_true, p_a)
    _, b = _validate_binary(y_true, p_b)
    return float(metric(y, a) - metric(y, b))


def _per_observation_losses(y, p, metric):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    if metric is log_loss:
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    if metric is brier_score:
        return (p - y) ** 2
    return None


def paired_bootstrap_difference(y_true, p_a, p_b, metric, n_boot=2000,
                                confidence=0.95, rng=None, chunk_size=256):
    """Estimate a paired bootstrap CI for metric(A) - metric(B)."""
    y, a = _validate_binary(y_true, p_a)
    _, b = _validate_binary(y_true, p_b)
    if not callable(metric):
        raise TypeError("metric must be callable")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    generator = np.random.default_rng(rng)
    n = len(y)
    estimates = np.empty(n_boot, dtype=float)
    offset = 0
    while offset < n_boot:
        size = min(chunk_size, n_boot - offset)
        indices = generator.integers(0, n, size=(size, n))
        estimates[offset:offset + size] = [
            _paired_metric(y[idx], a[idx], b[idx], metric) for idx in indices
        ]
        offset += size
    alpha = (1.0 - confidence) / 2.0
    return {
        "difference": _paired_metric(y, a, b, metric),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": float(confidence),
        "n_boot": int(n_boot),
    }


def paired_permutation_test(y_true, p_a, p_b, metric, n_permutations=5000, rng=None):
    """Run a paired randomisation test by swapping model labels per observation."""
    y, a = _validate_binary(y_true, p_a)
    _, b = _validate_binary(y_true, p_b)
    if not callable(metric):
        raise TypeError("metric must be callable")
    if n_permutations < 100:
        raise ValueError("n_permutations must be at least 100")
    observed = _paired_metric(y, a, b, metric)
    generator = np.random.default_rng(rng)
    extreme = 0
    for _ in range(n_permutations):
        swap = generator.integers(0, 2, size=len(y), dtype=np.int8).astype(bool)
        statistic = _paired_metric(y, np.where(swap, b, a), np.where(swap, a, b), metric)
        extreme += int(abs(statistic) >= abs(observed))
    return {
        "difference": observed,
        "p_value": float((1.0 + extreme) / (n_permutations + 1.0)),
        "n_permutations": int(n_permutations),
    }


def paired_effect_size(y_true, p_a, p_b, metric):
    """Return standardised paired loss difference for log loss or Brier score."""
    y, a = _validate_binary(y_true, p_a)
    _, b = _validate_binary(y_true, p_b)
    loss_a = _per_observation_losses(y, a, metric)
    loss_b = _per_observation_losses(y, b, metric)
    if loss_a is None:
        return {"effect_size": None}
    delta = loss_a - loss_b
    scale = float(np.std(delta, ddof=1))
    return {"effect_size": float(np.mean(delta) / scale) if scale > 0 else 0.0}


def holm_bonferroni(p_values, alpha=0.05):
    """Apply Holm's step-down correction and return adjusted p-values."""
    p = np.asarray(p_values, float)
    if p.ndim != 1 or not len(p) or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a non-empty finite vector in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    return {"adjusted_p_values": adjusted.tolist(), "alpha": float(alpha),
            "significant": (adjusted < alpha).tolist()}


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
        results.append({
            "model_a": a, "model_b": b, **bootstrap, **permutation,
            **paired_effect_size(y_true, predictions[a], predictions[b], metric)
        })
    correction = holm_bonferroni([r["p_value"] for r in results])
    for result, adjusted, significant in zip(
        results, correction["adjusted_p_values"], correction["significant"]
    ):
        result["p_value_holm"] = adjusted
        result["significant_holm"] = significant
    return results
