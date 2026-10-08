"""Model-agnostic probabilistic evaluation metrics."""
from __future__ import annotations
import numpy as np

def _validate_binary(y_true, y_prob):
    y, p = np.asarray(y_true, float), np.asarray(y_prob, float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true and y_prob must be one-dimensional arrays of equal length")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y_true must contain only 0 and 1")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")
    if not len(y):
        raise ValueError("at least one observation is required")
    return y, np.clip(p, 1e-15, 1 - 1e-15)

def log_loss(y_true, y_prob):
    """Mean binary log loss; lower is better."""
    y, p = _validate_binary(y_true, y_prob)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log1p(-p)))

def brier_score(y_true, y_prob):
    """Binary Brier score; lower is better."""
    y, p = _validate_binary(y_true, y_prob)
    return float(np.mean((p-y)**2))

def expected_calibration_error(y_true, y_prob, bins=10):
    """Equal-width expected calibration error; lower is better."""
    if bins < 1:
        raise ValueError("bins must be positive")
    y, p = _validate_binary(y_true, y_prob)
    edges = np.linspace(0, 1, bins+1)
    return float(sum(np.mean(mask)*abs(np.mean(y[mask])-np.mean(p[mask]))
                     for i in range(bins)
                     for mask in [(p >= edges[i]) & ((p < edges[i+1]) if i < bins-1 else (p <= edges[i+1]))]
                     if np.any(mask)))

def bootstrap_ci(values, statistic=np.mean, confidence=0.95, n_boot=2000, rng=None):
    """Percentile bootstrap confidence interval for a statistic."""
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("values must be a finite one-dimensional array with at least two entries")
    if not 0 < confidence < 1 or n_boot < 100:
        raise ValueError("confidence must be in (0, 1) and n_boot must be at least 100")
    rng = np.random.default_rng(rng)
    samples = rng.integers(0, len(x), size=(n_boot, len(x)))
    estimates = np.asarray([statistic(x[idx]) for idx in samples])
    alpha = (1-confidence)/2
    return float(statistic(x)), float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1-alpha))

def classification_report(y_true, y_prob, bins=10):
    """Return a JSON-friendly probabilistic evaluation summary."""
    y, p = _validate_binary(y_true, y_prob)
    return {"n": int(len(y)), "log_loss": log_loss(y,p), "brier": brier_score(y,p),
            "ece": expected_calibration_error(y,p,bins), "accuracy_at_0.5": float(np.mean((p >= .5) == y))}
