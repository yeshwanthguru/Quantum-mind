"""Tests for the model-agnostic evaluation layer."""
import numpy as np
import pytest
from quantum_mind.evaluation import brier_score, log_loss, expected_calibration_error, bootstrap_ci, classification_report
from benchmarks.hri_decision_benchmark import evaluate

def test_probabilistic_metrics_and_report():
    y = np.array([0,1,1,0])
    p = np.array([.1,.9,.8,.2])
    assert brier_score(y,p) < .05
    assert log_loss(y,p) < .25
    assert expected_calibration_error(y,p, bins=2) < .2
    report = classification_report(y,p)
    assert report["accuracy_at_0.5"] == 1.0

def test_bootstrap_is_reproducible():
    x = np.arange(20, dtype=float)
    assert bootstrap_ci(x, rng=7, n_boot=300) == bootstrap_ci(x, rng=7, n_boot=300)

def test_benchmark_scores_models_on_same_records():
    records = [{"y":int(i%2), "p_good": .9 if i%2 else .1, "p_bad": .6} for i in range(20)]
    result = evaluate(records, {"good":"p_good", "bad":"p_bad"}, seed=3, n_boot=300)
    assert result["good"]["log_loss"] < result["bad"]["log_loss"]
    assert result["good"]["brier"] < result["bad"]["brier"]
    assert len(result["good"]["decision_cost_ci95"]) == 2

def test_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError):
        brier_score([0,1], [-.1,.5])
    with pytest.raises(ValueError):
        log_loss([0,2], [.2,.8])
