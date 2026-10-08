"""Tests for the model-agnostic evaluation and research-validation layer."""
import numpy as np
import pytest

from quantum_mind.evaluation import (
    ModelProvenance, brier_score, bootstrap_ci, classification_report,
    compare_models, create_manifest, expected_calibration_error, log_loss,
    paired_permutation_test, sha256_file, validate_provenance,
)
from benchmarks.hri_decision_benchmark import evaluate

def test_probabilistic_metrics_and_report():
    y = np.array([0, 1, 1, 0])
    p = np.array([.1, .9, .8, .2])
    assert brier_score(y, p) < .05
    assert log_loss(y, p) < .25
    assert expected_calibration_error(y, p, bins=2) < .2
    assert classification_report(y, p)["accuracy_at_0.5"] == 1.0

def test_bootstrap_is_reproducible():
    x = np.arange(20, dtype=float)
    assert bootstrap_ci(x, rng=7, n_boot=300) == bootstrap_ci(x, rng=7, n_boot=300)

def test_benchmark_scores_models_and_paired_statistics():
    records = [{"y": int(i % 2), "p_good": .9 if i % 2 else .1, "p_bad": .6} for i in range(20)]
    result = evaluate(records, {"good": "p_good", "bad": "p_bad"},
                      seed=3, n_boot=300, n_permutations=300)
    assert result["good"]["log_loss"] < result["bad"]["log_loss"]
    assert result["good"]["brier"] < result["bad"]["brier"]
    assert len(result["good"]["decision_cost_ci95"]) == 2
    assert "log_loss" in result["_comparisons"]
    assert result["_comparisons"]["log_loss"][0]["model_a"] == "good"

def test_ask_resolution_is_explicit():
    records = [{"y": 1, "p_model": .51}, {"y": 0, "p_model": .49}]
    perfect = evaluate(records, {"model": "p_model"}, ask_cost=.1,
                       error_cost=5, ask_resolution=1, n_boot=100)
    partial = evaluate(records, {"model": "p_model"}, ask_cost=.1,
                       error_cost=5, ask_resolution=.0, n_boot=100)
    assert perfect["model"]["ask_resolution"] == 1.0
    assert partial["model"]["ask_resolution"] == 0.0

def test_paired_permutation_is_reproducible():
    y = np.array([0, 1, 0, 1] * 5)
    a = np.array([.1, .9, .2, .8] * 5)
    b = np.full(20, .5)
    one = paired_permutation_test(y, a, b, log_loss, n_permutations=300, rng=11)
    two = paired_permutation_test(y, a, b, log_loss, n_permutations=300, rng=11)
    assert one == two

def test_model_provenance():
    record = ModelProvenance(model_id="ql-order", category="quantum-like")
    assert validate_provenance(record)["model_id"] == "ql-order"
    with pytest.raises(ValueError):
        validate_provenance({"category": "baseline"})

def test_manifest_and_hash(tmp_path):
    data = tmp_path / "data.json"
    data.write_text('{"y": 1}\n')
    manifest = create_manifest(
        "demo", "demo-data", ["baseline", "quantum-like"],
        ["log_loss"], seed=7, data_source="synthetic",
        dataset_hash=sha256_file(data),
    )
    assert manifest["schema_version"] == "1.0"
    assert manifest["dataset_hash"] == sha256_file(data)

def test_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError):
        brier_score([0, 1], [-.1, .5])
    with pytest.raises(ValueError):
        log_loss([0, 2], [.2, .8])
