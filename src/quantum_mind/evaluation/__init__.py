"""Evaluation and reproducibility utilities for model comparisons."""
from .manifest import create_manifest, sha256_file, write_manifest
from .metrics import brier_score, bootstrap_ci, classification_report, expected_calibration_error, log_loss
from .provenance import ModelProvenance, validate_provenance
from .statistics import compare_models, holm_bonferroni, paired_bootstrap_difference, paired_effect_size, paired_permutation_test

__all__ = [
    "ModelProvenance", "brier_score", "bootstrap_ci", "classification_report",
    "compare_models", "create_manifest", "expected_calibration_error",
    "holm_bonferroni", "log_loss", "paired_bootstrap_difference",
    "paired_effect_size", "paired_permutation_test", "sha256_file",
    "validate_provenance", "write_manifest",
]
