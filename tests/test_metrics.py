from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import threshold_independent_metrics, threshold_metrics


def test_metrics_known_example():
    y = [0, 0, 1, 1]
    probability = [0.1, 0.4, 0.35, 0.8]
    independent = threshold_independent_metrics(y, probability, split_name="validation")
    default = threshold_metrics(y, probability, 0.5, split_name="validation")
    assert independent["pr_auc"] == pytest.approx(5 / 6)
    assert independent["roc_auc"] == pytest.approx(0.75)
    assert independent["ks"] == pytest.approx(0.5)
    assert default["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert default["specificity"] == 1.0


def test_empty_and_invalid_probability_rejected():
    with pytest.raises(ValueError, match="empty"):
        threshold_independent_metrics([], [], split_name="validation")
    with pytest.raises(ValueError, match="finite"):
        threshold_metrics([0], [math.inf], 0.5, split_name="validation")


def test_single_class_marks_roc_and_ks_undefined():
    result = threshold_independent_metrics([0, 0], [0.1, 0.2], split_name="validation")
    assert result["pr_auc"] is None and result["roc_auc"] is None and result["ks"] is None
    assert result["undefined_metrics"] == ["pr_auc", "roc_auc", "ks"]
