from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.guard import SealedTestSetError
from src.evaluation.metrics import (
    _compute_threshold_independent_metrics,
    _compute_threshold_metrics,
    threshold_independent_metrics,
    threshold_metrics,
)
from src.evaluation.sealed import ExactlyOnceTestPredictor, SealedEvaluationError
from src.experiments.sealed_executor import FROZEN_THRESHOLD, array_sha256


def test_ordinary_metrics_api_still_rejects_test():
    y = np.array([0, 1])
    probability = np.array([0.1, 0.9])
    with pytest.raises(SealedTestSetError):
        threshold_independent_metrics(y, probability, split_name="test")
    with pytest.raises(SealedTestSetError):
        threshold_metrics(y, probability, 0.5, split_name="test")


def test_sealed_metric_core_matches_public_validation_api():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    assert _compute_threshold_independent_metrics(y, probability) == (
        threshold_independent_metrics(y, probability, split_name="validation")
    )
    assert _compute_threshold_metrics(y, probability, FROZEN_THRESHOLD) == (
        threshold_metrics(
            y, probability, FROZEN_THRESHOLD, split_name="validation"
        )
    )


def test_exactly_once_test_predictor_calls_adapter_once():
    calls = []

    def predict(features):
        calls.append(len(features))
        return np.linspace(0.1, 0.9, len(features))

    predictor = ExactlyOnceTestPredictor(predict)
    probability = predictor.predict_proba(np.zeros((6, 2)))
    assert predictor.prediction_call_count == 1
    assert calls == [6]
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert len(array_sha256(probability)) == 64
    with pytest.raises(SealedEvaluationError, match="exactly once"):
        predictor.predict_proba(np.zeros((6, 2)))
    assert calls == [6]


def test_preflight_state_has_zero_prediction_calls():
    predictor = ExactlyOnceTestPredictor(lambda features: np.zeros(len(features)))
    assert predictor.prediction_call_count == 0


def test_fixed_threshold_confusion_matrix_totals_synthetic_rows():
    y = np.array([0, 0, 1, 1, 1])
    probability = np.array([0.2, 0.7, 0.4, 0.8, 0.9])
    metrics = _compute_threshold_metrics(y, probability, FROZEN_THRESHOLD)
    assert metrics["threshold"] == FROZEN_THRESHOLD
    assert sum(metrics["confusion_matrix"].values()) == len(y)


@pytest.mark.parametrize(
    "probability",
    [
        np.array([0.1, np.nan]),
        np.array([0.1, np.inf]),
        np.array([-0.1, 0.5]),
        np.array([0.5, 1.1]),
    ],
)
def test_sealed_metric_core_rejects_invalid_probability(probability):
    with pytest.raises(ValueError):
        _compute_threshold_independent_metrics(np.array([0, 1]), probability)
