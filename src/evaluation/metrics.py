"""Unified binary-classification metrics with positive label fixed to 1."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.guard import require_evaluation_split


def validate_binary_inputs(y_true: object, probability: object) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).reshape(-1)
    probability_array = np.asarray(probability, dtype="float64").reshape(-1)
    if len(y) == 0:
        raise ValueError("Metric input must not be empty")
    if len(y) != len(probability_array):
        raise ValueError("y_true and probability must have the same length")
    if not np.isfinite(probability_array).all():
        raise ValueError("Probabilities must be finite")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    unique = set(np.unique(y))
    if not unique.issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1")
    return y.astype("int8"), probability_array


def threshold_independent_metrics(
    y_true: object,
    probability: object,
    *,
    split_name: str,
) -> dict[str, Any]:
    require_evaluation_split(split_name)
    y, probability_array = validate_binary_inputs(y_true, probability)
    result: dict[str, Any] = {
        "pr_auc": None,
        "roc_auc": None,
        "ks": None,
        "undefined_metrics": [],
    }
    if len(np.unique(y)) < 2:
        result["undefined_metrics"] = ["pr_auc", "roc_auc", "ks"]
        return result
    result["pr_auc"] = float(average_precision_score(y, probability_array))
    result["roc_auc"] = float(roc_auc_score(y, probability_array))
    false_positive_rate, true_positive_rate, _ = roc_curve(y, probability_array, pos_label=1)
    result["ks"] = float(np.max(np.abs(true_positive_rate - false_positive_rate)))
    return result


def threshold_metrics(
    y_true: object,
    probability: object,
    threshold: float,
    *,
    split_name: str,
) -> dict[str, Any]:
    require_evaluation_split(split_name)
    y, probability_array = validate_binary_inputs(y_true, probability)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    prediction = (probability_array >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "predicted_positive_rate": float(prediction.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
