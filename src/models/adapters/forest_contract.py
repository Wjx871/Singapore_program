"""Shared RF/BRF contracts, including deterministic serial inference."""

from __future__ import annotations

from typing import Any

import numpy as np

RF_DEFAULT_CONTRACT = {
    "feature_set": "B",
    "preprocessing_strategy": "missing_plus_flag",
    "requires_scaled_features": False,
    "random_state": 42,
    "class_weight": None,
    "smote_allowed": False,
    "probability_output_required": True,
}

BRF_DEFAULT_CONTRACT = {
    "feature_set": "B",
    "preprocessing_strategy": "missing_plus_flag",
    "requires_scaled_features": False,
    "random_state": 42,
    "implementation": "imblearn.ensemble.BalancedRandomForestClassifier",
    "imbalance_strategy": "internal_balanced_sampling",
    "smote_allowed": False,
    "probability_output_required": True,
}


def serial_forest_predict_proba(estimator: Any, features: Any) -> np.ndarray:
    """Use native forest prediction serially, restoring training parallelism."""
    original_n_jobs = estimator.n_jobs
    try:
        estimator.set_params(n_jobs=1)
        return np.asarray(estimator.predict_proba(features))[:, 1]
    finally:
        estimator.set_params(n_jobs=original_n_jobs)
