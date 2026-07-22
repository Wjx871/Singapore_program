"""XGBoost handoff contract and Training-only class-ratio calculation."""

from __future__ import annotations

import numpy as np

XGBOOST_DEFAULT_CONTRACT = {
    "feature_set": "B",
    "preprocessing_strategy": "missing_plus_flag",
    "requires_scaled_features": False,
    "random_state": 42,
    "supports_validation_data": True,
    "supports_early_stopping": True,
    "early_stopping_split": "validation",
    "smote_allowed": False,
}


def compute_training_scale_pos_weight(y_train: object) -> float:
    values = np.asarray(y_train).reshape(-1)
    if len(values) == 0 or not set(np.unique(values)).issubset({0, 1}):
        raise ValueError("Training target must be a non-empty binary array")
    positive = int(values.sum())
    negative = len(values) - positive
    if positive == 0 or negative == 0:
        raise ValueError("scale_pos_weight requires both classes in Training")
    return negative / positive
