"""Strict model-specific user parameter schemas."""

from __future__ import annotations

import math
from typing import Any


ALLOWED_PARAMETERS: dict[str, set[str]] = {
    "logistic_regression": {"C", "penalty", "solver", "class_weight", "max_iter"},
    "random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "class_weight",
        "n_jobs",
    },
    "balanced_random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "sampling_strategy",
        "replacement",
        "bootstrap",
        "n_jobs",
    },
    "xgboost": {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "early_stopping_rounds",
        "n_jobs",
    },
    "lightgbm": set(),
}


def _is_finite_numeric(value: object) -> bool:
    """Return True when *value* is a finite int or float (not bool, NaN, or Inf)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def validate_model_parameters(model_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    if model_name not in ALLOWED_PARAMETERS:
        raise ValueError(f"Unknown model parameter schema: {model_name}")
    if model_name == "xgboost" and "scale_pos_weight" in parameters:
        raise ValueError("XGBoost scale_pos_weight is computed from Training target, not configured")
    unknown = set(parameters).difference(ALLOWED_PARAMETERS[model_name])
    if unknown:
        raise ValueError(f"Unknown {model_name} parameters: {sorted(unknown)}")
    validated = dict(parameters)

    # --- shared integer validators -------------------------------------------
    for key in ("n_estimators", "max_depth", "min_samples_leaf", "max_iter", "early_stopping_rounds"):
        if key in validated and validated[key] is not None:
            if isinstance(validated[key], bool) or not isinstance(validated[key], int) or validated[key] <= 0:
                raise ValueError(f"{model_name}.{key} must be a positive integer or null")

    if "n_jobs" in validated:
        if isinstance(validated["n_jobs"], bool) or not isinstance(validated["n_jobs"], int):
            raise ValueError(f"{model_name}.n_jobs must be a non-zero integer")
        if validated["n_jobs"] == 0:
            raise ValueError(f"{model_name}.n_jobs must be a non-zero integer")

    # --- shared numeric validators ------------------------------------------
    for key in ("C", "learning_rate", "min_child_weight"):
        if key in validated:
            if not _is_finite_numeric(validated[key]) or validated[key] <= 0:
                raise ValueError(f"{model_name}.{key} must be positive")

    # --- XGBoost-specific numeric validators --------------------------------
    if model_name == "xgboost":
        if "subsample" in validated:
            value = validated["subsample"]
            if not _is_finite_numeric(value) or not (0 < value <= 1):
                raise ValueError("xgboost.subsample must be in (0, 1]")
        if "colsample_bytree" in validated:
            value = validated["colsample_bytree"]
            if not _is_finite_numeric(value) or not (0 < value <= 1):
                raise ValueError("xgboost.colsample_bytree must be in (0, 1]")
        if "reg_lambda" in validated:
            value = validated["reg_lambda"]
            if not _is_finite_numeric(value) or value < 0:
                raise ValueError("xgboost.reg_lambda must be >= 0")

    # --- non-XGBoost subsample / colsample_bytree / reg_lambda ------------
    else:
        for key in ("subsample", "colsample_bytree", "reg_lambda"):
            if key in validated:
                if not _is_finite_numeric(validated[key]) or validated[key] <= 0:
                    raise ValueError(f"{model_name}.{key} must be positive")

    if "class_weight" in validated and validated["class_weight"] not in {
        None,
        "balanced",
        "balanced_subsample",
    }:
        raise ValueError(f"Invalid {model_name}.class_weight")
    if model_name == "balanced_random_forest" and "class_weight" in validated:
        raise ValueError("Balanced Random Forest uses internal sampling; class_weight/SMOTE is not allowed")

    if "max_features" in validated:
        value = validated["max_features"]
        if value not in {None, "sqrt", "log2"}:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Invalid {model_name}.max_features")
            if isinstance(value, float) and value > 1.0:
                raise ValueError(f"{model_name}.max_features float must be in (0, 1]")
    if model_name == "balanced_random_forest":
        if "sampling_strategy" in validated and validated["sampling_strategy"] != "all":
            raise ValueError("Balanced Random Forest sampling_strategy must be 'all'")
        for key in ("replacement", "bootstrap"):
            if key in validated and not isinstance(validated[key], bool):
                raise ValueError(f"balanced_random_forest.{key} must be boolean")
    return validated
