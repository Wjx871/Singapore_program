"""XGBoost implementation of the shared ModelAdapter contract.

Uses xgboost==3.0.5 with sklearn API (XGBClassifier).
Imbalance handled via Training-derived scale_pos_weight.
Early stopping uses Validation-only eval_set.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import xgboost as xgb

from src.models.adapters.xgboost_contract import compute_training_scale_pos_weight
from src.models.base import ModelAdapter


class XGBoostAdapter(ModelAdapter):
    model_name = "xgboost"
    model_family = "gradient_boosting"
    model_status = "implemented"
    supports_validation_data = True
    supports_early_stopping = True
    requires_scaled_features = False
    imbalance_strategy = "training_derived_scale_pos_weight"

    # Fixed internal parameters (not user-configurable)
    FIXED_INTERNAL: dict[str, object] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "importance_type": "gain",
        "verbosity": 0,
    }

    FIXED_BASELINE: dict[str, object] = {
        "n_estimators": 2000,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 50,
        "n_jobs": -1,
    }

    def __init__(self, parameters: dict[str, Any], random_seed: int) -> None:
        merged = {**self.FIXED_BASELINE, **parameters}
        super().__init__(merged, random_seed)
        self.estimator: xgb.XGBClassifier | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        x_train: Any,
        y_train: Any,
        *,
        x_validation: Any | None = None,
        y_validation: Any | None = None,
        validation_split_name: str = "validation",
        feature_names: Sequence[str] | None = None,
        groups: Any | None = None,
        sample_weight: Any | None = None,
    ) -> "XGBoostAdapter":
        # --- guard: Test must not enter fit ---------------------------------
        self.validate_validation_inputs(x_validation, y_validation, validation_split_name)

        # --- guard: Validation data is required ------------------------------
        if x_validation is None or y_validation is None:
            raise ValueError(
                "XGBoostAdapter requires Validation data (x_validation, y_validation) "
                "for early stopping"
            )

        # --- compute scale_pos_weight from Training ONLY --------------------
        y_train_arr = np.asarray(y_train).reshape(-1)
        scale_pos_weight = compute_training_scale_pos_weight(y_train_arr)

        # --- build the estimator -------------------------------------------
        train_params = dict(self.parameters)
        # In XGBoost 3.0.5, early_stopping_rounds is a constructor parameter
        # (passed via **kwargs), NOT a fit() parameter.
        early_stopping_rounds = int(train_params["early_stopping_rounds"])
        n_estimators = int(train_params["n_estimators"])

        self.estimator = xgb.XGBClassifier(
            **train_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_seed,
            **self.FIXED_INTERNAL,
        )

        # --- fit with Validation-only eval_set -----------------------------
        started = time.perf_counter()
        self.estimator.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            sample_weight=sample_weight,
            verbose=False,
        )
        training_time = time.perf_counter() - started

        # --- post-fit metadata ---------------------------------------------
        best_iteration = int(self.estimator.best_iteration)
        actual_boosting_rounds = best_iteration + 1

        if not isinstance(best_iteration, int) or best_iteration < 0:
            raise RuntimeError(
                f"XGBoost best_iteration must be a non-negative integer, got {best_iteration}"
            )
        if best_iteration >= n_estimators:
            raise RuntimeError(
                f"best_iteration ({best_iteration}) must be < n_estimators ({n_estimators})"
            )

        best_score_value: float | None = None
        if hasattr(self.estimator, "best_score") and self.estimator.best_score is not None:
            best_score_value = float(self.estimator.best_score)

        names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(self.estimator.n_features_in_)]
        )
        raw_importance = self.estimator.feature_importances_
        if not np.isfinite(raw_importance).all():
            raise RuntimeError("XGBoost feature importance contains non-finite values")
        if (raw_importance < 0).any():
            raise RuntimeError("XGBoost feature importance contains negative values")

        self._training_metadata = {
            "training_time_seconds": training_time,
            "feature_count": len(names),
            "feature_importance": {
                name: float(value)
                for name, value in zip(names, raw_importance, strict=True)
            },
            "feature_importance_type": "gain",
            "supports_validation_data": True,
            "supports_early_stopping": True,
            "early_stopping_rounds": early_stopping_rounds,
            "best_iteration": best_iteration,
            "actual_boosting_rounds": actual_boosting_rounds,
            "best_score": best_score_value,
            "negative_train_count": int((y_train_arr == 0).sum()),
            "positive_train_count": int((y_train_arr == 1).sum()),
            "scale_pos_weight": float(scale_pos_weight),
            "scale_pos_weight_formula": "negative_train_count / positive_train_count",
            "groups_supplied": groups is not None,
            "effective_random_state": self.random_seed,
            "effective_eval_metric": "aucpr",
            "effective_tree_method": "hist",
        }
        return self

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------
    def predict_proba(self, features: Any) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("XGBoostAdapter must be fitted before prediction")
        # XGBoost 3.0.5 auto-uses best_iteration in predict_proba
        raw = self.estimator.predict_proba(features)[:, 1]
        return self.validate_probability_output(raw, len(features))
