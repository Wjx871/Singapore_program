"""Random Forest implementation of the shared ModelAdapter contract."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.adapters.forest_contract import serial_forest_predict_proba
from src.models.base import ModelAdapter


class RandomForestAdapter(ModelAdapter):
    model_name = "random_forest"
    model_family = "bagging"
    model_status = "implemented"
    supports_validation_data = False
    supports_early_stopping = False
    requires_scaled_features = False
    imbalance_strategy = "optional_class_weight_no_smote"

    DEFAULT_PARAMETERS: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": None,
        "n_jobs": -1,
    }

    def __init__(self, parameters: dict[str, Any], random_seed: int) -> None:
        super().__init__({**self.DEFAULT_PARAMETERS, **parameters}, random_seed)
        self.estimator: RandomForestClassifier | None = None

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
    ) -> "RandomForestAdapter":
        self.validate_validation_inputs(x_validation, y_validation, validation_split_name)
        if x_validation is not None:
            raise ValueError("Random Forest does not consume Validation during fit")
        self.estimator = RandomForestClassifier(
            **self.parameters,
            random_state=self.random_seed,
        )
        started = time.perf_counter()
        self.estimator.fit(x_train, y_train, sample_weight=sample_weight)
        training_time = time.perf_counter() - started
        names = list(feature_names) if feature_names is not None else [
            f"feature_{index}" for index in range(self.estimator.n_features_in_)
        ]
        self._training_metadata = {
            "training_time_seconds": training_time,
            "feature_count": len(names),
            "feature_importance": {
                name: float(value)
                for name, value in zip(names, self.estimator.feature_importances_, strict=True)
            },
            "supports_validation_data": False,
            "supports_early_stopping": False,
            "best_iteration": None,
            "scale_pos_weight": None,
            "groups_supplied": groups is not None,
            "effective_random_state": self.random_seed,
            "training_n_jobs": self.estimator.n_jobs,
            "inference_n_jobs": 1,
            "deterministic_serial_inference": True,
        }
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("RandomForestAdapter must be fitted before prediction")
        raw = serial_forest_predict_proba(self.estimator, features)
        return self.validate_probability_output(raw, len(features))
