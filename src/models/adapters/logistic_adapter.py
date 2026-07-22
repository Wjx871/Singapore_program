"""Logistic Regression implementation of the shared ModelAdapter contract."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from src.models.base import ModelAdapter
from src.models.logistic_regression import build_logistic_pipeline


class LogisticRegressionAdapter(ModelAdapter):
    model_name = "logistic_regression"
    model_family = "linear_model"
    model_status = "implemented"
    supports_validation_data = False
    supports_early_stopping = False
    requires_scaled_features = True
    imbalance_strategy = "class_weight"

    def __init__(
        self,
        parameters: dict[str, Any],
        random_seed: int,
        *,
        binary_feature_names: Sequence[str] = (),
    ) -> None:
        super().__init__(parameters, random_seed)
        self.binary_feature_names = tuple(binary_feature_names)
        self.pipeline: Any | None = None

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
    ) -> "LogisticRegressionAdapter":
        self.validate_validation_inputs(x_validation, y_validation, validation_split_name)
        if x_validation is not None:
            raise ValueError("Logistic Regression does not consume Validation during fit")
        parameters = {**self.parameters, "random_state": self.random_seed}
        self.pipeline = build_logistic_pipeline(
            parameters,
            feature_names=feature_names,
            binary_feature_names=self.binary_feature_names,
        )
        started = time.perf_counter()
        fit_kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
        self.pipeline.fit(x_train, y_train, **fit_kwargs)
        training_time = time.perf_counter() - started
        self._training_metadata = {
            "training_time_seconds": training_time,
            "feature_count": len(feature_names) if feature_names is not None else None,
            "supports_validation_data": False,
            "supports_early_stopping": False,
            "best_iteration": None,
            "scale_pos_weight": None,
            "groups_supplied": groups is not None,
        }
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("LogisticRegressionAdapter must be fitted before prediction")
        raw = self.pipeline.predict_proba(features)[:, 1]
        return self.validate_probability_output(raw, len(features))
