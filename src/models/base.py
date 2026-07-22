"""Common model adapter contract, independent of data loading and evaluation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


class ModelAdapter(ABC):
    model_name: str
    model_family: str
    model_status: str
    supports_validation_data: bool
    supports_early_stopping: bool
    requires_scaled_features: bool
    imbalance_strategy: str

    def __init__(self, parameters: dict[str, Any], random_seed: int) -> None:
        self.parameters = dict(parameters)
        self.random_seed = random_seed
        self._training_metadata: dict[str, Any] = {}

    @abstractmethod
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
    ) -> "ModelAdapter":
        """Fit on Training only; optional evaluation data must be Validation."""

    @abstractmethod
    def predict_proba(self, features: Any) -> np.ndarray:
        """Return one-dimensional positive-class probabilities."""

    def get_parameters(self) -> dict[str, Any]:
        json.dumps(self.parameters, allow_nan=False)
        return dict(self.parameters)

    def get_training_metadata(self) -> dict[str, Any]:
        json.dumps(self._training_metadata, allow_nan=False)
        return dict(self._training_metadata)

    @staticmethod
    def validate_validation_inputs(
        x_validation: Any | None,
        y_validation: Any | None,
        validation_split_name: str,
    ) -> None:
        if validation_split_name != "validation":
            raise ValueError("ModelAdapter fit permits Validation data only; Test is sealed")
        if (x_validation is None) != (y_validation is None):
            raise ValueError("X_validation and y_validation must be provided together")

    @staticmethod
    def validate_probability_output(probability: Any, expected_rows: int) -> np.ndarray:
        values = np.asarray(probability, dtype="float64")
        if values.ndim != 1:
            raise ValueError("ModelAdapter predict_proba must return a one-dimensional array")
        if len(values) != expected_rows:
            raise ValueError(
                f"Probability length mismatch: expected {expected_rows}, got {len(values)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Model probabilities must be finite")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError("Model probabilities must lie in [0, 1]")
        return values
