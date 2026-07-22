"""ModelAdapter factory with explicit errors for contract-only models."""

from __future__ import annotations

from typing import Any, Sequence

from src.models.adapters.logistic_adapter import LogisticRegressionAdapter
from src.models.base import ModelAdapter
from src.models.contracts import validate_model_parameters
from src.models.registry import get_model_definition


class ModelNotImplementedError(NotImplementedError):
    pass


def create_model_adapter(
    model_name: str,
    parameters: dict[str, Any],
    *,
    random_seed: int,
    binary_feature_names: Sequence[str] = (),
) -> ModelAdapter:
    definition = get_model_definition(model_name)
    validated = validate_model_parameters(model_name, parameters)
    if definition.status != "implemented":
        raise ModelNotImplementedError(
            f"Model '{model_name}' is registered as {definition.status} but has no Stage 3.5 adapter"
        )
    if model_name == "logistic_regression":
        return LogisticRegressionAdapter(
            validated,
            random_seed,
            binary_feature_names=binary_feature_names,
        )
    raise ModelNotImplementedError(f"No adapter implementation is registered for {model_name}")
