"""Authoritative model names, statuses, and capability metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    family: str
    status: str
    requires_scaled_features: bool
    supports_validation_data: bool
    supports_early_stopping: bool
    imbalance_strategy: str
    core_model: bool


MODEL_REGISTRY: dict[str, ModelDefinition] = {
    "logistic_regression": ModelDefinition(
        "logistic_regression",
        "linear_model",
        "implemented",
        True,
        False,
        False,
        "class_weight",
        True,
    ),
    "random_forest": ModelDefinition(
        "random_forest",
        "bagging",
        "implemented",
        False,
        False,
        False,
        "optional_class_weight_no_smote",
        True,
    ),
    "balanced_random_forest": ModelDefinition(
        "balanced_random_forest",
        "imbalance_aware_bagging",
        "implemented",
        False,
        False,
        False,
        "internal_balanced_sampling_no_smote",
        True,
    ),
    "xgboost": ModelDefinition(
        "xgboost",
        "gradient_boosting",
        "contract_ready",
        False,
        True,
        True,
        "training_derived_scale_pos_weight",
        True,
    ),
    "lightgbm": ModelDefinition(
        "lightgbm",
        "gradient_boosting",
        "optional_legacy",
        False,
        True,
        True,
        "optional_legacy_only",
        False,
    ),
}


def get_model_definition(model_name: str) -> ModelDefinition:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model_name: {model_name}") from exc


def core_model_names() -> tuple[str, ...]:
    return tuple(name for name, definition in MODEL_REGISTRY.items() if definition.core_model)
