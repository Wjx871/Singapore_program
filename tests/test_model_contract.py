from __future__ import annotations

import numpy as np
import pytest

from src.models.adapters.forest_contract import BRF_DEFAULT_CONTRACT, RF_DEFAULT_CONTRACT
from src.models.adapters.xgboost_contract import (
    XGBOOST_DEFAULT_CONTRACT,
    compute_training_scale_pos_weight,
)
from src.models.base import ModelAdapter
from src.models.contracts import validate_model_parameters
from src.models.factory import ModelNotImplementedError, create_model_adapter
from src.models.registry import MODEL_REGISTRY, core_model_names


def test_registry_names_and_statuses():
    assert core_model_names() == (
        "logistic_regression",
        "random_forest",
        "balanced_random_forest",
        "xgboost",
    )
    assert MODEL_REGISTRY["logistic_regression"].status == "implemented"
    assert MODEL_REGISTRY["random_forest"].status == "implemented"
    assert MODEL_REGISTRY["balanced_random_forest"].status == "implemented"
    assert MODEL_REGISTRY["xgboost"].status == "contract_ready"
    assert MODEL_REGISTRY["lightgbm"].status == "optional_legacy"
    assert MODEL_REGISTRY["lightgbm"].core_model is False


def test_factory_creates_logistic_adapter():
    adapter = create_model_adapter(
        "logistic_regression",
        {"C": 1.0, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced", "max_iter": 1000},
        random_seed=42,
    )
    assert isinstance(adapter, ModelAdapter)
    assert adapter.requires_scaled_features is True


@pytest.mark.parametrize("name", ["xgboost"])
def test_factory_rejects_contract_only_models(name):
    with pytest.raises(ModelNotImplementedError, match="contract_ready"):
        create_model_adapter(name, {}, random_seed=42)


def test_unknown_parameters_are_rejected():
    with pytest.raises(ValueError, match="Unknown logistic_regression"):
        validate_model_parameters("logistic_regression", {"made_up": 1})
    with pytest.raises(ValueError, match="computed from Training"):
        validate_model_parameters("xgboost", {"scale_pos_weight": 13.0})


def test_contract_scaling_and_imbalance_rules():
    assert RF_DEFAULT_CONTRACT["requires_scaled_features"] is False
    assert BRF_DEFAULT_CONTRACT["requires_scaled_features"] is False
    assert XGBOOST_DEFAULT_CONTRACT["requires_scaled_features"] is False
    assert RF_DEFAULT_CONTRACT["smote_allowed"] is False
    assert BRF_DEFAULT_CONTRACT["smote_allowed"] is False
    assert XGBOOST_DEFAULT_CONTRACT["early_stopping_split"] == "validation"


def test_scale_pos_weight_uses_training_counts():
    assert compute_training_scale_pos_weight([0, 0, 0, 1]) == 3.0
    with pytest.raises(ValueError, match="both classes"):
        compute_training_scale_pos_weight([0, 0])


@pytest.mark.parametrize(
    ("probability", "rows", "message"),
    [
        ([[0.1], [0.2]], 2, "one-dimensional"),
        ([0.1], 2, "length mismatch"),
        ([0.1, np.inf], 2, "finite"),
        ([0.1, 1.2], 2, r"\[0, 1\]"),
    ],
)
def test_probability_contract(probability, rows, message):
    with pytest.raises(ValueError, match=message):
        ModelAdapter.validate_probability_output(probability, rows)


def test_adapter_rejects_test_as_validation_input():
    adapter = create_model_adapter(
        "logistic_regression",
        {"C": 1.0, "penalty": "l2", "solver": "liblinear", "class_weight": "balanced", "max_iter": 1000},
        random_seed=42,
    )
    with pytest.raises(ValueError, match="Test is sealed"):
        adapter.fit(
            np.ones((4, 2)),
            np.array([0, 1, 0, 1]),
            x_validation=np.ones((2, 2)),
            y_validation=np.array([0, 1]),
            validation_split_name="test",
            feature_names=["a", "b"],
        )
