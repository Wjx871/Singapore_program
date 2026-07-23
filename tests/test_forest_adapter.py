from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.models.adapters.balanced_forest_adapter import BalancedRandomForestAdapter
from src.models.adapters.forest_adapter import RandomForestAdapter
from src.models.contracts import validate_model_parameters
from src.models.factory import create_model_adapter


@pytest.fixture
def synthetic_binary_data():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(240, 5))
    target = (features[:, 0] + 0.4 * features[:, 1] > 1.0).astype("int8")
    return features, target


@pytest.mark.parametrize(
    ("model_name", "adapter_type"),
    [
        ("random_forest", RandomForestAdapter),
        ("balanced_random_forest", BalancedRandomForestAdapter),
    ],
)
def test_factory_and_probability_contract(model_name, adapter_type, synthetic_binary_data):
    features, target = synthetic_binary_data
    adapter = create_model_adapter(
        model_name,
        {"n_estimators": 20, "n_jobs": 1},
        random_seed=42,
    )
    assert isinstance(adapter, adapter_type)
    adapter.fit(features, target, feature_names=[f"x{i}" for i in range(5)])
    probability = adapter.predict_proba(features)
    assert probability.shape == (len(features),)
    assert np.isfinite(probability).all()
    assert ((probability >= 0.0) & (probability <= 1.0)).all()
    json.dumps(adapter.get_parameters(), allow_nan=False)
    json.dumps(adapter.get_training_metadata(), allow_nan=False)
    assert set(adapter.get_training_metadata()["feature_importance"]) == {
        "x0", "x1", "x2", "x3", "x4"
    }


@pytest.mark.parametrize("model_name", ["random_forest", "balanced_random_forest"])
def test_seed_is_deterministic(model_name, synthetic_binary_data):
    features, target = synthetic_binary_data
    parameters = {"n_estimators": 20, "n_jobs": 1}
    first = create_model_adapter(model_name, parameters, random_seed=42).fit(features, target)
    second = create_model_adapter(model_name, parameters, random_seed=42).fit(features, target)
    np.testing.assert_array_equal(first.predict_proba(features), second.predict_proba(features))


@pytest.mark.parametrize("model_name", ["random_forest", "balanced_random_forest"])
def test_predict_proba_is_serial_and_restores_training_n_jobs(
    model_name, synthetic_binary_data, monkeypatch
):
    features, target = synthetic_binary_data
    adapter = create_model_adapter(
        model_name, {"n_estimators": 20, "n_jobs": -1}, random_seed=42
    ).fit(features, target)
    observed_n_jobs = []
    native_predict = adapter.estimator.predict_proba

    def observing_predict(values):
        observed_n_jobs.append(adapter.estimator.n_jobs)
        return native_predict(values)

    monkeypatch.setattr(adapter.estimator, "predict_proba", observing_predict)
    adapter.predict_proba(features)
    assert observed_n_jobs == [1]
    assert adapter.estimator.n_jobs == -1


@pytest.mark.parametrize("model_name", ["random_forest", "balanced_random_forest"])
def test_predict_proba_restores_n_jobs_after_native_error(
    model_name, synthetic_binary_data, monkeypatch
):
    features, target = synthetic_binary_data
    adapter = create_model_adapter(
        model_name, {"n_estimators": 10, "n_jobs": -1}, random_seed=42
    ).fit(features, target)

    def failing_predict(_values):
        assert adapter.estimator.n_jobs == 1
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(adapter.estimator, "predict_proba", failing_predict)
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        adapter.predict_proba(features)
    assert adapter.estimator.n_jobs == -1


@pytest.mark.parametrize("model_name", ["random_forest", "balanced_random_forest"])
def test_serial_probability_hash_and_model_state_are_stable(
    model_name, synthetic_binary_data
):
    features, target = synthetic_binary_data
    adapter = create_model_adapter(
        model_name, {"n_estimators": 20, "n_jobs": -1}, random_seed=42
    ).fit(features, target)
    parameters_before = adapter.estimator.get_params(deep=True)
    importance_before = adapter.estimator.feature_importances_.copy()
    tree_state_before = _tree_state_sha256(adapter.estimator)

    first = adapter.predict_proba(features)
    second = adapter.predict_proba(features)

    np.testing.assert_array_equal(first, second)
    assert _array_sha256(first) == _array_sha256(second)
    assert first.shape == (len(features),)
    assert np.isfinite(first).all()
    assert ((first >= 0.0) & (first <= 1.0)).all()
    assert adapter.estimator.get_params(deep=True) == parameters_before
    np.testing.assert_array_equal(
        adapter.estimator.feature_importances_, importance_before
    )
    assert _tree_state_sha256(adapter.estimator) == tree_state_before


@pytest.mark.parametrize("model_name", ["random_forest", "balanced_random_forest"])
def test_metadata_records_training_and_inference_parallelism(
    model_name, synthetic_binary_data
):
    features, target = synthetic_binary_data
    adapter = create_model_adapter(
        model_name, {"n_estimators": 10, "n_jobs": -1}, random_seed=42
    ).fit(features, target)
    metadata = adapter.get_training_metadata()
    assert metadata["training_n_jobs"] == -1
    assert metadata["inference_n_jobs"] == 1
    assert metadata["deterministic_serial_inference"] is True


@pytest.mark.parametrize("model_name", ["random_forest", "balanced_random_forest"])
def test_adapter_rejects_test_as_validation(model_name, synthetic_binary_data):
    features, target = synthetic_binary_data
    adapter = create_model_adapter(model_name, {"n_estimators": 5}, random_seed=42)
    with pytest.raises(ValueError, match="Test is sealed"):
        adapter.fit(
            features,
            target,
            x_validation=features[:10],
            y_validation=target[:10],
            validation_split_name="test",
        )


@pytest.mark.parametrize(
    ("model_name", "parameters", "message"),
    [
        ("random_forest", {"n_estimators": 0}, "positive integer"),
        ("random_forest", {"max_features": "invalid"}, "max_features"),
        ("random_forest", {"max_features": 1.01}, r"\(0, 1\]"),
        ("balanced_random_forest", {"max_features": 2.0}, r"\(0, 1\]"),
        ("balanced_random_forest", {"sampling_strategy": "auto"}, "must be 'all'"),
        ("balanced_random_forest", {"replacement": "yes"}, "must be boolean"),
        ("balanced_random_forest", {"class_weight": "balanced"}, "Unknown balanced_random_forest"),
    ],
)
def test_invalid_parameters_are_rejected(model_name, parameters, message):
    with pytest.raises(ValueError, match=message):
        validate_model_parameters(model_name, parameters)


def test_fair_default_contracts():
    rf = RandomForestAdapter({}, 42).get_parameters()
    brf = BalancedRandomForestAdapter({}, 42).get_parameters()
    for key in ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "n_jobs"):
        assert rf[key] == brf[key]
    assert rf["class_weight"] is None
    assert brf["sampling_strategy"] == "all"
    assert brf["replacement"] is True
    assert brf["bootstrap"] is False


def _array_sha256(values):
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _tree_state_sha256(estimator):
    digest = hashlib.sha256()
    for tree in estimator.estimators_:
        for values in (
            tree.tree_.value,
            tree.tree_.threshold,
            tree.tree_.children_left,
            tree.tree_.children_right,
        ):
            array = np.ascontiguousarray(values)
            digest.update(array.view(np.uint8))
    return digest.hexdigest()
