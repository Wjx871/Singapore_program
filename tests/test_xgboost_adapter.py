"""XGBoost adapter contract tests (synthetic data only).

Tests cover:
  A. Adapter base contract
  B. scale_pos_weight
  C. Early Stopping
  D. Parameter contract
  E. Reproducibility
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.models.adapters.xgboost_adapter import XGBoostAdapter
from src.models.adapters.xgboost_contract import compute_training_scale_pos_weight
from src.models.contracts import validate_model_parameters
from src.models.factory import ModelNotImplementedError, create_model_adapter
from src.models.registry import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# synthetic fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_data():
    """Small synthetic binary dataset — sufficient for contract tests."""
    rng = np.random.default_rng(42)
    n = 200
    features = rng.normal(size=(n, 5))
    # Create imbalanced binary target
    logit = 0.5 * features[:, 0] + 0.3 * features[:, 1] - 1.5
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.random(n) < prob).astype("int8")
    features_val = rng.normal(size=(60, 5))
    logit_val = 0.5 * features_val[:, 0] + 0.3 * features_val[:, 1] - 1.5
    prob_val = 1.0 / (1.0 + np.exp(-logit_val))
    target_val = (rng.random(60) < prob_val).astype("int8")
    return features, target, features_val, target_val


def _fit_xgb(features, target, features_val, target_val, **overrides):
    """Quick-fit helper with small n_estimators for testing."""
    params = {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1,
        "early_stopping_rounds": 10,
        "n_jobs": 1,
    }
    params.update(overrides)
    adapter = XGBoostAdapter(params, random_seed=42)
    adapter.fit(
        features,
        target,
        x_validation=features_val,
        y_validation=target_val,
        validation_split_name="validation",
        feature_names=[f"x{i}" for i in range(features.shape[1])],
    )
    return adapter


# ===================================================================
# A. Adapter base contract
# ===================================================================
class TestAdapterBaseContract:
    def test_registry_status_implemented(self):
        assert MODEL_REGISTRY["xgboost"].status == "implemented"

    def test_factory_returns_xgboost_adapter(self):
        adapter = create_model_adapter(
            "xgboost",
            {"n_estimators": 20, "n_jobs": 1},
            random_seed=42,
        )
        assert isinstance(adapter, XGBoostAdapter)

    def test_factory_no_longer_rejects_xgboost(self):
        """XGBoost is now implemented — factory should succeed."""
        adapter = create_model_adapter(
            "xgboost",
            {"n_estimators": 20, "n_jobs": 1},
            random_seed=42,
        )
        assert adapter.model_status == "implemented"

    def test_predict_proba_shape(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        proba = adapter.predict_proba(features_val)
        assert proba.shape == (len(features_val),)

    def test_output_length_correct(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        proba = adapter.predict_proba(features_val)
        assert len(proba) == len(features_val)

    def test_probabilities_finite(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        proba = adapter.predict_proba(features_val)
        assert np.isfinite(proba).all()

    def test_probabilities_in_01(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        proba = adapter.predict_proba(features_val)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_predict_proba_before_fit_fails(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = XGBoostAdapter({"n_estimators": 10, "n_jobs": 1}, random_seed=42)
        with pytest.raises(RuntimeError, match="must be fitted"):
            adapter.predict_proba(features_val)

    def test_parameters_json_safe(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        params = adapter.get_parameters()
        json.dumps(params, allow_nan=False)

    def test_training_metadata_json_safe(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        json.dumps(meta, allow_nan=False)


# ===================================================================
# B. scale_pos_weight
# ===================================================================
class TestScalePosWeight:
    def test_computed_from_y_train_only(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        neg = int((target == 0).sum())
        pos = int((target == 1).sum())
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        expected = neg / pos
        assert meta["scale_pos_weight"] == pytest.approx(expected)

    def test_validation_labels_do_not_affect_scale_pos_weight(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter1 = _fit_xgb(features, target, features_val, target_val)
        sw1 = adapter1.get_training_metadata()["scale_pos_weight"]

        # Change validation labels — scale_pos_weight must stay the same
        target_val_swapped = 1 - target_val
        adapter2 = _fit_xgb(features, target, features_val, target_val_swapped)
        sw2 = adapter2.get_training_metadata()["scale_pos_weight"]

        assert sw1 == sw2

    def test_user_scale_pos_weight_rejected(self):
        with pytest.raises(ValueError, match="computed from Training"):
            validate_model_parameters("xgboost", {"scale_pos_weight": 13.0})

    def test_single_class_training_rejected(self):
        with pytest.raises(ValueError, match="both classes"):
            compute_training_scale_pos_weight([0, 0, 0, 0])

    def test_only_zero_and_one_allowed(self):
        with pytest.raises(ValueError, match="binary"):
            compute_training_scale_pos_weight([0, 1, 2])

    def test_result_is_finite_and_positive(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        sw = adapter.get_training_metadata()["scale_pos_weight"]
        assert np.isfinite(sw)
        assert sw > 0

    def test_metadata_records_counts_and_formula(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        assert "negative_train_count" in meta
        assert "positive_train_count" in meta
        assert meta["scale_pos_weight_formula"] == "negative_train_count / positive_train_count"
        assert meta["negative_train_count"] > 0
        assert meta["positive_train_count"] > 0


# ===================================================================
# C. Early Stopping
# ===================================================================
class TestEarlyStopping:
    def test_fails_without_validation(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = XGBoostAdapter({"n_estimators": 50, "n_jobs": 1}, random_seed=42)
        with pytest.raises(ValueError, match="requires Validation"):
            adapter.fit(features, target, feature_names=[f"x{i}" for i in range(5)])

    def test_validation_split_name_validation_ok(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        assert adapter.estimator is not None

    def test_validation_split_name_test_fails(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = XGBoostAdapter({"n_estimators": 10, "n_jobs": 1}, random_seed=42)
        with pytest.raises(ValueError, match="Test is sealed"):
            adapter.fit(
                features,
                target,
                x_validation=features_val,
                y_validation=target_val,
                validation_split_name="test",
            )

    def test_best_iteration_exists(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        assert "best_iteration" in meta
        assert meta["best_iteration"] is not None

    def test_best_iteration_less_than_n_estimators(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        assert meta["best_iteration"] < 50  # n_estimators used in _fit_xgb

    def test_actual_boosting_rounds_equals_best_iteration_plus_one(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        assert meta["actual_boosting_rounds"] == meta["best_iteration"] + 1

    def test_best_iteration_is_non_negative(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        assert isinstance(meta["best_iteration"], int)
        assert meta["best_iteration"] >= 0

    def test_early_stopping_rounds_recorded(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        assert meta["early_stopping_rounds"] == 10

    def test_best_score_recorded(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        assert meta["best_score"] is not None
        assert isinstance(meta["best_score"], float)


# ===================================================================
# D. Parameter contract
# ===================================================================
class TestParameterContract:
    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"subsample": 0}, r"subsample.*\(0, 1\]"),
            ({"subsample": 1.5}, r"subsample.*\(0, 1\]"),
            ({"subsample": -0.1}, r"subsample.*\(0, 1\]"),
            ({"colsample_bytree": 0}, r"colsample_bytree.*\(0, 1\]"),
            ({"colsample_bytree": 1.5}, r"colsample_bytree.*\(0, 1\]"),
            ({"n_estimators": 0}, "positive integer"),
            ({"n_estimators": -1}, "positive integer"),
            ({"max_depth": 0}, "positive integer"),
            ({"max_depth": -1}, "positive integer"),
            ({"early_stopping_rounds": 0}, "positive integer"),
            ({"early_stopping_rounds": -5}, "positive integer"),
            ({"learning_rate": 0}, "positive"),
            ({"learning_rate": -0.1}, "positive"),
            ({"reg_lambda": -1.0}, r"reg_lambda.*>= 0"),
            ({"unknown_param": 1}, "Unknown xgboost"),
            ({"n_estimators": True}, "positive integer"),
            ({"n_estimators": float("nan")}, "positive integer"),
            ({"subsample": True}, r"subsample.*\(0, 1\]"),
            ({"n_jobs": 0}, "non-zero integer"),
        ],
    )
    def test_invalid_parameters_rejected(self, params, message):
        with pytest.raises(ValueError, match=message):
            validate_model_parameters("xgboost", params)

    def test_reg_lambda_zero_is_accepted(self):
        # reg_lambda >= 0, so 0.0 is valid
        result = validate_model_parameters("xgboost", {"reg_lambda": 0.0})
        assert result["reg_lambda"] == 0.0

    def test_subsample_one_accepted(self):
        result = validate_model_parameters("xgboost", {"subsample": 1.0})
        assert result["subsample"] == 1.0

    def test_colsample_bytree_one_accepted(self):
        result = validate_model_parameters("xgboost", {"colsample_bytree": 1.0})
        assert result["colsample_bytree"] == 1.0

    def test_n_jobs_negative_one_accepted(self):
        result = validate_model_parameters("xgboost", {"n_jobs": -1})
        assert result["n_jobs"] == -1

    def test_n_jobs_positive_accepted(self):
        result = validate_model_parameters("xgboost", {"n_jobs": 2})
        assert result["n_jobs"] == 2

    def test_class_weight_none_accepted_in_spec(self):
        """XGBoost + class_weight=none should be accepted."""
        validate_model_parameters("xgboost", {})  # class_weight not in allowlist

    def test_boolean_disguised_as_int_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            validate_model_parameters("xgboost", {"n_estimators": True})

    def test_string_numeric_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_model_parameters("xgboost", {"learning_rate": "0.05"})


# ===================================================================
# E. Reproducibility & metadata
# ===================================================================
class TestReproducibility:
    def test_deterministic_with_seed(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1,
                  "early_stopping_rounds": 10, "n_jobs": 1}

        a1 = XGBoostAdapter(params, random_seed=42)
        a1.fit(features, target, x_validation=features_val, y_validation=target_val,
               validation_split_name="validation",
               feature_names=[f"x{i}" for i in range(5)])
        p1 = a1.predict_proba(features_val)

        a2 = XGBoostAdapter(params, random_seed=42)
        a2.fit(features, target, x_validation=features_val, y_validation=target_val,
               validation_split_name="validation",
               feature_names=[f"x{i}" for i in range(5)])
        p2 = a2.predict_proba(features_val)

        np.testing.assert_array_almost_equal(p1, p2)

    def test_feature_importance_matches_feature_names(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        fi = meta["feature_importance"]
        assert set(fi.keys()) == {"x0", "x1", "x2", "x3", "x4"}

    def test_feature_importance_finite_and_non_negative(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        values = list(meta["feature_importance"].values())
        assert all(np.isfinite(v) for v in values)
        assert all(v >= 0 for v in values)

    def test_feature_importance_type_is_gain(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        assert meta["feature_importance_type"] == "gain"

    def test_metadata_effective_random_state(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        meta = adapter.get_training_metadata()
        assert meta["effective_random_state"] == 42

    def test_groups_recorded(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = XGBoostAdapter({"n_estimators": 20, "n_jobs": 1}, random_seed=42)
        adapter.fit(
            features, target,
            x_validation=features_val, y_validation=target_val,
            validation_split_name="validation",
            groups=np.arange(len(features)),
        )
        assert adapter.get_training_metadata()["groups_supplied"] is True

    def test_groups_not_supplied_recorded(self, synthetic_data):
        features, target, features_val, target_val = synthetic_data
        adapter = _fit_xgb(features, target, features_val, target_val)
        assert adapter.get_training_metadata()["groups_supplied"] is False
