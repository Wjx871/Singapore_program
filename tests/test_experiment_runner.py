from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments.contracts import ExperimentSpec
from src.experiments.runner import SharedExperimentRunner
from src.models.factory import ModelNotImplementedError


def make_spec(config, **overrides):
    values = {
        "experiment_id": "synthetic_contract",
        "model_name": "logistic_regression",
        "feature_set": "A",
        "preprocessing_strategy": "missing_plus_flag",
        "class_weight": "balanced",
        "random_seed": 42,
        "config_path": config.config_path,
        "manifest_path": config.project_root / config.raw["split"]["manifest_path"],
        "expected_manifest_sha256": config.raw["split"]["frozen_manifest_sha256"],
    }
    values.update(overrides)
    return ExperimentSpec(**values)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model_name": "lightgbm"}, "optional/legacy"),
        ({"model_name": "made_up_model"}, "Unknown model_name"),
        ({"feature_set": "C3"}, "A or B"),
        ({"preprocessing_strategy": "zero_fill"}, "missing_plus_flag"),
        ({"class_weight": "auto"}, "balanced or none"),
        ({"evaluation_split": "test"}, "Validation evaluation only"),
    ],
)
def test_spec_rejects_unsupported_scope(experiment_config, override, message):
    with pytest.raises(ValueError, match=message):
        make_spec(experiment_config, **override)


def test_manifest_sha_guard_rejects_mismatch(experiment_config):
    runner = SharedExperimentRunner(experiment_config.config_path)
    spec = make_spec(experiment_config, expected_manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="differs from frozen"):
        runner.run(spec)


@pytest.mark.parametrize("model_name", ["xgboost"])
def test_contract_ready_models_fail_explicitly_at_factory(experiment_config, model_name):
    runner = SharedExperimentRunner(experiment_config.config_path)
    with pytest.raises(ModelNotImplementedError, match="contract_ready"):
        runner.run(make_spec(experiment_config, model_name=model_name))


def test_spec_accepts_controlled_model_parameter_overrides(experiment_config):
    spec = make_spec(experiment_config, model_parameters={"C": 0.3, "max_iter": 2000})
    assert spec.model_parameters == {"C": 0.3, "max_iter": 2000}
    with pytest.raises(ValueError, match="Unknown logistic_regression"):
        make_spec(experiment_config, model_parameters={"uncontrolled": True})


@pytest.mark.parametrize(
    ("class_weight", "estimator_class_weight"),
    [("none", None), ("balanced", "balanced")],
)
def test_random_forest_class_weight_reaches_estimator_and_result_metadata(
    experiment_config, class_weight, estimator_class_weight
):
    runner = SharedExperimentRunner(experiment_config.config_path)
    artifacts = runner.run(
        make_spec(
            experiment_config,
            experiment_id=f"rf_class_weight_{class_weight}",
            model_name="random_forest",
            class_weight=class_weight,
            model_parameters={"n_estimators": 5, "n_jobs": 1},
        )
    )
    assert artifacts.result.class_weight == class_weight
    assert artifacts.result.model_parameters["class_weight"] == estimator_class_weight


def test_balanced_random_forest_rejects_external_balanced_class_weight(experiment_config):
    runner = SharedExperimentRunner(experiment_config.config_path)
    with pytest.raises(ValueError, match="internal balanced sampling only"):
        runner.run(
            make_spec(
                experiment_config,
                model_name="balanced_random_forest",
                class_weight="balanced",
                model_parameters={"n_estimators": 5, "n_jobs": 1},
            )
        )


def test_balanced_random_forest_none_metadata_matches_internal_sampling(experiment_config):
    runner = SharedExperimentRunner(experiment_config.config_path)
    artifacts = runner.run(
        make_spec(
            experiment_config,
            experiment_id="brf_internal_sampling",
            model_name="balanced_random_forest",
            class_weight="none",
            model_parameters={"n_estimators": 5, "n_jobs": 1},
        )
    )
    assert artifacts.result.class_weight == "none"
    assert "class_weight" not in artifacts.result.model_parameters
    assert artifacts.result.imbalance_strategy == "internal_balanced_sampling_no_smote"
    assert artifacts.result.model_training_metadata["effective_sampling_strategy"] == "all"


def test_spec_defaults_to_shared_feature_and_preprocessing_contract(experiment_config):
    values = make_spec(experiment_config).__dict__.copy()
    values.pop("feature_set")
    values.pop("preprocessing_strategy")
    spec = ExperimentSpec(**values)
    assert spec.feature_set == "B"
    assert spec.preprocessing_strategy == "missing_plus_flag"


def test_formal_stage2_baseline_runs_and_is_json_safe(experiment_config, tmp_path):
    runner = SharedExperimentRunner(experiment_config.config_path)
    artifacts = runner.run(make_spec(experiment_config, experiment_id="stage2_reproduction"))
    payload = artifacts.result.to_dict()
    assert payload["manifest_sha256"] == experiment_config.raw["split"]["frozen_manifest_sha256"]
    assert payload["feature_count"] == 14
    assert payload["model_family"] == "linear_model"
    assert payload["model_status"] == "implemented"
    assert payload["model_parameters"]["class_weight"] == "balanced"
    assert payload["model_training_metadata"]["best_iteration"] is None
    assert payload["requires_scaled_features"] is True
    assert payload["supports_early_stopping"] is False
    assert payload["scale_pos_weight"] is None
    assert payload["validation_metrics"]["pr_auc"] == pytest.approx(0.3580201248072878)
    assert payload["operational_threshold"] == pytest.approx(0.42065105523373764)
    assert "test_probability" not in json.dumps(payload).lower()
    assert payload["test_access"].startswith("transform/schema")
    paths = runner.persist(artifacts, output_root=tmp_path)
    assert set(paths) == {"validation_metrics", "thresholds", "run_metadata"}
    assert not list(tmp_path.rglob("*probab*"))


def test_test_partition_changes_do_not_affect_validation(experiment_config):
    runner = SharedExperimentRunner(experiment_config.config_path)
    spec = make_spec(experiment_config, experiment_id="test_isolation_regression")
    first_artifacts = runner.run(spec)
    runner.partitions["test"].loc[:, "MonthlyIncome"] = 1e12
    second_artifacts = runner.run(spec)
    first = first_artifacts.result
    second = second_artifacts.result
    np.testing.assert_array_equal(
        first_artifacts.validation_probability, second_artifacts.validation_probability
    )
    assert first.validation_metrics == second.validation_metrics
    assert first.default_threshold_metrics == second.default_threshold_metrics
    assert first.operational_threshold == second.operational_threshold
    assert first.operational_threshold_metrics == second.operational_threshold_metrics
    assert first.preprocessing_metadata["medians"] == second.preprocessing_metadata["medians"]
