from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.contracts import ExperimentSpec
from src.experiments.runner import SharedExperimentRunner


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
        ({"model_name": "random_forest"}, "only logistic"),
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


def test_formal_stage2_baseline_runs_and_is_json_safe(experiment_config, tmp_path):
    runner = SharedExperimentRunner(experiment_config.config_path)
    artifacts = runner.run(make_spec(experiment_config, experiment_id="stage2_reproduction"))
    payload = artifacts.result.to_dict()
    assert payload["manifest_sha256"] == experiment_config.raw["split"]["frozen_manifest_sha256"]
    assert payload["feature_count"] == 14
    assert payload["validation_metrics"]["pr_auc"] == pytest.approx(0.3580201248072878)
    assert payload["operational_threshold"] == pytest.approx(0.42065105523373764)
    assert "test_probability" not in json.dumps(payload).lower()
    assert payload["test_access"].startswith("transform/schema")
