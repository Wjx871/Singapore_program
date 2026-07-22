from __future__ import annotations

import pytest

from scripts.run_lr_ablations import result_to_summary
from src.experiments.contracts import ExperimentSpec
from src.experiments.runner import SharedExperimentRunner


@pytest.fixture(scope="module")
def formal_runner():
    return SharedExperimentRunner("configs/experiment.yaml")


@pytest.mark.parametrize(
    ("experiment_id", "feature_set", "strategy", "class_weight", "feature_count"),
    [
        ("matrix_a", "A", "missing_plus_flag", "balanced", 14),
        ("matrix_b", "B", "missing_plus_flag", "balanced", 17),
        ("matrix_none", "B", "missing_plus_flag", "none", 17),
        ("matrix_keep", "B", "keep_raw", "balanced", 16),
    ],
)
def test_formal_ablation_variants_run(
    formal_runner, experiment_id, feature_set, strategy, class_weight, feature_count
):
    config = formal_runner.config
    spec = ExperimentSpec(
        experiment_id=experiment_id,
        model_name="logistic_regression",
        feature_set=feature_set,
        preprocessing_strategy=strategy,
        class_weight=class_weight,
        random_seed=42,
        config_path=config.config_path,
        manifest_path=formal_runner.manifest_path,
        expected_manifest_sha256=formal_runner.expected_manifest_sha256,
    )
    artifacts = formal_runner.run(spec)
    summary = result_to_summary(artifacts.result)
    assert summary["feature_count"] == feature_count
    assert summary["manifest_sha256"] == formal_runner.expected_manifest_sha256
    assert summary["operational_recall"] >= 0.75
    assert set(artifacts.validation_metrics) == {
        "split",
        "positive_label",
        "threshold_independent",
        "default_threshold",
        "operational_threshold",
    }
