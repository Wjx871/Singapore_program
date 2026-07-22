from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.eda import (
    EdaDataset,
    TRAIN_SCOPE,
    assert_validation_passed,
    build_abnormal_code_tables,
    build_class_distribution,
    build_duplicate_summary,
    build_missing_values,
    build_validation_checks,
    plot_class_distribution,
    plot_model_sensitivity,
    validate_scope,
)
from src.data.feature_hash import compute_feature_hashes


@pytest.fixture
def eda_dataset(experiment_config):
    rows = [
        [1, 0, 0.1, 45, 96, 0.3, 5000, 8, 98, 1, 96, 2],
        [2, 1, 0.1, 45, 96, 0.3, 5000, 8, 98, 1, 96, 2],
        [3, 1, 0.2, 50, 98, 0.4, None, 5, 98, 0, 98, None],
        [4, 0, 0.0, 0, 0, 0.1, 3000, 4, 0, 0, 0, 0],
        [5, 0, 0.5, 60, 1, 1.2, 7000, 10, 0, 2, 1, 1],
    ]
    columns = [
        experiment_config.data["id_column"],
        experiment_config.data["target_column"],
        *experiment_config.predictor_columns,
    ]
    frame = pd.DataFrame(rows, columns=columns)
    hashes = compute_feature_hashes(frame, experiment_config.predictor_columns)
    return EdaDataset(
        frame=frame,
        feature_hash_v1=hashes,
        scope=TRAIN_SCOPE,
        metadata={
            "manifest_sha256": experiment_config.raw["split"]["frozen_manifest_sha256"],
            "label_usage": "frozen Training partition only",
        },
    )


def test_scope_guard_has_no_validation_or_test_option():
    assert validate_scope("raw_audit") == "raw_audit"
    assert validate_scope("train") == "train"
    with pytest.raises(ValueError, match="scope must be one of"):
        validate_scope("validation")
    with pytest.raises(ValueError, match="scope must be one of"):
        validate_scope("test")


def test_core_eda_tables_are_consistent(eda_dataset, experiment_config):
    classes = build_class_distribution(eda_dataset, experiment_config)
    missing = build_missing_values(eda_dataset, experiment_config).set_index("feature")
    abnormal_counts, abnormal_summary = build_abnormal_code_tables(
        eda_dataset, experiment_config
    )
    duplicates = build_duplicate_summary(eda_dataset, experiment_config).set_index("metric")

    assert classes["count"].sum() == 5
    assert classes.loc[classes["target"] == 1, "count"].item() == 2
    assert missing.loc["MonthlyIncome", "missing_count"] == 1
    assert missing.loc["NumberOfDependents", "missing_count"] == 1
    assert abnormal_counts["affected_rows"].sum() == 9
    union = abnormal_summary.iloc[-1]
    assert union["unique_affected_rows"] == 3
    assert union["target_1"] == 2
    assert duplicates.loc["duplicate_predictor_excess_rows", "value"] == 1
    assert duplicates.loc["conflicting_target_groups", "value"] == 1
    assert duplicates.loc["rows_in_conflicting_target_groups", "value"] == 2


def test_validation_contract_passes_for_training_fixture(eda_dataset, experiment_config):
    checks = build_validation_checks(
        eda_dataset,
        experiment_config,
        raw_sha256_after=experiment_config.data["raw_labeled_sha256"],
    )
    assert set(checks["status"]) == {"PASS"}
    assert_validation_passed(checks)


def test_class_plot_is_written(eda_dataset, experiment_config, tmp_path):
    table = build_class_distribution(eda_dataset, experiment_config)
    path = tmp_path / "class_distribution.png"
    plot_class_distribution(table, path, scope=TRAIN_SCOPE)
    assert path.is_file()
    assert path.stat().st_size > 1_000


def test_shared_validation_sensitivity_plot_is_written(tmp_path):
    table = pd.DataFrame(
        {
            "preprocessing_strategy": ["missing_plus_flag", "keep_raw"],
            "pr_auc": [0.36, 0.30],
            "roc_auc": [0.83, 0.79],
            "ks": [0.51, 0.43],
            "operational_precision": [0.18, 0.14],
            "operational_recall": [0.75, 0.75],
        }
    )
    path = tmp_path / "abnormal_code_sensitivity_validation.png"
    plot_model_sensitivity(table, path)
    assert path.is_file()
    assert path.stat().st_size > 1_000
