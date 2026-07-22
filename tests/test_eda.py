from __future__ import annotations

import json
import re
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analysis.eda import (
    EdaDataset,
    EXPECTED_TRAIN_ROWS,
    RAW_AUDIT_SCOPE,
    SENSITIVITY_FINITE_COLUMNS,
    TRAIN_SCOPE,
    assert_validation_passed,
    build_abnormal_code_tables,
    build_class_distribution,
    build_duplicate_summary,
    build_missing_values,
    build_validation_checks,
    load_model_sensitivity_reference,
    plot_class_distribution,
    plot_model_sensitivity,
    run_eda,
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


def valid_sensitivity_frame(experiment_config):
    metric_values = {column: 0.5 for column in SENSITIVITY_FINITE_COLUMNS}
    rows = []
    for strategy, experiment_id, feature_count in (
        ("missing_plus_flag", "lr_b_balanced_missing_flag", 17),
        ("keep_raw", "lr_b_balanced_keep_raw", 16),
    ):
        rows.append(
            {
                "experiment_id": experiment_id,
                "model_name": "logistic_regression",
                "feature_set": "B",
                "preprocessing_strategy": strategy,
                "class_weight": "balanced",
                "feature_count": feature_count,
                "manifest_sha256": experiment_config.raw["split"][
                    "frozen_manifest_sha256"
                ],
                "config_sha256": experiment_config.config_sha256,
                **metric_values,
            }
        )
    return pd.DataFrame(rows)


def write_sensitivity(frame, project_root):
    path = project_root / "outputs" / "comparisons" / "abnormal_code_sensitivity.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def all_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_strings(child)
    elif isinstance(value, str):
        yield value


def is_absolute_path_string(value):
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("/")


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


def test_raw_abnormal_tables_have_no_target_fields(eda_dataset, experiment_config):
    raw_dataset = replace(eda_dataset, scope=RAW_AUDIT_SCOPE)
    counts, summary = build_abnormal_code_tables(raw_dataset, experiment_config)
    prohibited = {"target_0", "target_1", "positive_rate"}
    assert prohibited.isdisjoint(counts.columns)
    assert prohibited.isdisjoint(summary.columns)


def test_train_abnormal_tables_include_target_fields(eda_dataset, experiment_config):
    counts, summary = build_abnormal_code_tables(eda_dataset, experiment_config)
    expected = {"target_0", "target_1", "positive_rate"}
    assert expected.issubset(counts.columns)
    assert expected.issubset(summary.columns)


def test_validation_contract_requires_exact_training_row_count(
    eda_dataset, experiment_config
):
    checks = build_validation_checks(
        eda_dataset,
        experiment_config,
        raw_sha256_after=experiment_config.data["raw_labeled_sha256"],
    )
    row_check = checks.set_index("check_id").loc["row_count"]
    assert row_check["status"] == "FAIL"
    assert int(row_check["expected"]) == EXPECTED_TRAIN_ROWS == 95_995
    assert int(row_check["observed"]) == len(eda_dataset.frame)
    with pytest.raises(ValueError, match="row_count"):
        assert_validation_passed(checks)


def test_sensitivity_reference_accepts_exact_shared_contract(
    experiment_config, tmp_path
):
    config = replace(experiment_config, project_root=tmp_path)
    write_sensitivity(valid_sensitivity_frame(config), tmp_path)
    result = load_model_sensitivity_reference(config)
    assert set(result["preprocessing_strategy"]) == {
        "missing_plus_flag",
        "keep_raw",
    }
    assert set(result["source_file"]) == {
        "outputs/comparisons/abnormal_code_sensitivity.csv"
    }


@pytest.mark.parametrize(
    ("column", "row_index", "invalid_value", "message"),
    [
        ("manifest_sha256", 0, "wrong-manifest", "manifest_sha256"),
        ("config_sha256", 0, "wrong-config", "config_sha256"),
        ("model_name", 0, "xgboost", "model_name"),
        ("feature_set", 0, "A", "feature_set"),
        ("class_weight", 0, "none", "class_weight"),
        ("preprocessing_strategy", 1, "missing_plus_flag", "preprocessing_strategy"),
        ("experiment_id", 0, "wrong_experiment", "experiment_id"),
        ("feature_count", 0, 16, "feature_count"),
        ("pr_auc", 0, np.inf, "finite"),
    ],
)
def test_sensitivity_reference_rejects_contract_violations(
    experiment_config,
    tmp_path,
    column,
    row_index,
    invalid_value,
    message,
):
    config = replace(experiment_config, project_root=tmp_path)
    frame = valid_sensitivity_frame(config)
    frame.loc[row_index, column] = invalid_value
    write_sensitivity(frame, tmp_path)
    with pytest.raises(ValueError, match=message):
        load_model_sensitivity_reference(config)


def test_sensitivity_reference_is_required_when_requested(
    experiment_config, tmp_path
):
    config = replace(experiment_config, project_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="run_lr_ablations"):
        load_model_sensitivity_reference(config)


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


def test_run_eda_integration_smoke_and_metadata_has_no_absolute_paths(
    experiment_config, tmp_path
):
    scope_dir = run_eda(
        experiment_config.config_path,
        scope=TRAIN_SCOPE,
        output_root=tmp_path,
        include_strategy_sensitivity=False,
    )
    assert (scope_dir / "tables" / "dataset_summary.csv").is_file()
    assert (scope_dir / "figures" / "class_distribution.png").is_file()
    metadata = json.loads(
        (scope_dir / "metadata" / "eda_run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["analysis_row_count"] == EXPECTED_TRAIN_ROWS
    assert metadata["source"]["path"] == experiment_config.data["labeled_path"]
    assert metadata["split_metadata"]["source_path"] == experiment_config.data[
        "labeled_path"
    ]
    absolute_values = [
        value for value in all_strings(metadata) if is_absolute_path_string(value)
    ]
    assert absolute_values == []


def test_cli_reports_repository_relative_output_directories(
    experiment_config, monkeypatch, capsys
):
    import scripts.run_eda as cli

    output = experiment_config.project_root / "outputs" / "eda" / "train"
    monkeypatch.setattr(cli, "load_config", lambda _: experiment_config)
    monkeypatch.setattr(cli, "run_scopes", lambda *args, **kwargs: [output])
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_eda", "--scope", "train", "--skip-strategy-sensitivity"],
    )
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_directories"] == ["outputs/eda/train"]
