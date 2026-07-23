from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.evaluation.metrics import threshold_independent_metrics
from src.evaluation.model_comparison import (
    COMPARISON_COLUMNS,
    EXPECTED_VALIDATION_ROWS,
    ComparisonRun,
    build_formal_specs,
    generate_figures,
    REFERENCE_METRICS,
    select_recommended_model,
    validate_comparison_table,
    validate_probability_alignment,
    validate_reference_metrics,
    validate_specs,
    write_outputs,
)
from src.experiments.result_schema import ExperimentResult
from src.experiments.runner import ExperimentArtifacts


MANIFEST_SHA = "5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5"
RAW_SHA = "1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac"


def _comparison_table() -> pd.DataFrame:
    names = [
        ("Logistic Regression", "lr_b_balanced_missing_flag", 0.35, 0.82, 0.51, 0.17, 2.0),
        ("Random Forest", "rf_b_none_missing_flag", 0.39, 0.86, 0.57, 0.22, 4.0),
        (
            "Balanced Random Forest",
            "brf_b_none_missing_flag",
            0.38,
            0.87,
            0.59,
            0.23,
            5.0,
        ),
        ("XGBoost", "xgb_child5", 0.40, 0.88, 0.58, 0.24, 6.0),
    ]
    rows = []
    for model_name, experiment_id, pr_auc, roc_auc, ks, precision, runtime in names:
        row = {
            "model_name": model_name,
            "experiment_id": experiment_id,
            "model_family": "synthetic",
            "imbalance_strategy": "synthetic",
            "feature_set": "B",
            "preprocessing_strategy": "missing_plus_flag",
            "random_seed": 42,
            "evaluation_split": "validation",
            "positive_label": 1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "ks": ks,
            "training_time_seconds": runtime,
            "validation_inference_time_seconds": 0.1,
            "feature_count": 17,
            "manifest_sha256": MANIFEST_SHA,
            "config_sha256": "a" * 64,
            "git_commit_sha": "b" * 40,
            "raw_data_sha256": RAW_SHA,
            "package_versions": "{}",
        }
        for prefix, threshold in (("default", 0.5), ("operational", 0.4)):
            tn, fp, fn, tp = 20_000, 2_389, 402, 1_206
            row.update(
                {
                    f"{prefix}_threshold": threshold,
                    f"{prefix}_accuracy": (tn + tp) / EXPECTED_VALIDATION_ROWS,
                    f"{prefix}_precision": (
                        precision if prefix == "operational" else 0.2
                    ),
                    f"{prefix}_recall": tp / (tp + fn),
                    f"{prefix}_f1": 0.3,
                    f"{prefix}_specificity": tn / (tn + fp),
                    f"{prefix}_balanced_accuracy": 0.82,
                    f"{prefix}_predicted_positive_rate": (fp + tp)
                    / EXPECTED_VALIDATION_ROWS,
                    f"{prefix}_tn": tn,
                    f"{prefix}_fp": fp,
                    f"{prefix}_fn": fn,
                    f"{prefix}_tp": tp,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _artifact(
    model_name: str,
    experiment_id: str,
    probability: np.ndarray,
) -> ExperimentArtifacts:
    model_family = {
        "logistic_regression": "linear_model",
        "random_forest": "bagging",
        "balanced_random_forest": "imbalance_aware_bagging",
        "xgboost": "gradient_boosting",
    }[model_name]
    result = ExperimentResult(
        experiment_id=experiment_id,
        model_name=model_name,
        model_family=model_family,
        model_status="implemented",
        model_parameters={},
        model_training_metadata={"training_time_seconds": 1.0},
        imbalance_strategy="synthetic",
        requires_scaled_features=model_name == "logistic_regression",
        supports_validation_data=model_name == "xgboost",
        supports_early_stopping=model_name == "xgboost",
        best_iteration=None,
        scale_pos_weight=None,
        feature_set="B",
        preprocessing_strategy="missing_plus_flag",
        class_weight="none",
        random_seed=42,
        feature_count=17,
        feature_names=["feature"],
        train_row_count=95_995,
        validation_row_count=EXPECTED_VALIDATION_ROWS,
        validation_metrics={"pr_auc": 0.5, "roc_auc": 0.5, "ks": 0.0},
        default_threshold_metrics={},
        operational_threshold=0.4,
        operational_threshold_metrics={},
        training_time_seconds=1.0,
        validation_inference_time_seconds=0.1,
        raw_data_sha256=RAW_SHA,
        manifest_sha256=MANIFEST_SHA,
        config_sha256="a" * 64,
        git_commit_sha="b" * 40,
        package_versions={"numpy": "2.2.6"},
        preprocessing_metadata={},
        test_access="transform/schema/finite checks only; no Test prediction or metrics",
    )
    return ExperimentArtifacts(result, {}, pd.DataFrame(), probability)


def _figure_comparison() -> ComparisonRun:
    table = _comparison_table()
    y = np.tile(np.array([0, 0, 1], dtype="int8"), 8_000)[:EXPECTED_VALIDATION_ROWS]
    probability = np.linspace(0.0, 1.0, EXPECTED_VALIDATION_ROWS)
    models = [
        ("logistic_regression", "lr_b_balanced_missing_flag"),
        ("random_forest", "rf_b_none_missing_flag"),
        ("balanced_random_forest", "brf_b_none_missing_flag"),
        ("xgboost", "xgb_child5"),
    ]
    artifacts = tuple(_artifact(model, experiment, probability) for model, experiment in models)
    return ComparisonRun(
        table=table,
        artifacts=artifacts,
        validation_row_ids=np.arange(EXPECTED_VALIDATION_ROWS),
        y_validation=y,
        selected_experiment_id="xgb_child5",
        selection_considerations=(
            "pr_auc",
            "roc_auc",
            "operational_precision_with_recall_constraint",
            "runtime",
            "interpretability",
        ),
    )


def test_formal_comparison_contract_is_exactly_four_unique_models(experiment_config):
    specs = build_formal_specs(experiment_config)
    validate_specs(specs, experiment_config)
    assert len(specs) == 4
    assert len({spec.model_name for spec in specs}) == 4


def test_formal_specs_share_frozen_validation_contract(experiment_config):
    specs = build_formal_specs(experiment_config)
    assert {spec.feature_set for spec in specs} == {"B"}
    assert {spec.preprocessing_strategy for spec in specs} == {"missing_plus_flag"}
    assert {spec.evaluation_split for spec in specs} == {"validation"}
    assert {spec.expected_manifest_sha256 for spec in specs} == {MANIFEST_SHA}


def test_formal_specs_preserve_frozen_parameters(experiment_config):
    by_id = {spec.experiment_id: spec for spec in build_formal_specs(experiment_config)}
    assert by_id["rf_b_none_missing_flag"].model_parameters["n_estimators"] == 300
    assert by_id["brf_b_none_missing_flag"].model_parameters["replacement"] is True
    assert by_id["brf_b_none_missing_flag"].model_parameters["bootstrap"] is False
    assert by_id["xgb_child5"].model_parameters["min_child_weight"] == 5
    assert by_id["xgb_child5"].model_parameters["early_stopping_rounds"] == 50


def test_validate_specs_rejects_nonvalidation_or_wrong_feature_set(experiment_config):
    specs = list(build_formal_specs(experiment_config))
    with pytest.raises(ValueError, match="Feature Set B"):
        validate_specs([replace(specs[0], feature_set="A"), *specs[1:]], experiment_config)


@pytest.mark.parametrize(
    "probability",
    [
        np.zeros(EXPECTED_VALIDATION_ROWS - 1),
        np.r_[np.zeros(EXPECTED_VALIDATION_ROWS - 1), np.nan],
        np.r_[np.zeros(EXPECTED_VALIDATION_ROWS - 1), 1.1],
    ],
)
def test_probability_validation_rejects_length_nonfinite_and_range(probability):
    row_ids = np.arange(EXPECTED_VALIDATION_ROWS)
    with pytest.raises(ValueError):
        validate_probability_alignment(probability, row_ids, row_ids)


def test_probability_validation_requires_identical_row_order():
    row_ids = np.arange(EXPECTED_VALIDATION_ROWS)
    reversed_ids = row_ids[::-1]
    with pytest.raises(ValueError, match="row_id order"):
        validate_probability_alignment(
            np.zeros(EXPECTED_VALIDATION_ROWS), row_ids, reversed_ids
        )


def test_shared_pr_auc_is_average_precision_score():
    y = np.array([0, 1, 0, 1])
    probability = np.array([0.1, 0.8, 0.3, 0.6])
    metrics = threshold_independent_metrics(y, probability, split_name="validation")
    assert metrics["pr_auc"] == average_precision_score(y, probability)


def test_comparison_table_contract_and_confusion_totals():
    table = _comparison_table()
    validate_comparison_table(table, expected_manifest=MANIFEST_SHA)
    assert len(table) == 4
    assert table["model_name"].is_unique
    assert (table["operational_recall"] >= 0.75).all()


def test_comparison_table_rejects_test_fields():
    table = _comparison_table()
    table["test_pr_auc"] = 0.99
    with pytest.raises(ValueError, match="columns"):
        validate_comparison_table(table, expected_manifest=MANIFEST_SHA)


def test_comparison_table_rejects_bad_predicted_positive_rate():
    table = _comparison_table()
    table.loc[0, "operational_predicted_positive_rate"] = 0.0
    with pytest.raises(ValueError, match="predicted-positive"):
        validate_comparison_table(table, expected_manifest=MANIFEST_SHA)


def test_reference_metric_guard_reports_reproducibility_context():
    row = {
        "experiment_id": "xgb_child5",
        "pr_auc": 0.0,
        "roc_auc": 0.869935023,
        "ks": 0.585850356,
        "operational_threshold": 0.548155665,
        "operational_precision": 0.238865588,
        "operational_recall": 0.750313676,
        "package_versions": "{}",
        "git_commit_sha": "b" * 40,
        "manifest_sha256": MANIFEST_SHA,
        "config_sha256": "a" * 64,
    }
    with pytest.raises(RuntimeError, match="packages=.*git_sha=.*manifest_sha"):
        validate_reference_metrics(row)


def test_forest_references_use_deterministic_serial_inference_results():
    rf = REFERENCE_METRICS["rf_b_none_missing_flag"]
    brf = REFERENCE_METRICS["brf_b_none_missing_flag"]
    assert rf["pr_auc"] == pytest.approx(0.3917880699044836)
    assert rf["operational_threshold"] == pytest.approx(0.08617852913676516)
    assert brf["pr_auc"] == pytest.approx(0.3853522219300526)
    assert brf["operational_threshold"] == pytest.approx(0.4688175260543156)


def test_selection_uses_configured_order_and_selects_xgb(experiment_config):
    table = _comparison_table()
    order = experiment_config.raw["model_selection"]["ordered_considerations"]
    selected = select_recommended_model(table, order)
    assert selected["experiment_id"] == "xgb_child5"


def test_selection_ties_are_deterministic_and_use_runtime_then_interpretability():
    table = _comparison_table()
    table[["pr_auc", "roc_auc", "operational_precision"]] = 0.5
    table["training_time_seconds"] = 1.0
    selected = select_recommended_model(
        table,
        [
            "pr_auc",
            "roc_auc",
            "operational_precision_with_recall_constraint",
            "runtime",
            "interpretability",
        ],
    )
    assert selected["experiment_id"] == "lr_b_balanced_missing_flag"


def test_selection_rejects_test_metric_even_when_not_in_order():
    table = _comparison_table()
    table["independent_test_score"] = 1.0
    with pytest.raises(ValueError, match="forbids Test"):
        select_recommended_model(table, ["pr_auc"])


def test_all_formal_figures_generate_without_test_data(tmp_path):
    paths = generate_figures(_figure_comparison(), tmp_path)
    assert len(paths) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_pr_and_roc_figures_contain_four_auc_labels(monkeypatch, tmp_path):
    labels = []
    original = __import__("matplotlib").axes.Axes.plot

    def capture(self, *args, **kwargs):
        if kwargs.get("label"):
            labels.append(kwargs["label"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr("matplotlib.axes.Axes.plot", capture)
    generate_figures(_figure_comparison(), tmp_path)
    assert len([label for label in labels if "AP=" in label]) == 4
    assert len([label for label in labels if "ROC-AUC=" in label]) == 4


def test_output_summary_contains_handoff_and_quarantine_evidence(tmp_path):
    comparison = _figure_comparison()
    paths = write_outputs(
        comparison,
        output_dir=tmp_path / "outputs",
        figure_dir=tmp_path / "figures",
    )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["selected_experiment_id"] == "xgb_child5"
    assert len(summary["model_results"]) == 4
    assert {
        result["experiment_id"] for result in summary["model_results"]
    } == {
        "lr_b_balanced_missing_flag",
        "rf_b_none_missing_flag",
        "brf_b_none_missing_flag",
        "xgb_child5",
    }
    assert summary["test_evaluation_performed"] is False
    assert "accidentally viewed and quarantined" in summary[
        "forest_test_quarantine_statement"
    ]
    assert "inference-contract" in summary["forest_test_quarantine_statement"]
    assert not any("test_" in column.lower() for column in _comparison_table().columns)


def test_committed_handoff_contains_frozen_threshold_and_no_test_metrics():
    path = Path("docs/SEALED_EVALUATION_HANDOFF.md")
    if not path.is_file():
        pytest.skip("Handoff is generated after the formal comparison run")
    text = path.read_text(encoding="utf-8")
    assert "xgb_child5" in text
    assert "0.548155665" in text
    assert "不允许重新调整" in text or "must not be retuned" in text
    assert "Test PR-AUC" not in text
    assert "frozen Training only" in text
    assert "Validation only for early stopping" in text
    assert "n_estimators=2000" in text
    assert "early_stopping_rounds=50" in text
    assert "best_iteration=172" in text
    assert "actual_boosting_rounds=173" in text
    assert "abort before Test prediction" in text
    assert "不得合并 Training + Validation" in text
    assert "set n_estimators=173" not in text
    assert text.count("accidentally viewed and quarantined") == 1
