"""Unified four-model comparison on the frozen Validation partition only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import ExperimentConfig
from src.evaluation.metrics import validate_binary_inputs
from src.experiments.contracts import ExperimentSpec
from src.experiments.runner import ExperimentArtifacts

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


EXPECTED_VALIDATION_ROWS = 23_997
COMPARISON_MODEL_COUNT = 4
FORBIDDEN_RESULT_TOKEN = "test"

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "balanced_random_forest": "Balanced Random Forest",
    "xgboost": "XGBoost",
}

MODEL_INTERPRETABILITY = {
    "logistic_regression": 4,
    "random_forest": 3,
    "balanced_random_forest": 2,
    "xgboost": 1,
}

MODEL_COLORS = {
    "Logistic Regression": "#315D8A",
    "Random Forest": "#D88A27",
    "Balanced Random Forest": "#7A8F38",
    "XGBoost": "#A74D6E",
}

MODEL_LINESTYLES = {
    "Logistic Regression": "-",
    "Random Forest": "--",
    "Balanced Random Forest": "-.",
    "XGBoost": ":",
}

REFERENCE_METRICS: dict[str, dict[str, float]] = {
    "lr_b_balanced_missing_flag": {
        "pr_auc": 0.359228,
        "roc_auc": 0.827584,
        "ks": 0.514616,
        "operational_precision": 0.175753,
        "operational_recall": 0.750314,
    },
    "rf_b_none_missing_flag": {
        "pr_auc": 0.3917880699044836,
        "roc_auc": 0.8603043647082802,
        "ks": 0.5729567384633409,
        "operational_threshold": 0.08617852913676516,
        "operational_precision": 0.22802669208770257,
        "operational_recall": 0.7503136762860728,
    },
    "brf_b_none_missing_flag": {
        "pr_auc": 0.3853522219300526,
        "roc_auc": 0.8679775394169684,
        "ks": 0.582244597663503,
        "operational_threshold": 0.4688175260543156,
        "operational_precision": 0.235618597320725,
        "operational_recall": 0.7503136762860728,
    },
    "xgb_child5": {
        "pr_auc": 0.401962230,
        "roc_auc": 0.869935023,
        "ks": 0.585850356,
        "operational_threshold": 0.548155665,
        "operational_precision": 0.238865588,
        "operational_recall": 0.750313676,
    },
}

COMPARISON_COLUMNS = [
    "model_name",
    "experiment_id",
    "model_family",
    "imbalance_strategy",
    "feature_set",
    "preprocessing_strategy",
    "random_seed",
    "evaluation_split",
    "positive_label",
    "pr_auc",
    "roc_auc",
    "ks",
    "default_threshold",
    "default_accuracy",
    "default_precision",
    "default_recall",
    "default_f1",
    "default_specificity",
    "default_balanced_accuracy",
    "default_predicted_positive_rate",
    "default_tn",
    "default_fp",
    "default_fn",
    "default_tp",
    "operational_threshold",
    "operational_accuracy",
    "operational_precision",
    "operational_recall",
    "operational_f1",
    "operational_specificity",
    "operational_balanced_accuracy",
    "operational_predicted_positive_rate",
    "operational_tn",
    "operational_fp",
    "operational_fn",
    "operational_tp",
    "training_time_seconds",
    "validation_inference_time_seconds",
    "feature_count",
    "manifest_sha256",
    "config_sha256",
    "git_commit_sha",
    "raw_data_sha256",
    "package_versions",
]


@dataclass(frozen=True)
class ComparisonRun:
    table: pd.DataFrame
    artifacts: tuple[ExperimentArtifacts, ...]
    validation_row_ids: np.ndarray
    y_validation: np.ndarray
    selected_experiment_id: str
    selection_considerations: tuple[str, ...]


def build_formal_specs(config: ExperimentConfig) -> tuple[ExperimentSpec, ...]:
    """Return the four frozen comparison specifications."""
    shared = {
        "feature_set": "B",
        "preprocessing_strategy": "missing_plus_flag",
        "random_seed": int(config.raw["reproducibility"]["random_seed"]),
        "config_path": config.config_path,
        "manifest_path": config.project_root / config.raw["split"]["manifest_path"],
        "expected_manifest_sha256": config.raw["split"]["frozen_manifest_sha256"],
        "evaluation_split": "validation",
    }
    return (
        ExperimentSpec(
            experiment_id="lr_b_balanced_missing_flag",
            model_name="logistic_regression",
            class_weight="balanced",
            model_parameters={},
            **shared,
        ),
        ExperimentSpec(
            experiment_id="rf_b_none_missing_flag",
            model_name="random_forest",
            class_weight="none",
            model_parameters={
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "n_jobs": -1,
            },
            **shared,
        ),
        ExperimentSpec(
            experiment_id="brf_b_none_missing_flag",
            model_name="balanced_random_forest",
            class_weight="none",
            model_parameters={
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "sampling_strategy": "all",
                "replacement": True,
                "bootstrap": False,
                "n_jobs": -1,
            },
            **shared,
        ),
        ExperimentSpec(
            experiment_id="xgb_child5",
            model_name="xgboost",
            class_weight="none",
            model_parameters={
                "n_estimators": 2000,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "early_stopping_rounds": 50,
                "n_jobs": -1,
            },
            **shared,
        ),
    )


def validate_specs(specs: Sequence[ExperimentSpec], config: ExperimentConfig) -> None:
    if len(specs) != COMPARISON_MODEL_COUNT:
        raise ValueError("Unified comparison requires exactly four model specifications")
    if len({spec.model_name for spec in specs}) != COMPARISON_MODEL_COUNT:
        raise ValueError("Unified comparison model names must be unique")
    if {spec.model_name for spec in specs} != set(MODEL_DISPLAY_NAMES):
        raise ValueError("Unified comparison must contain the four configured core models")
    expected_manifest = config.raw["split"]["frozen_manifest_sha256"]
    for spec in specs:
        if spec.feature_set != "B":
            raise ValueError("Every comparison model must use Feature Set B")
        if spec.preprocessing_strategy != "missing_plus_flag":
            raise ValueError("Every comparison model must use missing_plus_flag")
        if spec.evaluation_split != "validation":
            raise ValueError("Unified comparison evaluation_split must be validation")
        if spec.expected_manifest_sha256 != expected_manifest:
            raise ValueError("Every comparison model must use the same frozen Manifest SHA")


def run_comparison(runner: Any) -> ComparisonRun:
    """Run all formal specs through the supplied SharedExperimentRunner."""
    specs = build_formal_specs(runner.config)
    validate_specs(specs, runner.config)
    validation = runner.partitions["validation"]
    target = runner.config.data["target_column"]
    row_ids = validation["row_id"].to_numpy(copy=True)
    y_validation = validation[target].to_numpy(copy=True)
    if len(row_ids) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(
            f"Frozen Validation must contain {EXPECTED_VALIDATION_ROWS} rows, got {len(row_ids)}"
        )

    artifacts: list[ExperimentArtifacts] = []
    rows: list[dict[str, Any]] = []
    for spec in specs:
        artifact = runner.run(spec)
        validate_probability_alignment(
            artifact.validation_probability,
            row_ids,
            validation["row_id"].to_numpy(),
        )
        row = comparison_row(artifact)
        validate_reference_metrics(row)
        artifacts.append(artifact)
        rows.append(row)

    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    validate_comparison_table(table, expected_manifest=runner.expected_manifest_sha256)
    considerations = tuple(runner.config.raw["model_selection"]["ordered_considerations"])
    selected = select_recommended_model(table, considerations)
    return ComparisonRun(
        table=table,
        artifacts=tuple(artifacts),
        validation_row_ids=row_ids,
        y_validation=y_validation,
        selected_experiment_id=str(selected["experiment_id"]),
        selection_considerations=considerations,
    )


def validate_probability_alignment(
    probability: object,
    expected_row_ids: object,
    actual_row_ids: object,
) -> np.ndarray:
    values = np.asarray(probability, dtype="float64")
    expected = np.asarray(expected_row_ids)
    actual = np.asarray(actual_row_ids)
    if values.shape != (EXPECTED_VALIDATION_ROWS,):
        raise ValueError(
            "Each Validation probability must have shape "
            f"({EXPECTED_VALIDATION_ROWS},), got {values.shape}"
        )
    if not np.array_equal(expected, actual):
        raise ValueError("Validation row_id order differs from the frozen reference order")
    if not np.isfinite(values).all():
        raise ValueError("Validation probabilities must be finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Validation probabilities must lie in [0, 1]")
    return values


def comparison_row(artifact: ExperimentArtifacts) -> dict[str, Any]:
    result = artifact.result
    independent = result.validation_metrics
    default = result.default_threshold_metrics
    operational = result.operational_threshold_metrics
    row: dict[str, Any] = {
        "model_name": MODEL_DISPLAY_NAMES[result.model_name],
        "experiment_id": result.experiment_id,
        "model_family": result.model_family,
        "imbalance_strategy": result.imbalance_strategy,
        "feature_set": result.feature_set,
        "preprocessing_strategy": result.preprocessing_strategy,
        "random_seed": result.random_seed,
        "evaluation_split": "validation",
        "positive_label": 1,
        "pr_auc": independent["pr_auc"],
        "roc_auc": independent["roc_auc"],
        "ks": independent["ks"],
        "training_time_seconds": result.training_time_seconds,
        "validation_inference_time_seconds": result.validation_inference_time_seconds,
        "feature_count": result.feature_count,
        "manifest_sha256": result.manifest_sha256,
        "config_sha256": result.config_sha256,
        "git_commit_sha": result.git_commit_sha,
        "raw_data_sha256": result.raw_data_sha256,
        "package_versions": json.dumps(result.package_versions, sort_keys=True),
    }
    _add_threshold_columns(row, "default", default)
    _add_threshold_columns(row, "operational", operational)
    return row


def _add_threshold_columns(
    row: dict[str, Any], prefix: str, metrics: Mapping[str, Any]
) -> None:
    for metric in (
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "predicted_positive_rate",
    ):
        row[f"{prefix}_{metric}"] = metrics[metric]
    matrix = metrics["confusion_matrix"]
    for cell in ("tn", "fp", "fn", "tp"):
        row[f"{prefix}_{cell}"] = matrix[cell]


def validate_reference_metrics(
    row: Mapping[str, Any],
    *,
    tolerance: float = 1e-6,
) -> None:
    experiment_id = str(row["experiment_id"])
    reference = REFERENCE_METRICS[experiment_id]
    mismatches = []
    for metric, expected in reference.items():
        actual = float(row[metric])
        difference = abs(actual - expected)
        if difference > tolerance:
            mismatches.append(
                f"{metric}: actual={actual:.12g}, reference={expected:.12g}, "
                f"difference={difference:.12g}"
            )
    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            f"Reference metric mismatch for {experiment_id}: {details}; "
            f"packages={row['package_versions']}; git_sha={row['git_commit_sha']}; "
            f"manifest_sha={row['manifest_sha256']}; config_sha={row['config_sha256']}"
        )


def validate_comparison_table(
    table: pd.DataFrame,
    *,
    expected_manifest: str,
) -> None:
    if list(table.columns) != COMPARISON_COLUMNS:
        raise ValueError("Comparison table columns differ from the frozen output contract")
    if len(table) != COMPARISON_MODEL_COUNT:
        raise ValueError("Comparison table must contain exactly four rows")
    if not table["model_name"].is_unique:
        raise ValueError("Comparison table model names must be unique")
    forbidden = [column for column in table.columns if FORBIDDEN_RESULT_TOKEN in column.lower()]
    if forbidden:
        raise ValueError(f"Comparison output contains forbidden Test fields: {forbidden}")
    if set(table["feature_set"]) != {"B"}:
        raise ValueError("Comparison output must use Feature Set B")
    if set(table["preprocessing_strategy"]) != {"missing_plus_flag"}:
        raise ValueError("Comparison output must use missing_plus_flag")
    if set(table["evaluation_split"]) != {"validation"}:
        raise ValueError("Comparison output must contain Validation results only")
    if set(table["positive_label"]) != {1}:
        raise ValueError("Positive class must be fixed to 1")
    if set(table["manifest_sha256"]) != {expected_manifest}:
        raise ValueError("Comparison rows must share the frozen Manifest SHA")
    numeric = table.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise ValueError("All numeric comparison metrics must be finite")
    if (table["operational_recall"] < 0.75).any():
        raise ValueError("Every operational threshold must achieve Recall >= 0.75")
    for prefix in ("default", "operational"):
        matrix_total = table[
            [f"{prefix}_{cell}" for cell in ("tn", "fp", "fn", "tp")]
        ].sum(axis=1)
        if not (matrix_total == EXPECTED_VALIDATION_ROWS).all():
            raise ValueError(
                f"Every {prefix} confusion matrix must total {EXPECTED_VALIDATION_ROWS}"
            )
        expected_rate = (
            table[f"{prefix}_fp"] + table[f"{prefix}_tp"]
        ) / EXPECTED_VALIDATION_ROWS
        if not np.allclose(
            expected_rate,
            table[f"{prefix}_predicted_positive_rate"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{prefix} predicted-positive rate differs from its confusion matrix"
            )


def select_recommended_model(
    table: pd.DataFrame,
    ordered_considerations: Sequence[str],
    *,
    tolerance: float = 1e-12,
) -> pd.Series:
    """Apply the configured ordered considerations as deterministic tie-breaks."""
    if not ordered_considerations:
        raise ValueError("model_selection.ordered_considerations must not be empty")
    forbidden = [
        column
        for column in table.columns
        if FORBIDDEN_RESULT_TOKEN in str(column).lower()
    ]
    if forbidden:
        raise ValueError(f"Model selection forbids Test fields: {forbidden}")
    metric_map = {
        "pr_auc": ("pr_auc", "max"),
        "roc_auc": ("roc_auc", "max"),
        "operational_precision_with_recall_constraint": (
            "operational_precision",
            "max",
        ),
        "runtime": ("training_time_seconds", "min"),
        "interpretability": ("interpretability_score", "max"),
    }
    unknown = [item for item in ordered_considerations if item not in metric_map]
    if unknown:
        raise ValueError(f"Unsupported model selection considerations: {unknown}")

    candidates = table.copy()
    candidates["interpretability_score"] = candidates["model_name"].map(
        {
            MODEL_DISPLAY_NAMES[key]: value
            for key, value in MODEL_INTERPRETABILITY.items()
        }
    )
    for consideration in ordered_considerations:
        column, direction = metric_map[consideration]
        best = (
            candidates[column].max()
            if direction == "max"
            else candidates[column].min()
        )
        candidates = candidates[
            np.isclose(candidates[column], best, rtol=0.0, atol=tolerance)
        ]
        if len(candidates) == 1:
            break
    return candidates.sort_values("experiment_id", kind="mergesort").iloc[0]


def write_outputs(
    comparison: ComparisonRun,
    *,
    output_dir: Path,
    figure_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "model_comparison.csv"
    operational_path = output_dir / "operational_comparison.csv"
    summary_path = output_dir / "comparison_summary.json"
    comparison.table.to_csv(table_path, index=False, lineterminator="\n")
    comparison.table[
        [
            "model_name",
            "operational_threshold",
            "operational_precision",
            "operational_recall",
            "operational_specificity",
            "operational_f1",
            "operational_balanced_accuracy",
            "operational_predicted_positive_rate",
            "operational_fp",
            "operational_fn",
        ]
    ].to_csv(operational_path, index=False, lineterminator="\n")
    summary = {
        "evaluation_split": "validation",
        "positive_label": 1,
        "model_count": len(comparison.table),
        "selected_experiment_id": comparison.selected_experiment_id,
        "selection_considerations": list(comparison.selection_considerations),
        "selected_model_result": next(
            artifact.result.to_dict()
            for artifact in comparison.artifacts
            if artifact.result.experiment_id == comparison.selected_experiment_id
        ),
        "model_results": [
            artifact.result.to_dict() for artifact in comparison.artifacts
        ],
        "test_evaluation_performed": False,
        "forest_test_quarantine_statement": (
            "Earlier Forest Test metrics were accidentally viewed and quarantined. "
            "They were not used for model, parameter, threshold, inference-contract, "
            "or comparison decisions."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    figures = generate_figures(comparison, figure_dir)
    return {
        "model_comparison": table_path,
        "operational_comparison": operational_path,
        "summary": summary_path,
        **figures,
    }


def generate_figures(
    comparison: ComparisonRun,
    figure_dir: Path,
) -> dict[str, Path]:
    _set_plot_style()
    paths = {
        "pr_curve": figure_dir / "validation_pr_curve.png",
        "roc_curve": figure_dir / "validation_roc_curve.png",
        "validation_metrics": figure_dir / "validation_metric_comparison.png",
        "operational_metrics": figure_dir / "operational_metric_comparison.png",
        "default_confusion": figure_dir / "confusion_matrices_default.png",
        "operational_confusion": figure_dir
        / "confusion_matrices_operational.png",
    }
    _plot_pr_curves(comparison, paths["pr_curve"])
    _plot_roc_curves(comparison, paths["roc_curve"])
    _plot_validation_metrics(comparison.table, paths["validation_metrics"])
    _plot_operational_metrics(comparison.table, paths["operational_metrics"])
    _plot_confusion_matrices(
        comparison.table,
        prefix="default",
        title="Validation Confusion Matrices — Default Threshold 0.5",
        path=paths["default_confusion"],
    )
    _plot_confusion_matrices(
        comparison.table,
        prefix="operational",
        title="Validation Confusion Matrices — Operational Thresholds",
        path=paths["operational_confusion"],
    )
    return paths


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#FBFCFE",
            "axes.facecolor": "#FBFCFE",
            "axes.edgecolor": "#30343B",
            "axes.labelcolor": "#30343B",
            "text.color": "#20242A",
            "xtick.color": "#4C535D",
            "ytick.color": "#4C535D",
            "grid.color": "#DDE2E8",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 11,
        }
    )


def _artifact_by_experiment(
    artifacts: Iterable[ExperimentArtifacts],
) -> dict[str, ExperimentArtifacts]:
    return {artifact.result.experiment_id: artifact for artifact in artifacts}


def _plot_pr_curves(comparison: ComparisonRun, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12.8, 7.2))
    artifact_map = _artifact_by_experiment(comparison.artifacts)
    prevalence = float(np.mean(comparison.y_validation))
    for row in comparison.table.itertuples(index=False):
        artifact = artifact_map[row.experiment_id]
        y, probability = validate_binary_inputs(
            comparison.y_validation, artifact.validation_probability
        )
        precision, recall, _ = precision_recall_curve(y, probability, pos_label=1)
        axis.plot(
            recall,
            precision,
            color=MODEL_COLORS[row.model_name],
            linestyle=MODEL_LINESTYLES[row.model_name],
            linewidth=2.3,
            label=f"{row.model_name}  AP={row.pr_auc:.6f}",
        )
    axis.axhline(
        prevalence,
        color="#59616B",
        linestyle=(0, (4, 3)),
        linewidth=1.4,
        label=f"Random baseline  prevalence={prevalence:.4f}",
    )
    axis.set(
        title="Validation Precision–Recall Curves",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(True, alpha=0.75)
    axis.legend(loc="upper right", frameon=False)
    _save_figure(fig, path)


def _plot_roc_curves(comparison: ComparisonRun, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12.8, 7.2))
    artifact_map = _artifact_by_experiment(comparison.artifacts)
    for row in comparison.table.itertuples(index=False):
        artifact = artifact_map[row.experiment_id]
        y, probability = validate_binary_inputs(
            comparison.y_validation, artifact.validation_probability
        )
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y, probability, pos_label=1
        )
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=MODEL_COLORS[row.model_name],
            linestyle=MODEL_LINESTYLES[row.model_name],
            linewidth=2.3,
            label=f"{row.model_name}  ROC-AUC={row.roc_auc:.6f}",
        )
    axis.plot(
        [0, 1],
        [0, 1],
        color="#59616B",
        linestyle=(0, (4, 3)),
        linewidth=1.4,
        label="Random classifier",
    )
    axis.set(
        title="Validation ROC Curves",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(True, alpha=0.75)
    axis.legend(loc="lower right", frameon=False)
    _save_figure(fig, path)


def _plot_validation_metrics(table: pd.DataFrame, path: Path) -> None:
    metrics = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("ks", "KS")]
    x = np.arange(len(table))
    width = 0.23
    fig, axis = plt.subplots(figsize=(12.8, 7.2))
    colors = ["#315D8A", "#D88A27", "#7A8F38"]
    for index, ((column, label), color) in enumerate(zip(metrics, colors, strict=True)):
        positions = x + (index - 1) * width
        bars = axis.bar(
            positions,
            table[column],
            width,
            label=label,
            color=color,
            edgecolor="#30343B",
            linewidth=0.6,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axis.set(
        title="Validation Threshold-independent Metrics",
        ylabel="Metric value",
        ylim=(0.0, 1.0),
    )
    axis.set_xticks(x, table["model_name"])
    axis.grid(axis="y", alpha=0.75)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    _save_figure(fig, path)


def _plot_operational_metrics(table: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(table))
    colors = [MODEL_COLORS[name] for name in table["model_name"]]

    def zoomed_limits(
        values: Sequence[float],
        *,
        include: float | None = None,
    ) -> tuple[float, float]:
        low = float(min(values))
        high = float(max(values))
        span = high - low
        padding = span * 0.25 if span > 0.0 else max(abs(high) * 0.01, 0.005)
        if include is not None:
            low = min(low, include)
            high = max(high, include)
        return max(0.0, low - padding), min(1.0, high + padding)

    fig, (precision_axis, recall_axis) = plt.subplots(
        1, 2, figsize=(12.8, 7.2)
    )
    panels = (
        (
            precision_axis,
            "operational_precision",
            "Operational Precision",
            None,
        ),
        (
            recall_axis,
            "operational_recall",
            "Operational Recall",
            0.75,
        ),
    )
    for axis, column, title, required_floor in panels:
        values = table[column].to_numpy(dtype="float64")
        bars = axis.bar(
            x,
            values,
            width=0.62,
            color=colors,
            edgecolor="#30343B",
            linewidth=0.6,
        )
        axis.set_xticks(x, table["model_name"], rotation=16, ha="right")
        axis.set_ylabel(title)
        axis.set_title(f"{title} — Validation (zoomed)")
        axis.set_ylim(zoomed_limits(values, include=required_floor))
        axis.grid(axis="y", alpha=0.75)
        axis.bar_label(bars, fmt="%.6f", padding=3, fontsize=9)

    recall_axis.axhline(
        0.75,
        color="#30343B",
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        label="Recall constraint = 0.75",
    )
    recall_axis.legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Validation Precision and Recall at Operational Thresholds",
        fontsize=16,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, path, tight=False)


def _plot_confusion_matrices(
    table: pd.DataFrame,
    *,
    prefix: str,
    title: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2))
    matrices = [
        np.array(
            [
                [row[f"{prefix}_tn"], row[f"{prefix}_fp"]],
                [row[f"{prefix}_fn"], row[f"{prefix}_tp"]],
            ],
            dtype="int64",
        )
        for row in table.to_dict(orient="records")
    ]
    maximum = max(int(matrix.max()) for matrix in matrices)
    for axis, matrix, row in zip(
        axes.flat, matrices, table.to_dict(orient="records"), strict=True
    ):
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        labels = (("TN", "FP"), ("FN", "TP"))
        for row_index in range(2):
            for column_index in range(2):
                value = int(matrix[row_index, column_index])
                text_color = "white" if value > maximum * 0.55 else "#20242A"
                axis.text(
                    column_index,
                    row_index,
                    f"{labels[row_index][column_index]}\n{value:,}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                    fontweight="semibold",
                )
        axis.set_title(
            f"{row['model_name']}\nthreshold={row[f'{prefix}_threshold']:.6f}",
            fontsize=11,
        )
        axis.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
        axis.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        axis.set_xlabel("Positive class = 1", fontsize=9)
        axis.grid(False)
    fig.suptitle(title, fontsize=16, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, path, tight=False)


def _save_figure(fig: Any, path: Path, *, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
