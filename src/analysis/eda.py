"""Auditable EDA tables and presentation-ready plots for the credit-risk project.

Two scopes are intentionally supported:

``raw_audit``
    A pre-declared, whole-file audit of schema, class balance, missingness,
    duplicate predictors, abnormal 96/98 codes, and univariate distributions.
    It must not be used for feature, model, threshold, or hyperparameter selection.

``train``
    Label-conditioned analysis of the frozen Training partition exposed by
    :mod:`src.analysis.train_only`.  There is deliberately no Validation/Test
    selector in this module.
"""

from __future__ import annotations

import hashlib
import json
import platform
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.train_only import load_train_partition, transform_train_with_strategy
from src.config import ExperimentConfig, load_config
from src.data.feature_hash import compute_feature_hashes
from src.data.load_data import load_labeled_data
from src.utils.reproducibility import git_commit_sha, package_versions, sha256_file

RAW_AUDIT_SCOPE = "raw_audit"
TRAIN_SCOPE = "train"
SUPPORTED_SCOPES = (RAW_AUDIT_SCOPE, TRAIN_SCOPE)
EXPECTED_TRAIN_ROWS = 95_995

SENSITIVITY_EXPERIMENTS = {
    "missing_plus_flag": {
        "experiment_id": "lr_b_balanced_missing_flag",
        "feature_count": 17,
    },
    "keep_raw": {
        "experiment_id": "lr_b_balanced_keep_raw",
        "feature_count": 16,
    },
}

SENSITIVITY_FINITE_COLUMNS = (
    "pr_auc",
    "roc_auc",
    "ks",
    "default_precision",
    "default_recall",
    "default_f1",
    "default_predicted_positive_rate",
    "operational_threshold",
    "operational_precision",
    "operational_recall",
    "operational_f1",
    "operational_predicted_positive_rate",
    "training_time_seconds",
    "validation_inference_time_seconds",
)

DISPLAY_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30–59 days past due",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines / loans",
    "NumberOfTimes90DaysLate": "90+ days late",
    "NumberRealEstateLoansOrLines": "Real-estate loans / lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89 days past due",
    "NumberOfDependents": "Dependents",
}

COLOR_PRIMARY = "#2455A4"
COLOR_ACCENT = "#E6862D"
COLOR_DARK = "#263238"
COLOR_LIGHT = "#D7E3F4"
COLOR_WARNING = "#C73E1D"


@dataclass(frozen=True)
class EdaDataset:
    """One explicitly scoped EDA dataset with traceability metadata."""

    frame: pd.DataFrame
    feature_hash_v1: pd.Series
    scope: str
    metadata: dict[str, Any]


def validate_scope(scope: str) -> str:
    """Reject any scope that could expose a sealed partition."""
    if scope not in SUPPORTED_SCOPES:
        raise ValueError(f"scope must be one of {SUPPORTED_SCOPES}; got {scope!r}")
    return scope


def repository_relative_path(path: str | Path, project_root: Path) -> str:
    """Return a portable repository-relative path or reject an external path."""
    resolved = Path(path).expanduser().resolve()
    root = project_root.expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside the repository root: {resolved}") from exc


def load_eda_dataset(
    config_path: str | Path = "configs/experiment.yaml",
    *,
    scope: str,
) -> tuple[ExperimentConfig, EdaDataset]:
    """Load either the locked raw audit or frozen Training-only EDA view."""
    validate_scope(scope)
    config = load_config(config_path)
    if scope == RAW_AUDIT_SCOPE:
        frame = load_labeled_data(config)
        hashes = compute_feature_hashes(frame, config.predictor_columns)
        metadata = {
            "split": None,
            "data_level": "raw labeled data before preprocessing",
            "source_path": repository_relative_path(
                config.labeled_path, config.project_root
            ),
            "source_sha256": config.data["raw_labeled_sha256"],
            "label_usage": "predeclared dataset audit only",
        }
    else:
        partition = load_train_partition(config.config_path)
        frame = pd.concat(
            [
                partition.row_id.rename(config.data["id_column"]),
                partition.target.rename(config.data["target_column"]),
                partition.predictors,
            ],
            axis=1,
        )
        hashes = partition.feature_hash_v1
        metadata = {
            **partition.split_metadata,
            "source_path": repository_relative_path(
                config.labeled_path, config.project_root
            ),
            "source_sha256": config.data["raw_labeled_sha256"],
            "label_usage": "frozen Training partition only",
        }
    return config, EdaDataset(
        frame=frame.reset_index(drop=True),
        feature_hash_v1=hashes.reset_index(drop=True).astype("string"),
        scope=scope,
        metadata=metadata,
    )


def _check_row(
    check_id: str,
    *,
    passed: bool,
    expected: object,
    observed: object,
    details: str,
    severity: str = "ERROR",
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "severity": severity,
        "status": "PASS" if passed else "FAIL",
        "expected": expected,
        "observed": observed,
        "details": details,
    }


def build_validation_checks(
    dataset: EdaDataset,
    config: ExperimentConfig,
    *,
    raw_sha256_after: str,
) -> pd.DataFrame:
    """Build machine-readable contract checks without mutating the data."""
    frame = dataset.frame
    predictors = list(config.predictor_columns)
    row_id = config.data["id_column"]
    target = config.data["target_column"]
    expected_schema = [row_id, target, *predictors]
    labels = sorted(frame[target].dropna().unique().tolist())
    missing_columns = sorted(frame[predictors].columns[frame[predictors].isna().any()].tolist())
    allowed_missing = ["MonthlyIncome", "NumberOfDependents"]
    infinite_count = int(np.isinf(frame[predictors].to_numpy(dtype="float64")).sum())
    rows: list[dict[str, object]] = [
        _check_row(
            "analysis_scope",
            passed=dataset.scope in SUPPORTED_SCOPES,
            expected=list(SUPPORTED_SCOPES),
            observed=dataset.scope,
            details="Only raw_audit and frozen Training are available; Validation/Test are sealed.",
        ),
        _check_row(
            "schema_and_order",
            passed=list(frame.columns) == expected_schema,
            expected=expected_schema,
            observed=list(frame.columns),
            details="The internal EDA schema is row_id + target + locked predictors.",
        ),
        _check_row(
            "row_count",
            passed=len(frame)
            == (
                int(config.data["expected_labeled_rows"])
                if dataset.scope == RAW_AUDIT_SCOPE
                else EXPECTED_TRAIN_ROWS
            ),
            expected=(
                int(config.data["expected_labeled_rows"])
                if dataset.scope == RAW_AUDIT_SCOPE
                else EXPECTED_TRAIN_ROWS
            ),
            observed=len(frame),
            details=(
                "Raw audit and frozen Training row counts must match their exact contracts."
            ),
        ),
        _check_row(
            "row_id_complete_and_unique",
            passed=not frame[row_id].isna().any() and frame[row_id].is_unique,
            expected="no missing values and all unique",
            observed=f"missing={int(frame[row_id].isna().sum())}; unique={frame[row_id].nunique()}",
            details="row_id is retained only for traceability.",
        ),
        _check_row(
            "binary_target",
            passed=not frame[target].isna().any() and labels == [0, 1],
            expected="complete labels in {0, 1}",
            observed=f"missing={int(frame[target].isna().sum())}; labels={labels}",
            details="The positive class is fixed to 1.",
        ),
        _check_row(
            "numeric_predictors",
            passed=all(pd.api.types.is_numeric_dtype(frame[column]) for column in predictors),
            expected="all predictors numeric",
            observed="all numeric"
            if all(pd.api.types.is_numeric_dtype(frame[column]) for column in predictors)
            else "one or more non-numeric predictors",
            details="Unexpected text columns are rejected before analysis.",
        ),
        _check_row(
            "missingness_columns",
            passed=set(missing_columns).issubset(allowed_missing),
            expected=allowed_missing,
            observed=missing_columns,
            details="Raw missing values may occur only in contract-declared fields.",
        ),
        _check_row(
            "no_infinite_predictors",
            passed=infinite_count == 0,
            expected=0,
            observed=infinite_count,
            details="NaN is audited separately; positive or negative infinity is invalid.",
        ),
        _check_row(
            "feature_hash_alignment",
            passed=len(dataset.feature_hash_v1) == len(frame)
            and not dataset.feature_hash_v1.isna().any(),
            expected=f"{len(frame)} non-missing hashes",
            observed=(
                f"rows={len(dataset.feature_hash_v1)}; "
                f"missing={int(dataset.feature_hash_v1.isna().sum())}"
            ),
            details="feature_hash_v1 is used for duplicate grouping, never as a model feature.",
        ),
        _check_row(
            "raw_file_immutable",
            passed=raw_sha256_after == config.data["raw_labeled_sha256"],
            expected=config.data["raw_labeled_sha256"],
            observed=raw_sha256_after,
            details="The raw CSV fingerprint must be unchanged after EDA.",
        ),
    ]
    if dataset.scope == TRAIN_SCOPE:
        expected_manifest = config.raw["split"]["frozen_manifest_sha256"]
        observed_manifest = dataset.metadata.get("manifest_sha256")
        rows.append(
            _check_row(
                "frozen_manifest_sha256",
                passed=observed_manifest == expected_manifest,
                expected=expected_manifest,
                observed=observed_manifest,
                details="Training EDA must use the approved frozen split manifest.",
            )
        )
    return pd.DataFrame(rows)


def assert_validation_passed(checks: pd.DataFrame) -> None:
    failed = checks[(checks["severity"] == "ERROR") & (checks["status"] != "PASS")]
    if not failed.empty:
        ids = ", ".join(failed["check_id"].astype(str))
        raise ValueError(f"EDA validation failed: {ids}")


def build_dataset_summary(dataset: EdaDataset, config: ExperimentConfig) -> pd.DataFrame:
    frame = dataset.frame
    target = config.data["target_column"]
    positive_count = int(frame[target].sum())
    rows = [
        ("analysis_scope", dataset.scope, "", "raw_audit is descriptive; train is label-conditioned"),
        ("source_sha256", config.data["raw_labeled_sha256"], "", "immutable raw CSV fingerprint"),
        ("row_count", len(frame), "rows", "rows included in this EDA scope"),
        ("column_count", frame.shape[1], "columns", "row_id + target + predictors"),
        ("predictor_count", len(config.predictor_columns), "features", "locked raw predictors"),
        ("negative_count", len(frame) - positive_count, "rows", "target=0"),
        ("positive_count", positive_count, "rows", "target=1"),
        ("positive_rate", positive_count / len(frame), "fraction", "target=1 / all rows"),
        ("age_non_positive_count", int((frame["age"] <= 0).sum()), "rows", "age <= 0"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit", "notes"])


def build_class_distribution(dataset: EdaDataset, config: ExperimentConfig) -> pd.DataFrame:
    target = config.data["target_column"]
    counts = dataset.frame[target].value_counts().reindex([0, 1], fill_value=0)
    return pd.DataFrame(
        {
            "target": [0, 1],
            "class_label": ["No serious delinquency", "Serious delinquency"],
            "count": counts.to_numpy(dtype="int64"),
            "rate": (counts / len(dataset.frame)).to_numpy(dtype="float64"),
        }
    )


def build_missing_values(dataset: EdaDataset, config: ExperimentConfig) -> pd.DataFrame:
    frame = dataset.frame
    rows = [
        {
            "feature": column,
            "missing_count": int(frame[column].isna().sum()),
            "missing_rate": float(frame[column].isna().mean()),
            "non_missing_count": int(frame[column].notna().sum()),
        }
        for column in config.predictor_columns
    ]
    return pd.DataFrame(rows).sort_values(
        ["missing_rate", "feature"], ascending=[False, True], ignore_index=True
    )


def build_descriptive_statistics(
    dataset: EdaDataset, config: ExperimentConfig
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in config.predictor_columns:
        series = dataset.frame[column].dropna().astype("float64")
        rows.append(
            {
                "feature": column,
                "count": int(series.size),
                "missing_count": int(dataset.frame[column].isna().sum()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "q95": float(series.quantile(0.95)),
                "q99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "zero_count": int((series == 0).sum()),
                "zero_rate_non_missing": float((series == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_abnormal_code_tables(
    dataset: EdaDataset, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = dataset.frame
    target = config.data["target_column"]
    abnormal_cfg = config.raw["preprocessing"]["abnormal_delinquency"]
    columns = list(abnormal_cfg["columns"])
    values = list(abnormal_cfg["abnormal_values"])
    include_target_statistics = dataset.scope == TRAIN_SCOPE
    count_rows: list[dict[str, object]] = []
    for column in columns:
        for code in values:
            mask = frame[column] == code
            affected = int(mask.sum())
            row: dict[str, object] = {
                "feature": column,
                "code": code,
                "affected_rows": affected,
                "row_rate": affected / len(frame),
            }
            if include_target_statistics:
                positives = int(frame.loc[mask, target].sum())
                row.update(
                    {
                        "target_0": affected - positives,
                        "target_1": positives,
                        "positive_rate": positives / affected if affected else np.nan,
                    }
                )
            count_rows.append(row)
    summary_rows: list[dict[str, object]] = []
    for label, codes in [(str(code), [code]) for code in values] + [
        ("_or_".join(str(code) for code in values), values)
    ]:
        any_mask = frame[columns].isin(codes).any(axis=1)
        all_same_mask = pd.Series(False, index=frame.index)
        for code in codes:
            all_same_mask |= frame[columns].eq(code).all(axis=1)
        affected = int(any_mask.sum())
        row = {
            "code": label,
            "unique_affected_rows": affected,
            "all_three_fields_same_code_rows": int(all_same_mask.sum()),
        }
        if include_target_statistics:
            positives = int(frame.loc[any_mask, target].sum())
            row.update(
                {
                    "target_0": affected - positives,
                    "target_1": positives,
                    "positive_rate": positives / affected if affected else np.nan,
                }
            )
        summary_rows.append(row)
    return pd.DataFrame(count_rows), pd.DataFrame(summary_rows)


def build_duplicate_summary(dataset: EdaDataset, config: ExperimentConfig) -> pd.DataFrame:
    frame = dataset.frame
    hashes = dataset.feature_hash_v1
    counts = hashes.value_counts()
    duplicate_hashes = counts[counts > 1].index
    target = config.data["target_column"]
    grouped = pd.DataFrame({"hash": hashes, "target": frame[target]}).groupby("hash")[
        "target"
    ]
    conflicts = grouped.agg(["nunique", "size"])
    conflicts = conflicts[conflicts["nunique"] > 1]
    values = {
        "row_count": len(frame),
        "unique_predictor_vectors": int(hashes.nunique()),
        "duplicate_predictor_excess_rows": int(len(frame) - hashes.nunique()),
        "duplicate_predictor_groups": int(len(duplicate_hashes)),
        "duplicate_predictor_plus_target_excess_rows": int(
            frame.duplicated(subset=[*config.predictor_columns, target]).sum()
        ),
        "conflicting_target_groups": int(len(conflicts)),
        "rows_in_conflicting_target_groups": int(conflicts["size"].sum()),
    }
    return pd.DataFrame({"metric": values.keys(), "value": values.values()})


def build_delinquency_zero_summary(
    dataset: EdaDataset, config: ExperimentConfig
) -> pd.DataFrame:
    frame = dataset.frame
    abnormal_cfg = config.raw["preprocessing"]["abnormal_delinquency"]
    abnormal_values = list(abnormal_cfg["abnormal_values"])
    rows: list[dict[str, object]] = []
    for column in abnormal_cfg["columns"]:
        abnormal = frame[column].isin(abnormal_values)
        zero = frame[column].eq(0)
        regular_positive = frame[column].gt(0) & ~abnormal
        rows.append(
            {
                "feature": column,
                "zero_count": int(zero.sum()),
                "zero_rate": float(zero.mean()),
                "regular_positive_count": int(regular_positive.sum()),
                "regular_positive_rate": float(regular_positive.mean()),
                "abnormal_count": int(abnormal.sum()),
                "abnormal_rate": float(abnormal.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_spearman_tables(
    dataset: EdaDataset, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    predictors = list(config.predictor_columns)
    correlation = dataset.frame[predictors].corr(method="spearman")
    correlation.index.name = "feature"
    correlation = correlation.reset_index()
    if dataset.scope != TRAIN_SCOPE:
        return correlation, None
    target = config.data["target_column"]
    associations = []
    for column in predictors:
        coefficient = dataset.frame[[column, target]].corr(method="spearman").iloc[0, 1]
        associations.append(
            {
                "feature": column,
                "spearman_with_target": float(coefficient),
                "absolute_spearman": abs(float(coefficient)),
            }
        )
    return correlation, pd.DataFrame(associations).sort_values(
        "absolute_spearman", ascending=False, ignore_index=True
    )


def build_preprocessing_sensitivity(
    config_path: str | Path = "configs/experiment.yaml",
) -> pd.DataFrame:
    """Compare the two approved 96/98 strategies through the shared preprocessor."""
    config = load_config(config_path)
    delinquency = list(config.raw["preprocessing"]["abnormal_delinquency"]["columns"])
    abnormal_values = list(
        config.raw["preprocessing"]["abnormal_delinquency"]["abnormal_values"]
    )
    rows: list[dict[str, object]] = []
    for strategy in ("missing_plus_flag", "keep_raw"):
        features, metadata = transform_train_with_strategy(
            config.config_path, strategy=strategy, feature_set="B"
        )
        values = features.to_numpy(dtype="float64")
        preprocessing = metadata["preprocessing"]
        rows.append(
            {
                "strategy": strategy,
                "feature_set": "B",
                "scope": metadata["split"],
                "row_count": len(features),
                "feature_count": features.shape[1],
                "all_values_finite": bool(np.isfinite(values).all()),
                "remaining_96_98_cells": int(
                    features[delinquency].isin(abnormal_values).sum().sum()
                ),
                "has_abnormal_flag": "HasAbnormalDelinquencyCode" in features.columns,
                "abnormal_flag_count": int(
                    features.get(
                        "HasAbnormalDelinquencyCode", pd.Series(0, index=features.index)
                    ).sum()
                ),
                "training_medians_json": json.dumps(
                    preprocessing["medians"], sort_keys=True, allow_nan=False
                ),
                "manifest_sha256": metadata["manifest_sha256"],
            }
        )
    return pd.DataFrame(rows)


def load_model_sensitivity_reference(config: ExperimentConfig) -> pd.DataFrame:
    """Load and strictly validate the shared Validation-only LR comparison."""
    project_root = config.project_root
    path = project_root / "outputs" / "comparisons" / "abnormal_code_sensitivity.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Required sensitivity result is missing: "
            f"{path.relative_to(project_root).as_posix()}. "
            "Run scripts.run_lr_ablations first or use --skip-strategy-sensitivity."
        )
    frame = pd.read_csv(path)
    required = {
        "experiment_id",
        "model_name",
        "feature_set",
        "preprocessing_strategy",
        "class_weight",
        "feature_count",
        "manifest_sha256",
        "config_sha256",
        *SENSITIVITY_FINITE_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Model sensitivity output is missing columns: {sorted(missing)}")

    violations: list[str] = []
    expected_strategies = set(SENSITIVITY_EXPERIMENTS)
    actual_strategies = set(frame["preprocessing_strategy"].astype(str))
    if len(frame) != len(expected_strategies) or actual_strategies != expected_strategies:
        violations.append(
            "preprocessing_strategy must contain exactly one missing_plus_flag and one keep_raw row"
        )
    if frame["preprocessing_strategy"].duplicated().any():
        violations.append("preprocessing_strategy rows must be unique")
    if set(frame["model_name"].astype(str)) != {"logistic_regression"}:
        violations.append("model_name must be logistic_regression")
    if set(frame["feature_set"].astype(str)) != {"B"}:
        violations.append("feature_set must be B")
    if set(frame["class_weight"].astype(str)) != {"balanced"}:
        violations.append("class_weight must be balanced")
    expected_manifest = config.raw["split"]["frozen_manifest_sha256"]
    if set(frame["manifest_sha256"].astype(str)) != {expected_manifest}:
        violations.append("manifest_sha256 does not match the frozen manifest")
    if set(frame["config_sha256"].astype(str)) != {config.config_sha256}:
        violations.append("config_sha256 does not match the loaded configuration")

    if actual_strategies == expected_strategies and not frame[
        "preprocessing_strategy"
    ].duplicated().any():
        indexed = frame.set_index("preprocessing_strategy")
        for strategy, expected in SENSITIVITY_EXPERIMENTS.items():
            row = indexed.loc[strategy]
            if str(row["experiment_id"]) != expected["experiment_id"]:
                violations.append(
                    f"{strategy} experiment_id must be {expected['experiment_id']}"
                )
            feature_count = pd.to_numeric(
                pd.Series([row["feature_count"]]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(feature_count) or feature_count != expected["feature_count"]:
                violations.append(
                    f"{strategy} feature_count must be {expected['feature_count']}"
                )

    numeric = frame.loc[:, list(SENSITIVITY_FINITE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        violations.append("all configured sensitivity metrics and runtimes must be finite")
    if violations:
        raise ValueError("Model sensitivity contract violation: " + "; ".join(violations))

    result = frame.copy()
    result.loc[:, list(SENSITIVITY_FINITE_COLUMNS)] = numeric
    result.insert(0, "source_file", path.relative_to(project_root).as_posix())
    return result


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#9AA4AE",
            "axes.labelcolor": COLOR_DARK,
            "axes.titlecolor": COLOR_DARK,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_class_distribution(table: pd.DataFrame, path: Path, *, scope: str) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(
        table["class_label"], table["count"], color=[COLOR_PRIMARY, COLOR_ACCENT], width=0.58
    )
    for bar, count, rate in zip(bars, table["count"], table["rate"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}\n{rate:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title(f"Class Distribution ({scope.replace('_', ' ').title()})", fontweight="bold")
    ax.set_ylabel("Number of records")
    ax.set_ylim(0, float(table["count"].max()) * 1.16)
    ax.ticklabel_format(style="plain", axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, path)


def plot_missing_values(table: pd.DataFrame, path: Path, *, scope: str) -> None:
    _set_plot_style()
    ordered = table.sort_values("missing_rate", ascending=True)
    labels = [DISPLAY_NAMES.get(value, value) for value in ordered["feature"]]
    colors = [COLOR_ACCENT if value > 0 else COLOR_LIGHT for value in ordered["missing_rate"]]
    fig, ax = plt.subplots(figsize=(10, 6.3))
    bars = ax.barh(labels, ordered["missing_rate"] * 100, color=colors)
    for bar, count, rate in zip(
        bars, ordered["missing_count"], ordered["missing_rate"], strict=True
    ):
        ax.text(
            bar.get_width() + 0.22,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.2%} ({int(count):,})",
            va="center",
            fontsize=9,
        )
    ax.set_title(f"Missing Values by Predictor ({scope.replace('_', ' ').title()})", fontweight="bold")
    ax.set_xlabel("Missing records (%)")
    ax.set_xlim(0, max(1.0, float(ordered["missing_rate"].max() * 100) * 1.30))
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, path)


def plot_abnormal_codes(table: pd.DataFrame, path: Path, *, scope: str) -> None:
    _set_plot_style()
    pivot = table.pivot(index="feature", columns="code", values="affected_rows").fillna(0)
    pivot = pivot.reindex(
        [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ]
    )
    labels = [DISPLAY_NAMES.get(value, value) for value in pivot.index]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars_96 = ax.bar(x - width / 2, pivot.get(96, 0), width, label="Code 96", color=COLOR_PRIMARY)
    bars_98 = ax.bar(x + width / 2, pivot.get(98, 0), width, label="Code 98", color=COLOR_ACCENT)
    for bars in (bars_96, bars_98):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height()):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Affected records")
    ax.set_title(f"Abnormal Delinquency Codes 96 and 98 ({scope.replace('_', ' ').title()})", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, path)


def plot_feature_distributions(
    dataset: EdaDataset, config: ExperimentConfig, path: Path
) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(2, 5, figsize=(18, 8.2))
    for ax, column in zip(axes.flat, config.predictor_columns, strict=True):
        series = dataset.frame[column].dropna().astype("float64")
        lower = float(series.quantile(0.01))
        upper = float(series.quantile(0.99))
        if lower == upper:
            lower = float(series.min())
            upper = float(series.max())
        display = series.clip(lower=lower, upper=upper)
        ax.hist(display, bins=32, color=COLOR_PRIMARY, alpha=0.88, edgecolor="white")
        title = textwrap.fill(DISPLAY_NAMES.get(column, column), width=24)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Records")
        ax.tick_params(axis="both", labelsize=8)
        ax.text(
            0.98,
            0.95,
            "display: p1–p99",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="#59636E",
        )
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Predictor Distributions ({dataset.scope.replace('_', ' ').title()})",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, path)


def plot_long_tail_features(dataset: EdaDataset, path: Path) -> None:
    _set_plot_style()
    columns = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, column in zip(axes, columns, strict=True):
        series = dataset.frame[column].dropna().clip(lower=0).astype("float64")
        transformed = np.log1p(series)
        ax.hist(transformed, bins=45, color=COLOR_ACCENT, alpha=0.88, edgecolor="white")
        ax.set_title(DISPLAY_NAMES[column], fontweight="bold")
        ax.set_xlabel("log1p(value), display only")
        ax.set_ylabel("Records")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Long-tailed Predictors ({dataset.scope.replace('_', ' ').title()})",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, path)


def plot_delinquency_zero_concentration(
    table: pd.DataFrame, path: Path, *, scope: str
) -> None:
    _set_plot_style()
    labels = [DISPLAY_NAMES.get(value, value) for value in table["feature"]]
    x = np.arange(len(labels))
    zero = table["zero_rate"].to_numpy() * 100
    positive = table["regular_positive_rate"].to_numpy() * 100
    abnormal = table["abnormal_rate"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x, zero, label="Zero", color=COLOR_PRIMARY)
    ax.bar(x, positive, bottom=zero, label="Regular positive", color=COLOR_LIGHT)
    ax.bar(x, abnormal, bottom=zero + positive, label="Code 96/98", color=COLOR_WARNING)
    for position, rate in zip(x, zero, strict=True):
        ax.text(position, rate / 2, f"{rate:.1f}% zero", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Share of records (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Zero Concentration in Delinquency Predictors ({scope.replace('_', ' ').title()})",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    ax.spines[["top", "right"]].set_visible(False)
    _save_figure(fig, path)


def plot_spearman_correlation(
    correlation: pd.DataFrame, path: Path, *, scope: str
) -> None:
    _set_plot_style()
    matrix = correlation.set_index("feature")
    labels = [DISPLAY_NAMES.get(value, value) for value in matrix.columns]
    short_labels = [textwrap.fill(label, width=16) for label in labels]
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix.to_numpy(dtype="float64"), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(short_labels)), short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(short_labels)), short_labels, fontsize=8)
    ax.set_title(f"Spearman Correlation ({scope.replace('_', ' ').title()})", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman coefficient")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_model_sensitivity(table: pd.DataFrame, path: Path) -> None:
    """Plot shared-runner Validation metrics without recomputing any metric."""
    _set_plot_style()
    order = ["missing_plus_flag", "keep_raw"]
    indexed = table.set_index("preprocessing_strategy").loc[order]
    labels = ["Missing + flag", "Keep raw"]
    colors = [COLOR_PRIMARY, COLOR_ACCENT]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    independent_metrics = ["pr_auc", "roc_auc", "ks"]
    independent_labels = ["PR-AUC", "ROC-AUC", "KS"]
    x = np.arange(len(independent_metrics))
    width = 0.34
    for offset, (strategy_label, color, (_, row)) in enumerate(
        zip(labels, colors, indexed.iterrows(), strict=True)
    ):
        values = row[independent_metrics].to_numpy(dtype="float64")
        bars = axes[0].bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label=strategy_label,
            color=color,
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set_xticks(x, independent_labels)
    axes[0].set_ylim(0, 0.92)
    axes[0].set_ylabel("Validation metric")
    axes[0].set_title("Threshold-independent metrics", fontweight="bold")

    threshold_metrics = ["operational_precision", "operational_recall"]
    threshold_labels = ["Precision", "Recall"]
    x = np.arange(len(threshold_metrics))
    for offset, (strategy_label, color, (_, row)) in enumerate(
        zip(labels, colors, indexed.iterrows(), strict=True)
    ):
        values = row[threshold_metrics].to_numpy(dtype="float64")
        bars = axes[1].bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label=strategy_label,
            color=color,
        )
        axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[1].set_xticks(x, threshold_labels)
    axes[1].set_ylim(0, 0.86)
    axes[1].set_ylabel("Validation metric")
    axes[1].set_title("Operational threshold (recall ≥ 0.75)", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Validation-only LR Sensitivity: 96/98 Handling (Feature Set B)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, path)


def _write_bilingual_notes(
    path: Path,
    *,
    dataset: EdaDataset,
    config: ExperimentConfig,
    class_distribution: pd.DataFrame,
    missing_values: pd.DataFrame,
    abnormal_summary: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    zero_summary: pd.DataFrame,
) -> None:
    values = duplicate_summary.set_index("metric")["value"]
    positive = class_distribution.loc[class_distribution["target"] == 1].iloc[0]
    missing = missing_values.set_index("feature")
    abnormal = abnormal_summary.iloc[-1]
    zero_rates = ", ".join(
        f"{DISPLAY_NAMES.get(row.feature, row.feature)} {row.zero_rate:.1%}"
        for row in zero_summary.itertuples()
    )
    noun_en = "dataset" if dataset.scope == RAW_AUDIT_SCOPE else "frozen Training partition"
    noun_zh = "数据集" if dataset.scope == RAW_AUDIT_SCOPE else "冻结的训练分区"
    if dataset.scope == TRAIN_SCOPE:
        abnormal_en = (
            f"Delinquency codes 96/98 affect {int(abnormal['unique_affected_rows']):,} "
            f"unique rows; their serious-delinquency rate is {abnormal['positive_rate']:.2%}."
        )
        abnormal_zh = (
            f"逾期字段中的 96/98 异常码共影响 "
            f"{int(abnormal['unique_affected_rows']):,} 行，"
            f"这些记录的严重违约率为 {abnormal['positive_rate']:.2%}。"
        )
    else:
        abnormal_en = (
            f"Delinquency codes 96/98 affect {int(abnormal['unique_affected_rows']):,} "
            "unique rows and occur together across all three delinquency fields."
        )
        abnormal_zh = (
            f"逾期字段中的 96/98 异常码共影响 "
            f"{int(abnormal['unique_affected_rows']):,} 行，"
            "并在三个逾期字段中同时出现。"
        )
    content = f"""# EDA Core Findings / EDA 核心结论

Scope / 口径: `{dataset.scope}`

## English (report-ready)

- The {noun_en} contains {len(dataset.frame):,} records and {len(config.predictor_columns)} predictors.
- The positive class accounts for only {positive['rate']:.3%} ({int(positive['count']):,} records), indicating severe class imbalance.
- `MonthlyIncome` has {int(missing.loc['MonthlyIncome', 'missing_count']):,} missing values ({missing.loc['MonthlyIncome', 'missing_rate']:.2%}).
- `NumberOfDependents` has {int(missing.loc['NumberOfDependents', 'missing_count']):,} missing values ({missing.loc['NumberOfDependents', 'missing_rate']:.2%}).
- {abnormal_en}
- There are {int((dataset.frame['age'] <= 0).sum()):,} records with `age <= 0`.
- Identical predictor vectors create {int(values['duplicate_predictor_excess_rows']):,} excess duplicate rows.
- There are {int(values['conflicting_target_groups']):,} predictor-identical groups with conflicting targets.
- Revolving utilization, debt ratio, and monthly income are strongly right-skewed and long-tailed.
- The three delinquency-count predictors are zero-inflated ({zero_rates}).

## 中文（可直接用于报告）

- 该{noun_zh}包含 {len(dataset.frame):,} 条记录和 {len(config.predictor_columns)} 个预测特征。
- 正类仅占 {positive['rate']:.3%}（{int(positive['count']):,} 条），存在严重的类别不平衡。
- `MonthlyIncome` 缺失 {int(missing.loc['MonthlyIncome', 'missing_count']):,} 条，缺失率为 {missing.loc['MonthlyIncome', 'missing_rate']:.2%}。
- `NumberOfDependents` 缺失 {int(missing.loc['NumberOfDependents', 'missing_count']):,} 条，缺失率为 {missing.loc['NumberOfDependents', 'missing_rate']:.2%}。
- {abnormal_zh}
- `age <= 0` 的记录有 {int((dataset.frame['age'] <= 0).sum()):,} 条。
- 相同 predictor 向量形成 {int(values['duplicate_predictor_excess_rows']):,} 条超额重复记录。
- 存在 {int(values['conflicting_target_groups']):,} 个 predictor 完全相同但 target 不同的冲突组。
- 循环额度使用率、负债比例和月收入呈明显右偏与长尾分布。
- 三个逾期次数字段的大多数取值为 0（{zero_rates}）。

> All figures and numbers are generated by code. The raw audit is descriptive only; model, feature, and threshold choices must use the approved Training/Validation workflow.
"""
    path.write_text(content, encoding="utf-8")


def _write_scope_readme(
    path: Path,
    *,
    dataset: EdaDataset,
    config: ExperimentConfig,
    class_distribution: pd.DataFrame,
) -> None:
    positive_rate = float(class_distribution.loc[class_distribution["target"] == 1, "rate"].iloc[0])
    notice = (
        "Full labeled data are used only for the pre-declared schema, class-balance, "
        "missingness, duplicate, abnormal-code, and univariate distribution audit. "
        "Do not use raw_audit outputs for feature, model, threshold, or hyperparameter selection."
        if dataset.scope == RAW_AUDIT_SCOPE
        else "All label-conditioned results use only the frozen Training partition. "
        "Validation and Test labels are not exposed by the EDA interface."
    )
    content = f"""# EDA Output — {dataset.scope}

- Rows analyzed: {len(dataset.frame):,}
- Positive rate: {positive_rate:.4%}
- Source SHA-256: `{config.data['raw_labeled_sha256']}`
- Label usage: `{dataset.metadata['label_usage']}`

All CSV, PNG, JSON, and Markdown files in this directory were generated by `scripts/run_eda.py`.

> {notice}
"""
    path.write_text(content, encoding="utf-8")


def _hash_output(path: Path, root: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def run_eda(
    config_path: str | Path = "configs/experiment.yaml",
    *,
    scope: str,
    output_root: str | Path = "outputs/eda",
    include_strategy_sensitivity: bool = False,
) -> Path:
    """Generate a complete EDA bundle and return its scope directory."""
    validate_scope(scope)
    config, dataset = load_eda_dataset(config_path, scope=scope)
    raw_sha_before = sha256_file(config.labeled_path)
    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        output_root_path = config.project_root / output_root_path
    scope_dir = output_root_path / scope
    tables_dir = scope_dir / "tables"
    figures_dir = scope_dir / "figures"
    metadata_dir = scope_dir / "metadata"
    for directory in (tables_dir, figures_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw_sha_after_load = sha256_file(config.labeled_path)
    checks = build_validation_checks(dataset, config, raw_sha256_after=raw_sha_after_load)
    assert_validation_passed(checks)
    dataset_summary = build_dataset_summary(dataset, config)
    class_distribution = build_class_distribution(dataset, config)
    missing_values = build_missing_values(dataset, config)
    descriptive = build_descriptive_statistics(dataset, config)
    abnormal_counts, abnormal_summary = build_abnormal_code_tables(dataset, config)
    duplicate_summary = build_duplicate_summary(dataset, config)
    zero_summary = build_delinquency_zero_summary(dataset, config)
    spearman, target_spearman = build_spearman_tables(dataset, config)

    tables: dict[str, pd.DataFrame] = {
        "validation_checks.csv": checks,
        "dataset_summary.csv": dataset_summary,
        "class_distribution.csv": class_distribution,
        "missing_values.csv": missing_values,
        "descriptive_statistics.csv": descriptive,
        "abnormal_code_counts.csv": abnormal_counts,
        "abnormal_code_summary.csv": abnormal_summary,
        "duplicate_summary.csv": duplicate_summary,
        "delinquency_zero_summary.csv": zero_summary,
        "spearman_correlation.csv": spearman,
    }
    if target_spearman is not None:
        tables["target_spearman_training_only.csv"] = target_spearman
    model_reference: pd.DataFrame | None = None
    if scope == TRAIN_SCOPE and include_strategy_sensitivity:
        model_reference = load_model_sensitivity_reference(config)
        tables["preprocessing_strategy_sensitivity.csv"] = build_preprocessing_sensitivity(
            config.config_path
        )
        tables["abnormal_code_model_sensitivity.csv"] = model_reference
    written: list[Path] = []
    for filename, table in tables.items():
        path = tables_dir / filename
        table.to_csv(path, index=False, lineterminator="\n")
        written.append(path)

    figure_paths = {
        "class_distribution.png": lambda path: plot_class_distribution(
            class_distribution, path, scope=scope
        ),
        "missing_values.png": lambda path: plot_missing_values(
            missing_values, path, scope=scope
        ),
        "abnormal_delinquency_codes.png": lambda path: plot_abnormal_codes(
            abnormal_counts, path, scope=scope
        ),
        "feature_distributions.png": lambda path: plot_feature_distributions(
            dataset, config, path
        ),
        "long_tail_features.png": lambda path: plot_long_tail_features(dataset, path),
        "delinquency_zero_concentration.png": lambda path: plot_delinquency_zero_concentration(
            zero_summary, path, scope=scope
        ),
        "spearman_correlation.png": lambda path: plot_spearman_correlation(
            spearman, path, scope=scope
        ),
    }
    for filename, plotter in figure_paths.items():
        path = figures_dir / filename
        plotter(path)
        written.append(path)
    if model_reference is not None:
        path = figures_dir / "abnormal_code_sensitivity_validation.png"
        plot_model_sensitivity(model_reference, path)
        written.append(path)

    readme_path = scope_dir / "README.md"
    _write_scope_readme(
        readme_path,
        dataset=dataset,
        config=config,
        class_distribution=class_distribution,
    )
    written.append(readme_path)
    notes_path = scope_dir / "PPT_NOTES_BILINGUAL.md"
    _write_bilingual_notes(
        notes_path,
        dataset=dataset,
        config=config,
        class_distribution=class_distribution,
        missing_values=missing_values,
        abnormal_summary=abnormal_summary,
        duplicate_summary=duplicate_summary,
        zero_summary=zero_summary,
    )
    written.append(notes_path)

    raw_sha_final = sha256_file(config.labeled_path)
    if raw_sha_final != raw_sha_before:
        raise RuntimeError("Raw labeled data changed during EDA")
    checks = build_validation_checks(dataset, config, raw_sha256_after=raw_sha_final)
    assert_validation_passed(checks)
    checks.to_csv(tables_dir / "validation_checks.csv", index=False, lineterminator="\n")

    runtime = package_versions()
    runtime["matplotlib"] = matplotlib.__version__
    runtime["platform"] = platform.platform()
    metadata = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": scope,
        "analysis_row_count": len(dataset.frame),
        "label_usage": dataset.metadata["label_usage"],
        "scope_notice": (
            "Pre-declared whole-file data audit only; not for model selection."
            if scope == RAW_AUDIT_SCOPE
            else "Label-conditioned analysis is restricted to the frozen Training partition."
        ),
        "source": {
            "path": repository_relative_path(config.labeled_path, config.project_root),
            "sha256_before": raw_sha_before,
            "sha256_after": raw_sha_final,
        },
        "split_metadata": dataset.metadata,
        "contract": {
            "id_column": config.data["id_column"],
            "target_column": config.data["target_column"],
            "predictor_columns": list(config.predictor_columns),
            "abnormal_columns": list(
                config.raw["preprocessing"]["abnormal_delinquency"]["columns"]
            ),
            "abnormal_values": list(
                config.raw["preprocessing"]["abnormal_delinquency"]["abnormal_values"]
            ),
        },
        "config_sha256": config.config_sha256,
        "git_commit_sha": git_commit_sha(config.project_root),
        "runtime": runtime,
        "outputs": [_hash_output(path, scope_dir) for path in sorted(set(written))],
    }
    metadata_path = metadata_dir / "eda_run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return scope_dir


def run_scopes(
    config_path: str | Path = "configs/experiment.yaml",
    *,
    scopes: Iterable[str] = SUPPORTED_SCOPES,
    output_root: str | Path = "outputs/eda",
    include_strategy_sensitivity: bool = True,
) -> list[Path]:
    """Run a declared sequence of scopes; useful for the command-line entrypoint."""
    return [
        run_eda(
            config_path,
            scope=scope,
            output_root=output_root,
            include_strategy_sensitivity=(
                include_strategy_sensitivity and scope == TRAIN_SCOPE
            ),
        )
        for scope in scopes
    ]
