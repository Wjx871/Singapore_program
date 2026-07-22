"""Read-only loading and validation of the Kaggle raw CSV files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ExperimentConfig
from src.data.feature_hash import compute_feature_hashes
from src.utils.reproducibility import package_versions, sha256_file


def _expected_columns(config: ExperimentConfig) -> list[str]:
    return [config.data["source_id_column"], config.data["target_column"], *config.predictor_columns]


def load_labeled_data(config: ExperimentConfig) -> pd.DataFrame:
    path = config.labeled_path
    before = sha256_file(path)
    expected_hash = config.data["raw_labeled_sha256"]
    if before != expected_hash:
        raise ValueError(f"Raw labeled data SHA-256 mismatch: expected {expected_hash}, got {before}")
    frame = pd.read_csv(path)
    after = sha256_file(path)
    if after != before:
        raise RuntimeError("Raw labeled data changed while it was being read")
    if frame.shape != (
        int(config.data["expected_labeled_rows"]),
        int(config.data["expected_raw_columns"]),
    ):
        raise ValueError(f"Unexpected labeled data shape: {frame.shape}")
    expected = _expected_columns(config)
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected labeled schema: {list(frame.columns)}")
    target = config.data["target_column"]
    if set(frame[target].dropna().unique()) != {0, 1} or frame[target].isna().any():
        raise ValueError("Labeled target must contain only 0 and 1 with no missing values")
    source_id = config.data["source_id_column"]
    row_id = config.data["id_column"]
    frame = frame.rename(columns={source_id: row_id})
    if frame[row_id].isna().any() or not frame[row_id].is_unique:
        raise ValueError("row_id must be non-missing and unique")
    frame[row_id] = frame[row_id].astype("int64")
    return frame


def validate_official_test_schema(config: ExperimentConfig) -> dict[str, Any]:
    frame = pd.read_csv(config.official_test_path)
    if len(frame) != int(config.data["expected_official_test_rows"]):
        raise ValueError(f"Unexpected official test row count: {len(frame)}")
    if list(frame.columns) != _expected_columns(config):
        raise ValueError("Official Kaggle test schema differs from labeled raw data")
    if not frame[config.data["target_column"]].isna().all():
        raise ValueError("Official Kaggle test target must be entirely unlabeled")
    return {"row_count": len(frame), "column_count": frame.shape[1], "target_all_missing": True}


def build_raw_data_summary(frame: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    target = config.data["target_column"]
    predictors = list(config.predictor_columns)
    delinquency = config.raw["preprocessing"]["abnormal_delinquency"]
    abnormal_values = set(delinquency["abnormal_values"])
    abnormal_counts = {
        column: int(frame[column].isin(abnormal_values).sum()) for column in delinquency["columns"]
    }
    hashes = compute_feature_hashes(frame, predictors)
    target_by_hash = pd.DataFrame({"hash": hashes, "target": frame[target]}).groupby("hash")["target"]
    conflicting_sizes = target_by_hash.agg(["nunique", "size"])
    conflicting = conflicting_sizes[conflicting_sizes["nunique"] > 1]
    positive_count = int(frame[target].sum())
    return {
        "source_path": config.data["labeled_path"],
        "sha256": config.data["raw_labeled_sha256"],
        "row_count": len(frame),
        "column_count": int(config.data["expected_raw_columns"]),
        "predictor_count": len(predictors),
        "target_column": target,
        "source_id_column": config.data["source_id_column"],
        "positive_count": positive_count,
        "negative_count": len(frame) - positive_count,
        "positive_rate": positive_count / len(frame),
        "missing_counts": {column: int(frame[column].isna().sum()) for column in predictors},
        "missing_rates": {column: float(frame[column].isna().mean()) for column in predictors},
        "abnormal_code_counts": abnormal_counts,
        "age_invalid_count": int((frame["age"] <= 0).sum()),
        "duplicate_statistics": {
            "unique_predictor_groups": int(hashes.nunique()),
            "duplicate_group_count": int((hashes.value_counts() > 1).sum()),
            "duplicate_group_rows": int(hashes.isin(hashes.value_counts()[lambda x: x > 1].index).sum()),
            "conflicting_target_group_count": len(conflicting),
            "conflicting_target_row_count": int(conflicting["size"].sum()),
        },
        "package_versions": package_versions(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
