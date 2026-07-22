"""Read-only access to the frozen raw Training partition for EDA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.runner import SharedExperimentRunner
from src.features.feature_sets import build_feature_set
from src.features.preprocessing import CreditRiskPreprocessor


@dataclass(frozen=True)
class TrainPartition:
    predictors: pd.DataFrame
    target: pd.Series
    row_id: pd.Series
    feature_hash_v1: pd.Series
    split_metadata: dict[str, Any]


def load_train_partition(
    config_path: str | Path = "configs/experiment.yaml",
) -> TrainPartition:
    """Return copies of raw Training data only after frozen-manifest validation."""
    runner = SharedExperimentRunner(config_path)
    return _partition_from_runner(runner)


def _partition_from_runner(runner: SharedExperimentRunner) -> TrainPartition:
    train = runner.partitions["train"]
    config = runner.config
    metadata = {
        "split": "train",
        "data_level": "raw predictors before preprocessing",
        "row_count": len(train),
        "manifest_sha256": runner.expected_manifest_sha256,
        "feature_hash_version": config.raw["feature_hash"]["version"],
        "source_path": config.data["labeled_path"],
    }
    return TrainPartition(
        predictors=train.loc[:, list(config.predictor_columns)].copy(deep=True),
        target=train[config.data["target_column"]].copy(deep=True),
        row_id=train[config.data["id_column"]].copy(deep=True),
        feature_hash_v1=train["feature_hash_v1"].copy(deep=True),
        split_metadata=metadata,
    )


def transform_train_with_strategy(
    config_path: str | Path = "configs/experiment.yaml",
    *,
    strategy: str,
    feature_set: str = "A",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the shared preprocessor on raw Training and return Training features only."""
    runner = SharedExperimentRunner(config_path)
    partition = _partition_from_runner(runner)
    config = runner.config
    preprocessing = config.raw["preprocessing"]
    processor = CreditRiskPreprocessor(
        predictor_columns=config.predictor_columns,
        delinquency_columns=preprocessing["abnormal_delinquency"]["columns"],
        abnormal_values=preprocessing["abnormal_delinquency"]["abnormal_values"],
        strategy=strategy,
        clipping_enabled=preprocessing["winsorization"]["enabled"],
        clipping_columns=preprocessing["winsorization"]["columns"],
        upper_quantile=preprocessing["winsorization"]["upper_quantile"],
    )
    cleaned = processor.fit_transform(partition.predictors)
    features = build_feature_set(cleaned, feature_set, config.predictor_columns, strategy)
    metadata = {
        **partition.split_metadata,
        "preprocessing": processor.metadata(),
        "feature_set": feature_set,
        "feature_count": features.shape[1],
        "feature_names": list(features.columns),
    }
    return features, metadata
