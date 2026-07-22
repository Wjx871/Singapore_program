from __future__ import annotations

import numpy as np

from src.analysis.train_only import load_train_partition, transform_train_with_strategy


def test_loader_returns_only_frozen_raw_training(experiment_config):
    partition = load_train_partition(experiment_config.config_path)
    assert len(partition.predictors) == len(partition.target) == 95995
    assert partition.split_metadata["split"] == "train"
    assert partition.split_metadata["data_level"] == "raw predictors before preprocessing"
    assert partition.split_metadata["manifest_sha256"] == (
        experiment_config.raw["split"]["frozen_manifest_sha256"]
    )
    assert list(partition.predictors.columns) == list(experiment_config.predictor_columns)
    assert partition.row_id.is_unique
    assert partition.feature_hash_v1.nunique() == 95586


def test_transform_reuses_shared_preprocessor(experiment_config):
    missing_features, missing_metadata = transform_train_with_strategy(
        experiment_config.config_path, strategy="missing_plus_flag", feature_set="A"
    )
    raw_features, raw_metadata = transform_train_with_strategy(
        experiment_config.config_path, strategy="keep_raw", feature_set="A"
    )
    assert missing_features.shape == (95995, 14)
    assert raw_features.shape == (95995, 13)
    assert "HasAbnormalDelinquencyCode" in missing_features
    assert "HasAbnormalDelinquencyCode" not in raw_features
    assert missing_metadata["preprocessing"]["abnormal_delinquency_strategy"] == (
        "missing_plus_flag"
    )
    assert raw_metadata["preprocessing"]["abnormal_delinquency_strategy"] == "keep_raw"
    assert np.isfinite(missing_features.to_numpy()).all()
    assert np.isfinite(raw_features.to_numpy()).all()
