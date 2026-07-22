#!/usr/bin/env python3
"""Run E1 Logistic Regression on Train/Validation while keeping Test sealed."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.feature_hash import compute_feature_hashes
from src.data.load_data import load_labeled_data
from src.data.manifest import manifest_sha256, validate_manifest
from src.evaluation.metrics import threshold_independent_metrics, threshold_metrics
from src.evaluation.thresholding import select_operational_threshold, threshold_table
from src.features.feature_sets import QUALITY_FLAGS, build_feature_set
from src.features.preprocessing import CreditRiskPreprocessor
from src.models.logistic_regression import build_logistic_pipeline
from src.utils.logging_utils import configure_stage2_logging
from src.utils.reproducibility import git_commit_sha, package_versions, set_random_seeds, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    logger, log_path = configure_stage2_logging(config.project_root)
    seed = int(config.raw["reproducibility"]["random_seed"])
    set_random_seeds(seed)
    raw_hash_before = sha256_file(config.labeled_path)
    raw = load_labeled_data(config)
    raw["feature_hash_v1"] = compute_feature_hashes(raw, config.predictor_columns)
    manifest_path = config.project_root / config.raw["split"]["manifest_path"]
    manifest = pd.read_csv(manifest_path, dtype={"feature_hash_v1": "string"})
    validate_manifest(manifest, raw, config)
    manifest_hash = manifest_sha256(manifest)
    joined = raw.merge(manifest[["row_id", "split"]], on="row_id", validate="one_to_one")
    partitions = {name: joined[joined["split"] == name].copy() for name in ("train", "validation", "test")}
    target = config.data["target_column"]
    preprocessing = config.raw["preprocessing"]
    cleaner = CreditRiskPreprocessor(
        predictor_columns=config.predictor_columns,
        delinquency_columns=preprocessing["abnormal_delinquency"]["columns"],
        abnormal_values=preprocessing["abnormal_delinquency"]["abnormal_values"],
        clipping_enabled=preprocessing["winsorization"]["enabled"],
        clipping_columns=preprocessing["winsorization"]["columns"],
        upper_quantile=preprocessing["winsorization"]["upper_quantile"],
    )
    cleaned_train = cleaner.fit_transform(partitions["train"])
    cleaned_validation = cleaner.transform(partitions["validation"])
    cleaned_test = cleaner.transform(partitions["test"])
    x_train = build_feature_set(cleaned_train, "A", config.predictor_columns)
    x_validation = build_feature_set(cleaned_validation, "A", config.predictor_columns)
    x_test_schema_only = build_feature_set(cleaned_test, "A", config.predictor_columns)
    if tuple(x_train.columns) != tuple(x_validation.columns) or tuple(x_train.columns) != tuple(x_test_schema_only.columns):
        raise RuntimeError("Train/Validation/Test transformed schemas differ")
    if not np.isfinite(x_test_schema_only.to_numpy()).all():
        raise RuntimeError("Test transform contains NaN or infinity")

    parameters = config.raw["models"]["logistic_regression"]["baseline"]
    pipeline = build_logistic_pipeline(
        parameters, feature_names=x_train.columns, binary_feature_names=QUALITY_FLAGS
    )
    fit_started = time.perf_counter()
    pipeline.fit(x_train, partitions["train"][target].to_numpy())
    training_time = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    validation_probability = pipeline.predict_proba(x_validation)[:, 1]
    validation_inference_time = time.perf_counter() - inference_started
    independent = threshold_independent_metrics(
        partitions["validation"][target], validation_probability, split_name="validation"
    )
    default = threshold_metrics(
        partitions["validation"][target], validation_probability, 0.5, split_name="validation"
    )
    recall_minimum = float(config.raw["thresholds"]["operational"]["recall_minimum"])
    thresholds = threshold_table(
        partitions["validation"][target],
        validation_probability,
        recall_minimum=recall_minimum,
        split_name="validation",
    )
    operational_threshold = select_operational_threshold(thresholds)
    operational = threshold_metrics(
        partitions["validation"][target],
        validation_probability,
        operational_threshold,
        split_name="validation",
    )
    validation_metrics = {
        "split": "validation",
        "positive_label": 1,
        "threshold_independent": independent,
        "default_threshold": default,
        "operational_threshold": operational,
    }
    run_metadata = {
        "stage": 2,
        "experiment_id": "E1_logistic_regression_feature_set_A",
        "model_name": "logistic_regression",
        "feature_set": "A",
        "parameters": parameters,
        "class_weight": parameters["class_weight"],
        "random_seed": seed,
        "raw_data_sha256": raw_hash_before,
        "config_sha256": config.config_sha256,
        "manifest_sha256": manifest_hash,
        "git_commit_sha": git_commit_sha(config.project_root),
        "package_versions": package_versions(),
        "train_row_count": len(partitions["train"]),
        "validation_row_count": len(partitions["validation"]),
        "test_row_count": len(partitions["test"]),
        "feature_count": x_train.shape[1],
        "feature_names": list(x_train.columns),
        "preprocessing": cleaner.metadata(),
        "training_time_seconds": training_time,
        "validation_inference_time_seconds": validation_inference_time,
        "validation_metrics": validation_metrics,
        "default_threshold": 0.5,
        "operational_threshold": operational_threshold,
        "recall_constraint": recall_minimum,
        "threshold_selection_rule": "maximize precision subject to recall >= 0.75; ties: recall, threshold",
        "test_evaluation_enabled": False,
        "test_access": "transform/schema/finite checks only; no Test prediction or metrics",
        "log_path": str(log_path.relative_to(config.project_root)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_paths = {
        "metrics": config.output_path("logistic_validation_metrics"),
        "thresholds": config.output_path("logistic_validation_thresholds"),
        "run": config.output_path("logistic_run_metadata"),
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    output_paths["metrics"].write_text(json.dumps(validation_metrics, indent=2) + "\n", encoding="utf-8")
    thresholds.to_csv(output_paths["thresholds"], index=False, lineterminator="\n")
    output_paths["run"].write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    raw_hash_after = sha256_file(config.labeled_path)
    if raw_hash_after != raw_hash_before:
        raise RuntimeError("Raw labeled data changed during the Logistic Regression run")
    logger.info("stage=2 branch=feat/core-pipeline-lr commit=%s", run_metadata["git_commit_sha"])
    logger.info("raw_hash=%s config_hash=%s manifest_hash=%s", raw_hash_before, config.config_sha256, manifest_hash)
    logger.info("rows train=%d validation=%d test=%d features=%d", len(partitions["train"]), len(partitions["validation"]), len(partitions["test"]), x_train.shape[1])
    logger.info("validation_metrics=%s", json.dumps(validation_metrics, sort_keys=True))
    logger.info("operational_threshold=%.12g recall_constraint=%.3f", operational_threshold, recall_minimum)
    logger.info("test_evaluation_enabled=false; Test transform/schema/finite checks passed")
    logger.info("outputs=%s elapsed_fit=%.6f elapsed_validation_inference=%.6f", output_paths, training_time, validation_inference_time)


if __name__ == "__main__":
    main()
