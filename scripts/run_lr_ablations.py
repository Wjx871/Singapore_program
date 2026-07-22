#!/usr/bin/env python3
"""Run the locked Stage 3 Logistic Regression ablation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.experiments.contracts import ExperimentSpec
from src.experiments.result_schema import ExperimentResult
from src.experiments.runner import SharedExperimentRunner
from src.utils.logging_utils import configure_experiment_logging
from src.utils.reproducibility import sha256_file


def result_to_summary(result: ExperimentResult) -> dict[str, object]:
    return {
        "experiment_id": result.experiment_id,
        "model_name": result.model_name,
        "feature_set": result.feature_set,
        "preprocessing_strategy": result.preprocessing_strategy,
        "class_weight": result.class_weight,
        "feature_count": result.feature_count,
        "pr_auc": result.validation_metrics["pr_auc"],
        "roc_auc": result.validation_metrics["roc_auc"],
        "ks": result.validation_metrics["ks"],
        "default_precision": result.default_threshold_metrics["precision"],
        "default_recall": result.default_threshold_metrics["recall"],
        "default_f1": result.default_threshold_metrics["f1"],
        "default_predicted_positive_rate": result.default_threshold_metrics[
            "predicted_positive_rate"
        ],
        "operational_threshold": result.operational_threshold,
        "operational_precision": result.operational_threshold_metrics["precision"],
        "operational_recall": result.operational_threshold_metrics["recall"],
        "operational_f1": result.operational_threshold_metrics["f1"],
        "operational_predicted_positive_rate": result.operational_threshold_metrics[
            "predicted_positive_rate"
        ],
        "training_time_seconds": result.training_time_seconds,
        "validation_inference_time_seconds": result.validation_inference_time_seconds,
        "manifest_sha256": result.manifest_sha256,
        "config_sha256": result.config_sha256,
        "git_commit_sha": result.git_commit_sha,
    }


def write_comparison_outputs(summary: pd.DataFrame, root: Path) -> dict[str, Path]:
    output_dir = root / "outputs" / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = {
        "lr_feature_set_comparison.csv": [
            "lr_a_balanced_missing_flag",
            "lr_b_balanced_missing_flag",
        ],
        "lr_class_weight_ablation.csv": [
            "lr_b_none_missing_flag",
            "lr_b_balanced_missing_flag",
        ],
        "abnormal_code_sensitivity.csv": [
            "lr_b_balanced_missing_flag",
            "lr_b_balanced_keep_raw",
        ],
        "lr_stage3_summary.csv": list(summary["experiment_id"]),
    }
    paths: dict[str, Path] = {}
    for filename, experiment_ids in selections.items():
        path = output_dir / filename
        indexed = summary.set_index("experiment_id")
        comparison = indexed.loc[experiment_ids].reset_index()
        comparison.to_csv(path, index=False, lineterminator="\n")
        paths[filename] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    logger, log_path = configure_experiment_logging(config.project_root, stage=3)
    raw_hash_before = sha256_file(config.labeled_path)
    runner = SharedExperimentRunner(config.config_path)
    summary_rows: list[dict[str, object]] = []
    for item in config.raw["experiments"]:
        spec = ExperimentSpec(
            experiment_id=item["id"],
            model_name=item["model"],
            feature_set=item["feature_set"],
            preprocessing_strategy=item["preprocessing_strategy"],
            class_weight=item["class_weight"],
            random_seed=config.raw["reproducibility"]["random_seed"],
            config_path=config.config_path,
            manifest_path=runner.manifest_path,
            expected_manifest_sha256=runner.expected_manifest_sha256,
        )
        logger.info("running experiment=%s", spec.experiment_id)
        artifacts = runner.run(spec)
        paths = runner.persist(artifacts)
        summary_rows.append(result_to_summary(artifacts.result))
        logger.info(
            "completed experiment=%s pr_auc=%.12g operational_threshold=%.12g outputs=%s",
            spec.experiment_id,
            artifacts.result.validation_metrics["pr_auc"],
            artifacts.result.operational_threshold,
            paths,
        )
    summary = pd.DataFrame(summary_rows)
    comparison_paths = write_comparison_outputs(summary, config.project_root)
    if sha256_file(config.labeled_path) != raw_hash_before:
        raise RuntimeError("Raw labeled data changed during Stage 3 experiments")
    logger.info(
        "stage3 complete manifest_sha=%s test_metrics=false comparison_outputs=%s",
        runner.expected_manifest_sha256,
        comparison_paths,
    )
    print(
        json.dumps(
            {
                "experiments": summary_rows,
                "comparison_outputs": {key: str(value) for key, value in comparison_paths.items()},
                "log_path": str(log_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
