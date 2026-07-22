"""Shared frozen-manifest runner for Validation-only experiments."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig, load_config
from src.data.feature_hash import compute_feature_hashes
from src.data.load_data import load_labeled_data
from src.data.manifest import validate_manifest
from src.evaluation.metrics import threshold_independent_metrics, threshold_metrics
from src.evaluation.thresholding import select_operational_threshold, threshold_table
from src.experiments.contracts import ExperimentSpec
from src.experiments.result_schema import ExperimentResult
from src.features.feature_sets import build_feature_set, quality_flags_for_strategy
from src.features.preprocessing import CreditRiskPreprocessor
from src.models.logistic_regression import build_logistic_pipeline
from src.utils.reproducibility import git_commit_sha, package_versions, set_random_seeds, sha256_file


@dataclass(frozen=True)
class ExperimentArtifacts:
    result: ExperimentResult
    validation_metrics: dict[str, object]
    thresholds: pd.DataFrame


class SharedExperimentRunner:
    """Load immutable data once and run multiple Validation-only specifications."""

    def __init__(self, config_path: str | Path = "configs/experiment.yaml") -> None:
        self.config: ExperimentConfig = load_config(config_path)
        self.manifest_path = self.config.project_root / self.config.raw["split"]["manifest_path"]
        self.expected_manifest_sha256 = self.config.raw["split"]["frozen_manifest_sha256"]
        self._verify_manifest_file(self.manifest_path, self.expected_manifest_sha256)
        self.raw = load_labeled_data(self.config)
        self.raw["feature_hash_v1"] = compute_feature_hashes(
            self.raw, self.config.predictor_columns
        )
        self.manifest = pd.read_csv(
            self.manifest_path, dtype={"feature_hash_v1": "string"}
        )
        validate_manifest(self.manifest, self.raw, self.config)
        joined = self.raw.merge(
            self.manifest[["row_id", "split"]], on="row_id", validate="one_to_one"
        )
        self.partitions = {
            name: joined[joined["split"] == name].copy()
            for name in ("train", "validation", "test")
        }

    @staticmethod
    def _verify_manifest_file(path: Path, expected_sha256: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen split manifest does not exist: {path}")
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(
                f"Frozen manifest SHA-256 mismatch: expected {expected_sha256}, got {actual}. "
                "Stage 3 will not regenerate or replace the manifest."
            )

    def run(self, spec: ExperimentSpec) -> ExperimentArtifacts:
        self._validate_spec_paths(spec)
        set_random_seeds(spec.random_seed)
        config = self.config
        preprocessing_cfg = config.raw["preprocessing"]
        cleaner = CreditRiskPreprocessor(
            predictor_columns=config.predictor_columns,
            delinquency_columns=preprocessing_cfg["abnormal_delinquency"]["columns"],
            abnormal_values=preprocessing_cfg["abnormal_delinquency"]["abnormal_values"],
            strategy=spec.preprocessing_strategy,
            clipping_enabled=preprocessing_cfg["winsorization"]["enabled"],
            clipping_columns=preprocessing_cfg["winsorization"]["columns"],
            upper_quantile=preprocessing_cfg["winsorization"]["upper_quantile"],
        )
        cleaned_train = cleaner.fit_transform(self.partitions["train"])
        cleaned_validation = cleaner.transform(self.partitions["validation"])
        cleaned_test = cleaner.transform(self.partitions["test"])
        x_train = build_feature_set(
            cleaned_train, spec.feature_set, config.predictor_columns, spec.preprocessing_strategy
        )
        x_validation = build_feature_set(
            cleaned_validation,
            spec.feature_set,
            config.predictor_columns,
            spec.preprocessing_strategy,
        )
        x_test_schema_only = build_feature_set(
            cleaned_test, spec.feature_set, config.predictor_columns, spec.preprocessing_strategy
        )
        if not (
            tuple(x_train.columns)
            == tuple(x_validation.columns)
            == tuple(x_test_schema_only.columns)
        ):
            raise RuntimeError("Train/Validation/Test transformed schemas differ")
        if not np.isfinite(x_test_schema_only.to_numpy()).all():
            raise RuntimeError("Test transform contains NaN or infinity")

        parameters = dict(config.raw["models"]["logistic_regression"]["baseline"])
        parameters["class_weight"] = None if spec.class_weight == "none" else "balanced"
        parameters["random_state"] = spec.random_seed
        pipeline = build_logistic_pipeline(
            parameters,
            feature_names=x_train.columns,
            binary_feature_names=quality_flags_for_strategy(spec.preprocessing_strategy),
        )
        target = config.data["target_column"]
        fit_started = time.perf_counter()
        pipeline.fit(x_train, self.partitions["train"][target].to_numpy())
        training_time = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        validation_probability = pipeline.predict_proba(x_validation)[:, 1]
        inference_time = time.perf_counter() - inference_started
        independent = threshold_independent_metrics(
            self.partitions["validation"][target],
            validation_probability,
            split_name="validation",
        )
        default = threshold_metrics(
            self.partitions["validation"][target],
            validation_probability,
            0.5,
            split_name="validation",
        )
        recall_minimum = float(
            config.raw["thresholds"]["operational"]["recall_minimum"]
        )
        thresholds = threshold_table(
            self.partitions["validation"][target],
            validation_probability,
            recall_minimum=recall_minimum,
            split_name="validation",
        )
        operational_threshold = select_operational_threshold(thresholds)
        operational = threshold_metrics(
            self.partitions["validation"][target],
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
        result = ExperimentResult(
            experiment_id=spec.experiment_id,
            model_name=spec.model_name,
            feature_set=spec.feature_set,
            preprocessing_strategy=spec.preprocessing_strategy,
            class_weight=spec.class_weight,
            random_seed=spec.random_seed,
            feature_count=x_train.shape[1],
            feature_names=list(x_train.columns),
            train_row_count=len(x_train),
            validation_row_count=len(x_validation),
            validation_metrics=independent,
            default_threshold_metrics=default,
            operational_threshold=operational_threshold,
            operational_threshold_metrics=operational,
            training_time_seconds=training_time,
            validation_inference_time_seconds=inference_time,
            raw_data_sha256=config.data["raw_labeled_sha256"],
            manifest_sha256=self.expected_manifest_sha256,
            config_sha256=config.config_sha256,
            git_commit_sha=git_commit_sha(config.project_root),
            package_versions=package_versions(),
            preprocessing_metadata=cleaner.metadata(),
            test_access="transform/schema/finite checks only; no Test prediction or metrics",
        )
        result.to_dict()
        return ExperimentArtifacts(result, validation_metrics, thresholds)

    def persist(self, artifacts: ExperimentArtifacts) -> dict[str, Path]:
        output_dir = (
            self.config.project_root / "outputs" / "experiments" / artifacts.result.experiment_id
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "validation_metrics": output_dir / "validation_metrics.json",
            "thresholds": output_dir / "thresholds.csv",
            "run_metadata": output_dir / "run_metadata.json",
        }
        paths["validation_metrics"].write_text(
            json.dumps(artifacts.validation_metrics, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        artifacts.thresholds.to_csv(paths["thresholds"], index=False, lineterminator="\n")
        paths["run_metadata"].write_text(
            json.dumps(artifacts.result.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return paths

    def _validate_spec_paths(self, spec: ExperimentSpec) -> None:
        if spec.config_path.expanduser().resolve() != self.config.config_path:
            raise ValueError("ExperimentSpec config_path differs from the loaded runner config")
        if spec.manifest_path.expanduser().resolve() != self.manifest_path.resolve():
            raise ValueError("ExperimentSpec must use the frozen configured manifest path")
        if spec.expected_manifest_sha256 != self.expected_manifest_sha256:
            raise ValueError("ExperimentSpec expected manifest SHA differs from frozen configuration")
        self._verify_manifest_file(spec.manifest_path.expanduser().resolve(), spec.expected_manifest_sha256)
