"""Strict loader for the locked Stage 2 experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.utils.paths import project_root_from_config, resolve_repository_path
from src.utils.reproducibility import stable_json_sha256

EXPECTED_CORE_MODELS = (
    "logistic_regression",
    "random_forest",
    "balanced_random_forest",
    "xgboost",
)


def _require(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"Missing required configuration field: {path}")
        current = current[part]
    return current


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    config_path: Path
    project_root: Path
    config_sha256: str

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def predictor_columns(self) -> tuple[str, ...]:
        return tuple(self.data["predictor_columns"])

    @property
    def labeled_path(self) -> Path:
        return resolve_repository_path(
            self.project_root, self.data["labeled_path"], must_exist=True
        )

    @property
    def official_test_path(self) -> Path:
        return resolve_repository_path(
            self.project_root,
            self.data["official_unlabeled_test_path"],
            must_exist=True,
        )

    def output_path(self, key: str) -> Path:
        return resolve_repository_path(
            self.project_root, self.raw["outputs"][key], must_exist=False
        )


def validate_config(raw: dict[str, Any]) -> None:
    required = [
        "data.labeled_path",
        "data.official_unlabeled_test_path",
        "data.raw_labeled_sha256",
        "data.target_column",
        "data.source_id_column",
        "data.id_column",
        "data.predictor_columns",
        "reproducibility.random_seed",
        "reproducibility.split_seed_test",
        "reproducibility.split_seed_validation",
        "preprocessing.abnormal_delinquency.abnormal_values",
        "preprocessing.abnormal_delinquency_strategy",
        "feature_hash.version",
        "split.ratios",
        "split.frozen_manifest_sha256",
        "feature_sets",
        "thresholds.operational.recall_minimum",
        "metrics.primary",
        "metrics.pr_auc_implementation",
        "models.core_models",
        "models.optional_models",
        "test_access.test_evaluation_enabled",
        "preprocessing.winsorization.enabled",
        "outputs.raw_data_summary",
        "outputs.logistic_validation_metrics",
        "outputs.logistic_validation_thresholds",
        "outputs.logistic_run_metadata",
        "experiments",
    ]
    for path in required:
        _require(raw, path)

    ratios = _require(raw, "split.ratios")
    expected_splits = {"train", "validation", "test"}
    if set(ratios) != expected_splits or abs(sum(float(v) for v in ratios.values()) - 1.0) > 1e-12:
        raise ValueError("split.ratios must contain train/validation/test and sum to 1")

    predictors = list(_require(raw, "data.predictor_columns"))
    if len(predictors) != 10 or len(set(predictors)) != len(predictors):
        raise ValueError("data.predictor_columns must contain 10 unique columns")
    target = _require(raw, "data.target_column")
    source_id = _require(raw, "data.source_id_column")
    if target in predictors:
        raise ValueError("Target column must not appear in predictor columns")
    if source_id in predictors or _require(raw, "data.id_column") in predictors:
        raise ValueError("Source/row ID must not appear in predictor columns")

    core_models = tuple(_require(raw, "models.core_models"))
    if len(core_models) != len(set(core_models)):
        raise ValueError("models.core_models must not contain duplicates")
    if set(core_models) != set(EXPECTED_CORE_MODELS):
        raise ValueError(f"models.core_models must be exactly {EXPECTED_CORE_MODELS}")
    if "lightgbm" in core_models:
        raise ValueError("LightGBM must not be a core model")
    optional_models = set(_require(raw, "models.optional_models"))
    if optional_models.intersection(core_models):
        raise ValueError("Optional models must not overlap core models")
    if "lightgbm" not in optional_models:
        raise ValueError("LightGBM must remain explicitly optional/legacy")

    recall = float(_require(raw, "thresholds.operational.recall_minimum"))
    if not 0.0 <= recall <= 1.0:
        raise ValueError("Operational recall constraint must be between 0 and 1")
    if _require(raw, "metrics.primary") != "pr_auc":
        raise ValueError("Primary metric must be pr_auc")
    if _require(raw, "metrics.pr_auc_implementation") != "average_precision_score":
        raise ValueError("PR-AUC implementation must be average_precision_score")
    if _require(raw, "test_access.test_evaluation_enabled") is not False:
        raise ValueError("Stage 2 requires test_evaluation_enabled: false")
    if _require(raw, "preprocessing.winsorization.enabled") is not False:
        raise ValueError("Stage 2 baseline requires clipping/winsorization disabled")
    strategy = _require(raw, "preprocessing.abnormal_delinquency_strategy")
    if strategy not in {"missing_plus_flag", "keep_raw"}:
        raise ValueError(
            "preprocessing.abnormal_delinquency_strategy must be missing_plus_flag or keep_raw"
        )
    if _require(raw, "feature_hash.version") != "feature_hash_v1":
        raise ValueError("Feature hash version must be feature_hash_v1")
    frozen_manifest = _require(raw, "split.frozen_manifest_sha256")
    if not isinstance(frozen_manifest, str) or len(frozen_manifest) != 64:
        raise ValueError("split.frozen_manifest_sha256 must be a 64-character SHA-256")
    experiments = _require(raw, "experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a non-empty list")
    experiment_ids = [item.get("id") for item in experiments if isinstance(item, Mapping)]
    if len(experiment_ids) != len(experiments) or len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("experiments must contain unique IDs")
    for item in experiments:
        if item.get("model") not in EXPECTED_CORE_MODELS:
            raise ValueError(
                f"Stage 3 experiment matrix supports only core models: {EXPECTED_CORE_MODELS}"
            )
        if item.get("feature_set") not in {"A", "B"}:
            raise ValueError("Stage 3 experiment feature_set must be A or B")
        if item.get("preprocessing_strategy") not in {"missing_plus_flag", "keep_raw"}:
            raise ValueError("Stage 3 experiment preprocessing strategy is invalid")
        if item.get("class_weight") not in {"balanced", "none"}:
            raise ValueError("Stage 3 experiment class_weight must be balanced or none")


def load_config(path: str | Path = "configs/experiment.yaml") -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Experiment config root must be a mapping")
    validate_config(raw)
    root = project_root_from_config(config_path)
    for configured in (
        raw["data"]["labeled_path"],
        raw["data"]["official_unlabeled_test_path"],
    ):
        resolve_repository_path(root, configured, must_exist=True)
    return ExperimentConfig(raw, config_path, root, stable_json_sha256(raw))
