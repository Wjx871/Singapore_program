"""Validated, immutable experiment request contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    model_name: str
    feature_set: str
    preprocessing_strategy: str
    class_weight: str
    random_seed: int
    config_path: Path
    manifest_path: Path
    expected_manifest_sha256: str
    evaluation_split: str = "validation"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.experiment_id):
            raise ValueError("experiment_id must use lowercase letters, digits, hyphens, or underscores")
        if self.model_name != "logistic_regression":
            raise ValueError("Stage 3 runner currently supports only logistic_regression")
        if self.feature_set not in {"A", "B"}:
            raise ValueError("Stage 3 feature_set must be A or B")
        if self.preprocessing_strategy not in {"missing_plus_flag", "keep_raw"}:
            raise ValueError("preprocessing_strategy must be missing_plus_flag or keep_raw")
        if self.class_weight not in {"balanced", "none"}:
            raise ValueError("class_weight must be balanced or none")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.evaluation_split != "validation":
            raise ValueError("Stage 3 ExperimentSpec permits Validation evaluation only")
        if not re.fullmatch(r"[0-9a-f]{64}", self.expected_manifest_sha256):
            raise ValueError("expected_manifest_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
