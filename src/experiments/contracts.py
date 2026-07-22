"""Validated, immutable experiment request contract."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.contracts import validate_model_parameters
from src.models.registry import get_model_definition


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    model_name: str
    class_weight: str
    random_seed: int
    config_path: Path
    manifest_path: Path
    expected_manifest_sha256: str
    feature_set: str = "B"
    preprocessing_strategy: str = "missing_plus_flag"
    evaluation_split: str = "validation"
    model_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.experiment_id):
            raise ValueError("experiment_id must use lowercase letters, digits, hyphens, or underscores")
        definition = get_model_definition(self.model_name)
        if not definition.core_model:
            raise ValueError(f"{self.model_name} is optional/legacy and is not a core experiment model")
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
        if not isinstance(self.model_parameters, dict):
            raise TypeError("model_parameters must be a dictionary")
        parameters = validate_model_parameters(self.model_name, self.model_parameters)
        json.dumps(parameters, allow_nan=False)
        object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "model_parameters", parameters)

    @property
    def model_family(self) -> str:
        return get_model_definition(self.model_name).family

    @property
    def model_status(self) -> str:
        return get_model_definition(self.model_name).status

    @property
    def imbalance_strategy(self) -> str:
        return get_model_definition(self.model_name).imbalance_strategy
