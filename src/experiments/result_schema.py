"""JSON-safe unified Validation experiment result."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    model_name: str
    feature_set: str
    preprocessing_strategy: str
    class_weight: str
    random_seed: int
    feature_count: int
    feature_names: list[str]
    train_row_count: int
    validation_row_count: int
    validation_metrics: dict[str, Any]
    default_threshold_metrics: dict[str, Any]
    operational_threshold: float
    operational_threshold_metrics: dict[str, Any]
    training_time_seconds: float
    validation_inference_time_seconds: float
    raw_data_sha256: str
    manifest_sha256: str
    config_sha256: str
    git_commit_sha: str | None
    package_versions: dict[str, str]
    preprocessing_metadata: dict[str, Any]
    test_access: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        forbidden = [key for key in payload if key.startswith("test_") and key != "test_access"]
        if forbidden:
            raise ValueError(f"ExperimentResult contains forbidden Test result fields: {forbidden}")
        json.dumps(payload, allow_nan=False)
        return payload
