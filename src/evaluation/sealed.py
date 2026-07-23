"""Authorization and one-call primitives for the sealed Independent Test."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

AUTHORIZATION_PHRASE = "OPEN_INDEPENDENT_TEST_ONCE"
INTERACTIVE_PHRASE = "EXECUTE ONE-TIME INDEPENDENT TEST EVALUATION"
EVALUATION_MODE = "authorized_one_time_sealed_evaluation"


class SealedEvaluationError(RuntimeError):
    """Raised before or during the explicitly authorized sealed path."""


@dataclass(frozen=True)
class SealedEvaluationAuthorization:
    authorization_phrase: str
    expected_executor_sha: str
    expected_manifest_sha256: str
    expected_raw_sha256: str
    expected_config_sha256: str
    frozen_operational_threshold: float
    run_id: str

    def validate_shape(self) -> None:
        if self.authorization_phrase != AUTHORIZATION_PHRASE:
            raise SealedEvaluationError("Invalid sealed evaluation authorization phrase")
        if not re.fullmatch(r"[0-9a-f]{40}", self.expected_executor_sha):
            raise SealedEvaluationError("Expected executor SHA must be an exact 40-character Git SHA")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", self.run_id):
            raise SealedEvaluationError("run_id must be unique, immutable, and filesystem-safe")

    @property
    def authorization_phrase_sha256(self) -> str:
        return hashlib.sha256(self.authorization_phrase.encode("utf-8")).hexdigest()


def verify_local_authorization(root: Path, authorization: SealedEvaluationAuthorization) -> None:
    authorization.validate_shape()
    if os.getenv("CI", "").lower() == "true" or os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        raise SealedEvaluationError("Sealed evaluation is forbidden in CI")
    head = _git(root, "rev-parse", "HEAD")
    if head != authorization.expected_executor_sha:
        raise SealedEvaluationError(
            f"Current HEAD {head} does not match expected executor SHA "
            f"{authorization.expected_executor_sha}"
        )
    if _git(root, "status", "--porcelain"):
        raise SealedEvaluationError("Working tree must be completely clean")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


class ExactlyOnceTestPredictor:
    """Permit a model's Independent Test probability call exactly once."""

    def __init__(self, predict: Callable[[Any], np.ndarray]) -> None:
        self._predict = predict
        self.prediction_call_count = 0

    def predict_proba(self, features: Any) -> np.ndarray:
        if self.prediction_call_count != 0:
            raise SealedEvaluationError("Independent Test predict_proba may execute exactly once")
        self.prediction_call_count = 1
        return np.asarray(self._predict(features), dtype="float64")
