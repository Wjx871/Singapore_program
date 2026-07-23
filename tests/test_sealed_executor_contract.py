from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_sealed_evaluation import main
from src.evaluation.sealed import (
    AUTHORIZATION_PHRASE,
    EVALUATION_MODE,
    SealedEvaluationAuthorization,
    SealedEvaluationError,
    verify_local_authorization,
)
from src.experiments.sealed_executor import (
    CONFIG_SHA,
    FROZEN_THRESHOLD,
    MANIFEST_SHA,
    RAW_SHA,
)


def authorization(**overrides):
    values = {
        "authorization_phrase": AUTHORIZATION_PHRASE,
        "expected_executor_sha": "a" * 40,
        "expected_manifest_sha256": MANIFEST_SHA,
        "expected_raw_sha256": RAW_SHA,
        "expected_config_sha256": CONFIG_SHA,
        "frozen_operational_threshold": FROZEN_THRESHOLD,
        "run_id": "sealed-xgb-20260723-170000",
    }
    values.update(overrides)
    return SealedEvaluationAuthorization(**values)


def test_authorization_rejects_wrong_phrase():
    with pytest.raises(SealedEvaluationError, match="authorization"):
        authorization(authorization_phrase="wrong").validate_shape()


def test_authorization_rejects_missing_or_invalid_executor_sha():
    with pytest.raises(SealedEvaluationError, match="SHA"):
        authorization(expected_executor_sha="").validate_shape()


def test_cli_rejects_missing_authorization_without_running_preflight(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not run for missing CLI authorization")

    monkeypatch.setattr(
        "src.experiments.sealed_executor.SealedEvaluationExecutor.preflight", forbidden
    )
    assert main([]) == 1


def test_local_authorization_rejects_ci(monkeypatch, tmp_path):
    monkeypatch.setenv("CI", "true")
    with pytest.raises(SealedEvaluationError, match="CI"):
        verify_local_authorization(tmp_path, authorization())


def test_local_authorization_rejects_head_mismatch(monkeypatch, tmp_path):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr("src.evaluation.sealed._git", lambda root, *args: "b" * 40)
    with pytest.raises(SealedEvaluationError, match="does not match"):
        verify_local_authorization(tmp_path, authorization())


def test_local_authorization_rejects_dirty_tree(monkeypatch, tmp_path):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def fake_git(root, *args):
        return "a" * 40 if args == ("rev-parse", "HEAD") else " M dirty.py"

    monkeypatch.setattr("src.evaluation.sealed._git", fake_git)
    with pytest.raises(SealedEvaluationError, match="clean"):
        verify_local_authorization(tmp_path, authorization())


@pytest.mark.parametrize(
    "existing",
    [
        "outputs/sealed_evaluation/sealed_evaluation_receipt.json",
        "outputs/sealed_evaluation/sealed_evaluation_report.json",
        "docs/INDEPENDENT_TEST_EVALUATION.md",
        "docs/SEALED_EVALUATION_RECEIPT.json",
    ],
)
def test_output_preparation_rejects_existing_receipt_or_report(tmp_path, existing):
    from src.experiments.sealed_executor import SealedEvaluationExecutor

    path = tmp_path / existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(SealedEvaluationError, match="already exists"):
        SealedEvaluationExecutor._prepare_output_paths(tmp_path, smoke_only=True)


def test_receipt_governance_shape_has_no_row_level_data():
    receipt = {
        "status": "completed",
        "evaluation_mode": EVALUATION_MODE,
        "prediction_call_count": 1,
        "model_reselection_performed": False,
        "threshold_reselection_performed": False,
        "calibration_performed": False,
        "legacy_forest_test_quarantined": True,
    }
    serialized = json.dumps(receipt, allow_nan=False)
    assert "probabilities" not in serialized
    assert "predictions" not in serialized
    assert receipt["evaluation_mode"] == "authorized_one_time_sealed_evaluation"
