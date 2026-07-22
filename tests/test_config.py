from __future__ import annotations

from copy import deepcopy

import pytest

from src.config import EXPECTED_CORE_MODELS, load_config, validate_config


@pytest.fixture
def config():
    return load_config()


def test_repository_config_loads(config):
    assert config.labeled_path.is_file()
    assert config.official_test_path.is_file()
    assert tuple(config.raw["models"]["core_models"]) == EXPECTED_CORE_MODELS


def test_split_ratios_sum_to_one(config):
    assert sum(config.raw["split"]["ratios"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["split"]["ratios"].update(test=0.21), "sum to 1"),
        (lambda raw: raw["models"]["core_models"].append("lightgbm"), "exactly"),
        (lambda raw: raw["test_access"].update(test_evaluation_enabled=True), "requires"),
        (lambda raw: raw["preprocessing"]["winsorization"].update(enabled=True), "disabled"),
        (
            lambda raw: raw["preprocessing"].update(abnormal_delinquency_strategy="zero_fill"),
            "missing_plus_flag or keep_raw",
        ),
    ],
)
def test_invalid_config_is_rejected(config, mutation, message):
    raw = deepcopy(config.raw)
    mutation(raw)
    with pytest.raises(ValueError, match=message):
        validate_config(raw)


def test_target_and_source_id_are_not_predictors(config):
    predictors = config.predictor_columns
    assert config.data["target_column"] not in predictors
    assert config.data["source_id_column"] not in predictors
    assert config.data["id_column"] not in predictors


def test_lightgbm_is_optional_only(config):
    assert "lightgbm" not in config.raw["models"]["core_models"]
    assert config.raw["models"]["optional_models"]["lightgbm"]["tier"] == (
        "legacy_or_optional_appendix_only"
    )
