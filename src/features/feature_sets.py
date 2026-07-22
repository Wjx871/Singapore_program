"""Deterministic Feature Set A/B/C schema construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

COMMON_QUALITY_FLAGS = (
    "MonthlyIncomeMissingFlag",
    "DependentsMissingFlag",
    "AgeInvalidFlag",
)
QUALITY_FLAGS = (*COMMON_QUALITY_FLAGS[:2], "HasAbnormalDelinquencyCode", COMMON_QUALITY_FLAGS[2])
SAFE_ENGINEERED = ("IncomePerDependent", "LogMonthlyIncome", "HighDebtFlag")
DELINQUENCY_COLUMNS = (
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
)


def quality_flags_for_strategy(strategy: str) -> tuple[str, ...]:
    if strategy == "missing_plus_flag":
        return QUALITY_FLAGS
    if strategy == "keep_raw":
        return COMMON_QUALITY_FLAGS
    raise ValueError("strategy must be missing_plus_flag or keep_raw")


def feature_set_schema(
    name: str, predictor_columns: Sequence[str], strategy: str = "missing_plus_flag"
) -> tuple[str, ...]:
    set_a = (*predictor_columns, *quality_flags_for_strategy(strategy))
    set_b = (*set_a, *SAFE_ENGINEERED)
    schemas = {
        "A": set_a,
        "B": set_b,
        "C1": set_b,
        "C2": (*[column for column in set_b if column not in DELINQUENCY_COLUMNS], "DelinquencyScore"),
        "C3": (*set_b, "DelinquencyScore"),
    }
    if name not in schemas:
        raise ValueError(f"Unknown feature set: {name}")
    return tuple(schemas[name])


def build_feature_set(
    cleaned: pd.DataFrame,
    name: str,
    predictor_columns: Sequence[str],
    strategy: str = "missing_plus_flag",
) -> pd.DataFrame:
    expected_a = feature_set_schema("A", predictor_columns, strategy)
    missing = set(expected_a).difference(cleaned.columns)
    if missing:
        raise ValueError(f"Cleaned frame is missing required columns: {sorted(missing)}")
    features = cleaned.loc[:, list(expected_a)].copy()
    if name != "A":
        denominator = features["NumberOfDependents"] + 1.0
        if (denominator <= 0).any():
            raise ValueError("IncomePerDependent denominator must be positive")
        features["IncomePerDependent"] = features["MonthlyIncome"] / denominator
        features["LogMonthlyIncome"] = np.log1p(features["MonthlyIncome"].clip(lower=0.0))
        features["HighDebtFlag"] = (features["DebtRatio"] > 1.0).astype("float64")
    if name in {"C2", "C3"}:
        features["DelinquencyScore"] = (
            features[DELINQUENCY_COLUMNS[0]]
            + 3.0 * features[DELINQUENCY_COLUMNS[1]]
            + 5.0 * features[DELINQUENCY_COLUMNS[2]]
        )
    schema = feature_set_schema(name, predictor_columns, strategy)
    features = features.loc[:, list(schema)]
    values = features.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError(f"Feature Set {name} contains NaN or infinity")
    if len(features.columns) != len(set(features.columns)):
        raise ValueError(f"Feature Set {name} contains duplicate columns")
    return features.astype("float64")
