"""Training-only fitted common tabular cleaning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class CreditRiskPreprocessor:
    predictor_columns: Sequence[str]
    delinquency_columns: Sequence[str]
    abnormal_values: Sequence[float] = (96, 98)
    clipping_enabled: bool = False
    clipping_columns: Sequence[str] = ()
    upper_quantile: float = 0.99
    medians_: dict[str, float] = field(default_factory=dict, init=False)
    clipping_upper_: dict[str, float] = field(default_factory=dict, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, train: pd.DataFrame) -> "CreditRiskPreprocessor":
        self._validate_input(train)
        working = self._mark_invalid_as_missing(train)
        median_columns = ["MonthlyIncome", "NumberOfDependents", "age", *self.delinquency_columns]
        self.medians_ = {}
        for column in median_columns:
            median = float(working[column].median())
            if not np.isfinite(median):
                raise ValueError(f"Training median is not finite for {column}")
            self.medians_[column] = median
        imputed = working.copy()
        for column, median in self.medians_.items():
            imputed[column] = imputed[column].fillna(median)
        if self.clipping_enabled:
            self.clipping_upper_ = {
                column: float(imputed[column].quantile(self.upper_quantile))
                for column in self.clipping_columns
            }
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("CreditRiskPreprocessor must be fitted on Training before transform")
        self._validate_input(frame)
        original = frame.loc[:, list(self.predictor_columns)].copy()
        output = self._mark_invalid_as_missing(original)
        output["MonthlyIncomeMissingFlag"] = original["MonthlyIncome"].isna().astype("int8")
        output["DependentsMissingFlag"] = original["NumberOfDependents"].isna().astype("int8")
        output["HasAbnormalDelinquencyCode"] = original.loc[:, list(self.delinquency_columns)].isin(
            self.abnormal_values
        ).any(axis=1).astype("int8")
        output["AgeInvalidFlag"] = (original["age"] <= 0).astype("int8")
        for column, median in self.medians_.items():
            output[column] = output[column].fillna(median)
        for column, upper in self.clipping_upper_.items():
            output[column] = output[column].clip(upper=upper)
        output = output.loc[:, [
            *self.predictor_columns,
            "MonthlyIncomeMissingFlag",
            "DependentsMissingFlag",
            "HasAbnormalDelinquencyCode",
            "AgeInvalidFlag",
        ]]
        values = output.to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError("Preprocessed features contain NaN or infinity")
        return output.astype("float64")

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    def _validate_input(self, frame: pd.DataFrame) -> None:
        missing = set(self.predictor_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing preprocessing predictors: {sorted(missing)}")

    def _mark_invalid_as_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.loc[:, list(self.predictor_columns)].copy()
        for column in self.delinquency_columns:
            output.loc[output[column].isin(self.abnormal_values), column] = np.nan
        output.loc[output["age"] <= 0, "age"] = np.nan
        return output

    def metadata(self) -> dict[str, object]:
        if not self.fitted_:
            raise RuntimeError("Preprocessor metadata is unavailable before fit")
        return {
            "fit_scope": "training_only",
            "medians": self.medians_,
            "clipping_enabled": self.clipping_enabled,
            "clipping_upper": self.clipping_upper_,
            "abnormal_values": list(self.abnormal_values),
        }
