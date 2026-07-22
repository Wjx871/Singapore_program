from __future__ import annotations

import numpy as np
import pytest

from src.features.preprocessing import CreditRiskPreprocessor


def make_preprocessor(config, strategy="missing_plus_flag"):
    preprocessing = config.raw["preprocessing"]
    return CreditRiskPreprocessor(
        predictor_columns=config.predictor_columns,
        delinquency_columns=preprocessing["abnormal_delinquency"]["columns"],
        abnormal_values=preprocessing["abnormal_delinquency"]["abnormal_values"],
        strategy=strategy,
        clipping_enabled=preprocessing["winsorization"]["enabled"],
    )


def test_training_only_statistics(predictor_frame, experiment_config):
    train = predictor_frame.iloc[[0, 2, 3]]
    validation = predictor_frame.iloc[[1]].copy()
    processor = make_preprocessor(experiment_config).fit(train)
    medians_before = processor.medians_.copy()
    validation["MonthlyIncome"] = 999999999
    processor.transform(validation)
    assert processor.medians_ == medians_before
    assert processor.medians_["MonthlyIncome"] == 5000


def test_abnormal_and_age_values_become_median_not_zero(predictor_frame, experiment_config):
    processor = make_preprocessor(experiment_config).fit(predictor_frame.iloc[[0, 2, 3]])
    transformed = processor.transform(predictor_frame.iloc[[1]])
    assert transformed["HasAbnormalDelinquencyCode"].iloc[0] == 1
    assert transformed["AgeInvalidFlag"].iloc[0] == 1
    for column in processor.delinquency_columns:
        assert transformed[column].iloc[0] == processor.medians_[column]
    assert transformed["age"].iloc[0] == processor.medians_["age"]
    assert transformed["MonthlyIncomeMissingFlag"].iloc[0] == 1
    assert transformed["DependentsMissingFlag"].iloc[0] == 1


def test_output_has_no_nan_or_inf(predictor_frame, experiment_config):
    transformed = make_preprocessor(experiment_config).fit_transform(predictor_frame)
    assert np.isfinite(transformed.to_numpy()).all()


def test_keep_raw_preserves_codes_and_omits_flag(predictor_frame, experiment_config):
    processor = make_preprocessor(experiment_config, "keep_raw").fit(predictor_frame.iloc[[0, 2, 3]])
    transformed = processor.transform(predictor_frame.iloc[[1]])
    assert "HasAbnormalDelinquencyCode" not in transformed
    assert transformed["NumberOfTime30-59DaysPastDueNotWorse"].iloc[0] == 96
    assert transformed["NumberOfTimes90DaysLate"].iloc[0] == 98
    assert not set(processor.delinquency_columns).intersection(processor.medians_)
    assert processor.metadata()["output_schema"] == list(transformed.columns)


def test_strategy_change_after_fit_is_rejected(predictor_frame, experiment_config):
    processor = make_preprocessor(experiment_config).fit(predictor_frame)
    processor.strategy = "keep_raw"
    with pytest.raises(RuntimeError, match="cannot change"):
        processor.transform(predictor_frame)
