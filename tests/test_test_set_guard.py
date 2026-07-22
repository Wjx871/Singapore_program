from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.guard import SealedTestSetError
from src.evaluation.metrics import threshold_independent_metrics
from src.evaluation.thresholding import threshold_table
from src.features.feature_sets import QUALITY_FLAGS, build_feature_set
from src.features.preprocessing import CreditRiskPreprocessor
from src.models.logistic_regression import build_logistic_pipeline


def test_test_metrics_and_threshold_selection_are_blocked(experiment_config):
    assert experiment_config.raw["test_access"]["test_evaluation_enabled"] is False
    with pytest.raises(SealedTestSetError):
        threshold_independent_metrics([0, 1], [0.1, 0.9], split_name="test")
    with pytest.raises(SealedTestSetError):
        threshold_table([0, 1], [0.1, 0.9], recall_minimum=0.75, split_name="test")


def test_test_changes_do_not_affect_training_pipeline_or_validation_threshold(
    predictor_frame, experiment_config
):
    cfg = experiment_config.raw["preprocessing"]
    train = predictor_frame.iloc[[0, 2, 3]].copy()
    validation = predictor_frame.iloc[[0, 3]].copy()
    test_a = predictor_frame.iloc[[1]].copy()
    test_b = test_a.copy()
    test_b.loc[:, "MonthlyIncome"] = 1e12

    def run(test_frame):
        cleaner = CreditRiskPreprocessor(
            predictor_columns=experiment_config.predictor_columns,
            delinquency_columns=cfg["abnormal_delinquency"]["columns"],
            abnormal_values=cfg["abnormal_delinquency"]["abnormal_values"],
        )
        x_train = build_feature_set(cleaner.fit_transform(train), "A", experiment_config.predictor_columns)
        x_validation = build_feature_set(cleaner.transform(validation), "A", experiment_config.predictor_columns)
        build_feature_set(cleaner.transform(test_frame), "A", experiment_config.predictor_columns)
        y_train = np.array([0, 1, 0])
        model = build_logistic_pipeline(
            experiment_config.raw["models"]["logistic_regression"]["baseline"],
            feature_names=x_train.columns,
            binary_feature_names=QUALITY_FLAGS,
        )
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_validation)[:, 1]
        table = threshold_table([0, 1], probability, recall_minimum=0.75, split_name="validation")
        selected = float(table.loc[table["selected_operational_threshold"], "threshold"].iloc[0])
        scaler = model.named_steps["preprocessing"].named_transformers_["scale"]
        return cleaner.medians_, scaler.mean_, model.named_steps["model"].coef_, selected

    first = run(test_a)
    second = run(test_b)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    assert first[3] == second[3]
