from __future__ import annotations

import numpy as np
import pytest

from src.features.feature_sets import DELINQUENCY_COLUMNS, build_feature_set, feature_set_schema
from src.features.preprocessing import CreditRiskPreprocessor


@pytest.fixture
def cleaned(predictor_frame, experiment_config):
    cfg = experiment_config.raw["preprocessing"]
    processor = CreditRiskPreprocessor(
        predictor_columns=experiment_config.predictor_columns,
        delinquency_columns=cfg["abnormal_delinquency"]["columns"],
        abnormal_values=cfg["abnormal_delinquency"]["abnormal_values"],
    )
    return processor.fit_transform(predictor_frame)


@pytest.mark.parametrize(("name", "count"), [("A", 14), ("B", 17), ("C1", 17), ("C2", 15), ("C3", 18)])
def test_feature_set_schema(name, count, cleaned, experiment_config):
    features = build_feature_set(cleaned, name, experiment_config.predictor_columns)
    assert tuple(features.columns) == feature_set_schema(name, experiment_config.predictor_columns)
    assert features.shape[1] == count
    assert np.isfinite(features.to_numpy()).all()
    assert not {"row_id", "target", "feature_hash_v1"}.intersection(features.columns)


def test_c2_drops_raw_delinquency_and_c3_keeps_it(cleaned, experiment_config):
    c2 = build_feature_set(cleaned, "C2", experiment_config.predictor_columns)
    c3 = build_feature_set(cleaned, "C3", experiment_config.predictor_columns)
    assert not set(DELINQUENCY_COLUMNS).intersection(c2.columns)
    assert set(DELINQUENCY_COLUMNS).issubset(c3.columns)
    assert "DelinquencyScore" in c2 and "DelinquencyScore" in c3
