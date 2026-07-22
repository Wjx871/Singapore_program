from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.feature_hash import canonical_payload, compute_feature_hashes, hash_predictor_vector


def test_hash_is_deterministic(predictor_frame, experiment_config):
    first = compute_feature_hashes(predictor_frame, experiment_config.predictor_columns)
    second = compute_feature_hashes(predictor_frame.copy(), experiment_config.predictor_columns)
    assert first.equals(second)
    assert first.str.fullmatch(r"[0-9a-f]{64}").all()


def test_equivalent_zero_and_integer_values_hash_identically():
    columns = ["x"]
    hashes = [hash_predictor_vector([value], columns) for value in (0, 0.0, -0.0)]
    assert len(set(hashes)) == 1
    assert hash_predictor_vector([1], columns) == hash_predictor_vector([1.0], columns)


def test_missing_sentinel_is_stable():
    assert canonical_payload([None], ["x"]) == canonical_payload([np.nan], ["x"])


def test_changed_value_changes_canonical_input():
    assert canonical_payload([1.25], ["x"]) != canonical_payload([1.5], ["x"])


def test_hash_uses_declared_column_order(predictor_frame, experiment_config):
    columns = experiment_config.predictor_columns
    normal = compute_feature_hashes(predictor_frame, columns)
    reversed_hash = compute_feature_hashes(predictor_frame, tuple(reversed(columns)))
    assert not normal.equals(reversed_hash)
