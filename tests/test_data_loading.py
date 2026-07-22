from __future__ import annotations

from src.data.load_data import load_labeled_data, validate_official_test_schema
from src.utils.reproducibility import sha256_file


def test_raw_labeled_data_hash_schema_and_target(experiment_config):
    before = sha256_file(experiment_config.labeled_path)
    frame = load_labeled_data(experiment_config)
    after = sha256_file(experiment_config.labeled_path)
    assert before == after == experiment_config.data["raw_labeled_sha256"]
    assert frame.shape == (150000, 12)
    assert frame[experiment_config.data["id_column"]].is_unique
    assert set(frame[experiment_config.data["target_column"]].unique()) == {0, 1}
    assert set(experiment_config.predictor_columns).issubset(frame.columns)


def test_official_test_is_unlabeled_schema_only(experiment_config):
    result = validate_official_test_schema(experiment_config)
    assert result == {"row_count": 101503, "column_count": 12, "target_all_missing": True}
