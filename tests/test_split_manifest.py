from __future__ import annotations

import pandas as pd

from src.data.feature_hash import compute_feature_hashes
from src.data.load_data import load_labeled_data
from src.data.manifest import manifest_sha256, validate_manifest
from src.data.split_data import build_group_aware_split


def synthetic_grouped_frame() -> pd.DataFrame:
    rows = []
    for group in range(100):
        target = int(group % 10 == 0)
        repeats = 2 if group in {5, 10, 15, 20} else 1
        for repeat in range(repeats):
            rows.append(
                {"row_id": group * 10 + repeat, "feature_hash_v1": f"g{group:03d}", "target": target}
            )
    rows.append({"row_id": 1001, "feature_hash_v1": "g010", "target": 0})
    return pd.DataFrame(rows)


def split_synthetic(frame: pd.DataFrame, test_seed: int = 42):
    return build_group_aware_split(
        frame,
        target_column="target",
        row_id_column="row_id",
        group_column="feature_hash_v1",
        test_seed=test_seed,
        validation_seed=43,
    )[0]


def test_group_isolation_and_conflicting_group_not_split():
    manifest = split_synthetic(synthetic_grouped_frame())
    assert manifest["row_id"].is_unique
    assert manifest.groupby("feature_hash_v1")["split"].nunique().max() == 1
    assert manifest.loc[manifest["feature_hash_v1"] == "g010", "split"].nunique() == 1


def test_split_is_reproducible_and_input_order_independent():
    frame = synthetic_grouped_frame()
    first = split_synthetic(frame).sort_values("row_id").reset_index(drop=True)
    second = split_synthetic(frame.sample(frac=1, random_state=9)).sort_values("row_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    assert manifest_sha256(first) == manifest_sha256(second)


def test_changed_seed_changes_split():
    frame = synthetic_grouped_frame()
    first = split_synthetic(frame, 42).set_index("row_id")["split"]
    second = split_synthetic(frame, 99).set_index("row_id")["split"]
    assert not first.equals(second)


def test_formal_manifest_matches_raw_data(experiment_config):
    path = experiment_config.project_root / experiment_config.raw["split"]["manifest_path"]
    manifest = pd.read_csv(path, dtype={"feature_hash_v1": "string"})
    raw = load_labeled_data(experiment_config)
    raw["feature_hash_v1"] = compute_feature_hashes(raw, experiment_config.predictor_columns)
    result = validate_manifest(manifest, raw, experiment_config)
    assert all(value == 0 for value in result["row_overlap_check"].values())
    assert all(value == 0 for value in result["group_overlap_check"].values())
