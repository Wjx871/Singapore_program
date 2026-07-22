"""Deterministic two-stage StratifiedGroupKFold split selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class FoldCandidate:
    fold_index: int
    row_count: int
    row_fraction: float
    positive_rate: float
    group_count: int
    group_fraction: float
    row_ratio_deviation: float
    positive_rate_deviation: float
    group_ratio_deviation: float

    @property
    def score(self) -> tuple[float, float, float, int]:
        return (
            self.row_ratio_deviation,
            self.positive_rate_deviation,
            self.group_ratio_deviation,
            self.fold_index,
        )


def _select_fold(
    frame: pd.DataFrame,
    *,
    target_column: str,
    group_column: str,
    n_splits: int,
    seed: int,
    target_fraction: float,
    reference_positive_rate: float,
) -> tuple[np.ndarray, list[FoldCandidate], int]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates: list[FoldCandidate] = []
    held_out_indices: list[np.ndarray] = []
    group_total = frame[group_column].nunique()
    for fold_index, (_, held_out) in enumerate(
        splitter.split(frame, frame[target_column], groups=frame[group_column])
    ):
        fold = frame.iloc[held_out]
        row_fraction = len(fold) / len(frame)
        group_fraction = fold[group_column].nunique() / group_total
        positive_rate = float(fold[target_column].mean())
        candidates.append(
            FoldCandidate(
                fold_index=fold_index,
                row_count=len(fold),
                row_fraction=row_fraction,
                positive_rate=positive_rate,
                group_count=int(fold[group_column].nunique()),
                group_fraction=group_fraction,
                row_ratio_deviation=abs(row_fraction - target_fraction),
                positive_rate_deviation=abs(positive_rate - reference_positive_rate),
                group_ratio_deviation=abs(group_fraction - target_fraction),
            )
        )
        held_out_indices.append(held_out)
    selected = min(candidates, key=lambda item: item.score)
    return held_out_indices[selected.fold_index], candidates, selected.fold_index


def build_group_aware_split(
    frame: pd.DataFrame,
    *,
    target_column: str,
    row_id_column: str,
    group_column: str,
    test_seed: int,
    validation_seed: int,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ordered = frame.sort_values(row_id_column, kind="mergesort").reset_index(drop=True)
    global_rate = float(ordered[target_column].mean())
    test_positions, test_candidates, selected_test = _select_fold(
        ordered,
        target_column=target_column,
        group_column=group_column,
        n_splits=n_splits,
        seed=test_seed,
        target_fraction=0.20,
        reference_positive_rate=global_rate,
    )
    is_test = np.zeros(len(ordered), dtype=bool)
    is_test[test_positions] = True
    development = ordered.loc[~is_test].reset_index(drop=True)
    validation_positions, validation_candidates, selected_validation = _select_fold(
        development,
        target_column=target_column,
        group_column=group_column,
        n_splits=n_splits,
        seed=validation_seed,
        target_fraction=0.20,
        reference_positive_rate=global_rate,
    )
    validation_ids = set(development.iloc[validation_positions][row_id_column])
    split = np.where(
        is_test,
        "test",
        np.where(ordered[row_id_column].isin(validation_ids), "validation", "train"),
    )
    manifest = pd.DataFrame(
        {
            "row_id": ordered[row_id_column].astype("int64"),
            "split": split,
            "feature_hash_v1": ordered[group_column].astype("string"),
            "target": ordered[target_column].astype("int8"),
        }
    )
    selection = {
        "fold_selection_rule": [
            "row_ratio_deviation",
            "positive_rate_deviation",
            "group_ratio_deviation",
            "fold_index",
        ],
        "selected_test_fold": selected_test,
        "selected_validation_fold": selected_validation,
        "test_candidates": [asdict(item) for item in test_candidates],
        "validation_candidates": [asdict(item) for item in validation_candidates],
        "input_order_rule": "stable sort by row_id before splitting",
    }
    return manifest, selection
