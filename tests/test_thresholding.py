from __future__ import annotations

import pytest

from src.evaluation.thresholding import select_operational_threshold, threshold_table


def test_threshold_maximizes_precision_subject_to_recall():
    table = threshold_table(
        [1, 1, 1, 0, 0], [0.9, 0.8, 0.2, 0.7, 0.1], recall_minimum=0.75, split_name="validation"
    )
    threshold = select_operational_threshold(table)
    selected = table[table["selected_operational_threshold"]].iloc[0]
    assert threshold == pytest.approx(0.2)
    assert selected["recall"] == 1.0
    assert selected["precision"] == pytest.approx(0.75)


def test_tie_break_prefers_higher_recall_then_threshold():
    table = threshold_table(
        [1, 1, 0, 0], [0.8, 0.7, 0.8, 0.1], recall_minimum=0.5, split_name="validation"
    )
    selected = table[table["selected_operational_threshold"]].iloc[0]
    assert selected["threshold"] == pytest.approx(0.7)
    assert selected["recall"] == 1.0


def test_all_positive_candidate_exists():
    table = threshold_table([1, 0, 0], [0.9, 0.5, 0.1], recall_minimum=1.0, split_name="validation")
    assert (table["recall"] == 1.0).any()


def test_empty_and_single_class_rejected():
    with pytest.raises(ValueError, match="empty"):
        threshold_table([], [], recall_minimum=0.75, split_name="validation")
    with pytest.raises(ValueError, match="both"):
        threshold_table([1, 1], [0.2, 0.3], recall_minimum=0.75, split_name="validation")
