"""Validation-only operational threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.guard import require_evaluation_split
from src.evaluation.metrics import validate_binary_inputs


def threshold_table(
    y_true: object,
    probability: object,
    *,
    recall_minimum: float,
    split_name: str,
) -> pd.DataFrame:
    require_evaluation_split(split_name)
    y, probabilities = validate_binary_inputs(y_true, probability)
    if len(np.unique(y)) < 2:
        raise ValueError("Operational threshold selection requires both target classes")
    if not 0.0 <= recall_minimum <= 1.0:
        raise ValueError("recall_minimum must be between 0 and 1")

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_p = probabilities[order]
    sorted_y = y[order]
    unique_probabilities, first_indices = np.unique(sorted_p, return_index=True)
    group_starts = first_indices[np.argsort(-unique_probabilities)]
    group_ends = np.r_[group_starts[1:], len(sorted_p)]
    total_positive = int(y.sum())
    total_negative = len(y) - total_positive
    tp = fp = 0
    rows: list[dict[str, object]] = [
        _threshold_row(
            float(np.nextafter(sorted_p[0], np.inf)),
            tp=0,
            fp=0,
            total_positive=total_positive,
            total_negative=total_negative,
            recall_minimum=recall_minimum,
        )
    ]
    for start, end in zip(group_starts, group_ends, strict=True):
        labels = sorted_y[start:end]
        tp += int(labels.sum())
        fp += int(len(labels) - labels.sum())
        rows.append(
            _threshold_row(
                float(sorted_p[start]),
                tp=tp,
                fp=fp,
                total_positive=total_positive,
                total_negative=total_negative,
                recall_minimum=recall_minimum,
            )
        )
    table = pd.DataFrame(rows)
    eligible = table[table["meets_recall_constraint"]]
    if eligible.empty:
        raise RuntimeError("No threshold satisfies recall constraint; all-positive candidate is missing")
    selected_index = max(
        eligible.index,
        key=lambda index: (
            table.at[index, "precision"],
            table.at[index, "recall"],
            table.at[index, "threshold"],
        ),
    )
    table["selected_operational_threshold"] = False
    table.loc[selected_index, "selected_operational_threshold"] = True
    return table.sort_values("threshold", ascending=False, kind="mergesort").reset_index(drop=True)


def _threshold_row(
    threshold: float,
    *,
    tp: int,
    fp: int,
    total_positive: int,
    total_negative: int,
    recall_minimum: float,
) -> dict[str, object]:
    fn = total_positive - tp
    tn = total_negative - fp
    predicted_positive = tp + fp
    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / total_positive
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / total_negative if total_negative else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "predicted_positive_rate": predicted_positive / (total_positive + total_negative),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "meets_recall_constraint": bool(recall >= recall_minimum),
    }


def select_operational_threshold(table: pd.DataFrame) -> float:
    selected = table[table["selected_operational_threshold"]]
    if len(selected) != 1:
        raise ValueError("Threshold table must contain exactly one selected operational threshold")
    return float(selected.iloc[0]["threshold"])
