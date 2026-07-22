"""Fixed Logistic Regression E1 baseline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_pipeline(
    parameters: Mapping[str, Any],
    *,
    feature_names: Sequence[str] | None = None,
    binary_feature_names: Sequence[str] = (),
) -> Pipeline:
    allowed = {"C", "penalty", "solver", "class_weight", "max_iter", "random_state"}
    unknown = set(parameters).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown Logistic Regression parameters: {sorted(unknown)}")
    if feature_names is None:
        preprocessing: Any = StandardScaler()
    else:
        binary = [name for name in binary_feature_names if name in feature_names]
        numeric = [name for name in feature_names if name not in binary]
        preprocessing = ColumnTransformer(
            [("scale", StandardScaler(), numeric), ("binary", "passthrough", binary)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(**dict(parameters))),
        ]
    )
