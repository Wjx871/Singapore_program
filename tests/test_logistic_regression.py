from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.logistic_regression import build_logistic_pipeline


def test_logistic_pipeline_fits_and_predicts():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(100, 4))
    x = pd.DataFrame(values, columns=["a", "b", "c", "flag"])
    x["flag"] = (x["flag"] > 0).astype(float)
    y = (values[:, 0] + 0.5 * values[:, 1] > 0).astype(int)
    parameters = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42,
    }
    pipeline = build_logistic_pipeline(
        parameters, feature_names=x.columns, binary_feature_names=["flag"]
    ).fit(x, y)
    probability = pipeline.predict_proba(x)[:, 1]
    assert probability.shape == (100,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    transformer = pipeline.named_steps["preprocessing"]
    assert hasattr(transformer.named_transformers_["scale"], "mean_")
    assert "flag" in transformer.transformers_[1][2]
