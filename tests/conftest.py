from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture(scope="session")
def experiment_config():
    return load_config()


@pytest.fixture
def predictor_frame(experiment_config):
    columns = experiment_config.predictor_columns
    rows = [
        [0.1, 45, 0, 0.3, 5000, 8, 0, 1, 0, 2],
        [0.2, 0, 96, 1.2, None, 4, 98, 0, 96, None],
        [0.0, 60, 2, 0.1, 8000, 12, 1, 2, 0, 0],
        [0.7, 35, 1, 2.0, 3000, 3, 0, 0, 1, 1],
    ]
    return pd.DataFrame(rows, columns=columns)
