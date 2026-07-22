"""Versioned canonical serialization for raw predictor vectors."""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real
from typing import Sequence

import pandas as pd

HASH_VERSION = "feature_hash_v1"
MISSING_TOKEN = "<NA>"
FIELD_SEPARATOR = "\x1f"


def canonicalize_value(value: object) -> str:
    if pd.isna(value):
        return MISSING_TOKEN
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Predictor values may not contain positive or negative infinity")
        if number == 0.0:
            return "0"
        if number.is_integer():
            return str(int(number))
        return number.hex()
    raise TypeError(f"Unsupported predictor value type: {type(value).__name__}")


def canonical_payload(values: Sequence[object], columns: Sequence[str]) -> str:
    if len(values) != len(columns):
        raise ValueError("Predictor values and columns must have the same length")
    fields = [HASH_VERSION]
    fields.extend(
        f"{column}={canonicalize_value(value)}"
        for column, value in zip(columns, values, strict=True)
    )
    return FIELD_SEPARATOR.join(fields)


def hash_predictor_vector(values: Sequence[object], columns: Sequence[str]) -> str:
    payload = canonical_payload(values, columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_feature_hashes(frame: pd.DataFrame, predictor_columns: Sequence[str]) -> pd.Series:
    missing = set(predictor_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing predictor columns for feature hash: {sorted(missing)}")
    ordered = frame.loc[:, list(predictor_columns)]
    hashes = [
        hash_predictor_vector(row, predictor_columns)
        for row in ordered.itertuples(index=False, name=None)
    ]
    return pd.Series(hashes, index=frame.index, name=HASH_VERSION, dtype="string")
