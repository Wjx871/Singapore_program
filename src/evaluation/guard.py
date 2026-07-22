"""Structural protection for the sealed Independent Test Set."""

from __future__ import annotations


class SealedTestSetError(RuntimeError):
    """Raised when Stage 2 code attempts model evaluation on the sealed test set."""


def require_evaluation_split(split_name: str) -> None:
    if split_name == "test":
        raise SealedTestSetError(
            "Independent Test evaluation is sealed in Stage 2; only schema and transform checks are allowed"
        )
    if split_name not in {"train", "validation"}:
        raise ValueError(f"Unknown evaluation split: {split_name}")
