"""Split manifest validation, hashing, persistence, and metadata."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config import ExperimentConfig
from src.utils.reproducibility import git_commit_sha, package_versions

SPLITS = ("train", "validation", "test")


def manifest_csv_bytes(manifest: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    manifest.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def manifest_sha256(manifest: pd.DataFrame) -> str:
    return hashlib.sha256(manifest_csv_bytes(manifest)).hexdigest()


def validate_manifest(
    manifest: pd.DataFrame,
    raw: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, Any]:
    required = ["row_id", "split", "feature_hash_v1", "target"]
    if list(manifest.columns) != required:
        raise ValueError(f"Manifest columns must be {required}")
    if len(manifest) != len(raw) or len(manifest) != int(config.data["expected_labeled_rows"]):
        raise ValueError("Manifest must cover every labeled raw row exactly once")
    if not manifest["row_id"].is_unique:
        raise ValueError("Manifest row_id values must be unique")
    raw_ids = set(raw[config.data["id_column"]])
    if set(manifest["row_id"]) != raw_ids:
        raise ValueError("Manifest contains missing or unknown row_id values")
    if set(manifest["split"]) != set(SPLITS):
        raise ValueError(f"Manifest split values must be exactly {SPLITS}")

    raw_targets = raw.set_index(config.data["id_column"])[config.data["target_column"]]
    aligned = manifest.set_index("row_id")["target"].sort_index().to_numpy(dtype="int8")
    expected_targets = raw_targets.sort_index().to_numpy(dtype="int8")
    if not (aligned == expected_targets).all():
        raise ValueError("Manifest target values do not match raw data")
    if "feature_hash_v1" in raw.columns:
        aligned_hashes = manifest.set_index("row_id")["feature_hash_v1"].sort_index().astype(str)
        expected_hashes = (
            raw.set_index(config.data["id_column"])["feature_hash_v1"].sort_index().astype(str)
        )
        if not (aligned_hashes.to_numpy() == expected_hashes.to_numpy()).all():
            raise ValueError("Manifest feature_hash_v1 values do not match raw predictors")
    if manifest.groupby("feature_hash_v1")["split"].nunique().max() != 1:
        raise ValueError("A feature_hash_v1 group crosses split boundaries")

    ratios = config.raw["split"]["ratios"]
    tolerances = config.raw["split"]["tolerances"]
    global_rate = float(manifest["target"].mean())
    stats: dict[str, Any] = {}
    row_sets: dict[str, set[int]] = {}
    group_sets: dict[str, set[str]] = {}
    for split_name in SPLITS:
        part = manifest[manifest["split"] == split_name]
        fraction = len(part) / len(manifest)
        positive_rate = float(part["target"].mean())
        if abs(fraction - float(ratios[split_name])) > float(tolerances["sample_fraction_absolute"]):
            raise ValueError(f"{split_name} row ratio is outside tolerance: {fraction}")
        if abs(positive_rate - global_rate) > float(tolerances["positive_rate_absolute"]):
            raise ValueError(f"{split_name} positive rate is outside tolerance: {positive_rate}")
        positives = int(part["target"].sum())
        stats[split_name] = {
            "row_count": len(part),
            "row_fraction": fraction,
            "group_count": int(part["feature_hash_v1"].nunique()),
            "positive_count": positives,
            "negative_count": len(part) - positives,
            "positive_rate": positive_rate,
        }
        row_sets[split_name] = set(part["row_id"])
        group_sets[split_name] = set(part["feature_hash_v1"])

    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    row_overlap = {f"{a}_{b}": len(row_sets[a] & row_sets[b]) for a, b in pairs}
    group_overlap = {f"{a}_{b}": len(group_sets[a] & group_sets[b]) for a, b in pairs}
    if any(row_overlap.values()) or any(group_overlap.values()):
        raise ValueError("Manifest row/group isolation check failed")
    return {"splits": stats, "row_overlap_check": row_overlap, "group_overlap_check": group_overlap}


def build_split_metadata(
    manifest: pd.DataFrame,
    raw: pd.DataFrame,
    config: ExperimentConfig,
    selection: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    conflicts = (
        manifest.groupby("feature_hash_v1")["target"].agg(["nunique", "size"])
    )
    conflicts = conflicts[conflicts["nunique"] > 1]
    versions = package_versions()
    return {
        "raw_data_sha256": config.data["raw_labeled_sha256"],
        "manifest_sha256": manifest_sha256(manifest),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": versions["python"],
        "pandas_version": versions["pandas"],
        "numpy_version": versions["numpy"],
        "scikit_learn_version": versions["scikit-learn"],
        "split_seeds": {
            "test": config.raw["reproducibility"]["split_seed_test"],
            "validation": config.raw["reproducibility"]["split_seed_validation"],
        },
        "split_ratios": config.raw["split"]["ratios"],
        **selection,
        **validation,
        "conflicting_target_group_count": len(conflicts),
        "conflicting_target_row_count": int(conflicts["size"].sum()),
        "feature_hash_version": config.raw["feature_hash"]["version"],
        "config_hash": config.config_sha256,
        "source_commit_sha": git_commit_sha(config.project_root),
    }
