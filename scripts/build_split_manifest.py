#!/usr/bin/env python3
"""Generate the locked Stage 2 split manifest and local metadata."""

from __future__ import annotations

import argparse
import json
import logging

from src.config import load_config
from src.data.feature_hash import compute_feature_hashes
from src.data.load_data import build_raw_data_summary, load_labeled_data, validate_official_test_schema
from src.data.manifest import build_split_metadata, manifest_csv_bytes, validate_manifest
from src.data.split_data import build_group_aware_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    raw = load_labeled_data(config)
    validate_official_test_schema(config)
    raw["feature_hash_v1"] = compute_feature_hashes(raw, config.predictor_columns)
    manifest, selection = build_group_aware_split(
        raw,
        target_column=config.data["target_column"],
        row_id_column=config.data["id_column"],
        group_column="feature_hash_v1",
        test_seed=config.raw["reproducibility"]["split_seed_test"],
        validation_seed=config.raw["reproducibility"]["split_seed_validation"],
        n_splits=config.raw["split"]["n_splits"],
    )
    validation = validate_manifest(manifest, raw, config)
    metadata = build_split_metadata(manifest, raw, config, selection, validation)
    summary = build_raw_data_summary(raw, config)
    paths = {
        "manifest": config.output_path("raw_data_summary").parents[2] / "data/processed/split_manifest.csv",
        "metadata": config.output_path("raw_data_summary").parents[2] / "data/processed/split_metadata.json",
        "summary": config.output_path("raw_data_summary"),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_bytes(manifest_csv_bytes(manifest))
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Generated manifest %s (%s)", paths["manifest"], metadata["manifest_sha256"])


if __name__ == "__main__":
    main()
