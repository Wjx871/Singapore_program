#!/usr/bin/env python3
"""Validate a generated manifest against immutable raw labeled data."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import load_config
from src.data.feature_hash import compute_feature_hashes
from src.data.load_data import load_labeled_data
from src.data.manifest import manifest_sha256, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = config.project_root / config.raw["split"]["manifest_path"]
    manifest = pd.read_csv(manifest_path, dtype={"feature_hash_v1": "string"})
    raw = load_labeled_data(config)
    raw["feature_hash_v1"] = compute_feature_hashes(raw, config.predictor_columns)
    result = validate_manifest(manifest, raw, config)
    result["manifest_sha256"] = manifest_sha256(manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
