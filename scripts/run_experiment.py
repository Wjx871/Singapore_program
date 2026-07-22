#!/usr/bin/env python3
"""Run one configured Stage 3 Validation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.experiments.contracts import ExperimentSpec
from src.experiments.runner import SharedExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model", default="logistic_regression")
    parser.add_argument("--feature-set", choices=["A", "B"], required=True)
    parser.add_argument(
        "--preprocessing-strategy",
        choices=["missing_plus_flag", "keep_raw"],
        required=True,
    )
    parser.add_argument("--class-weight", choices=["balanced", "none"], required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = config.project_root / config.raw["split"]["manifest_path"]
    spec = ExperimentSpec(
        experiment_id=args.experiment_id,
        model_name=args.model,
        feature_set=args.feature_set,
        preprocessing_strategy=args.preprocessing_strategy,
        class_weight=args.class_weight,
        random_seed=args.random_seed,
        config_path=config.config_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=config.raw["split"]["frozen_manifest_sha256"],
    )
    runner = SharedExperimentRunner(config.config_path)
    artifacts = runner.run(spec)
    paths = runner.persist(artifacts)
    print(json.dumps({"result": artifacts.result.to_dict(), "outputs": {k: str(v) for k, v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
