#!/usr/bin/env python3
"""Generate the Data Analysis member's auditable EDA deliverables."""

from __future__ import annotations

import argparse
import json

from src.analysis.eda import RAW_AUDIT_SCOPE, TRAIN_SCOPE, run_scopes
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate raw-audit and/or frozen-Training EDA outputs. "
            "Validation and Test scopes are intentionally unavailable."
        )
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--scope",
        choices=(RAW_AUDIT_SCOPE, TRAIN_SCOPE, "all"),
        default="all",
    )
    parser.add_argument("--output-root", default="outputs/eda")
    parser.add_argument(
        "--skip-strategy-sensitivity",
        action="store_true",
        help="Skip the train-only preprocessing strategy comparison.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    scopes = (
        (RAW_AUDIT_SCOPE, TRAIN_SCOPE)
        if args.scope == "all"
        else (args.scope,)
    )
    paths = run_scopes(
        config.config_path,
        scopes=scopes,
        output_root=args.output_root,
        include_strategy_sensitivity=not args.skip_strategy_sensitivity,
    )
    relative_paths: list[str] = []
    for path in paths:
        try:
            relative_paths.append(
                path.resolve().relative_to(config.project_root.resolve()).as_posix()
            )
        except ValueError as exc:
            raise ValueError(
                f"CLI output directory must be inside the repository root: {path}"
            ) from exc
    print(
        json.dumps(
            {
                "generated_scopes": list(scopes),
                "output_directories": relative_paths,
                "eda_direct_validation_or_test_label_access": False,
                "validation_model_metrics_may_be_reused_from_shared_runner": True,
                "test_model_metrics_generated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
