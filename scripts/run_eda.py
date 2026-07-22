#!/usr/bin/env python3
"""Generate the Data Analysis member's auditable EDA deliverables."""

from __future__ import annotations

import argparse
import json

from src.analysis.eda import RAW_AUDIT_SCOPE, TRAIN_SCOPE, run_scopes


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
    scopes = (
        (RAW_AUDIT_SCOPE, TRAIN_SCOPE)
        if args.scope == "all"
        else (args.scope,)
    )
    paths = run_scopes(
        args.config,
        scopes=scopes,
        output_root=args.output_root,
        include_strategy_sensitivity=not args.skip_strategy_sensitivity,
    )
    print(
        json.dumps(
            {
                "generated_scopes": list(scopes),
                "output_directories": [str(path) for path in paths],
                "eda_direct_validation_or_test_label_access": False,
                "validation_model_metrics_may_be_reused_from_shared_runner": True,
                "test_model_metrics_generated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
