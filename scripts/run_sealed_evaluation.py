#!/usr/bin/env python3
"""CLI for the authorized one-time Independent Test evaluation."""

from __future__ import annotations

import argparse
import json
import sys

from src.evaluation.sealed import SealedEvaluationAuthorization, SealedEvaluationError
from src.experiments.sealed_executor import (
    CONFIG_SHA, FROZEN_THRESHOLD, MANIFEST_SHA, RAW_SHA, SealedEvaluationExecutor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--authorize")
    parser.add_argument("--expected-executor-sha")
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    executor = SealedEvaluationExecutor(args.config)
    try:
        if args.preflight_only:
            if any((args.authorize, args.expected_executor_sha, args.run_id)):
                raise SealedEvaluationError("Preflight-only must not include execution authorization")
            artifacts = executor.preflight()
            print("SEALED PREFLIGHT PASSED — NO TEST PREDICTION PERFORMED")
            print(json.dumps({
                "hashes": artifacts.hashes,
                "validation_metrics": artifacts.validation_metrics,
                "best_iteration": artifacts.training_metadata["best_iteration"],
                "actual_boosting_rounds": artifacts.training_metadata["actual_boosting_rounds"],
            }, indent=2, allow_nan=False))
            return 0
        if not all((args.authorize, args.expected_executor_sha, args.run_id)):
            raise SealedEvaluationError(
                "Execution requires --authorize, --expected-executor-sha, and --run-id"
            )
        authorization = SealedEvaluationAuthorization(
            args.authorize, args.expected_executor_sha, MANIFEST_SHA, RAW_SHA,
            CONFIG_SHA, FROZEN_THRESHOLD, args.run_id
        )
        artifacts = executor.preflight()
        executor.verify_authorization(artifacts, authorization)
        print("SEALED EVALUATION FINAL CHECK")
        for line in executor.final_check_lines(artifacts, authorization):
            print(f"- {line}")
        print("TYPE EXACTLY:\nEXECUTE ONE-TIME INDEPENDENT TEST EVALUATION")
        phrase = input().strip()
        result = executor.execute(
            authorization,
            interactive_phrase=phrase,
            preflight_artifacts=artifacts,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(f"SEALED EVALUATION ABORTED BEFORE/AT GATE: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
