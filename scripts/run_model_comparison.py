#!/usr/bin/env python3
"""Run the frozen four-model comparison on Validation only."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.model_comparison import run_comparison, write_outputs
from src.experiments.runner import SharedExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run unified four-model frozen Validation comparison"
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    runner = SharedExperimentRunner(args.config)
    comparison = run_comparison(runner)
    output_dir = runner.config.project_root / "outputs" / "model_comparison"
    figure_dir = (
        runner.config.project_root / "docs" / "assets" / "model_comparison"
    )
    paths = write_outputs(
        comparison,
        output_dir=output_dir,
        figure_dir=figure_dir,
    )

    print(f"Completed {len(comparison.table)}/4 frozen Validation models.")
    print(f"Recommended experiment: {comparison.selected_experiment_id}")
    print("No Independent Test prediction or evaluation was performed.")
    for name, path in paths.items():
        print(f"{name}: {Path(path).relative_to(runner.config.project_root)}")


if __name__ == "__main__":
    main()
