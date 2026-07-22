#!/usr/bin/env python3
"""Run pre-declared 8-candidate XGBoost validation search.

Each candidate is an ExperimentSpec run through the SharedExperimentRunner.
The script does NOT directly instantiate XGBClassifier or XGBoostAdapter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.experiments.contracts import ExperimentSpec
from src.experiments.runner import SharedExperimentRunner

# ---------------------------------------------------------------------------
# Fixed common parameters (not varied across candidates)
# ---------------------------------------------------------------------------
FIXED_COMMON: dict[str, object] = {
    "n_estimators": 2000,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Pre-declared 8-candidate matrix (one-factor-at-a-time)
# ---------------------------------------------------------------------------
CANDIDATE_MATRIX: list[dict[str, Any]] = [
    # 1. baseline
    {
        "id": "xgb_baseline",
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    # 2. depth=3
    {
        "id": "xgb_depth3",
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    # 3. depth=5
    {
        "id": "xgb_depth5",
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    # 4. learning_rate=0.03
    {
        "id": "xgb_lr003",
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    # 5. min_child_weight=5
    {
        "id": "xgb_child5",
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    # 6. min_child_weight=10
    {
        "id": "xgb_child10",
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    # 7. subsample=1.0
    {
        "id": "xgb_full_rows",
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
    },
    # 8. colsample_bytree=1.0
    {
        "id": "xgb_full_cols",
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
    },
]


# ---------------------------------------------------------------------------
# Candidate selection rule (pre-declared, fixed)
# ---------------------------------------------------------------------------
def select_best_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Primary: max Validation PR-AUC.
    Tie-break (PR-AUC diff <= 1e-6):
      1. higher ROC-AUC
      2. higher KS
      3. shorter training_time_seconds
      4. earlier position in CANDIDATE_MATRIX (by experiment_id)
    """
    candidate_order = [c["id"] for c in CANDIDATE_MATRIX]

    def sort_key(row: dict[str, Any]) -> tuple[float, ...]:
        return (
            row["pr_auc"],
            row["roc_auc"],
            row["ks"],
            -float(row["training_time_seconds"]),  # shorter = better → larger negative
            -candidate_order.index(row["experiment_id"]),  # earlier index = better → less negative
        )

    return max(results, key=sort_key)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run XGBoost validation search")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = config.project_root / config.raw["split"]["manifest_path"]
    manifest_sha = config.raw["split"]["frozen_manifest_sha256"]
    runner = SharedExperimentRunner(config.config_path)

    output_root = config.project_root / "outputs" / "xgboost_search"
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for candidate in CANDIDATE_MATRIX:
        exp_id = candidate["id"]
        model_params: dict[str, Any] = {**FIXED_COMMON}
        for key in ("max_depth", "learning_rate", "min_child_weight",
                     "subsample", "colsample_bytree"):
            model_params[key] = candidate[key]

        print(f"--- Running {exp_id} ---")
        spec = ExperimentSpec(
            experiment_id=exp_id,
            model_name="xgboost",
            feature_set="B",
            preprocessing_strategy="missing_plus_flag",
            class_weight="none",
            random_seed=42,
            config_path=config.config_path,
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            model_parameters=model_params,
        )
        artifacts = runner.run(spec)
        runner.persist(artifacts)

        result = artifacts.result
        metrics = result.validation_metrics
        op_metrics = result.operational_threshold_metrics
        default_metrics = result.default_threshold_metrics
        meta = result.model_training_metadata

        row = {
            "experiment_id": result.experiment_id,
            "model_name": result.model_name,
            "feature_set": result.feature_set,
            "preprocessing_strategy": result.preprocessing_strategy,
            "class_weight": result.class_weight,
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "ks": metrics["ks"],
            "default_threshold": default_metrics["threshold"],
            "default_precision": default_metrics["precision"],
            "default_recall": default_metrics["recall"],
            "default_f1": default_metrics["f1"],
            "operational_threshold": result.operational_threshold,
            "operational_precision": op_metrics["precision"],
            "operational_recall": op_metrics["recall"],
            "operational_f1": op_metrics["f1"],
            "operational_specificity": op_metrics["specificity"],
            "operational_balanced_accuracy": op_metrics["balanced_accuracy"],
            "operational_predicted_positive_rate": op_metrics["predicted_positive_rate"],
            "confusion_matrix_tn": op_metrics["confusion_matrix"]["tn"],
            "confusion_matrix_fp": op_metrics["confusion_matrix"]["fp"],
            "confusion_matrix_fn": op_metrics["confusion_matrix"]["fn"],
            "confusion_matrix_tp": op_metrics["confusion_matrix"]["tp"],
            "training_time_seconds": result.training_time_seconds,
            "validation_inference_time_seconds": result.validation_inference_time_seconds,
            "best_iteration": meta.get("best_iteration"),
            "best_score": meta.get("best_score"),
            "actual_boosting_rounds": meta.get("actual_boosting_rounds"),
            "scale_pos_weight": meta.get("scale_pos_weight"),
            "negative_train_count": meta.get("negative_train_count"),
            "positive_train_count": meta.get("positive_train_count"),
            "manifest_sha256": result.manifest_sha256,
            "config_sha256": result.config_sha256,
            "git_commit_sha": result.git_commit_sha,
        }
        all_results.append(row)
        print(f"  PR-AUC: {row['pr_auc']:.6f}  "
              f"best_iter: {row['best_iteration']}  "
              f"time: {row['training_time_seconds']:.1f}s")

    total_time = time.perf_counter() - start_time

    # --- Persist candidate results CSV ---
    df = pd.DataFrame(all_results)
    df.to_csv(output_root / "candidate_results.csv", index=False, lineterminator="\n")

    # --- Select best candidate ---
    selected = select_best_candidate(all_results)
    selected_candidate = CANDIDATE_MATRIX[
        [c["id"] for c in CANDIDATE_MATRIX].index(selected["experiment_id"])
    ]
    selected_payload = {
        "selected_experiment_id": selected["experiment_id"],
        "selection_metric": "validation_pr_auc",
        "selection_rule": (
            "max Validation PR-AUC; "
            "tie-break: ROC-AUC > KS > shorter training_time > earlier experiment_id"
        ),
        "selected_parameters": {
            **FIXED_COMMON,
            **{k: v for k, v in selected_candidate.items() if k != "id"},
        },
        "validation_metrics": {
            "pr_auc": selected["pr_auc"],
            "roc_auc": selected["roc_auc"],
            "ks": selected["ks"],
        },
        "default_threshold_metrics": {
            "threshold": 0.5,
            "precision": selected["default_precision"],
            "recall": selected["default_recall"],
            "f1": selected["default_f1"],
        },
        "operational_threshold": selected["operational_threshold"],
        "operational_metrics": {
            "precision": selected["operational_precision"],
            "recall": selected["operational_recall"],
            "f1": selected["operational_f1"],
            "specificity": selected["operational_specificity"],
            "balanced_accuracy": selected["operational_balanced_accuracy"],
            "predicted_positive_rate": selected["operational_predicted_positive_rate"],
            "confusion_matrix": {
                "tn": selected["confusion_matrix_tn"],
                "fp": selected["confusion_matrix_fp"],
                "fn": selected["confusion_matrix_fn"],
                "tp": selected["confusion_matrix_tp"],
            },
        },
        "best_iteration": selected["best_iteration"],
        "scale_pos_weight": selected["scale_pos_weight"],
        "training_time_seconds": selected["training_time_seconds"],
        "inference_time_seconds": selected["validation_inference_time_seconds"],
        "manifest_sha256": selected["manifest_sha256"],
        "config_sha256": selected["config_sha256"],
        "git_commit_sha": selected["git_commit_sha"],
    }
    (output_root / "selected_candidate.json").write_text(
        json.dumps(selected_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # --- Feature importance from the best candidate ---
    # We need to re-read the persisted run metadata for the selected candidate.
    run_meta_path = (
        config.project_root / "outputs" / "experiments"
        / selected["experiment_id"] / "run_metadata.json"
    )
    if run_meta_path.is_file():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        imp = run_meta.get("model_training_metadata", {}).get("feature_importance", {})
        if imp:
            fi_df = pd.DataFrame(
                {"feature": list(imp.keys()), "gain_importance": list(imp.values())}
            )
            fi_df = fi_df.sort_values("gain_importance", ascending=False)
            fi_df.to_csv(
                output_root / "feature_importance.csv", index=False, lineterminator="\n"
            )

    # --- Search metadata ---
    search_meta = {
        "total_candidates": len(CANDIDATE_MATRIX),
        "successful": len(all_results),
        "total_wall_time_seconds": total_time,
        "fixed_common_parameters": FIXED_COMMON,
        "shared_spec": {
            "feature_set": "B",
            "preprocessing_strategy": "missing_plus_flag",
            "class_weight": "none",
            "random_seed": 42,
            "evaluation_split": "validation",
        },
        "candidate_order": [c["id"] for c in CANDIDATE_MATRIX],
    }
    (output_root / "search_metadata.json").write_text(
        json.dumps(search_meta, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"\n=== Search complete: {len(all_results)}/{len(CANDIDATE_MATRIX)} candidates ===")
    print(f"Best candidate: {selected['experiment_id']}")
    print(f"  PR-AUC: {selected['pr_auc']:.6f}")
    print(f"  ROC-AUC: {selected['roc_auc']:.6f}")
    print(f"  KS: {selected['ks']:.6f}")
    print(f"Outputs: {output_root}")


if __name__ == "__main__":
    main()
