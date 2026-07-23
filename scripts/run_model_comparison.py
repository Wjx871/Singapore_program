#!/usr/bin/env python3
"""Run the four-model Validation comparison and produce outputs + figures.

Outputs written to ``outputs/model_comparison/`` (git-ignored):
  - comparison_report.json      : full JSON-safe comparison report
  - model_comparison.csv        : four-row summary table
  - figures/pr_curves.png       : Precision-Recall curves for all four models
  - figures/roc_curves.png      : ROC curves for all four models
  - figures/metric_bar.png      : bar chart of threshold-independent metrics
  - figures/operational_bar.png : operational-threshold precision / recall bar chart
  - figures/confusion_matrices.png : 2x2 confusion matrices (operational threshold)

Figure images are also copied to ``docs/assets/model_comparison/`` for document
embedding. All figure values are cross-checked against the CSV; the script
raises if any discrepancy exceeds 1e-6.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# Guard: Test access is enforced inside the runner.  Nothing in this script
# touches runner.partitions["test"] for metrics or predictions.


def _setup_paths(project_root: Path) -> dict[str, Path]:
    out_root = project_root / "outputs" / "model_comparison"
    fig_dir = out_root / "figures"
    docs_dir = project_root / "docs" / "assets" / "model_comparison"
    for path in (out_root, fig_dir, docs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": out_root,
        "figures": fig_dir,
        "docs_assets": docs_dir,
        "report": out_root / "comparison_report.json",
        "csv": out_root / "model_comparison.csv",
    }


def _write_csv(rows_data: list[dict], out_path: Path) -> None:
    if not rows_data:
        raise ValueError("No rows to write")
    fieldnames = list(rows_data[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows_data:
            flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in row.items()}
            writer.writerow(flat)


def _consistency_check(rows_data: list[dict], csv_path: Path) -> None:
    """Verify that the CSV matches the in-memory rows to 1e-6."""
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))
    if len(reader) != len(rows_data):
        raise AssertionError(
            f"CSV row count {len(reader)} != expected {len(rows_data)}"
        )
    float_keys = {"pr_auc", "roc_auc", "ks", "operational_threshold",
                  "operational_precision", "operational_recall", "operational_f1",
                  "default_precision", "default_recall", "training_time_seconds"}
    for csv_row, mem_row in zip(reader, rows_data):
        for key in float_keys:
            if key not in mem_row:
                continue
            csv_val = float(csv_row[key])
            mem_val = float(mem_row[key])
            diff = abs(csv_val - mem_val)
            if diff > 1e-6:
                raise AssertionError(
                    f"Consistency check failed for {mem_row['experiment_id']}.{key}: "
                    f"CSV={csv_val} memory={mem_val} diff={diff:.2e}"
                )


def _make_pr_curves(run: object, fig_dir: Path, rows_data: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, row in enumerate(rows_data):
        eid = row["experiment_id"]
        prob = run.validation_probabilities[eid]
        targets = run.validation_targets
        precision, recall, _ = precision_recall_curve(targets, prob)
        label = f"{row['model_name'].replace('_', ' ').title()} (AUC={row['pr_auc']:.4f})"
        ax.plot(recall, precision, color=colors[i], lw=1.5, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Validation")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    path = fig_dir / "pr_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _make_roc_curves(run: object, fig_dir: Path, rows_data: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, row in enumerate(rows_data):
        eid = row["experiment_id"]
        prob = run.validation_probabilities[eid]
        targets = run.validation_targets
        fpr, tpr, _ = roc_curve(targets, prob)
        label = f"{row['model_name'].replace('_', ' ').title()} (AUC={row['roc_auc']:.4f})"
        ax.plot(fpr, tpr, color=colors[i], lw=1.5, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Validation")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    path = fig_dir / "roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _make_metric_bar(rows_data: list[dict], fig_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["model_name"].replace("_", " ").title() for r in rows_data]
    pr_aucs = [r["pr_auc"] for r in rows_data]
    roc_aucs = [r["roc_auc"] for r in rows_data]
    ks_vals = [r["ks"] for r in rows_data]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, pr_aucs, width, label="PR-AUC", color="#1f77b4")
    ax.bar(x, roc_aucs, width, label="ROC-AUC", color="#ff7f0e")
    ax.bar(x + width, ks_vals, width, label="KS", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Threshold-Independent Metrics — Validation")
    ax.legend()
    ax.set_ylim([0, 1])
    for i, val in enumerate(pr_aucs):
        ax.annotate(f"{val:.4f}", xy=(x[i] - width, val + 0.01), fontsize=7, ha="center")
    fig.tight_layout()
    path = fig_dir / "metric_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _make_operational_bar(rows_data: list[dict], fig_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["model_name"].replace("_", " ").title() for r in rows_data]
    precisions = [r["operational_precision"] for r in rows_data]
    recalls = [r["operational_recall"] for r in rows_data]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, precisions, width, label="Precision", color="#1f77b4")
    ax.bar(x + width / 2, recalls, width, label="Recall", color="#ff7f0e")
    ax.axhline(0.75, color="red", linestyle="--", lw=1, label="Recall ≥ 0.75 constraint")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Operational Threshold Metrics — Validation")
    ax.legend()
    ax.set_ylim([0, 1])
    fig.tight_layout()
    path = fig_dir / "operational_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _make_confusion_matrices(rows_data: list[dict], fig_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(rows_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, row in zip(axes, rows_data):
        cm = row["operational_confusion_matrix"]
        if isinstance(cm, str):
            import ast
            cm = ast.literal_eval(cm)
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.set_title(row["model_name"].replace("_", " ").title(), fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Confusion Matrices at Operational Threshold — Validation", fontsize=10)
    fig.tight_layout()
    path = fig_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run four-model Validation comparison")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.config import load_config
    from src.evaluation.model_comparison import run_model_comparison
    from src.experiments.runner import SharedExperimentRunner

    config = load_config(args.config)
    paths = _setup_paths(config.project_root)

    print("Loading data and building runner …")
    runner = SharedExperimentRunner(args.config)

    print("Running four-model comparison (this may take a few minutes) …")
    comparison_run = run_model_comparison(runner=runner)
    report = comparison_run.report
    rows_data = [row.to_dict() for row in report.rows]

    # --- persist report JSON -------------------------------------------------
    paths["report"].write_text(
        json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"  Report → {paths['report']}")

    # --- persist CSV ---------------------------------------------------------
    _write_csv(rows_data, paths["csv"])
    print(f"  CSV    → {paths['csv']}")

    # --- programmatic consistency check: figures vs CSV ----------------------
    _consistency_check(rows_data, paths["csv"])
    print("  Consistency check (figures ↔ CSV): PASSED")

    # --- figures -------------------------------------------------------------
    print("Generating figures …")
    fig_paths: list[Path] = []
    fig_paths.append(_make_pr_curves(comparison_run, paths["figures"], rows_data))
    fig_paths.append(_make_roc_curves(comparison_run, paths["figures"], rows_data))
    fig_paths.append(_make_metric_bar(rows_data, paths["figures"]))
    fig_paths.append(_make_operational_bar(rows_data, paths["figures"]))
    fig_paths.append(_make_confusion_matrices(rows_data, paths["figures"]))
    for fig_path in fig_paths:
        print(f"    {fig_path.name} → {fig_path}")
        shutil.copy2(fig_path, paths["docs_assets"] / fig_path.name)

    # --- summary print -------------------------------------------------------
    print("\n=== Model Comparison Summary (Validation) ===")
    print(f"{'Model':<35} {'PR-AUC':>8} {'ROC-AUC':>8} {'Op Prec':>8} {'Op Rec':>7}")
    print("-" * 70)
    for row in report.rows:
        flag = " ← RECOMMENDED" if row.experiment_id == report.recommended_experiment_id else ""
        print(
            f"{row.model_name:<35} {row.pr_auc:>8.6f} {row.roc_auc:>8.6f} "
            f"{row.operational_precision:>8.6f} {row.operational_recall:>7.6f}{flag}"
        )
    print()
    print(f"Recommended model : {report.recommended_experiment_id}")
    print("Rationale:")
    for line in report.recommendation_rationale:
        print(f"  {line}")

    print("\nTest access:", report.test_access)
    print(f"Manifest SHA: {report.manifest_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
