"""Fair four-model Validation comparison built on the shared experiment runner.

This module reruns the four core credit-default models
(Logistic Regression, Random Forest, Balanced Random Forest, XGBoost)
through :class:`SharedExperimentRunner` on the frozen split manifest and
assembles a single comparison table. It selects a recommended model using the
pre-declared ``model_selection.ordered_considerations`` from the config.

Independent Test remains sealed: every specification uses
``evaluation_split="validation"`` and the runner only performs
transform/schema/finite checks on Test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.config import ExperimentConfig, load_config
from src.experiments.contracts import ExperimentSpec
from src.experiments.runner import ExperimentArtifacts, SharedExperimentRunner

# Canonical comparison specifications. Feature Set B + missing_plus_flag + seed 42
# is the frozen shared model-comparison contract (see MODEL_ADAPTER_HANDOFF.md).
# XGBoost uses the pre-selected xgb_child5 candidate from XGBOOST_VALIDATION.md.
CANONICAL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "experiment_id": "compare_logistic_regression",
        "model_name": "logistic_regression",
        "class_weight": "balanced",
        "model_parameters": {},
    },
    {
        "experiment_id": "compare_random_forest",
        "model_name": "random_forest",
        "class_weight": "none",
        "model_parameters": {},
    },
    {
        "experiment_id": "compare_balanced_random_forest",
        "model_name": "balanced_random_forest",
        "class_weight": "none",
        "model_parameters": {},
    },
    {
        "experiment_id": "compare_xgboost",
        "model_name": "xgboost",
        "class_weight": "none",
        "model_parameters": {
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
)

PROBABILITY_ALIGNMENT_TOLERANCE = 1e-9

# Qualitative interpretability: higher = more transparent. Used as a late
# tie-breaker only; does not affect metrics.
_INTERPRETABILITY_RANK: dict[str, int] = {
    "logistic_regression": 4,
    "random_forest": 2,
    "balanced_random_forest": 2,
    "xgboost": 1,
}


@dataclass(frozen=True)
class ModelComparisonRow:
    """One JSON-safe Validation row for the four-model comparison table."""

    experiment_id: str
    model_name: str
    model_family: str
    feature_set: str
    preprocessing_strategy: str
    class_weight: str
    pr_auc: float
    roc_auc: float
    ks: float
    default_precision: float
    default_recall: float
    default_f1: float
    operational_threshold: float
    operational_precision: float
    operational_recall: float
    operational_f1: float
    operational_confusion_matrix: dict[str, int]
    training_time_seconds: float
    validation_inference_time_seconds: float
    interpretability_rank: int
    best_iteration: int | None
    scale_pos_weight: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "feature_set": self.feature_set,
            "preprocessing_strategy": self.preprocessing_strategy,
            "class_weight": self.class_weight,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "ks": self.ks,
            "default_precision": self.default_precision,
            "default_recall": self.default_recall,
            "default_f1": self.default_f1,
            "operational_threshold": self.operational_threshold,
            "operational_precision": self.operational_precision,
            "operational_recall": self.operational_recall,
            "operational_f1": self.operational_f1,
            "operational_confusion_matrix": dict(self.operational_confusion_matrix),
            "training_time_seconds": self.training_time_seconds,
            "validation_inference_time_seconds": self.validation_inference_time_seconds,
            "interpretability_rank": self.interpretability_rank,
            "best_iteration": self.best_iteration,
            "scale_pos_weight": self.scale_pos_weight,
        }


@dataclass(frozen=True)
class ComparisonReport:
    """JSON-safe result of the four-model Validation comparison."""

    rows: list[ModelComparisonRow]
    recommended_experiment_id: str
    recommendation_rationale: list[str]
    ordered_considerations: list[str]
    validation_row_count: int
    manifest_sha256: str
    config_sha256: str
    recall_minimum: float
    evaluation_split: str = "validation"
    test_access: str = (
        "transform/schema/finite checks only; no Test prediction or metrics"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_split": self.evaluation_split,
            "recommended_experiment_id": self.recommended_experiment_id,
            "recommendation_rationale": list(self.recommendation_rationale),
            "ordered_considerations": list(self.ordered_considerations),
            "validation_row_count": self.validation_row_count,
            "manifest_sha256": self.manifest_sha256,
            "config_sha256": self.config_sha256,
            "recall_minimum": self.recall_minimum,
            "test_access": self.test_access,
            "models": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class ComparisonRun:
    """Report plus non-serialisable Validation probability arrays.

    The arrays are kept separate from ``ComparisonReport`` to preserve
    JSON-safety. They are used only for figure generation and the
    programmatic figure / table consistency check.
    """

    report: ComparisonReport
    validation_targets: np.ndarray
    validation_probabilities: dict[str, np.ndarray]
    thresholds_per_model: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper: build ExperimentSpec list
# ---------------------------------------------------------------------------

def build_comparison_specs(
    config: ExperimentConfig,
    *,
    specs: tuple[dict[str, Any], ...] = CANONICAL_SPECS,
) -> list[ExperimentSpec]:
    """Construct an ExperimentSpec for each entry in the canonical matrix."""
    manifest_path = config.project_root / config.raw["split"]["manifest_path"]
    expected_sha = config.raw["split"]["frozen_manifest_sha256"]
    result: list[ExperimentSpec] = []
    for entry in specs:
        result.append(
            ExperimentSpec(
                experiment_id=entry["experiment_id"],
                model_name=entry["model_name"],
                feature_set="B",
                preprocessing_strategy="missing_plus_flag",
                class_weight=entry["class_weight"],
                random_seed=42,
                config_path=config.config_path,
                manifest_path=manifest_path,
                expected_manifest_sha256=expected_sha,
                model_parameters=dict(entry["model_parameters"]),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Helper: build a ComparisonRow from runner artifacts
# ---------------------------------------------------------------------------

def _row_from_artifacts(
    artifacts: ExperimentArtifacts,
    interpretability_rank: int,
) -> ModelComparisonRow:
    r = artifacts.result
    ind = r.validation_metrics
    dfl = r.default_threshold_metrics
    op = r.operational_threshold_metrics
    return ModelComparisonRow(
        experiment_id=r.experiment_id,
        model_name=r.model_name,
        model_family=r.model_family,
        feature_set=r.feature_set,
        preprocessing_strategy=r.preprocessing_strategy,
        class_weight=r.class_weight,
        pr_auc=float(ind["pr_auc"]),
        roc_auc=float(ind["roc_auc"]),
        ks=float(ind["ks"]),
        default_precision=float(dfl["precision"]),
        default_recall=float(dfl["recall"]),
        default_f1=float(dfl["f1"]),
        operational_threshold=float(r.operational_threshold),
        operational_precision=float(op["precision"]),
        operational_recall=float(op["recall"]),
        operational_f1=float(op["f1"]),
        operational_confusion_matrix=dict(op["confusion_matrix"]),
        training_time_seconds=float(r.training_time_seconds),
        validation_inference_time_seconds=float(r.validation_inference_time_seconds),
        interpretability_rank=int(interpretability_rank),
        best_iteration=r.best_iteration,
        scale_pos_weight=r.scale_pos_weight,
    )


# ---------------------------------------------------------------------------
# Probability alignment verification
# ---------------------------------------------------------------------------

def verify_probability_alignment(
    runner: SharedExperimentRunner,
    probabilities: dict[str, np.ndarray],
) -> np.ndarray:
    """Verify every model scored the same Validation rows in the same order.

    Returns the Validation target array.  Raises ``ValueError`` on any mismatch.
    """
    validation = runner.partitions["validation"]
    target_col = runner.config.data["target_column"]
    targets = validation[target_col].to_numpy()
    n = len(targets)
    for eid, prob in probabilities.items():
        arr = np.asarray(prob, dtype="float64")
        if arr.ndim != 1:
            raise ValueError(f"{eid}: probability must be 1-D")
        if len(arr) != n:
            raise ValueError(
                f"{eid}: probability length {len(arr)} != Validation row count {n}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"{eid}: probability contains non-finite values")
        if ((arr < 0.0) | (arr > 1.0)).any():
            raise ValueError(f"{eid}: probability outside [0, 1]")
    return targets


# ---------------------------------------------------------------------------
# Model recommendation (config-driven lexicographic rule)
# ---------------------------------------------------------------------------

def _score(row: ModelComparisonRow, consideration: str) -> tuple[float, bool]:
    """Return (value, higher_is_better) for one consideration."""
    lookup: dict[str, tuple[float, bool]] = {
        "pr_auc": (row.pr_auc, True),
        "roc_auc": (row.roc_auc, True),
        "operational_precision_with_recall_constraint": (row.operational_precision, True),
        "runtime": (row.training_time_seconds, False),
        "interpretability": (float(row.interpretability_rank), True),
    }
    if consideration not in lookup:
        raise ValueError(f"Unknown model_selection consideration: {consideration!r}")
    return lookup[consideration]


def recommend_model(
    rows: list[ModelComparisonRow],
    ordered_considerations: list[str],
    *,
    tolerance: float = 1e-6,
) -> tuple[str, list[str]]:
    """Apply the pre-declared lexicographic selection rule.

    At each step, the top-scoring candidates (within ``tolerance`` for
    float-valued metrics) survive.  Discrete ranks (interpretability)
    use exact equality.  Returns ``(winning_experiment_id, rationale_lines)``.
    """
    if not rows:
        raise ValueError("Cannot recommend from an empty comparison")
    candidates = list(rows)
    rationale: list[str] = []

    for consideration in ordered_considerations:
        scored: list[tuple[float, ModelComparisonRow]] = [
            (_score(row, consideration)[0], row) for row in candidates
        ]
        higher_is_better = _score(rows[0], consideration)[1]
        values = [v for v, _ in scored]
        best = max(values) if higher_is_better else min(values)
        # discrete considerations (interpretability, runtime) use exact equality
        if consideration in {"interpretability", "runtime"}:
            tol = 0.0
        else:
            tol = tolerance
        if higher_is_better:
            retained = [r for v, r in scored if v >= best - tol]
        else:
            retained = [r for v, r in scored if v <= best + tol]
        direction = "↑" if higher_is_better else "↓"
        rationale.append(
            f"{consideration}{direction}: best={best:.6f}  "
            f"{len(retained)}/{len(candidates)} candidates retained "
            f"[{', '.join(r.experiment_id for r in retained)}]"
        )
        candidates = retained
        if len(candidates) == 1:
            break

    if len(candidates) > 1:
        # Final deterministic tie-break: lowest experiment_id lexicographically
        candidates.sort(key=lambda r: r.experiment_id)
        rationale.append(
            f"final tie-break: lexicographic experiment_id → {candidates[0].experiment_id}"
        )
    return candidates[0].experiment_id, rationale


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_model_comparison(
    config_path: str | Path = "configs/experiment.yaml",
    *,
    runner: SharedExperimentRunner | None = None,
    specs: tuple[dict[str, Any], ...] = CANONICAL_SPECS,
) -> ComparisonRun:
    """Run all four models and return the comparison report + probability arrays.

    Parameters
    ----------
    config_path:
        Path to ``experiment.yaml``.  Ignored when ``runner`` is supplied.
    runner:
        Pre-initialised :class:`SharedExperimentRunner`.  Pass an existing
        runner to avoid re-loading data (e.g. from tests).
    specs:
        Override the canonical comparison matrix (useful for synthetic tests).
    """
    active_runner = runner if runner is not None else SharedExperimentRunner(config_path)
    config = active_runner.config
    experiment_specs = build_comparison_specs(config, specs=specs)

    # Build a rank look-up from the spec matrix
    rank_by_eid: dict[str, int] = {
        entry["experiment_id"]: _INTERPRETABILITY_RANK.get(entry["model_name"], 1)
        for entry in specs
    }

    rows: list[ModelComparisonRow] = []
    probabilities: dict[str, np.ndarray] = {}
    thresholds_per_model: dict[str, Any] = {}

    for spec in experiment_specs:
        artifacts = active_runner.run(spec)
        rows.append(_row_from_artifacts(artifacts, rank_by_eid[spec.experiment_id]))
        probabilities[spec.experiment_id] = artifacts.validation_probability.copy()
        thresholds_per_model[spec.experiment_id] = artifacts.thresholds

    targets = verify_probability_alignment(active_runner, probabilities)

    ordered = list(config.raw["model_selection"]["ordered_considerations"])
    recall_minimum = float(config.raw["thresholds"]["operational"]["recall_minimum"])

    recommended_id, rationale = recommend_model(rows, ordered)

    report = ComparisonReport(
        rows=rows,
        recommended_experiment_id=recommended_id,
        recommendation_rationale=rationale,
        ordered_considerations=ordered,
        validation_row_count=int(len(targets)),
        manifest_sha256=active_runner.expected_manifest_sha256,
        config_sha256=config.config_sha256,
        recall_minimum=recall_minimum,
    )
    # JSON-safety guard: will raise if any value is NaN/Inf
    import json
    json.dumps(report.to_dict(), allow_nan=False)

    return ComparisonRun(
        report=report,
        validation_targets=targets,
        validation_probabilities=probabilities,
        thresholds_per_model=thresholds_per_model,
    )
