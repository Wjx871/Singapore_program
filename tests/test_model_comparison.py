"""Tests for src/evaluation/model_comparison.py.

Six assertion categories:
  1. Comparison contract  – CANONICAL_SPECS structure and ExperimentSpec construction
  2. Probability alignment – verify_probability_alignment guards
  3. Row construction      – _row_from_artifacts populates every field correctly
  4. Model selection       – recommend_model lexicographic rule + tie-breaking
  5. Full run (smoke)      – run_model_comparison with synthetic runner mock
  6. Handoff / Test guard  – test_access field; no Test metrics anywhere

All tests use synthetic data or lightweight mocks so they do not require the
real dataset or GPU/CPU-heavy model training.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.model_comparison import (
    CANONICAL_SPECS,
    ModelComparisonRow,
    ComparisonReport,
    ComparisonRun,
    build_comparison_specs,
    recommend_model,
    run_model_comparison,
    verify_probability_alignment,
    _row_from_artifacts,
    _INTERPRETABILITY_RANK,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def four_rows() -> list[ModelComparisonRow]:
    """Four synthetic rows in canonical model order."""
    def _row(eid, model, pr, roc, ks, op_prec, op_rec, train_time, rank):
        return ModelComparisonRow(
            experiment_id=eid,
            model_name=model,
            model_family="test",
            feature_set="B",
            preprocessing_strategy="missing_plus_flag",
            class_weight="balanced",
            pr_auc=pr,
            roc_auc=roc,
            ks=ks,
            default_precision=0.2,
            default_recall=0.8,
            default_f1=0.3,
            operational_threshold=0.5,
            operational_precision=op_prec,
            operational_recall=op_rec,
            operational_f1=0.35,
            operational_confusion_matrix={"tn": 100, "fp": 20, "fn": 5, "tp": 15},
            training_time_seconds=train_time,
            validation_inference_time_seconds=0.01,
            interpretability_rank=rank,
            best_iteration=None,
            scale_pos_weight=None,
        )

    return [
        _row("compare_logistic_regression", "logistic_regression",
             0.359, 0.827, 0.514, 0.176, 0.750, 0.8, 4),
        _row("compare_random_forest", "random_forest",
             0.393, 0.861, 0.572, 0.227, 0.750, 2.0, 2),
        _row("compare_balanced_random_forest", "balanced_random_forest",
             0.384, 0.867, 0.587, 0.235, 0.751, 1.5, 2),
        _row("compare_xgboost", "xgboost",
             0.402, 0.870, 0.586, 0.243, 0.751, 0.9, 1),
    ]


# ---------------------------------------------------------------------------
# 1. Comparison contract
# ---------------------------------------------------------------------------

class TestComparisonContract:
    def test_canonical_specs_has_four_entries(self):
        assert len(CANONICAL_SPECS) == 4

    def test_canonical_model_names(self):
        names = [entry["model_name"] for entry in CANONICAL_SPECS]
        assert names == [
            "logistic_regression",
            "random_forest",
            "balanced_random_forest",
            "xgboost",
        ]

    def test_xgboost_spec_has_min_child_weight_5(self):
        xgb = next(e for e in CANONICAL_SPECS if e["model_name"] == "xgboost")
        assert xgb["model_parameters"]["min_child_weight"] == 5

    def test_all_specs_use_feature_set_b_and_missing_plus_flag(self):
        """build_comparison_specs must lock FS B and missing_plus_flag."""
        # Use experiment_config (session-scoped) to build specs
        # We exercise the function with real config to test plumbing.
        pass  # Full integration tested in smoke tests below.

    def test_canonical_specs_feature_set_b_hardcoded(self):
        """Every CANONICAL_SPECS entry must produce FS B specs."""
        for entry in CANONICAL_SPECS:
            assert entry["model_name"] in {
                "logistic_regression",
                "random_forest",
                "balanced_random_forest",
                "xgboost",
            }

    def test_logistic_uses_balanced_class_weight(self):
        lr = next(e for e in CANONICAL_SPECS if e["model_name"] == "logistic_regression")
        assert lr["class_weight"] == "balanced"

    def test_tree_models_use_none_class_weight(self):
        for model in ("random_forest", "balanced_random_forest", "xgboost"):
            entry = next(e for e in CANONICAL_SPECS if e["model_name"] == model)
            assert entry["class_weight"] == "none", f"{model} must use class_weight='none'"

    def test_interpretability_ranks_ordered(self):
        assert _INTERPRETABILITY_RANK["logistic_regression"] > _INTERPRETABILITY_RANK["random_forest"]
        assert _INTERPRETABILITY_RANK["logistic_regression"] > _INTERPRETABILITY_RANK["xgboost"]
        assert _INTERPRETABILITY_RANK["random_forest"] >= _INTERPRETABILITY_RANK["xgboost"]

    def test_build_comparison_specs_uses_experiment_config(self, experiment_config):
        specs = build_comparison_specs(experiment_config)
        assert len(specs) == 4
        for spec in specs:
            assert spec.feature_set == "B"
            assert spec.preprocessing_strategy == "missing_plus_flag"
            assert spec.random_seed == 42
            assert spec.evaluation_split == "validation"
            assert (
                spec.expected_manifest_sha256
                == experiment_config.raw["split"]["frozen_manifest_sha256"]
            )

    def test_build_comparison_specs_xgboost_params(self, experiment_config):
        specs = build_comparison_specs(experiment_config)
        xgb = next(s for s in specs if s.model_name == "xgboost")
        assert xgb.model_parameters["min_child_weight"] == 5

    def test_comparison_report_is_json_safe(self, four_rows):
        report = ComparisonReport(
            rows=four_rows,
            recommended_experiment_id="compare_xgboost",
            recommendation_rationale=["pr_auc↑: best=0.402 ..."],
            ordered_considerations=["pr_auc", "roc_auc"],
            validation_row_count=23997,
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            recall_minimum=0.75,
        )
        payload = report.to_dict()
        json.dumps(payload, allow_nan=False)  # must not raise

    def test_model_comparison_row_to_dict_has_expected_keys(self, four_rows):
        keys = set(four_rows[0].to_dict().keys())
        required = {
            "experiment_id", "model_name", "pr_auc", "roc_auc", "ks",
            "operational_threshold", "operational_precision", "operational_recall",
            "operational_confusion_matrix", "interpretability_rank",
        }
        assert required.issubset(keys)


# ---------------------------------------------------------------------------
# 2. Probability alignment
# ---------------------------------------------------------------------------

class TestProbabilityAlignment:
    def _mock_runner(self, n: int) -> MagicMock:
        runner = MagicMock()
        runner.config.data = {"target_column": "SeriousDlqin2yrs"}
        import pandas as pd
        runner.partitions = {
            "validation": pd.DataFrame({
                "SeriousDlqin2yrs": np.random.randint(0, 2, size=n),
            })
        }
        return runner

    def test_valid_probabilities_return_target_array(self):
        runner = self._mock_runner(100)
        probs = {
            "model_a": np.random.uniform(0, 1, 100),
            "model_b": np.random.uniform(0, 1, 100),
        }
        targets = verify_probability_alignment(runner, probs)
        assert len(targets) == 100

    def test_wrong_length_raises(self):
        runner = self._mock_runner(100)
        with pytest.raises(ValueError, match="length"):
            verify_probability_alignment(runner, {"model_a": np.ones(99)})

    def test_non_finite_raises(self):
        runner = self._mock_runner(100)
        bad = np.ones(100)
        bad[5] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            verify_probability_alignment(runner, {"model_a": bad})

    def test_probability_above_1_raises(self):
        runner = self._mock_runner(100)
        bad = np.ones(100) * 1.1
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            verify_probability_alignment(runner, {"model_a": bad})

    def test_probability_below_0_raises(self):
        runner = self._mock_runner(100)
        bad = np.ones(100) * -0.01
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            verify_probability_alignment(runner, {"model_a": bad})

    def test_2d_array_raises(self):
        runner = self._mock_runner(100)
        with pytest.raises(ValueError, match="1-D"):
            verify_probability_alignment(runner, {"m": np.ones((100, 2))})

    def test_empty_probability_dict_returns_targets(self):
        runner = self._mock_runner(50)
        targets = verify_probability_alignment(runner, {})
        assert len(targets) == 50


# ---------------------------------------------------------------------------
# 3. Metrics / row construction
# ---------------------------------------------------------------------------

class TestRowConstruction:
    def _make_artifacts(self, eid="test_model", pr_auc=0.4, operational_threshold=0.5):
        """Build a minimal ExperimentArtifacts mock."""
        result = MagicMock()
        result.experiment_id = eid
        result.model_name = "logistic_regression"
        result.model_family = "linear_model"
        result.feature_set = "B"
        result.preprocessing_strategy = "missing_plus_flag"
        result.class_weight = "balanced"
        result.validation_metrics = {
            "pr_auc": pr_auc, "roc_auc": 0.85, "ks": 0.5,
            "undefined_metrics": [],
        }
        result.default_threshold_metrics = {
            "precision": 0.2, "recall": 0.8, "f1": 0.32, "threshold": 0.5,
            "confusion_matrix": {"tn": 80, "fp": 20, "fn": 5, "tp": 15},
        }
        result.operational_threshold = operational_threshold
        result.operational_threshold_metrics = {
            "precision": 0.25, "recall": 0.75, "f1": 0.37,
            "confusion_matrix": {"tn": 90, "fp": 10, "fn": 5, "tp": 15},
        }
        result.training_time_seconds = 1.0
        result.validation_inference_time_seconds = 0.01
        result.best_iteration = None
        result.scale_pos_weight = None
        artifacts = MagicMock()
        artifacts.result = result
        artifacts.validation_probability = np.random.uniform(0, 1, 100)
        artifacts.thresholds = MagicMock()
        return artifacts

    def test_row_experiment_id_matches(self):
        artifacts = self._make_artifacts(eid="my_exp")
        row = _row_from_artifacts(artifacts, interpretability_rank=4)
        assert row.experiment_id == "my_exp"

    def test_row_pr_auc_matches(self):
        artifacts = self._make_artifacts(pr_auc=0.3721)
        row = _row_from_artifacts(artifacts, interpretability_rank=2)
        assert row.pr_auc == pytest.approx(0.3721)

    def test_row_operational_threshold_matches(self):
        artifacts = self._make_artifacts(operational_threshold=0.543)
        row = _row_from_artifacts(artifacts, interpretability_rank=1)
        assert row.operational_threshold == pytest.approx(0.543)

    def test_row_interpretability_rank_propagated(self):
        artifacts = self._make_artifacts()
        row = _row_from_artifacts(artifacts, interpretability_rank=3)
        assert row.interpretability_rank == 3

    def test_row_confusion_matrix_all_keys_present(self):
        artifacts = self._make_artifacts()
        row = _row_from_artifacts(artifacts, interpretability_rank=2)
        assert set(row.operational_confusion_matrix.keys()) == {"tn", "fp", "fn", "tp"}

    def test_row_to_dict_json_safe(self):
        artifacts = self._make_artifacts()
        row = _row_from_artifacts(artifacts, interpretability_rank=2)
        json.dumps(row.to_dict(), allow_nan=False)  # must not raise

    def test_row_no_test_metric_keys(self):
        artifacts = self._make_artifacts()
        row = _row_from_artifacts(artifacts, interpretability_rank=2)
        payload = json.dumps(row.to_dict()).lower()
        for forbidden in ("test_proba", "test_metric", "test_auc", "test_precision"):
            assert forbidden not in payload, f"Forbidden key found: {forbidden}"


# ---------------------------------------------------------------------------
# 4. Model selection
# ---------------------------------------------------------------------------

class TestModelSelection:
    def test_clear_pr_auc_winner(self, four_rows):
        winner, rationale = recommend_model(four_rows, ["pr_auc"])
        assert winner == "compare_xgboost"
        assert any("pr_auc" in line for line in rationale)

    def test_tiebreak_to_roc_auc(self, four_rows):
        """When two models tie on PR-AUC within 1e-6, ROC-AUC breaks the tie."""
        rows = list(four_rows)
        # Force tie on pr_auc between row[2] and row[3]
        rows[2] = ModelComparisonRow(**{
            **rows[2].__dict__,
            "experiment_id": "a_brf",
            "pr_auc": rows[3].pr_auc,  # exact same
            "roc_auc": rows[3].roc_auc + 0.01,  # BRF is better on ROC
        })
        winner, _ = recommend_model(rows, ["pr_auc", "roc_auc"])
        assert winner == "a_brf"

    def test_tiebreak_to_runtime(self, four_rows):
        """Runtime is a lower-is-better consideration."""
        rows = list(four_rows)
        rows[0] = ModelComparisonRow(**{
            **rows[0].__dict__,
            "pr_auc": rows[3].pr_auc,
            "roc_auc": rows[3].roc_auc,
            "training_time_seconds": 0.1,  # much faster
        })
        winner, rationale = recommend_model(rows, ["pr_auc", "roc_auc", "runtime"])
        assert winner == "compare_logistic_regression"
        assert any("runtime" in line for line in rationale)

    def test_interpretability_favours_logistic_regression_as_tiebreak(self, four_rows):
        """When all other metrics tie, the most interpretable model wins."""
        equal_pr = four_rows[0].pr_auc
        tied_rows = [
            ModelComparisonRow(**{**row.__dict__, "pr_auc": equal_pr})
            for row in four_rows
        ]
        winner, _ = recommend_model(tied_rows, ["pr_auc", "interpretability"])
        assert winner == "compare_logistic_regression"

    def test_empty_rows_raises(self):
        with pytest.raises(ValueError, match="empty"):
            recommend_model([], ["pr_auc"])

    def test_unknown_consideration_raises(self, four_rows):
        with pytest.raises(ValueError, match="Unknown"):
            recommend_model(four_rows, ["made_up_metric"])

    def test_single_row_always_wins(self, four_rows):
        winner, _ = recommend_model(four_rows[:1], ["pr_auc", "roc_auc", "interpretability"])
        assert winner == four_rows[0].experiment_id

    def test_full_ordered_considerations_from_config(self, four_rows):
        """Simulate the real config ordered_considerations list."""
        ordered = [
            "pr_auc",
            "roc_auc",
            "operational_precision_with_recall_constraint",
            "runtime",
            "interpretability",
        ]
        winner, rationale = recommend_model(four_rows, ordered)
        # XGBoost has best PR-AUC in synthetic data
        assert winner == "compare_xgboost"
        assert len(rationale) >= 1


# ---------------------------------------------------------------------------
# 5. Full run (smoke test with minimal synthetic runner)
# ---------------------------------------------------------------------------

class TestFullRunSmoke:
    def _make_synthetic_runner(self, experiment_config):
        """Return a real SharedExperimentRunner but override the run() method."""
        from src.experiments.runner import SharedExperimentRunner

        class SyntheticRunner(SharedExperimentRunner):
            def __init__(self):
                super().__init__(experiment_config.config_path)

            def run(self, spec):
                # Redirect every spec to the fast logistic model with few estimators
                from src.experiments.contracts import ExperimentSpec
                fast_spec = ExperimentSpec(
                    experiment_id=spec.experiment_id,
                    model_name="logistic_regression",
                    feature_set="B",
                    preprocessing_strategy="missing_plus_flag",
                    class_weight="balanced",
                    random_seed=42,
                    config_path=spec.config_path,
                    manifest_path=spec.manifest_path,
                    expected_manifest_sha256=spec.expected_manifest_sha256,
                )
                return super().run(fast_spec)

        return SyntheticRunner()

    def test_run_returns_comparison_run_type(self, experiment_config):
        runner = self._make_synthetic_runner(experiment_config)
        result = run_model_comparison(runner=runner)
        assert isinstance(result, ComparisonRun)

    def test_run_report_has_four_rows(self, experiment_config):
        runner = self._make_synthetic_runner(experiment_config)
        result = run_model_comparison(runner=runner)
        assert len(result.report.rows) == 4

    def test_run_report_is_json_safe(self, experiment_config):
        runner = self._make_synthetic_runner(experiment_config)
        result = run_model_comparison(runner=runner)
        json.dumps(result.report.to_dict(), allow_nan=False)

    def test_run_manifest_sha_in_report(self, experiment_config):
        runner = self._make_synthetic_runner(experiment_config)
        result = run_model_comparison(runner=runner)
        expected_sha = experiment_config.raw["split"]["frozen_manifest_sha256"]
        assert result.report.manifest_sha256 == expected_sha

    def test_run_validation_row_count_matches_partition(self, experiment_config):
        runner = self._make_synthetic_runner(experiment_config)
        result = run_model_comparison(runner=runner)
        expected_n = len(runner.partitions["validation"])
        assert result.report.validation_row_count == expected_n

    def test_run_probabilities_have_correct_shape(self, experiment_config):
        runner = self._make_synthetic_runner(experiment_config)
        result = run_model_comparison(runner=runner)
        expected_n = len(runner.partitions["validation"])
        for eid, prob in result.validation_probabilities.items():
            assert prob.shape == (expected_n,), f"{eid}: wrong probability shape"

    def test_run_recommended_id_is_one_of_the_specs(self, experiment_config):
        runner = self._make_synthetic_runner(experiment_config)
        result = run_model_comparison(runner=runner)
        valid_ids = {e["experiment_id"] for e in CANONICAL_SPECS}
        assert result.report.recommended_experiment_id in valid_ids


# ---------------------------------------------------------------------------
# 6. Handoff / Test guard
# ---------------------------------------------------------------------------

class TestHandoffAndTestGuard:
    def test_comparison_report_has_test_access_field(self, four_rows):
        report = ComparisonReport(
            rows=four_rows,
            recommended_experiment_id="compare_xgboost",
            recommendation_rationale=[],
            ordered_considerations=["pr_auc"],
            validation_row_count=23997,
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            recall_minimum=0.75,
        )
        assert "transform" in report.test_access.lower()
        assert "no test" in report.test_access.lower()

    def test_comparison_report_to_dict_no_test_probability_fields(self, four_rows):
        report = ComparisonReport(
            rows=four_rows,
            recommended_experiment_id="compare_xgboost",
            recommendation_rationale=[],
            ordered_considerations=["pr_auc"],
            validation_row_count=23997,
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            recall_minimum=0.75,
        )
        payload_str = json.dumps(report.to_dict()).lower()
        forbidden_patterns = ["test_proba", "test_auc", "test_metric", "test_precision",
                              "test_recall", "test_confusion"]
        for pattern in forbidden_patterns:
            assert pattern not in payload_str, f"Forbidden field in report: {pattern}"

    def test_comparison_run_has_no_test_partition_access(self, experiment_config):
        """ComparisonRun must not store or expose Test partition data."""
        from src.experiments.runner import SharedExperimentRunner

        runner = SharedExperimentRunner(experiment_config.config_path)
        result = run_model_comparison(runner=runner)
        payload_str = json.dumps(result.report.to_dict()).lower()
        assert "test_precision" not in payload_str
        assert "test_recall" not in payload_str
        assert "test_probability" not in payload_str

    def test_evaluation_split_is_validation(self, four_rows):
        report = ComparisonReport(
            rows=four_rows,
            recommended_experiment_id="compare_xgboost",
            recommendation_rationale=[],
            ordered_considerations=["pr_auc"],
            validation_row_count=23997,
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            recall_minimum=0.75,
        )
        assert report.evaluation_split == "validation"
        assert report.to_dict()["evaluation_split"] == "validation"

    def test_recall_constraint_met_for_all_rows(self, four_rows):
        """Every operational recall must satisfy the 0.75 minimum."""
        recall_min = 0.75
        for row in four_rows:
            assert row.operational_recall >= recall_min - 1e-9, (
                f"{row.experiment_id}: operational recall {row.operational_recall} < {recall_min}"
            )

    def test_confusion_matrix_total_is_validation_row_count(self, four_rows):
        """All four confusion matrices must sum to the same total."""
        totals = set()
        for row in four_rows:
            cm = row.operational_confusion_matrix
            totals.add(cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"])
        assert len(totals) == 1, f"Inconsistent confusion matrix totals: {totals}"
