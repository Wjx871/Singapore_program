"""XGBoost search script contract tests.

Tests cover:
  F. Search script contract — candidate count, uniqueness, parameter changes,
     selection rule tie-breaking (including PR-AUC ≤ 1e-6 threshold).
"""

from __future__ import annotations

import pytest

from scripts.run_xgboost_search import CANDIDATE_MATRIX, select_best_candidate


# ===================================================================
# F. Search script contract
# ===================================================================
class TestSearchCandidates:
    """Tests that verify the 8 candidate matrix definition independently."""

    FIXED_COMMON = {
        "n_estimators": 2000, "reg_lambda": 1.0,
        "early_stopping_rounds": 50, "n_jobs": -1,
    }

    def test_candidate_count_is_eight(self):
        assert len(CANDIDATE_MATRIX) == 8

    def test_candidate_ids_are_unique(self):
        ids = [c["id"] for c in CANDIDATE_MATRIX]
        assert len(ids) == len(set(ids))

    def test_each_candidate_changes_only_declared_parameters(self):
        baseline_params = {
            "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }
        for candidate in CANDIDATE_MATRIX:
            diff = {
                k for k in baseline_params
                if candidate.get(k) != baseline_params[k]
            }
            if candidate["id"] == "xgb_baseline":
                assert diff == set(), f"{candidate['id']} should match baseline"
            else:
                assert len(diff) == 1, (
                    f"{candidate['id']} changes {diff} parameters, expected exactly 1"
                )

    def test_fixed_common_parameters_present(self):
        for candidate in CANDIDATE_MATRIX:
            for key in self.FIXED_COMMON:
                assert key not in candidate, (
                    f"{candidate['id']} should not override fixed param {key}"
                )


class TestSelectionRule:
    """Pre-declared candidate selection rule — imports the real function."""

    def test_tie_break_order_matches_declared(self):
        """Identical PR-AUC → ROC-AUC tie-break, then KS."""
        results = [
            {"experiment_id": "xgb_child5", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 120.0},
            {"experiment_id": "xgb_depth3", "pr_auc": 0.40, "roc_auc": 0.86, "ks": 0.55, "training_time_seconds": 120.0},
            {"experiment_id": "xgb_full_rows", "pr_auc": 0.40, "roc_auc": 0.86, "ks": 0.56, "training_time_seconds": 120.0},
        ]
        selected = select_best_candidate(results)
        # All PR-AUC tied → xgb_full_rows wins: higher ROC-AUC than xgb_child5,
        # same ROC-AUC as xgb_depth3 but higher KS
        assert selected["experiment_id"] == "xgb_full_rows"

    def test_training_time_tie_break(self):
        results = [
            {"experiment_id": "xgb_lr003", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 200.0},
            {"experiment_id": "xgb_depth3", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 100.0},
        ]
        selected = select_best_candidate(results)
        assert selected["experiment_id"] == "xgb_depth3"

    def test_experiment_id_order_tie_break(self):
        """All metrics identical → earlier in CANDIDATE_MATRIX wins."""
        results = [
            {"experiment_id": "xgb_child10", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 100.0},
            {"experiment_id": "xgb_baseline", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 100.0},
        ]
        selected = select_best_candidate(results)
        # xgb_baseline is index 0 in CANDIDATE_MATRIX, xgb_child10 is index 5
        assert selected["experiment_id"] == "xgb_baseline"

    def test_higher_pr_auc_wins_regardless(self):
        results = [
            {"experiment_id": "xgb_depth5", "pr_auc": 0.30, "roc_auc": 0.99, "ks": 0.99, "training_time_seconds": 10.0},
            {"experiment_id": "xgb_child5", "pr_auc": 0.35, "roc_auc": 0.80, "ks": 0.50, "training_time_seconds": 999.0},
        ]
        selected = select_best_candidate(results)
        assert selected["experiment_id"] == "xgb_child5"

    def test_pr_auc_diff_5e_7_uses_roc_auc_tiebreak(self):
        """PR-AUC diff = 5e-7 (within 1e-6) → ROC-AUC tie-break applies."""
        results = [
            {"experiment_id": "xgb_depth3", "pr_auc": 0.4000005, "roc_auc": 0.82, "ks": 0.50, "training_time_seconds": 100.0},
            {"experiment_id": "xgb_baseline", "pr_auc": 0.4000000, "roc_auc": 0.88, "ks": 0.50, "training_time_seconds": 100.0},
        ]
        selected = select_best_candidate(results)
        # Diff = 5e-7 ≤ 1e-6 → tie-break; xgb_baseline has higher ROC-AUC
        assert selected["experiment_id"] == "xgb_baseline"

    def test_pr_auc_diff_2e_6_higher_pr_auc_wins(self):
        """PR-AUC diff = 2e-6 (exceeds 1e-6) → higher PR-AUC wins outright."""
        results = [
            {"experiment_id": "xgb_depth5", "pr_auc": 0.400002, "roc_auc": 0.70, "ks": 0.40, "training_time_seconds": 999.0},
            {"experiment_id": "xgb_lr003", "pr_auc": 0.400000, "roc_auc": 0.99, "ks": 0.99, "training_time_seconds": 10.0},
        ]
        selected = select_best_candidate(results)
        # Diff = 2e-6 > 1e-6 → xgb_depth5 wins despite worse secondary metrics
        assert selected["experiment_id"] == "xgb_depth5"

    def test_all_identical_uses_matrix_order(self):
        """All metrics identical → fixed candidate matrix order decides."""
        results = [
            {"experiment_id": "xgb_child10", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 100.0},
            {"experiment_id": "xgb_baseline", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 100.0},
            {"experiment_id": "xgb_depth3", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 100.0},
        ]
        selected = select_best_candidate(results)
        # CANDIDATE_MATRIX order: xgb_baseline (0), xgb_depth3 (1), xgb_child10 (5)
        assert selected["experiment_id"] == "xgb_baseline"

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="empty"):
            select_best_candidate([])
