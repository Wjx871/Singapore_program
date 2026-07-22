"""XGBoost search script contract tests.

Tests cover:
  F. Search script contract — candidate count, uniqueness, parameter changes,
     selection rule tie-breaking, output contract.
"""

from __future__ import annotations


# ===================================================================
# F. Search script contract (unit-level, no full data needed)
# ===================================================================
class TestSearchCandidates:
    """Tests that verify the 8 candidate matrix definition independently."""

    CANDIDATES = [
        {
            "id": "xgb_baseline",
            "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 0.8,
        },
        {
            "id": "xgb_depth3",
            "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 0.8,
        },
        {
            "id": "xgb_depth5",
            "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 0.8,
        },
        {
            "id": "xgb_lr003",
            "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 0.8,
        },
        {
            "id": "xgb_child5",
            "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.8,
        },
        {
            "id": "xgb_child10",
            "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 10,
            "subsample": 0.8, "colsample_bytree": 0.8,
        },
        {
            "id": "xgb_full_rows",
            "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1,
            "subsample": 1.0, "colsample_bytree": 0.8,
        },
        {
            "id": "xgb_full_cols",
            "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 1.0,
        },
    ]

    FIXED_COMMON = {
        "n_estimators": 2000, "reg_lambda": 1.0,
        "early_stopping_rounds": 50, "n_jobs": -1,
    }

    def test_candidate_count_is_eight(self):
        assert len(self.CANDIDATES) == 8

    def test_candidate_ids_are_unique(self):
        ids = [c["id"] for c in self.CANDIDATES]
        assert len(ids) == len(set(ids))

    def test_each_candidate_changes_only_declared_parameters(self):
        baseline_params = {
            "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }
        for candidate in self.CANDIDATES:
            diff = {
                k for k in baseline_params
                if candidate.get(k) != baseline_params[k]
            }
            if candidate["id"] == "xgb_baseline":
                assert diff == set(), f"{candidate['id']} should match baseline"
            else:
                # Each non-baseline candidate changes exactly ONE parameter
                assert len(diff) == 1, (
                    f"{candidate['id']} changes {diff} parameters, expected exactly 1"
                )

    def test_fixed_common_parameters_present(self):
        for candidate in self.CANDIDATES:
            for key, value in self.FIXED_COMMON.items():
                # Fixed common params are not in the per-candidate dict,
                # but we verify the candidate dict doesn't override them.
                assert key not in candidate, (
                    f"{candidate['id']} should not override fixed param {key}"
                )


class TestSelectionRule:
    """Pre-declared candidate selection rule tests."""

    def test_tie_break_order_matches_declared(self):
        """Verify tie-breaking: PR-AUC, then ROC-AUC, then KS,
        then training_time, then experiment_id order."""
        # Mock results with identical PR-AUC
        results = [
            {"experiment_id": "c", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 120.0},
            {"experiment_id": "a", "pr_auc": 0.40, "roc_auc": 0.86, "ks": 0.55, "training_time_seconds": 120.0},
            {"experiment_id": "b", "pr_auc": 0.40, "roc_auc": 0.86, "ks": 0.56, "training_time_seconds": 120.0},
        ]
        selected = _select_candidate(results)
        # b wins: same pr_auc, same roc_auc as a, higher ks
        assert selected["experiment_id"] == "b"

    def test_training_time_tie_break(self):
        results = [
            {"experiment_id": "slow", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 200.0},
            {"experiment_id": "fast", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 100.0},
        ]
        selected = _select_candidate(results)
        assert selected["experiment_id"] == "fast"

    def test_experiment_id_order_tie_break(self):
        results = [
            {"experiment_id": "a_first", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 100.0},
            {"experiment_id": "z_last", "pr_auc": 0.40, "roc_auc": 0.85, "ks": 0.55, "training_time_seconds": 100.0},
        ]
        selected = _select_candidate(results)
        assert selected["experiment_id"] == "a_first"

    def test_higher_pr_auc_wins_regardless(self):
        results = [
            {"experiment_id": "low", "pr_auc": 0.30, "roc_auc": 0.99, "ks": 0.99, "training_time_seconds": 10.0},
            {"experiment_id": "high", "pr_auc": 0.35, "roc_auc": 0.80, "ks": 0.50, "training_time_seconds": 999.0},
        ]
        selected = _select_candidate(results)
        assert selected["experiment_id"] == "high"


def _select_candidate(results: list[dict]) -> dict:
    """Replica of the pre-declared selection rule used in the search script."""
    # Primary: max PR-AUC
    # Tie-break: 1) ROC-AUC, 2) KS, 3) shorter training_time, 4) earlier in candidate list
    candidate_order = [r["experiment_id"] for r in results]

    def sort_key(r):
        return (
            r["pr_auc"],
            r["roc_auc"],
            r["ks"],
            -r["training_time_seconds"],  # negated → shorter = larger
            -candidate_order.index(r["experiment_id"]),  # negated → earlier = larger
        )

    return max(results, key=sort_key)
