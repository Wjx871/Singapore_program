# Sealed Evaluation Handoff

> **Branch**: `feat/model-evaluation`
> **Base**: `integration/stage2`
> **Purpose**: Govern the sealed Independent Test set boundary and isolate legacy
> Forest Test artifacts from the Stage 3 comparison pipeline.

## 1. Recommended Model for Stage 3 Evaluation

**Recommended model: XGBoost (`xgb_child5`)**

Basis: highest Validation PR-AUC (0.402297) under the pre-declared
`model_selection.ordered_considerations` rule from `configs/experiment.yaml`.
No Independent Test metric was used for this selection. Full evidence is in
`MODEL_COMPARISON_VALIDATION.md`.

Selected parameters:

| Parameter | Value |
|---|---:|
| max_depth | 4 |
| learning_rate | 0.05 |
| min_child_weight | 5 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| n_estimators | 2000 |
| reg_lambda | 1.0 |
| early_stopping_rounds | 50 |
| n_jobs | -1 |

Selected Validation metrics: PR-AUC 0.402297, ROC-AUC 0.869722, KS 0.586536.

Operational threshold (Validation, Recall ≥ 0.75): **0.553241**.
Operational Precision: 0.243002, Operational Recall: 0.751568.

## 2. Independent Test Governance

The Independent Test set is sealed in Stage 2 and Stage 3. The following
actions are **not permitted** until the Technical Lead explicitly opens the
Test evaluation gate:

| Forbidden action | Enforced by |
|---|---|
| `predict_proba` on Test partition | `SealedTestSetError` guard in `evaluation/guard.py` |
| Computing any metric on Test | `require_evaluation_split("test")` raises |
| Threshold selection on Test | Same guard |
| Model selection on Test | Config: `allow_test_based_reselection: false` |
| Calibration fitting on Test | Config: `allow_test_for_calibration_fit: false` |
| Tuning on Test | Config: `allow_test_for_tuning: false` |
| Early stopping on Test | `XGBoostAdapter` rejects `validation_split_name="test"` |

All six of the above are structurally enforced; they cannot be bypassed by
editing model or evaluation code without also modifying the guard contracts.

The runner performs only three non-metric operations on Test:

1. Apply the fitted preprocessor (`cleaner.transform`).
2. Build Feature Set B features from the cleaned Test partition.
3. Check that the resulting matrix contains no NaN or infinity.

No prediction probabilities or derived metrics leave the runner for the Test
partition.

## 3. Frozen Manifest Governance

The split manifest (`data/processed/split_manifest.csv`) is generated once
and frozen. Its SHA-256 is stored in `configs/experiment.yaml`:

```
frozen_manifest_sha256: 5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5
```

The `SharedExperimentRunner` verifies this SHA on every instantiation and on
every `ExperimentSpec.run()` call. Any manifest regeneration would invalidate
the frozen SHA and all downstream experiments.

## 4. Isolation of Legacy Forest Test Artifacts

Previous development branches (`feat/forest-models`) may contain Test-partition
artifacts (confusion matrices, metric files, probability arrays) that were
generated under a different governance policy or an earlier test-access setting.
Those artifacts:

- **Are not used** in the Stage 3 comparison pipeline.
- **Are not imported** by any module under `src/evaluation/` or `scripts/`.
- **Are not cited** in `MODEL_COMPARISON_VALIDATION.md` or other Stage 3 docs.
- Will be reviewed and, if necessary, quarantined by the Technical Lead during
  the PR review before `integration/stage2` merge.

The Stage 3 comparison produces exclusively Validation metrics from fresh
`SharedExperimentRunner` runs on the frozen manifest. It does not inherit or
replay any result from a prior branch.

## 5. When Test Can Be Opened

Independent Test may only be evaluated after all of the following conditions
are met and the Technical Lead has explicitly opened the gate:

1. Model selection is complete and the recommended model is frozen.
2. The operational threshold is frozen (derived from Validation only).
3. The preprocessor and feature set are frozen (fitted on Training only).
4. No further hyperparameter, architecture, or threshold changes are permitted
   after Test is opened.
5. The Technical Lead updates `test_access.test_evaluation_enabled` in
   `configs/experiment.yaml` to `true` and removes the `SealedTestSetError`
   guard or wraps it with an explicit gating flag.

The Test evaluation must be run exactly once, producing a single final report.
It must not be iterated to improve Test metrics.

## 6. Operational Deployment Caveat

The recommended model and threshold are educational ML results derived from a
Kaggle dataset. They must not be used for real credit-risk or lending decisions.
The model has not been audited for fairness, calibration, or regulatory
compliance.
