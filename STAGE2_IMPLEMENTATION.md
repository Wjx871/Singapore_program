# Stage 2 Implementation

## 1. Scope

Stage 2 implements the leakage-safe data pipeline and the E1 Logistic Regression vertical slice. Random Forest, Balanced Random Forest, XGBoost, LightGBM, model search, and sealed Independent Test evaluation are out of scope.

## 2. Git Baseline

- Development branch: `feat/core-pipeline-lr`
- Baseline commit: `5638b9847b82d5cd012a183d34c6d20ebf583dc5`
- Target integration branch: `integration/stage2`
- `main` and `integration/stage2` were not directly modified.

## 3. Files Added

- `src/config.py` and `src/utils/`: configuration, paths, hashing, versions, logging, and seeds.
- `src/data/`: raw validation, `feature_hash_v1`, group-aware split, manifest validation and metadata.
- `src/features/`: Training-fitted preprocessing and Feature Sets A/B/C.
- `src/evaluation/`: reusable metrics, threshold selection, and sealed-test guard.
- `src/models/logistic_regression.py`: fixed E1 estimator pipeline.
- `scripts/`: manifest build/validation and Logistic Regression run entry points.
- `tests/`: synthetic unit tests and formal-data smoke checks.
- `requirements-stage2.txt`: locked Stage 2 environment.

## 4. Configuration Design

`configs/experiment.yaml` is loaded without silent defaults. The loader rejects missing fields, absolute/escaping paths, invalid split ratios, target/ID leakage, duplicate or incorrect core models, LightGBM in the core tier, incorrect PR-AUC implementation, enabled Test evaluation, enabled Stage 2 clipping, and any hash version other than `feature_hash_v1`.

## 5. Raw Data Validation

The labeled CSV is read-only and SHA-256 checked before and after loading. Verified facts:

- Rows: 150,000; raw columns: 12; predictors: 10.
- Positives: 10,026; negatives: 139,974; positive rate: 0.06684.
- SHA-256: `1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac`.
- Official Kaggle test: 101,503 rows with an entirely missing target; schema check only.

## 6. Feature Hash Specification

`feature_hash_v1` serializes the ten raw predictors in configured order. Each payload contains the version, column names, ASCII Unit Separator delimiters, `<NA>` for missing data, canonical decimal integers, and `float.hex()` for finite non-integral binary64 values. `0`, `0.0`, and `-0.0` normalize to the same value. SHA-256 produces a 64-character lowercase hexadecimal digest. `row_id`, target, imputation, clipping, scaling, and engineered features are excluded.

## 7. Split Manifest Generation

Run:

```bash
.venv/bin/python -m scripts.build_split_manifest --config configs/experiment.yaml
.venv/bin/python -m scripts.validate_split_manifest --config configs/experiment.yaml
```

The implementation stable-sorts by `row_id` before splitting, so input row reordering does not change row-to-split assignments.

## 8. Fold Selection Rule

Both stages enumerate all five `StratifiedGroupKFold` held-out folds. Selection uses the lexicographic tuple: absolute row-ratio deviation, absolute positive-rate deviation, absolute group-ratio deviation, then fold index. The selected test fold was 0 with seed 42; the selected validation fold was 0 with seed 43. Model predictions are never consulted.

## 9. Split Verification

Manifest SHA-256: `5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5`.

| Split | Rows | Fraction | Groups | Positive | Negative | Positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | 95,995 | 0.6399667 | 95,586 | 6,454 | 89,541 | 0.0672327 |
| Validation | 23,997 | 0.1599800 | 23,893 | 1,594 | 22,403 | 0.0664250 |
| Independent Test | 30,008 | 0.2000533 | 29,875 | 1,978 | 28,030 | 0.0659158 |

All row and group pairwise overlaps are zero. All ratios and split positive-rate deviations satisfy the locked tolerances. The 37 conflicting-target groups (145 rows) remain intact.

## 10. Leakage Prevention Rules

All learned values come from Training only. Validation and Test call `transform` only. `row_id`, target, and `feature_hash_v1` never enter a model matrix. Clipping is implemented as an optional Training-fitted operation but is disabled for E1.

## 11. Preprocessing Design

- `MonthlyIncome`: missing flag and Training median (actual median 5,400).
- `NumberOfDependents`: missing flag and Training median (actual median 0).
- Delinquency 96/98: shared abnormal flag, conversion to missing, then per-column Training medians.
- `age <= 0`: invalid flag, conversion to missing, then Training median (actual median 52).
- Logistic Regression: StandardScaler on non-binary numeric/count features; quality flags pass through unchanged.

## 12. Feature Set Schemas

- A: ten cleaned predictors plus four quality flags = 14 columns.
- B/C1: A plus `IncomePerDependent`, `LogMonthlyIncome`, `HighDebtFlag` = 17 columns.
- C2: B without three raw delinquency fields plus `DelinquencyScore` = 15 columns.
- C3: B plus `DelinquencyScore` = 18 columns.

All schemas are ordered, unique, numeric, finite, and exclude identity/target/hash columns. E1 uses Feature Set A only.

## 13. Logistic Regression Pipeline

Actual parameters: L2 penalty, `liblinear`, `C=1.0`, `class_weight="balanced"`, `max_iter=5000`, and `random_state=42`. No search, SMOTE, or Test-based decision was performed.

## 14. Validation Evaluation

The generated Validation-only metrics are PR-AUC (Average Precision), ROC-AUC, KS, and threshold-dependent metrics. Actual final values are recorded in `outputs/metrics/logistic_regression_validation_metrics.json`; the exact values are also summarized under Actual Runtime Results after the final reproducibility run.

## 15. Threshold Selection

Candidate thresholds cover all unique Validation probabilities plus an all-negative boundary; the lowest unique probability supplies the all-positive case. Among candidates with recall at least 0.75, the selector maximizes precision, then recall, then threshold. There is no F1 fallback and no relaxation of the recall constraint.

## 16. Test Set Guard

`test_evaluation_enabled` is false. Evaluation and threshold APIs reject `split_name="test"` with `SealedTestSetError`. The runner transforms Test predictors only to verify schema equality and absence of NaN/Inf; it does not call `predict_proba` for Test and never passes Test labels into metric or threshold functions.

## 17. Reproducibility Metadata

Run metadata records the raw/config/manifest hashes, Git SHA, package versions, seeds, sample counts, feature schema, Training-fitted preprocessing values, model parameters, timings, Validation results, threshold rule, and Test access status.

## 18. Automated Tests

Tests cover config/schema failures, data integrity, stable hashing, equivalent numeric serialization, group isolation, seed sensitivity, row-order invariance, manifest stability, conflicting groups, Training-only preprocessing, abnormal and age handling, A/B/C schemas, metrics, threshold edge cases, LR fit/predict, and sealed-test protection.

## 19. Actual Runtime Results

Actual E1 Validation results from the Training-only pipeline run:

| Metric | Value |
|---|---:|
| PR-AUC (Average Precision) | 0.3580201248 |
| ROC-AUC | 0.8265052303 |
| KS | 0.5095997853 |

At threshold 0.5: accuracy 0.8577322165, precision 0.2621536853, recall 0.6292346299, F1 0.3701107011, specificity 0.8739900906, balanced accuracy 0.7516123602, and predicted-positive rate 0.1594365962. The confusion matrix is TN=19,580, FP=2,823, FN=591, TP=1,003.

The selected operational threshold is 0.4206510552. On Validation it gives precision 0.1740142587, recall 0.7503136763, F1 0.2825085627, specificity 0.7465964380, balanced accuracy 0.7484550571, and predicted-positive rate 0.2864108014. The confusion matrix is TN=16,726, FP=5,677, FN=398, TP=1,196.

These are Validation results, not Independent Test results. Runtime fields are regenerated on every run and are retained only in local metadata/logs.

## 20. Generated Local Artifacts

- `data/processed/split_manifest.csv`
- `data/processed/split_metadata.json`
- `outputs/metadata/raw_data_summary.json`
- `outputs/metrics/logistic_regression_validation_metrics.json`
- `outputs/metrics/logistic_regression_validation_thresholds.csv`
- `outputs/metadata/logistic_regression_run.json`
- `logs/stage2_*.log`

All are ignored by Git.

## 21. Known Limitations

Only fixed-parameter LR + Feature Set A is executed. Feature comparison, class-weight ablation, tree/boosting models, calibration, fairness analysis, and Independent Test evaluation remain pending. The model is educational and is not suitable for automated lending decisions.

## 22. Stage 3 Entry Conditions

Review and freeze the manifest SHA, Feature Set B contract, evaluation API, guard behavior, and automated tests. RF/BRF/XGBoost must then reuse these contracts and must not generate alternative splits, preprocessing statistics, metrics, or Test loaders.
