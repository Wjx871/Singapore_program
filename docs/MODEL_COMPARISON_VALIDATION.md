# Model Comparison Validation Report

> Branch: `feat/model-evaluation`
>
> Base: `integration/stage2`
>
> Evaluation split: frozen `validation` only
>
> Independent Test: sealed; no prediction or evaluation performed

## 1. Scope

This report compares Logistic Regression (LR), Random Forest (RF), Balanced
Random Forest (BRF), and the frozen XGBoost candidate `xgb_child5`. Every result
was produced by `SharedExperimentRunner`; legacy tables were not copied into
the comparison.

## 2. Frozen data and execution contract

| Item | Frozen value |
|---|---|
| Feature set | B |
| Preprocessing | `missing_plus_flag` |
| Random seed | 42 |
| Training / Validation rows | 95,995 / 23,997 |
| Feature count | 17 |
| Manifest SHA-256 | `5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5` |
| Raw data SHA-256 | `1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac` |
| Config SHA-256 | `896e0f677764c5e456b55a46791e4099fa0767865bd50e46f22df88a2572b86a` |
| Evaluation code SHA | `2d20fe0caed9536b5706ba428ea620cd7bd4b7fa` |

Environment: Python 3.12.13, NumPy 2.2.6, pandas 2.3.3,
scikit-learn 1.7.2, imbalanced-learn 0.14.0, XGBoost 3.0.5,
PyYAML 6.0.3, and joblib 1.5.2.

## 3. Frozen model specifications

| Model | Imbalance handling | Frozen parameters |
|---|---|---|
| LR | `class_weight="balanced"` | C=1.0, L2, liblinear, max_iter=5000 |
| RF | none | 300 trees, max_depth=None, min_samples_leaf=2, max_features="sqrt", random_state=42 |
| BRF | internal balanced sampling | RF settings plus sampling_strategy="all", replacement=True, bootstrap=False |
| XGBoost | Training-derived `scale_pos_weight` | depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, lambda=1 |

No model parameter, feature, preprocessing step, split, threshold rule, metric
implementation, or tolerance was changed during provenance correction.

## 4. RF and BRF provenance governance

The RF metrics reported by PR #6 cannot be reproduced from its declared
historical commit `a3bdfc12925d8509b7047be9b70bac03efa5cdb9`, locked raw
data/config/manifest, and stated estimator parameters. Running that historical
commit in the current unified environment agrees with member C's rerun to
approximately 1e-8. Therefore, the discrepancy is not explained by a current
code or parameter change.

PR #6 did not preserve the actual package versions, OS/architecture,
`run_metadata.json`, input-matrix hashes, or Validation-probability provenance.
Its RF and BRF values are retained only as non-authoritative historical
references. Fresh unified-environment reruns are the authoritative Validation
results below.

## 5. Deterministic Forest Inference Contract

RF and BRF fitting remains parallel with `n_jobs=-1`. Their formal Validation,
reproduction, and future sealed inference uses the estimator's native
`predict_proba` with `n_jobs=1`; a `finally` block restores `n_jobs=-1`
afterward, including on failure.

Parallel `predict_proba` can change floating-point accumulation order and
produce probability differences near 1e-16. Many tied or nearly tied
probabilities make average precision sensitive to these last-bit ordering
differences; BRF PR-AUC consequently varied by about 1.75e-6. Tree structure,
feature importance, operational threshold, and both confusion matrices did not
change. Serial inference standardizes execution order only; it is not model
tuning, and the 1e-6 reference tolerance was not relaxed.

Two independent fits of each forest produced identical input hashes, row-id
hashes, feature order, estimator parameters, tree hashes, feature importance,
serial Validation probability hashes, thresholds, confusion matrices, and
metrics:

| Model | Tree SHA-256 (run 1 = run 2) | Probability SHA-256 (run 1 = run 2) |
|---|---|---|
| RF | `d5d1ce08ff81b4aa9ee196d749bac686c4ff7dc39bcbd45b6d3f13fa37699595` | `9a6c3d0a0a61cb588e1d68184b75f2f9e318d7750bfe941b376bdfa28bc00364` |
| BRF | `9525b062afb8af7158a01eb8f6a8eeac7880c7d3708cb14029a85097d8af7291` | `94ec16cbc16d9dad4e5932b7703c7f2e1c8a8a3065136b2a9510d5fa1ee539f4` |

Common input provenance:

| Input | Shape / dtype | SHA-256 |
|---|---|---|
| X_train | (95,995, 17) / float64 | `bb5c1670053cd63ac2839d4969b68f22d5f198c6e0f51fcf61c4c92b7f64e344` |
| X_validation | (23,997, 17) / float64 | `e0b27f3478f75dcb5409f69be2c72142b24e5b39e3363d5bb9b404c7f68ba421` |
| Training row_id | 95,995 rows | `5404e9df77d5474483bb2beb1afd964f716ba8eeff6ea832bc47918b8ba07052` |
| Validation row_id | 23,997 rows | `b3da505a7ef76d7a288dfe4770c7a7258eff37e9ebbadfc04a847e08a3096804` |

Probability hashes remain diagnostic metadata. The acceptance contract also
requires identical inputs, row IDs, feature order, estimator parameters,
thresholds and confusion matrices; core metric differences must be <= 1e-6,
and feature importance must agree within floating-point precision.

## 6. Feature names and order

The shared 17-column order is:

1. `RevolvingUtilizationOfUnsecuredLines`
2. `age`
3. `NumberOfTime30-59DaysPastDueNotWorse`
4. `DebtRatio`
5. `MonthlyIncome`
6. `NumberOfOpenCreditLinesAndLoans`
7. `NumberOfTimes90DaysLate`
8. `NumberRealEstateLoansOrLines`
9. `NumberOfTime60-89DaysPastDueNotWorse`
10. `NumberOfDependents`
11. `MonthlyIncomeMissingFlag`
12. `DependentsMissingFlag`
13. `HasAbnormalDelinquencyCode`
14. `AgeInvalidFlag`
15. `IncomePerDependent`
16. `LogMonthlyIncome`
17. `HighDebtFlag`

## 7. Threshold-independent Validation results

| Model | PR-AUC | ROC-AUC | KS |
|---|---:|---:|---:|
| Logistic Regression | 0.359228055 | 0.827583572 | 0.514616170 |
| Random Forest | 0.391788070 | 0.860304365 | 0.572956738 |
| Balanced Random Forest | 0.385352222 | 0.867977539 | 0.582244598 |
| **XGBoost (`xgb_child5`)** | **0.401962230** | **0.869935023** | **0.585850356** |

XGBoost is highest on all three threshold-independent metrics. Its strict
reference check passed at the unchanged 1e-6 tolerance.

## 8. Default threshold metrics

Default threshold is 0.5 for all models.

| Model | Accuracy | Precision | Recall | F1 | Specificity | Balanced accuracy | Positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| LR | 0.857941 | 0.263487 | 0.634253 | 0.372307 | 0.873856 | 0.754055 | 0.159895 |
| RF | 0.937409 | 0.600877 | 0.171895 | 0.267317 | 0.991876 | 0.581885 | 0.019002 |
| BRF | 0.841313 | 0.255088 | 0.723338 | 0.377167 | 0.849708 | 0.786523 | 0.188357 |
| XGBoost | 0.790390 | 0.211454 | 0.789837 | 0.333598 | 0.790430 | 0.790133 | 0.248114 |

## 9. Default threshold confusion matrices

Each matrix totals 23,997 Validation rows; positive class is 1.

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| LR | 19,577 | 2,826 | 583 | 1,011 |
| RF | 22,221 | 182 | 1,320 | 274 |
| BRF | 19,036 | 3,367 | 441 | 1,153 |
| XGBoost | 17,708 | 4,695 | 335 | 1,259 |

## 10. Operational threshold rule

The unchanged rule maximizes Validation Precision subject to Recall >= 0.75,
then breaks ties by higher Recall and higher threshold. Thresholds are derived
from Validation only.

## 11. Operational Validation results

| Model | Threshold | Precision | Recall | F1 | Specificity | Balanced accuracy | Positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| LR | 0.422535492 | 0.175753 | 0.750314 | 0.284796 | 0.749632 | 0.749973 | 0.283577 |
| RF | 0.086178529 | 0.228027 | 0.750314 | 0.349759 | 0.819265 | 0.784789 | 0.218569 |
| BRF | 0.468817526 | 0.235619 | 0.750314 | 0.358621 | 0.826809 | 0.788561 | 0.211526 |
| **XGBoost** | **0.548155665** | **0.238866** | **0.750314** | **0.362369** | **0.829889** | **0.790101** | **0.208651** |

## 12. Operational confusion matrices

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| LR | 16,794 | 5,609 | 398 | 1,196 |
| RF | 18,354 | 4,049 | 398 | 1,196 |
| BRF | 18,523 | 3,880 | 398 | 1,196 |
| XGBoost | 18,592 | 3,811 | 398 | 1,196 |

At the common recall floor, XGBoost has the highest precision and fewest false
positives.

## 13. Runtime

Representative formal-run timings (seconds):

| Model | Training time | Validation inference time | Execution note |
|---|---:|---:|---|
| LR | 0.241 | 0.002 | native inference |
| RF | 10.039 | 0.745 | training `n_jobs=-1`; serial inference `n_jobs=1` |
| BRF | 2.550 | 0.647 | training `n_jobs=-1`; serial inference `n_jobs=1` |
| XGBoost | 0.924 | 0.008 | `n_jobs=-1` |

Runtime is machine- and load-dependent and is a later selection consideration,
not a reference metric.

## 14. Interpretability

LR is most directly interpretable through signed coefficients. RF provides
global impurity-based feature importance, BRF provides the same type of
importance under balanced sampling, and XGBoost provides ensemble feature
importance but has the greatest model complexity. Importance vectors and their
feature alignment are preserved in each model's run metadata.

The five largest formal-run forest importances were:

| Rank | RF feature (importance) | BRF feature (importance) |
|---:|---|---|
| 1 | RevolvingUtilizationOfUnsecuredLines (0.166004) | RevolvingUtilizationOfUnsecuredLines (0.248456) |
| 2 | DebtRatio (0.122752) | DebtRatio (0.097548) |
| 3 | NumberOfTimes90DaysLate (0.108705) | NumberOfTime30-59DaysPastDueNotWorse (0.096208) |
| 4 | age (0.099480) | age (0.094171) |
| 5 | IncomePerDependent (0.088860) | NumberOfTimes90DaysLate (0.089927) |

## 15. PR, ROC, and KS comparison

![Validation PR curves](assets/model_comparison/validation_pr_curve.png)

![Validation ROC curves](assets/model_comparison/validation_roc_curve.png)

![Validation PR-AUC, ROC-AUC, and KS](assets/model_comparison/validation_metric_comparison.png)

## 16. Operational comparison figures

![Operational metrics](assets/model_comparison/operational_metric_comparison.png)

![Default-threshold confusion matrices](assets/model_comparison/confusion_matrices_default.png)

![Operational-threshold confusion matrices](assets/model_comparison/confusion_matrices_operational.png)

All chart values are generated from the same validated comparison table.

## 17. Model selection

**Recommended Validation model: XGBoost (`xgb_child5`).**

The frozen ordered considerations are PR-AUC, ROC-AUC, operational precision at
the recall constraint, runtime, and interpretability. XGBoost wins at the first
criterion (PR-AUC), by more than the 1e-6 tolerance, and also leads ROC-AUC, KS,
and operational precision. This is a Validation-only recommendation, not a
claim about Independent Test performance.

## 18. XGBoost strict verification

`xgb_child5` reproduced PR-AUC 0.401962230, ROC-AUC 0.869935023, KS
0.585850356, operational threshold 0.548155665, precision 0.238865588, and
recall 0.750313676 within 1e-6. Early stopping selected iteration 172 (173
boosting rounds); Training-derived `scale_pos_weight` was
13.873721722962504.

## 19. Test isolation and quarantine

Independent Test probabilities, metrics, thresholds, confusion matrices, and
model-selection evidence were neither generated nor read. The shared runner
only applies the fitted transformation and performs schema/finite checks on
Test features.

Earlier Forest Test metrics were accidentally viewed and quarantined. They
were not used for model, parameter, or threshold selection.

Earlier Forest Test metrics were accidentally viewed and quarantined.
They were not used for model, parameter, threshold, inference-contract,
or comparison decisions.

## 20. Limitations and handoff

This is a single frozen Validation split, not cross-validation. Calibration,
fairness, distribution shift, and production suitability are out of scope.
The selected model, preprocessing, feature order, XGBoost boosting round, and
Validation-derived operational threshold are frozen in
`SEALED_EVALUATION_HANDOFF.md`. Independent Test remains sealed until an
explicitly authorized one-time evaluation.
