# Sealed Evaluation Handoff

> Branch: `feat/model-evaluation`
>
> Base: `integration/stage2`
>
> Status: model selected and frozen on Validation; Independent Test remains sealed

## 1. Frozen recommendation

The recommended model is **XGBoost `xgb_child5`**. Selection used only the
frozen Validation partition and the pre-declared ordered considerations:
PR-AUC, ROC-AUC, operational precision subject to Recall >= 0.75, runtime, and
interpretability.

## 2. Frozen model parameters

| Parameter | Value |
|---|---:|
| `n_estimators` | 2000 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `min_child_weight` | 5 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 1.0 |
| `early_stopping_rounds` | 50 |
| `n_jobs` | -1 |
| `random_state` | 42 |

The executor must preserve the full estimator contract:
`n_estimators=2000`, `max_depth=4`, `learning_rate=0.05`,
`min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`,
`reg_lambda=1.0`, `early_stopping_rounds=50`, `n_jobs=-1`, and
`random_state=42`.

Early stopping selected `best_iteration=172` and
`actual_boosting_rounds=173`. Training imbalance handling remains
`scale_pos_weight = 89541 / 6454 = 13.873721722962504`, calculated only from
the frozen Training target.

## 3. Frozen data and preprocessing

| Item | Value |
|---|---|
| Feature set | B |
| Preprocessing | `missing_plus_flag` |
| Feature count | 17 |
| Random seed | 42 |
| Manifest SHA-256 | `5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5` |
| Raw data SHA-256 | `1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac` |
| Config SHA-256 | `896e0f677764c5e456b55a46791e4099fa0767865bd50e46f22df88a2572b86a` |
| Formal Validation comparison code SHA | `2d20fe0caed9536b5706ba428ea620cd7bd4b7fa` |

The preprocessing algorithm and feature contract must remain unchanged. Refit
the preprocessor deterministically on frozen Training only (95,995 rows), and
verify the resulting feature names, order, and metadata before Test prediction.
Do not merge Validation into Training.

The final sealed evaluation executor SHA is a separate governance artifact. It
must be frozen only after the project lead implements and reviews the one-time
sealed evaluation script. The Formal Validation comparison code SHA above must
not be treated as the sealed executor SHA.

## 4. Frozen Validation evidence

| Metric | Value |
|---|---:|
| PR-AUC | 0.401962230 |
| ROC-AUC | 0.869935023 |
| KS | 0.585850356 |
| Operational threshold | **0.548155665398** |
| Operational Precision | 0.238865588 |
| Operational Recall | 0.750313676 |

The operational threshold is Validation-derived and must not be retuned
(不允许重新调整) from Independent Test results.

## 5. Package contract

Python 3.12.13; NumPy 2.2.6; pandas 2.3.3; scikit-learn 1.7.2;
imbalanced-learn 0.14.0; XGBoost 3.0.5; PyYAML 6.0.3; joblib 1.5.2.

## 6. Forest inference contract

Although XGBoost is selected, any authorized future RF or BRF reproduction or
sealed inference must preserve the deterministic forest contract: fit with
`n_jobs=-1`, call native `predict_proba` with temporary `n_jobs=1`, and restore
the training value in `finally`. This is an execution-order contract, not a
parameter change.

## 7. Independent Test prohibitions

Until the project lead explicitly opens the one-time evaluation gate:

- no `predict_proba` on Independent Test;
- no Test metric or confusion matrix;
- no threshold selection, tuning, calibration fitting, or early stopping on Test;
- no model reselection based on Test;
- no reading legacy Test result files as decision evidence.

The shared runner's Test interaction is limited to transformation,
feature-schema consistency, and finite-value checks. It produces no Test
probability or evaluation result.

## 8. One-time sealed evaluation procedure

After explicit authorization, the designated executor must follow this exact
procedure:

1. Refit the preprocessing algorithm deterministically on frozen Training only
   (95,995 rows). Do not combine frozen Validation with Training.
2. Refit the frozen `xgb_child5` pipeline on frozen Training only, using frozen
   Validation only for early stopping as the `eval_set`.
3. Keep the complete frozen estimator parameters, including
   `n_estimators=2000` and `early_stopping_rounds=50`.
4. Compute `scale_pos_weight` only from the frozen Training target:
   `89541 / 6454 = 13.873721722962504`.
5. Before any Independent Test prediction, assert all of the following:
   `best_iteration=172`, `actual_boosting_rounds=173`,
   `scale_pos_weight=13.873721722962504`, `feature_count=17`, frozen feature
   names and order, Raw/Manifest/Config SHA values, and every package version.
6. The executor must abort before Test prediction unless every assertion
   passes.
7. Only after all checks pass, execute exactly one Independent Test
   `predict_proba`. With XGBoost 3.0.5, `predict_proba` automatically uses the
   frozen best iteration, which means the first 173 boosting rounds.
8. Apply the frozen Operational Threshold `0.548155665398` unchanged and
   publish one immutable final evaluation report.

In normative form: Refit the frozen `xgb_child5` pipeline on frozen Training
only, using frozen Validation only for early stopping. Keep
`n_estimators=2000` and `early_stopping_rounds=50`. Abort before Independent
Test prediction unless `best_iteration=172` and
`actual_boosting_rounds=173`. XGBoost `predict_proba` then uses the frozen best
iteration automatically.

The following alternative procedure is explicitly forbidden:

- 不得合并 Training + Validation 后重新拟合；
- 不得把 `n_estimators` 从 2000 改成 173；
- do not recompute `scale_pos_weight` on combined samples;
- do not use Independent Test for early stopping.

Independent Test results must not be used to reselect `best_iteration`, change
boosting rounds, recalculate the threshold, tune parameters, reselect the
model, fit calibration, or modify features or preprocessing.

## 9. Governance evidence

The comparison code rejects non-Validation experiment specs, rejects result
columns containing Test fields, validates all four models against frozen
references, checks probability alignment to frozen Validation row order, and
requires every confusion matrix to total 23,997 rows.

Earlier Forest Test metrics were accidentally viewed and quarantined.
They were not used for model, parameter, threshold, inference-contract,
or comparison decisions.

## 10. Ownership

The project lead owns the decision to open the Independent Test gate. The
evaluation executor owns hash/version verification and the single immutable
run. Reviewers must reject any handoff that changes the frozen contract or
uses Test evidence for another model-selection decision.

This handoff records a Validation recommendation only. It contains no
Independent Test metric.
