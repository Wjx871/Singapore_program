# XGBoost Validation Experiment Report

> **Branch**: `feat/xgboost`
> **Base**: `integration/stage2`
> **Author**: Member A
> **Date**: 2026-07-22

## 1. Scope

This report documents the XGBoost model adapter implementation and the 8-candidate
one-factor-at-a-time Validation search. All metrics are produced by the shared
experiment runner on the frozen 64/16/20 split manifest. Independent Test remains
sealed.

## 2. Frozen Data Contract

| Item | Value |
|---|---|
| Feature Set | B |
| Preprocessing | missing_plus_flag |
| Random seed | 42 |
| Evaluation split | validation |
| Manifest SHA-256 | `5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5` |
| Raw data SHA-256 | `1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac` |
| Training rows | 95,995 |
| Validation rows | 23,997 |
| Independent Test rows | 30,008 |

## 3. XGBoost Adapter Design

`XGBoostAdapter` implements the shared `ModelAdapter` contract
([src/models/adapters/xgboost_adapter.py](../src/models/adapters/xgboost_adapter.py)).

```python
class XGBoostAdapter(ModelAdapter):
    model_name = "xgboost"
    model_family = "gradient_boosting"
    model_status = "implemented"
    supports_validation_data = True
    supports_early_stopping = True
    requires_scaled_features = False
    imbalance_strategy = "training_derived_scale_pos_weight"
```

It uses `xgboost.XGBClassifier` (v3.0.5) with the sklearn API.
Fixed internal parameters are locked in `FIXED_INTERNAL`:

- `objective = "binary:logistic"`
- `eval_metric = "aucpr"`
- `tree_method = "hist"`
- `importance_type = "gain"`
- `verbosity = 0`

## 4. Training-derived scale_pos_weight

Computed in `fit()` from **Training labels only**:

```
scale_pos_weight = negative_train_count / positive_train_count
                 = 89,541 / 6,454
                 ≈ 13.873721722962504
```

Validations enforced:
- y_train contains only 0 and 1
- positive_count > 0
- negative_count > 0
- Result is finite and positive
- User cannot override via configuration

## 5. Validation-only Early Stopping

- `eval_set` contains only `[(x_validation, y_validation)]`
- Training is never mixed with Validation for early stop purposes
- Test is never passed to `eval_set` or `fit()`
- `validation_split_name` must be exactly `"validation"`; `"test"` is rejected
- XGBoost 3.0.5 uses `early_stopping_rounds` as a constructor parameter

XGBoost internal `eval_metric="aucpr"` drives early stopping. The final candidate
ranking uses the shared `average_precision_score` via the Runner — these are
different implementations and must not be confused.

`predict_proba()` in XGBoost 3.0.5 automatically uses `best_iteration` — no
explicit `iteration_range` is required.

## 6. Fixed Baseline

| Parameter | Value |
|---|---|
| n_estimators | 2000 |
| max_depth | 4 |
| learning_rate | 0.05 |
| min_child_weight | 1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_lambda | 1.0 |
| early_stopping_rounds | 50 |
| n_jobs | -1 |

## 7. Pre-declared Candidate Matrix

All candidates share: `n_estimators=2000`, `reg_lambda=1.0`,
`early_stopping_rounds=50`, `n_jobs=-1`.

One-factor-at-a-time variations from baseline:

| # | experiment_id | max_depth | learning_rate | min_child_weight | subsample | colsample_bytree |
|---|---|---|---|---|---|---|
| 1 | xgb_baseline | 4 | 0.05 | 1 | 0.8 | 0.8 |
| 2 | xgb_depth3 | 3 | 0.05 | 1 | 0.8 | 0.8 |
| 3 | xgb_depth5 | 5 | 0.05 | 1 | 0.8 | 0.8 |
| 4 | xgb_lr003 | 4 | 0.03 | 1 | 0.8 | 0.8 |
| 5 | xgb_child5 | 4 | 0.05 | 5 | 0.8 | 0.8 |
| 6 | xgb_child10 | 4 | 0.05 | 10 | 0.8 | 0.8 |
| 7 | xgb_full_rows | 4 | 0.05 | 1 | 1.0 | 0.8 |
| 8 | xgb_full_cols | 4 | 0.05 | 1 | 0.8 | 1.0 |

## 8. Candidate Selection Rule

**Primary**: highest Validation PR-AUC.

**Tie-break** (when PR-AUC difference ≤ 1e-6):
1. Higher ROC-AUC
2. Higher KS
3. Shorter training_time_seconds
4. Earlier position in candidate matrix

The rule is implemented in `scripts/run_xgboost_search.py` and is unit-tested
in `tests/test_xgboost_search.py`.

## 9. Candidate Results

Results are produced by `python -m scripts.run_xgboost_search` and written to
`outputs/xgboost_search/candidate_results.csv`. This file is git-ignored.

*(Table populated after formal run with real data.)*

## 10. Selected Validation Candidate

Written to `outputs/xgboost_search/selected_candidate.json`. Contains:

- `selected_experiment_id`
- `selection_metric` and `selection_rule`
- Full parameter set
- Validation PR-AUC / ROC-AUC / KS
- Default threshold metrics
- Operational threshold and metrics
- Confusion matrix
- `best_iteration`, `best_score`, `scale_pos_weight`
- Training and inference time
- Manifest / config / git SHA

## 11. Default Threshold Metrics

Threshold = 0.5. All metrics computed by shared evaluation functions.

*(Populated after formal run.)*

## 12. Operational Threshold Metrics

Selected by shared `select_operational_threshold()`:
- Recall ≥ 0.75 constraint
- Maximize Precision
- Tie-break: higher Recall → higher Threshold

*(Populated after formal run.)*

## 13. Confusion Matrix

Positive class = 1 (SeriousDlqin2yrs = 1). Columns: TN, FP, FN, TP.

*(Populated after formal run.)*

## 14. Runtime

Measured with `time.perf_counter()`:
- `training_time_seconds`: elapsed during `adapter.fit()`
- `validation_inference_time_seconds`: elapsed during `adapter.predict_proba()` on Validation

*(Populated after formal run.)*

## 15. best_iteration

- Recorded from `estimator.best_iteration`
- Must be non-negative integer
- Must be < n_estimators
- actual_boosting_rounds = best_iteration + 1
- `best_score` is the XGBoost internal aucpr value at best_iteration

## 16. Feature Importance

- Type: `gain` (fixed via `importance_type="gain"`)
- All values ≥ 0 and finite
- Aligned 1:1 with Feature Set B feature names
- No features silently dropped
- Written to `outputs/xgboost_search/feature_importance.csv`

## 17. Comparison with LR/RF/BRF

The four-model comparison will be performed by Member C after all models are
integrated. Preliminary reference values from prior runs:

| Model | PR-AUC | ROC-AUC | KS | Operational Precision | Operational Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.359228 | 0.827584 | 0.514616 | 0.175753 | 0.750314 |
| Random Forest | 0.393716 | 0.861110 | 0.572985 | 0.227506 | 0.750314 |
| Balanced Random Forest | 0.384743 | 0.867921 | 0.587148 | 0.235398 | 0.750941 |
| **XGBoost** | *(TBD)* | *(TBD)* | *(TBD)* | *(TBD)* | *(TBD)* |

## 18. Test Isolation

**This branch does NOT access Independent Test for:**
- `predict_proba`
- Metrics computation
- Confusion matrix
- Threshold selection
- Parameter/model selection
- Early stopping

The shared Runner performs only `transform`, schema consistency checks, and
finite-value checks on Test features.

## 19. Reproducibility

- `random_seed = 42` locked via `set_random_seeds()`
- `n_jobs = 1` for deterministic tests
- XGBoost `random_state = 42` injected per adapter
- Determinism verified on synthetic data
- All metadata includes manifest/config/git SHA

## 20. Limitations

1. **Single frozen Validation split**: Results are from one 16% holdout;
   no cross-validation variance is reported.
2. **XGBoost internal aucpr ≠ shared PR-AUC**: Early stopping uses XGBoost's
   internal `eval_metric="aucpr"`; final selection uses scikit-learn
   `average_precision_score`. These are distinct implementations.
3. **Feature importance ≠ causality**: Gain-based importance reflects
   predictive utility in the fitted model, not causal effects.
4. **No fairness analysis**: The model was not evaluated for demographic
   parity, equal opportunity, or other fairness criteria.
5. **No calibration analysis**: Brier score and calibration curves are
   deferred to the presentation-ready phase (E7).
6. **No probability calibration**: Raw XGBoost probabilities are used
   directly; no Platt scaling or isotonic regression was applied.
7. **Independent Test sealed**: Generalization performance on the 20%
   holdout is not measured in this task.
8. **Not for real credit decisions**: This is an educational ML project.
   The model must not be used for actual lending decisions.
9. **Bounded search only**: Only 8 pre-declared candidates were evaluated.
   The global optimum within the parameter space is not guaranteed.
10. **No SMOTE**: Per the MVP contract, synthetic minority oversampling
    is not used. Only `scale_pos_weight` addresses class imbalance.
