# XGBoost Validation Experiment Report

> **Branch**: `feat/xgboost`
> **Base**: `integration/stage2`
> **Author**: Member A
> **Formal run date**: 2026-07-23
> **Formal-run Git SHA**: `3a16ae3ca691c083105b4121c054b2c423d13ba1`

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
| Parsed config SHA-256 | `896e0f677764c5e456b55a46791e4099fa0767865bd50e46f22df88a2572b86a` |
| Config file SHA-256 | `1e7c95b016c3a6a807bcd28cde9313f708d3a73b479301ad58369c1e2bffbf02` |
| Training rows | 95,995 |
| Validation rows | 23,997 |
| Independent Test rows | 30,008 |

The formal run used macOS 26.5.1 on arm64, Python 3.12.13, NumPy 2.2.6,
pandas 2.3.3, scikit-learn 1.7.2, PyYAML 6.0.3, joblib 1.5.2,
XGBoost 3.0.5, and imbalanced-learn 0.14.0. Homebrew libomp 22.1.8
provided the macOS OpenMP runtime required by XGBoost.

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

All eight candidates completed successfully in their pre-declared order. Times
are local wall-clock measurements and are not portable hardware benchmarks.

| experiment_id | PR-AUC | ROC-AUC | KS | Default P | Default R | Op threshold | Op P | Op R | best iter | rounds | train s | infer s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgb_baseline | 0.401323324 | 0.869435127 | 0.583440048 | 0.211911591 | 0.787954831 | 0.541890383 | 0.234234234 | 0.750313676 | 180 | 181 | 0.943825 | 0.011836 |
| xgb_depth3 | 0.397022241 | 0.869508257 | 0.586883529 | 0.213981245 | 0.787327478 | 0.547832549 | 0.240837696 | 0.750313676 | 356 | 357 | 1.491328 | 0.011711 |
| xgb_depth5 | 0.399135710 | 0.868807872 | 0.580511852 | 0.216783217 | 0.777917189 | 0.536598265 | 0.236374408 | 0.750941029 | 170 | 171 | 0.995409 | 0.010273 |
| xgb_lr003 | 0.400976356 | 0.868962337 | 0.589342646 | 0.211155378 | 0.797992472 | 0.544766068 | 0.235502261 | 0.751568381 | 204 | 205 | 1.048716 | 0.010224 |
| xgb_child5 | 0.401962230 | 0.869935023 | 0.585850356 | 0.211454484 | 0.789836888 | 0.548155665 | 0.238865588 | 0.750313676 | 172 | 173 | 0.920203 | 0.008357 |
| xgb_child10 | 0.401652256 | 0.869682996 | 0.583923745 | 0.211525538 | 0.789836888 | 0.545511842 | 0.237123613 | 0.750941029 | 173 | 174 | 0.916323 | 0.008218 |
| xgb_full_rows | 0.400755434 | 0.869360191 | 0.585473855 | 0.214151748 | 0.787954831 | 0.542938292 | 0.238961039 | 0.750313676 | 225 | 226 | 1.083177 | 0.009811 |
| xgb_full_cols | 0.401118462 | 0.869564501 | 0.584437461 | 0.213084112 | 0.786700125 | 0.546213329 | 0.239775461 | 0.750313676 | 189 | 190 | 0.979775 | 0.007667 |

The fixed baseline produced Validation PR-AUC 0.401323324, ROC-AUC
0.869435127, and KS 0.583440048. At threshold 0.5 its accuracy was
0.791265575, precision 0.211911591, recall 0.787954831, F1 0.333998139,
specificity 0.791501138, balanced accuracy 0.789727984, and predicted-positive
rate 0.246989207. Its confusion matrix was TN=17,732, FP=4,671, FN=338,
TP=1,256. Its operational threshold was 0.541890383, with precision
0.234234234 and recall 0.750313676.

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

Independent recomputation of the pre-declared rule selected `xgb_child5`,
matching `selected_candidate.json`. It was the unique candidate within 1e-6 of
the maximum PR-AUC, so no secondary tie-break was needed.

Selected parameters:

| Parameter | Value |
|---|---:|
| n_estimators | 2000 |
| max_depth | 4 |
| learning_rate | 0.05 |
| min_child_weight | 5 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_lambda | 1.0 |
| early_stopping_rounds | 50 |
| n_jobs | -1 |

Selected Validation metrics: PR-AUC 0.401962230, ROC-AUC 0.869935023, and
KS 0.585850356.

## 11. Default Threshold Metrics

Threshold = 0.5. All metrics computed by shared evaluation functions.

| Metric | Value |
|---|---:|
| Accuracy | 0.790390465 |
| Precision | 0.211454484 |
| Recall | 0.789836888 |
| F1 | 0.333598304 |
| Specificity | 0.790429853 |
| Balanced accuracy | 0.790133371 |
| Predicted-positive rate | 0.248114348 |

## 12. Operational Threshold Metrics

Selected by shared `select_operational_threshold()`:
- Recall ≥ 0.75 constraint
- Maximize Precision
- Tie-break: higher Recall → higher Threshold

The selected threshold was 0.548155665.

| Metric | Value |
|---|---:|
| Precision | 0.238865588 |
| Recall | 0.750313676 |
| F1 | 0.362369338 |
| Specificity | 0.829888854 |
| Balanced accuracy | 0.790101265 |
| Predicted-positive rate | 0.208651081 |

## 13. Confusion Matrix

Positive class = 1 (SeriousDlqin2yrs = 1). Columns: TN, FP, FN, TP.

| Threshold | TN | FP | FN | TP | Total |
|---|---:|---:|---:|---:|---:|
| Default (0.5) | 17,708 | 4,695 | 335 | 1,259 | 23,997 |
| Operational (0.548155665) | 18,592 | 3,811 | 398 | 1,196 | 23,997 |

## 14. Runtime

Measured with `time.perf_counter()`:
- `training_time_seconds`: elapsed during `adapter.fit()`
- `validation_inference_time_seconds`: elapsed during `adapter.predict_proba()` on Validation

For `xgb_child5`, training took 0.920203 seconds and Validation inference took
0.008357 seconds. The complete 8-candidate search wall time was 11.728801
seconds in this local environment.

## 15. best_iteration

- Recorded from `estimator.best_iteration`
- Must be non-negative integer
- Must be < n_estimators
- actual_boosting_rounds = best_iteration + 1
- `best_score` is the XGBoost internal aucpr value at best_iteration

For `xgb_child5`, `best_iteration=172`, `actual_boosting_rounds=173`, and
`best_score=0.401592894`. The latter is XGBoost's internal Validation aucpr,
not the shared scikit-learn PR-AUC used for candidate selection.

## 16. Feature Importance

- Type: `gain` (fixed via `importance_type="gain"`)
- All values ≥ 0 and finite
- Aligned 1:1 with Feature Set B feature names
- No features silently dropped
- Written to `outputs/xgboost_search/feature_importance.csv`

All 17 Feature Set B features were present exactly once. The top ten were:

| Rank | Feature | Gain importance |
|---:|---|---:|
| 1 | RevolvingUtilizationOfUnsecuredLines | 0.263594270 |
| 2 | NumberOfTimes90DaysLate | 0.203387424 |
| 3 | NumberOfTime30-59DaysPastDueNotWorse | 0.191988274 |
| 4 | NumberOfTime60-89DaysPastDueNotWorse | 0.098271936 |
| 5 | HasAbnormalDelinquencyCode | 0.043177735 |
| 6 | NumberRealEstateLoansOrLines | 0.041822847 |
| 7 | age | 0.029721349 |
| 8 | NumberOfOpenCreditLinesAndLoans | 0.024378022 |
| 9 | MonthlyIncomeMissingFlag | 0.016617920 |
| 10 | LogMonthlyIncome | 0.016191671 |

Gain importance measures predictive usage in this fitted model and does not
establish causality.

## 17. Comparison with LR/RF/BRF

The following values are all formal Validation results on the same frozen
contract. They are not Independent Test or production results.

| Model | PR-AUC | ROC-AUC | KS | Operational Precision | Operational Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.359228 | 0.827584 | 0.514616 | 0.175753 | 0.750314 |
| Random Forest | 0.393716 | 0.861110 | 0.572985 | 0.227506 | 0.750314 |
| Balanced Random Forest | 0.384743 | 0.867921 | 0.587148 | 0.235398 | 0.750941 |
| **XGBoost (`xgb_child5`)** | **0.401962** | **0.869935** | 0.585850 | **0.238866** | 0.750314 |

XGBoost is the current best Validation PR-AUC model among these four. The
pre-declared XGBoost candidate selection used only the XGBoost rule in Section
8; these cross-model values did not trigger any parameter change. BRF retains a
slightly higher KS (0.587148 versus 0.585850).

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
- Formal search used the pre-declared `n_jobs = -1`; synthetic tests use
  `n_jobs = 1` where deterministic unit-level behavior is required
- XGBoost `random_state = 42` injected per adapter
- Determinism verified on synthetic data
- All metadata includes manifest/config/git SHA
- Training labels were 89,541 negative and 6,454 positive; every candidate
  recorded `scale_pos_weight = 13.873721722962504`
- `python -m compileall -q src scripts tests` passed
- XGBoost-specific set: 99 passed, 0 skipped, 0 warnings, 0 failures/errors
- Full repository: 182 passed, 0 skipped, 0 warnings, 0 failures/errors
- Generated CSV/JSON artifacts remained under git-ignored `outputs/`; no model
  binary or Validation/Test probability array was saved

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
