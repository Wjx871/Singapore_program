# Stage 3.5 Model Integration Contract

## 1. Scope

Stage 3.5 establishes a strict model adapter, registry, factory, unified result metadata, and team handoff. It migrates the existing Logistic Regression path without changing its estimator or experiment protocol. It does not implement or formally train Random Forest, Balanced Random Forest, or XGBoost.

## 2. Git Baseline

Development started from `integration/stage2` commit `7ff9ee664b466fb935485c414e4cf0f4a34122f7` on `feat/model-adapter-contract`. No work was performed on `main` or directly on `integration/stage2`.

## 3. Frozen Contracts

- Manifest SHA-256: `5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5`.
- Default model-comparison input: Feature Set B and `missing_plus_flag`.
- Partition sizes: Training 95,995; Validation 23,997; Test 30,008.
- Primary metric: PR-AUC; operational threshold is Validation-selected with Recall >= 0.75.
- Preprocessing is fitted on Training only; Test model evaluation remains sealed.
- Models may not resplit data or replace shared preprocessing, metrics, or threshold rules.

## 4. ModelAdapter Protocol

`src/models/base.py` defines `ModelAdapter`. Every implementation declares its name, family, status, imbalance strategy, scaling requirement, Validation support, early-stopping support, and random seed. `fit` accepts Training data and optional explicitly named Validation data, feature names, groups, and sample weights. `predict_proba` returns a one-dimensional, finite positive-class probability array in `[0, 1]` with one value per row.

Parameters and training metadata must be JSON-safe. Adapters do not load data, read the manifest, compute evaluation metrics, select thresholds, persist experiment outputs, or access Test.

## 5. Model Registry

The authoritative registry contains:

| Model | Family | Stage 3.5 status | Core |
|---|---|---|---|
| `logistic_regression` | linear model | `implemented` | yes |
| `random_forest` | bagging | `contract_ready` | yes |
| `balanced_random_forest` | imbalance-aware bagging | `contract_ready` | yes |
| `xgboost` | gradient boosting | `contract_ready` | yes |
| `lightgbm` | gradient boosting | `optional_legacy` | no |

## 6. Model Factory

`create_model_adapter` validates model-specific parameters before construction. It creates the Logistic Regression adapter and raises `ModelNotImplementedError` for every contract-only model. It never substitutes Logistic Regression or a dummy estimator for another registered name.

## 7. Logistic Regression Migration

The adapter delegates estimator construction to the existing `build_logistic_pipeline`; no training logic was copied. The shared runner provides the same parameters, preprocessing, features, seed, metrics, threshold rule, and Test Guard as before.

The locked Feature Set B + `missing_plus_flag` + balanced Validation regression remains PR-AUC 0.3592280553, ROC-AUC 0.8275835722, and KS 0.5146161696 within floating-point tolerance.

## 8. Random Forest Contract

Random Forest is the standard bagging baseline. Its default contract uses Feature Set B, `missing_plus_flag`, no scaling, seed 42, no class weighting by default, positive-class probabilities, and no SMOTE. Its controlled schema permits only the documented forest parameters.

## 9. Balanced Random Forest Contract

Balanced Random Forest uses `imbalanced-learn==0.14.0`, `BalancedRandomForestClassifier`, Feature Set B, `missing_plus_flag`, no scaling, seed 42, and internal balanced sampling. It must not combine its sampling with SMOTE or introduce a different feature pipeline from Random Forest.

## 10. XGBoost Contract

XGBoost is cost-sensitive boosting with no StandardScaler. Its `scale_pos_weight` is computed only from the Training target as negative count divided by positive count and recorded in metadata. Validation may be used for early stopping; Test may not. The adapter must record `best_iteration` and must reject any non-Validation early-stopping split.

## 11. Runner Integration

`SharedExperimentRunner` now resolves controlled parameters and creates models through the factory. The runner remains the sole owner of data loading, manifest verification, train-only preprocessing, feature construction, metrics, thresholding, metadata, persistence, and Test protection. It passes Validation to `fit` only when the adapter declares `supports_validation_data`; it never passes Test to `fit`, prediction, metrics, or threshold selection.

Scaling remains model-owned through the adapter contract: the LR adapter reuses the existing selective scaler pipeline, while tree contracts declare no scaling. Future adapters must not add a separate shared preprocessing path.

## 12. Result Schema

`ExperimentResult` now records model family/status, controlled parameters, adapter training metadata, imbalance strategy, scaling and Validation capabilities, early-stopping support, `best_iteration`, and `scale_pos_weight`, in addition to the existing features, timing, Validation metrics, thresholds, hashes, commit, and package versions. Non-applicable values are `null`. Test metrics, probabilities, confusion matrices, thresholds, and rankings are forbidden.

## 13. Test Set Protection

The adapter validates that optional evaluation input is named `validation`. Runner construction passes only Training and Validation objects to the adapter. Test is transformed only for schema and finite-value consistency, preserving the pre-existing guard, and is not predicted or evaluated. Changing Test values does not alter Training, Validation probabilities, metrics, preprocessing metadata, or thresholds.

## 14. Automated Tests

Tests cover registry states, factory behavior, controlled parameter schemas, capability flags, Training-derived XGBoost weighting, probability validation, Test-as-Validation rejection, LR adapter reuse, Runner contract-only failures, default experiment contracts, JSON-safe result metadata, exact LR Validation regression, manifest guards, and Test isolation. RF, BRF, and XGBoost are not trained.

## 15. Team Handoff

Implementation ownership, commands, PR evidence, output paths, and review gates are defined in [docs/MODEL_ADAPTER_HANDOFF.md](docs/MODEL_ADAPTER_HANDOFF.md).

## 16. Known Limitations

- RF, BRF, and XGBoost have contracts but no Stage 3.5 adapter implementation.
- The generic CLI does not yet expose arbitrary model parameter JSON; model members must add narrow validated plumbing if needed.
- Formal cross-model Validation comparison, bounded tuning, feature importance, and sealed evaluation remain future work.
- Deep immutability of the dictionary stored in a frozen `ExperimentSpec` is enforced by validation and copying, not by an immutable mapping type.

## 17. Stage 4/5 Entry Conditions

Each model must implement `ModelAdapter`, pass synthetic and full repository tests, use only controlled parameters, reproduce manifest and preprocessing contracts, emit the unified result schema, and prove no Test model access. Its Draft PR must be reviewed by the Technical Lead before formal comparison. The sealed Independent Test may be opened only after all model choices, parameters, feature contracts, and the operational threshold rule are frozen.
