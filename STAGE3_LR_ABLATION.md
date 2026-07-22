# Stage 3 Shared Runner and Logistic Regression Ablation

## 1. Scope

Stage 3 freezes the Stage 2 data contract, introduces a shared Validation-only experiment runner, supports two 96/98 strategies, exposes a Train-only EDA interface, and executes Logistic Regression feature, class-weight, and abnormal-code ablations. No RF, BRF, XGBoost, LightGBM, parameter search, or Independent Test model evaluation is performed.

## 2. Git Baseline

- Base branch: `integration/stage2`
- Base merge commit: `92b528c7e0a6a5e6bc4082c475eb2bacf9836b2f`
- Development branch: `feat/shared-runner-lr-ablation`
- `main` and `integration/stage2` are not directly modified.

## 3. Frozen Manifest

All experiments reuse `data/processed/split_manifest.csv` with SHA-256 `5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5`. The Runner verifies the file bytes before loading and fails rather than regenerating it. Counts remain Train=95,995, Validation=23,997, Test=30,008, with zero row/group overlap.

## 4. Shared Runner Design

`ExperimentSpec` accepts only Logistic Regression, Feature Set A/B, `missing_plus_flag`/`keep_raw`, `balanced`/`none`, and `evaluation_split="validation"`. `SharedExperimentRunner` loads raw data and the frozen manifest once, reuses Stage 2 preprocessing/features/metrics/thresholding, and returns a JSON-safe `ExperimentResult`. Future models should extend the estimator factory rather than duplicate the runner.

## 5. Preprocessing Strategies

`missing_plus_flag` converts delinquency codes 96/98 to missing, adds `HasAbnormalDelinquencyCode`, and fills each affected field with its Training median. `keep_raw` preserves 96/98, does not fit delinquency medians, and omits the abnormal flag. Both strategies keep the income/dependents missing flags and `AgeInvalidFlag`. Strategy changes after fit raise an exception. Clipping remains disabled.

## 6. Feature Set Comparison

With balanced class weight and `missing_plus_flag`, Feature Set B improves PR-AUC from 0.3580201 to 0.3592281, ROC-AUC from 0.8265052 to 0.8275836, and KS from 0.5095998 to 0.5146162. The improvement is small but directionally consistent, so B remains the default shared feature contract.

## 7. Class Weight Ablation

On Feature Set B + `missing_plus_flag`, `class_weight=None` produces high default-threshold precision (0.6039120) but only 0.1549561 recall. Balanced weighting raises default recall to 0.6342535 with precision 0.2634871. At the Validation-selected operational threshold, both meet approximately 0.7503 recall, while balanced achieves higher precision (0.1757531 vs 0.1659498) and a lower predicted-positive rate (0.2835771 vs 0.3003292).

## 8. 96/98 Sensitivity

With Feature Set B and balanced weighting, `missing_plus_flag` achieves PR-AUC 0.3592281, while `keep_raw` achieves 0.3046232. ROC-AUC is 0.8275836 vs 0.7888492 and KS is 0.5146162 vs 0.4326962. This supports `missing_plus_flag` for subsequent Validation experiments without asserting any known business meaning for 96/98.

## 9. Train-only EDA Interface

`src.analysis.train_only.load_train_partition` returns copies of raw Training predictors, target, `row_id`, `feature_hash_v1`, and split metadata only. `transform_train_with_strategy` reuses the production Preprocessor for Training-only transformed views. Usage is documented in `docs/DATA_ANALYSIS_HANDOFF.md`.

## 10. Validation Metrics

| Experiment | Features | Strategy | Weight | PR-AUC | ROC-AUC | KS |
|---|---:|---|---|---:|---:|---:|
| LR A balanced | 14 | missing_plus_flag | balanced | 0.3580201 | 0.8265052 | 0.5095998 |
| LR B balanced | 17 | missing_plus_flag | balanced | 0.3592281 | 0.8275836 | 0.5146162 |
| LR B none | 17 | missing_plus_flag | none | 0.3540479 | 0.8204618 | 0.4999517 |
| LR B keep raw | 16 | keep_raw | balanced | 0.3046232 | 0.7888492 | 0.4326962 |

## 11. Threshold Trade-offs

| Experiment | Default precision | Default recall | Operational threshold | Operational precision | Operational recall | Operational predicted-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| LR A balanced | 0.2621537 | 0.6292346 | 0.4206511 | 0.1740143 | 0.7503137 | 0.2864108 |
| LR B balanced | 0.2634871 | 0.6342535 | 0.4225355 | 0.1757531 | 0.7503137 | 0.2835771 |
| LR B none | 0.6039120 | 0.1549561 | 0.0563819 | 0.1659498 | 0.7503137 | 0.3003292 |
| LR B keep raw | 0.1686224 | 0.6580928 | 0.4519404 | 0.1351626 | 0.7509410 | 0.3690461 |

## 12. Test Set Guard

The Runner has no Test evaluation option. Specs reject `evaluation_split="test"`; shared metric and threshold APIs reject Test; Test is transformed only for schema/finite checks; no Test probability file or metric field is produced. Tests verify that changing Test values does not change Training medians, Validation probabilities/metrics, or operational threshold.

## 13. Automated Tests

Tests cover both strategy schemas, immutable fit strategy, frozen manifest SHA, spec scope, unified result serialization, Stage 2 reproduction, all four formal configurations, Train-only access, shared preprocessing reuse, output filenames, and Test isolation. Formal full-matrix runs are performed once outside repeated unit tests.

## 14. Actual Runtime Results

The tables above come from the actual run of `.venv/bin/python -m scripts.run_lr_ablations --config configs/experiment.yaml`. Runtime varies by machine and is recorded per experiment in local CSV/JSON metadata. All result rows share the frozen manifest SHA and configuration hash. No Test model metric was generated.

## 15. Selected Defaults for Stage 4/5

Based only on Validation evidence, use Feature Set B, `missing_plus_flag`, and balanced class weighting as the default Logistic Regression and shared preprocessing contract. RF/BRF/XGBoost should start from Feature Set B and `missing_plus_flag`; their model-specific imbalance strategies remain governed by the locked experiment design.

## 16. Known Limitations

The comparison uses fixed LR parameters and one frozen split. It does not establish causal meaning, business deployment fitness, fairness, calibration, or Independent Test performance. The small A/B improvement should not be overstated, while the keep-raw degradation is specific to this model/configuration and Validation split.

## 17. Handoff to Data Analysis

Use `load_train_partition` for label-conditioned EDA and `transform_train_with_strategy` for shared strategy views. Do not create new split logic, read Validation/Test labels, copy preprocessing rules, or commit generated figures/data.

## 18. Handoff to RF/BRF/XGBoost

Add estimator adapters to the shared runner while reusing the frozen manifest, Feature Set B, `missing_plus_flag`, metrics, operational threshold rule, result schema, and Test Guard. XGBoost early stopping may use Validation only. No model branch may create a private split or Test loader.
