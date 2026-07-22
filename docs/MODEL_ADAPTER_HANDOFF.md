# Model Adapter Implementation Handoff

This document is the implementation boundary for the Forest and XGBoost members. All model work must reuse the frozen data, preprocessing, feature, evaluation, threshold, and result contracts. A model branch must not create a parallel experiment pipeline.

## Shared Contract

- Base branch: latest reviewed `integration/stage2` containing the Stage 3.5 adapter contract.
- Default feature contract: Feature Set B with `missing_plus_flag`.
- Frozen manifest SHA-256: `5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5`.
- Training rows: 95,995; Validation rows: 23,997; sealed Test rows: 30,008.
- Primary metric: Validation PR-AUC.
- Operational rule: maximize Precision subject to Validation Recall >= 0.75.
- SMOTE is not allowed in the MVP.

## File Ownership

### Random Forest member

Use branch `feat/forest-models`. The member may add `src/models/adapters/forest_adapter.py` and `tests/test_forest_adapter.py`, and may make the smallest reviewed integration edits needed in `src/models/registry.py` and `src/models/factory.py`. Random Forest must implement the existing `ModelAdapter` contract with no scaler, `random_state=42`, optional allowed `class_weight`, and positive-class probability output.

### Balanced Random Forest member

Use the same `feat/forest-models` branch when RF and BRF have the same owner. The member may add `src/models/adapters/balanced_forest_adapter.py` and `tests/test_balanced_forest_adapter.py`, plus the same narrow registry/factory integration edits. Use `imbalanced-learn==0.14.0` and `BalancedRandomForestClassifier`. Imbalance handling is internal balanced sampling; do not add SMOTE or unapproved feature engineering.

### XGBoost member

Use branch `feat/xgboost`. The member may add `src/models/adapters/xgboost_adapter.py` and `tests/test_xgboost_adapter.py`, plus narrow reviewed registry/factory integration edits. Compute `scale_pos_weight = negative_train_count / positive_train_count` from the Training target inside the adapter and record it in training metadata. Early stopping may receive Validation only, and `best_iteration` must be recorded.

## Frozen Public Files

Do not directly redesign or duplicate the following shared files:

- `src/experiments/contracts.py`
- `src/experiments/runner.py`
- `src/experiments/result_schema.py`
- `src/models/base.py`
- `src/models/contracts.py`
- data loading, manifest, preprocessing, feature, metric, threshold, and Test Guard modules
- `configs/experiment.yaml` frozen split and evaluation fields

If an adapter cannot be implemented without changing a frozen interface, describe the requested change in the Draft PR and ask the Technical Lead to approve it. Registry status, factory mapping, dependencies, and model-specific parameter fields are the only expected shared integration changes.

## Synthetic Smoke Test

Run contract tests before using the project data:

```bash
.venv/bin/python -m pytest tests/test_model_contract.py tests/test_<model>_adapter.py -q
```

The model-specific tests must use synthetic arrays and verify probability shape, length, finiteness, range, deterministic seed handling, JSON-safe parameters/metadata, and invalid-parameter rejection. XGBoost tests must also prove that only a split explicitly named `validation` can be used for early stopping.

## Validation Experiment

After synthetic tests pass and the registry status is intentionally changed to `implemented`, run one bounded Validation experiment through the shared entry point:

```bash
.venv/bin/python scripts/run_experiment.py \
  --experiment-id <safe_experiment_id> \
  --model <random_forest|balanced_random_forest|xgboost> \
  --feature-set B \
  --preprocessing-strategy missing_plus_flag \
  --class-weight none
```

Do not call the estimator from an ad hoc training script. Model-specific parameters must be supplied through the controlled `ExperimentSpec.model_parameters` path; add CLI/config plumbing only as a small reviewed change.

Generated results belong under `outputs/experiments/<experiment_id>/` and remain ignored by Git. The standard files are `validation_metrics.json`, `thresholds.csv`, and `run_metadata.json`. Do not manually edit these files and do not commit them.

## Proof That Test Was Not Accessed

Include all of the following evidence in the PR:

1. `ExperimentSpec.evaluation_split` remains `validation`.
2. The adapter receives only Training and, if supported, Validation in `fit`.
3. No adapter call receives `runner.partitions["test"]`.
4. Test Guard and runner isolation tests pass.
5. `run_metadata.json` states `test_access: transform/schema/finite checks only; no Test prediction or metrics`.
6. No Test metric, probability, confusion matrix, threshold, or ranking file exists in the diff.

## Draft Pull Request

Push the member branch and create a Draft PR against `integration/stage2`:

```bash
git push -u origin <branch>
gh pr create --draft \
  --base integration/stage2 \
  --head <branch> \
  --title "feat: implement <model> adapter"
```

The PR description must report changed files, controlled parameter scope, synthetic tests, full test result, data-leakage implications, Test access status, manifest SHA, Validation PR-AUC/ROC-AUC/KS, default-threshold metrics, operational threshold and metrics, training/inference time, actual imbalance parameters, and early-stopping metadata when applicable. Metrics must be generated by the shared runner, clearly labeled Validation, and never copied from legacy reports.

Do not merge the Draft PR. The Technical Lead reviews contract compliance, reproducibility, result schema, Test protection, and comparison fairness before it becomes ready for review.
