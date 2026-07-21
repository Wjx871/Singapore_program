# Team Workflow

This document defines the capability-based ownership model, dependency gates, and Git workflow for the six-person team. Except for 王家兴, member names are not yet confirmed and therefore remain role-based placeholders.

## Team Structure

The project does not use an equal one-model-per-person assignment. One Technical Lead owns the shared experimental pipeline and Logistic Regression; the remaining roles are assigned according to technical difficulty and dependency boundaries.

The four core models remain Logistic Regression, Random Forest, Balanced Random Forest, and XGBoost. LightGBM remains legacy/optional only and is excluded from the MVP and primary presentation.

| Role | Branch | Primary ownership | Expected delivery files |
|---|---|---|---|
| 王家兴 — Technical Lead and Core Pipeline Developer | `feat/core-pipeline-lr` | Configuration, `feature_hash_v1`, group-aware split, split manifest, leakage-safe preprocessing, Feature Sets A/B/C, Logistic Regression, Test Set Guard, integration, code review, sealed evaluation, result consistency review | `configs/experiment.yaml`, `src/credit_risk/config.py`, `src/credit_risk/data.py`, `src/credit_risk/features.py`, `src/credit_risk/preprocessing.py`, `src/credit_risk/models.py`, `src/credit_risk/experiment.py`, `scripts/run_experiment.py`, pipeline tests |
| XGBoost Member | `feat/xgboost` | XGBoost wrapper, `scale_pos_weight`, validation early stopping, bounded search, feature importance, optional SHAP, runtime analysis | XGBoost section/module under `src/credit_risk/`, model tests, configuration additions, generated-output code only |
| Forest Member | `feat/forest-models` | Random Forest, Balanced Random Forest, fair RF/BRF comparison, balanced sampling analysis, feature importance, false-positive trade-off | Forest section/module under `src/credit_risk/`, model tests, configuration additions, generated-output code only |
| Evaluation Member | `feat/evaluation` | PR-AUC, ROC-AUC, KS, threshold metrics, operational threshold, ROC/PR curves, confusion matrix, model comparison outputs | `src/credit_risk/metrics.py`, `src/credit_risk/thresholding.py`, `src/credit_risk/plots.py`, `tests/test_metrics.py`, threshold/test-isolation tests |
| Data Analysis Member | `feat/data-analysis` | EDA, class distribution, missingness, 96/98 analysis, predictor distributions, presentation-ready data plots, simple sensitivity-result organization | `scripts/audit_data.py`, data-plot generation code, EDA tests or validation checks; generated figures remain uncommitted |
| Documentation and Presentation Member | `docs/presentation` | README and reproducibility guidance, experiment log structure, references, presentation outline, English script, Q&A preparation, cross-machine reproduction, visual consistency | `README.md`, `docs/`, reproducibility instructions, presentation source/outline when approved; no manually entered formal metrics |

If the Stage 2 implementation keeps a single `models.py`, model contributors must coordinate small, reviewable changes to that file. Splitting it into model-specific modules requires Technical Lead approval and must not change the locked experiment contract.

## Stage 2 Dependency Gates

### Work That Can Start Immediately

- **Technical Lead:** configuration loader/schema, `feature_hash_v1`, split builder, manifest validation, preprocessing interfaces, Feature Set interfaces, Logistic Regression vertical slice, and Test Set Guard.
- **Evaluation Member:** pure metric and threshold APIs plus unit tests using synthetic labels/probabilities; no independent test data is required.
- **Data Analysis Member:** repository-relative EDA and plotting code for class balance, missingness, 96/98 codes, and predictor distributions. Raw data remains local and generated figures are not committed.
- **Documentation Member:** reproducibility skeleton, experiment-log template, references, presentation outline, Q&A inventory, and cross-machine checklist without model-result claims.
- **XGBoost/Forest Members:** review the shared estimator contract and prepare design notes or synthetic smoke tests, without running formal searches or creating an alternative data pipeline.

### Work That Must Wait for Shared Interfaces

- Random Forest, Balanced Random Forest, and XGBoost integration must wait until the Technical Lead freezes the manifest loader, feature matrix/target/group contract, Feature Set B schema, and model wrapper interface.
- Formal tuning, runtime comparison, feature importance, sensitivity analysis, and model comparison must wait until the common preprocessing and metric APIs pass integrated validation.
- No model branch may generate its own split, imputation statistics, feature definitions, metrics implementation, or test-set loader.

### Gate Sequence

1. **G1 — Configuration contract:** YAML loader, schema, paths, seeds, and dependency decisions are reviewable.
2. **G2 — Data identity:** `feature_hash_v1` and the group-aware manifest pass row/group isolation checks.
3. **G3 — Feature contract:** train-only preprocessing and Feature Sets A/B/C pass schema and leakage tests.
4. **G4 — Evaluation contract:** unified metrics, operational threshold, output schema, and synthetic tests pass.
5. **G5 — Model integration:** LR first, then RF/BRF and XGBoost through the same interfaces.
6. **G6 — Sealed evaluation:** only after configuration, features, models, and threshold rules are frozen.

## Branch and Pull Request Flow

```text
personal feature branch
        ↓ Pull Request
integration/stage2
        ↓ Stage 2 integrated validation
        ↓ Pull Request
main
        ↓
stage2-complete tag
```

Rules:

1. `main` contains only reviewed, stable stage baselines; direct development on `main` is prohibited.
2. `integration/stage2` is maintained by the Technical Lead. Ordinary members do not push directly to it.
3. Each contributor creates a personal branch from the latest `origin/integration/stage2` and merges through a Pull Request.
4. Every PR must use `.github/pull_request_template.md` and describe changed files, experiment scope, tests, leakage implications, and test-set access.
5. `main` receives Stage 2 only after integrated validation through a separate PR.
6. Once generated and approved, the split manifest must not be regenerated by a personal branch.
7. All models must reuse the same manifest, feature contract, preprocessing, metrics, and output schema.
8. Raw/generated data, model binaries, logs, caches, credentials, and local environments must not be committed.
9. Formal metrics must be generated by code and must never be entered or edited manually.

## Branch Creation for Other Members

After `integration/stage2` exists, each member should create their own branch after accepting the task:

```bash
git fetch origin
git switch --track origin/integration/stage2
git switch -c <assigned-branch>
git push -u origin <assigned-branch>
```

Assigned branch names are:

- `feat/xgboost`
- `feat/forest-models`
- `feat/evaluation`
- `feat/data-analysis`
- `docs/presentation`

These branches are intentionally not pre-created by the Technical Lead, avoiding unowned and stale remote branches.

## Independent Test Set Access

- Only the Technical Lead may open the independent test set after the experiment configuration, manifest, features, model settings, and operational threshold rule are frozen.
- Personal branches and routine integration tests must not load independent test rows or labels.
- Any approved sealed evaluation must be declared in the PR checklist and recorded in immutable run metadata.
- Test performance cannot trigger retuning, feature changes, threshold changes, or a new model ranking.

## Q&A Ownership

| Topic | Primary owner | Supporting owner |
|---|---|---|
| Research question, protocol, leakage prevention, split, sealed test | Technical Lead | Data Analysis Member |
| Logistic Regression and Feature Sets | Technical Lead | Evaluation Member |
| Random Forest vs Balanced Random Forest | Forest Member | Evaluation Member |
| XGBoost, weighting, early stopping, runtime | XGBoost Member | Technical Lead |
| Metrics, thresholds, confusion matrices, false alarms | Evaluation Member | Forest Member |
| Dataset, missing values, 96/98 codes, EDA plots | Data Analysis Member | Technical Lead |
| Reproducibility, limitations, presentation narrative, references | Documentation and Presentation Member | All members |

The Documentation and Presentation Member maintains the shared Q&A list, but each technical owner is responsible for the correctness of answers in their area.
