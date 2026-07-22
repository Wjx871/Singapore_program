# Credit Default Risk Prediction under Class Imbalance

## Overview

This project studies the question:

> How do different model families and imbalance-handling strategies affect the identification of high-risk borrowers?

The work is based on the Kaggle **Give Me Some Credit** dataset. The repository is being rebuilt as a leakage-safe, reproducible experiment pipeline; the original implementation is retained only for auditability.

## Research Objective

The task is binary credit-default prediction under severe class imbalance. The comparison focuses on predictive performance, interpretability, and the trade-off between identifying high-risk borrowers and creating false alarms.

## Core Models

- Logistic Regression
- Random Forest
- Balanced Random Forest
- XGBoost

LightGBM is retained only as a legacy or optional appendix model. It is not part of the MVP or primary presentation.

## Dataset

The labeled Kaggle training set contains 150,000 records, with a positive-class rate of approximately 6.684%. Raw data is not included in this repository and must be downloaded separately from Kaggle.

Place local dataset files according to [data/README.md](data/README.md). Do not commit raw, interim, or processed data.

## Current Status

- Stage 0 Repository Audit: completed
- Stage 1 Experiment Design: completed
- Stage 2 Data Pipeline Implementation and Logistic Regression vertical slice: completed on `feat/core-pipeline-lr`
- Stage 3 Shared Experiment Runner and Logistic Regression ablations: completed on `feat/shared-runner-lr-ablation`
- Only validation results have been produced; the independent test set remains sealed for model evaluation

Legacy metrics in old reports or scripts are not treated as current, verified results.

## Repository Structure

```text
.
├── PROJECT_AUDIT.md          # Stage 0 findings
├── EXPERIMENT_DESIGN.md     # Locked Stage 1 protocol
├── IMPLEMENTATION_PLAN.md   # Staged remediation plan
├── configs/                 # Experiment configuration
├── data/                    # Local data placement instructions only
├── docs/                    # Team collaboration guidance
├── submission/              # Legacy implementation retained for traceability
├── src/                     # Leakage-safe Stage 2 implementation
├── scripts/                 # Manifest and LR experiment entry points
└── tests/                   # Automated leakage and reproducibility checks
```

`submission/` contains the original implementation used for the legacy report. It includes known reproducibility and evaluation issues documented in [PROJECT_AUDIT.md](PROJECT_AUDIT.md) and is retained only for traceability.

## Experimental Principles

- Group-aware 64/16/20 train/validation/test split
- Preprocessing fitted only on training data or the current training fold
- Sealed independent test set
- PR-AUC (Average Precision) as the primary metric
- Operational threshold selected only on validation data
- No SMOTE in the MVP

See [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for the complete protocol.

## Team Workflow

Development and review rules are defined in [docs/TEAM_WORKFLOW.md](docs/TEAM_WORKFLOW.md).

## Reproducibility

Stage 2 dependencies are locked in `requirements-stage2.txt`. Generated manifests, metrics, logs, and model artifacts remain local and ignored by Git. See [STAGE2_IMPLEMENTATION.md](STAGE2_IMPLEMENTATION.md) for commands and verified runtime details.

## Data Analysis / EDA

Member D's EDA code uses two explicit scopes: a pre-declared `raw_audit` of the
whole labeled file and a label-conditioned `train` analysis loaded only through
the frozen-manifest Training interface. Validation and independent Test scopes
are not available.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eda.txt
.\.venv\Scripts\python.exe -m scripts.build_split_manifest --config configs/experiment.yaml
.\.venv\Scripts\python.exe -m scripts.run_eda --config configs/experiment.yaml --scope all
```

Generated tables, figures, bilingual PPT notes, and run metadata are written to
`outputs/eda/` and remain uncommitted. See [docs/EDA_OUTPUT_SPEC.md](docs/EDA_OUTPUT_SPEC.md)
for the output contract and [docs/EDA_FINDINGS_BILINGUAL.md](docs/EDA_FINDINGS_BILINGUAL.md)
for reviewed report-ready wording.

## Disclaimer

This is an educational machine learning project. It must not be used as an automated lending decision system.
Give Me Some Credit
