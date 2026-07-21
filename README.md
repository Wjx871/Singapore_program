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
- Stage 2 Data Pipeline Implementation: pending
- No official model results have been produced by the new pipeline yet

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
├── src/                     # Future leakage-safe implementation
├── scripts/                 # Future experiment entry points
└── tests/                   # Future automated checks
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

Formal dependency locks and executable commands will be added during Stage 2. Until then, this repository records the audited baseline and experiment design only; it does not claim new model results.

## Disclaimer

This is an educational machine learning project. It must not be used as an automated lending decision system.
Give Me Some Credit
