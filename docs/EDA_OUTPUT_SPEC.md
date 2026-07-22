# EDA Output Specification

## Purpose and ownership

This module is the Data Analysis member's deliverable for dataset validation,
class balance, missingness, 96/98 codes, predictor distributions, duplicate
predictors, and simple sensitivity-result organization. It reuses the Technical
Lead's configuration, feature hash, frozen manifest, and preprocessing code.

## Allowed scopes

| Scope | Rows | Allowed use | Forbidden use |
|---|---|---|---|
| `raw_audit` | All 150,000 labeled rows | Pre-declared schema, class-balance, missingness, duplicate, abnormal-code, and univariate audit | Feature, model, threshold, or hyperparameter selection |
| `train` | Frozen Training partition only | Label-conditioned EDA, predictor-target associations, preprocessing strategy diagnostics | Validation/Test analysis, private splitting, independent cleaning logic |

There is no Validation or Test selector in `src.analysis.eda`. Training access
is provided only through `src.analysis.train_only`, which verifies the frozen
manifest SHA before returning rows.

## Command

```powershell
.\.venv\Scripts\python.exe -m scripts.run_eda --config configs/experiment.yaml --scope all
```

Use `--skip-strategy-sensitivity` only for a fast plot-only run. Generated data,
figures, logs, and metadata are local artifacts and remain ignored by Git.

## Directory contract

```text
outputs/eda/<scope>/
├── README.md
├── PPT_NOTES_BILINGUAL.md
├── figures/
│   ├── class_distribution.png
│   ├── missing_values.png
│   ├── abnormal_delinquency_codes.png
│   ├── feature_distributions.png
│   ├── long_tail_features.png
│   ├── delinquency_zero_concentration.png
│   ├── spearman_correlation.png
│   └── abnormal_code_sensitivity_validation.png  # train, when available
├── tables/
│   ├── validation_checks.csv
│   ├── dataset_summary.csv
│   ├── class_distribution.csv
│   ├── missing_values.csv
│   ├── descriptive_statistics.csv
│   ├── abnormal_code_counts.csv
│   ├── abnormal_code_summary.csv
│   ├── duplicate_summary.csv
│   ├── delinquency_zero_summary.csv
│   └── spearman_correlation.csv
└── metadata/
    └── eda_run_metadata.json
```

The `train` scope additionally emits `target_spearman_training_only.csv` and,
when enabled, a Feature Set B `preprocessing_strategy_sensitivity.csv`. The
required shared LR result at `outputs/comparisons/abnormal_code_sensitivity.csv`
is organized as `abnormal_code_model_sensitivity.csv` after validation.

`raw_audit` abnormal-code tables contain occurrence fields only. `target_0`,
`target_1`, and `positive_rate` are emitted only by the `train` scope. When
sensitivity is enabled, the shared comparison file is required and must match
the frozen manifest SHA, current config SHA, Logistic Regression, Feature Set B,
balanced class weight, the two declared strategy/experiment pairs, and finite
metric/runtime values. A missing or non-conforming file fails the run explicitly.

## Output rules

- CSV files are UTF-8, comma-delimited, use a header row, and use `\n` line endings.
- Rates are stored as numeric fractions, not formatted strings.
- Counts remain integer-valued.
- PNG figures use English titles and labels and are saved at 180 DPI.
- Display clipping or `log1p` is used only inside plots and is explicitly labeled;
  source values and tabular summaries are never modified.
- Every run records raw-data hash, manifest metadata, configuration hash, Git
  commit, package versions, file hashes, and validation status.
- Metadata source paths and CLI output directories are repository-relative and
  portable; machine-specific absolute paths are forbidden.
- The frozen Training EDA contract requires exactly 95,995 rows.
- A failed ERROR-level validation stops the run before the bundle is accepted.
- Formal model metrics are read only from shared code-generated comparison files;
  EDA code does not recompute Validation/Test model results.
