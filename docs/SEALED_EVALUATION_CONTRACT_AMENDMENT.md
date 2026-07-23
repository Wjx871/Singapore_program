# Sealed Evaluation Contract Amendment

## Authorization

The project owner explicitly authorized this amendment on 2026-07-24 before
any Independent Test probability or metric was generated.

## Amended frozen threshold

- Previous rounded display value: `0.548155665398`
- Authoritative Validation-selected value: `0.548155665397644`
- Classification rule: `probability >= threshold`

The full-precision value is recorded in the authoritative Validation artifacts:

- `outputs/experiments/xgb_child5/thresholds.csv`
- `outputs/experiments/xgb_child5/validation_metrics.json`
- `outputs/model_comparison/model_comparison.csv`

The rounded display value is slightly greater than the underlying float32
boundary probability. It therefore excludes one positive Validation sample
and cannot reproduce the frozen operational Precision, Recall, or confusion
matrix within the fixed `1e-6` tolerance.

## Governance effect

This amendment restores the threshold originally selected from Validation. It
does not perform threshold selection, parameter tuning, calibration, model
selection, or any other decision using Independent Test.

At publication:

- Independent Test `predict_proba` call count was zero.
- No Independent Test probability hash existed.
- No Independent Test metric or confusion matrix existed.
- The frozen model, parameters, preprocessing, feature order, and Validation
  evidence were unchanged.

The executor must use `0.548155665397644` without decimal rounding.
