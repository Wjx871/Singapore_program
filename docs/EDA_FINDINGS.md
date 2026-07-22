# Exploratory Data Analysis Findings

## Scope and evidence policy

This report separates the first-stage whole-file audit from the analysis used to
support feature handling:

- `raw_audit` covers all 150,000 labeled rows only for the pre-declared schema,
  class-balance, missingness, duplicate, 96/98, and univariate-distribution audit.
- `train` is loaded through the frozen-manifest Train-only interface and is the
  only scope used for new label-conditioned EDA.
- The 96/98 model comparison is read from the shared Logistic Regression runner.
  It uses Feature Set B, the shared preprocessing/metrics/threshold APIs, and
  Validation only. No Test probability or model metric is produced.

The generated source tables, figures, bilingual notes, and run metadata are in
the ignored `outputs/eda/` directory. The frozen split manifest SHA-256 is
`5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5`.

## 1. Raw audit (descriptive baseline)

The raw labeled file contains 150,000 rows and 10 predictors. There are 139,974
negative records and 10,026 positive records, giving a positive rate of 6.684%.
This confirms severe class imbalance.

Missingness is concentrated in two fields:

| Predictor | Missing rows | Missing rate |
|---|---:|---:|
| MonthlyIncome | 29,731 | 19.82% |
| NumberOfDependents | 3,924 | 2.62% |

Codes 96/98 occur together across all three delinquency fields and affect 269
unique raw rows. The Raw abnormal-code tables report occurrence counts and rates
only; they intentionally contain no target counts or positive rates. Any
label-conditioned interpretation is restricted to the frozen Training partition.

Additional data-quality findings are:

- one row has `age = 0`;
- 149,354 unique predictor vectors imply 646 excess duplicate rows;
- 37 predictor-identical groups contain conflicting targets, covering 145 rows;
- revolving utilization, debt ratio, and monthly income are strongly
  right-skewed and long-tailed;
- the three delinquency-count predictors are dominated by zeros.

These duplicate/conflict findings support the already frozen group-aware split:
identical predictor vectors must remain in one partition to prevent leakage.

## 2. Train-only label-conditioned EDA

The frozen Training partition contains 95,995 rows: 89,541 negatives and 6,454
positives. Its positive rate is 6.723%, close to the whole-file descriptive rate.

| Training finding | Result |
|---|---:|
| MonthlyIncome missing | 18,966 (19.76%) |
| NumberOfDependents missing | 2,563 (2.67%) |
| 96/98 affected rows | 173 |
| Positive rows among 96/98 records | 91 |
| Positive rate among 96/98 records | 52.60% |
| Excess duplicate predictor rows | 409 |
| Conflicting-target predictor groups | 27 |
| `age <= 0` rows | 0 |

The zero rates of the Training delinquency predictors are 83.89% for 30–59 days,
94.94% for 60–89 days, and 94.32% for 90+ days. The strongest univariate
Training Spearman associations with the target are the three delinquency fields,
followed by revolving utilization. These are associations, not causal effects.

## 3. Shared preprocessing sensitivity

Both approved strategies were applied to the same 95,995 Training rows through
the shared preprocessor and Feature Set B:

| Strategy | Features | Remaining 96/98 cells | Abnormal flag rows | Finite output |
|---|---:|---:|---:|---|
| missing_plus_flag | 17 | 0 | 173 | Yes |
| keep_raw | 16 | 519 | 0 | Yes |

`missing_plus_flag` converts each 96/98 occurrence to missing, imputes from
Training-only medians, and adds `HasAbnormalDelinquencyCode`. `keep_raw` preserves
the codes and omits that flag. The EDA module does not duplicate either rule.

## 4. Validation-only LR comparison

The shared LR runner compares both strategies with Feature Set B and balanced
class weighting. All metrics below are Validation results.

| Strategy | PR-AUC | ROC-AUC | KS | Operational precision | Operational recall |
|---|---:|---:|---:|---:|---:|
| missing_plus_flag | 0.359228 | 0.827584 | 0.514616 | 0.175753 | 0.750314 |
| keep_raw | 0.304623 | 0.788849 | 0.432696 | 0.135163 | 0.750941 |

Under this fixed LR/Feature Set B comparison, `missing_plus_flag` improves PR-AUC
by 0.054605 and also improves ROC-AUC, KS, and operational precision at the
pre-declared recall constraint. This Validation evidence supports retaining
`missing_plus_flag` as the shared default. It does not establish a business
meaning for codes 96/98 and must not be generalized beyond this controlled
experiment without further evidence.

## 5. Presentation-ready conclusions

- The data are severely imbalanced, so accuracy alone is not an informative
  evaluation metric; PR-AUC and the shared operational threshold are required.
- Missingness is concentrated in income and dependents and is handled with
  Training-only medians plus explicit missingness flags.
- Codes 96/98 are rare but label-associated in Training. The shared
  `missing_plus_flag` strategy materially outperforms `keep_raw` for the fixed
  Validation LR comparison.
- Heavy tails motivate robust visual summaries. Plot-level p1–p99 clipping and
  `log1p` are display-only and never modify model inputs.
- Duplicate and conflicting predictor groups are retained and isolated by the
  frozen group-aware split rather than manually deleted.

## Limitations

EDA cannot establish causality, fairness, business deployment fitness, or the
semantic meaning of 96/98. The strategy result comes from one frozen split and a
fixed Logistic Regression configuration. Independent Test results remain sealed
and cannot be used to revise preprocessing, features, thresholds, or model
selection.
