# EDA Findings / EDA 核心结论

The values below are the reviewed whole-file `raw_audit` snapshot. The runnable
source of truth is `outputs/eda/raw_audit/`; `PPT_NOTES_BILINGUAL.md` is regenerated
from the CSV tables on every EDA run.

## English

- The dataset contains 150,000 records and 10 predictive features.
- The positive class accounts for only 6.684% of the data, indicating severe class imbalance.
- `MonthlyIncome` has 29,731 missing values (19.82%).
- `NumberOfDependents` has 3,924 missing values (2.62%).
- Delinquency codes 96 and 98 affect 269 unique rows and occur together across all three delinquency fields. Raw abnormal-code outputs contain no target-conditioned statistics.
- One record has `age = 0`, which is invalid under the shared preprocessing contract.
- Identical predictor vectors create 646 excess duplicate rows.
- There are 37 predictor-identical groups with conflicting target labels.
- Revolving utilization, debt ratio, and monthly income have strongly right-skewed, long-tailed distributions.
- Most values in all three delinquency-count predictors are zero.

## 中文

- 数据集包含 150,000 条记录和 10 个预测特征。
- 正类仅占 6.684%，存在严重的类别不平衡。
- `MonthlyIncome` 缺失 29,731 条，缺失率为 19.82%。
- `NumberOfDependents` 缺失 3,924 条，缺失率为 2.62%。
- 逾期字段中的 96 和 98 异常码共影响 269 行，并在三个逾期字段中同时出现。Raw 异常码输出不包含任何 target 条件统计。
- `age = 0` 有 1 条，按统一预处理规则属于无效值。
- 相同 predictor 向量形成 646 条超额重复记录。
- 存在 37 个 predictor 完全相同但 target 不同的冲突组。
- 循环额度使用率、负债比例和月收入呈明显右偏与长尾分布。
- 三个逾期次数字段的大多数取值为 0。

## PPT-ready short version

**English:** The dataset contains 150,000 borrowers and 10 predictors, with a
positive rate of only 6.684%. Missingness is concentrated in monthly income
(19.82%) and number of dependents (2.62%). Codes 96/98 in the delinquency fields
affect 269 rows and occur together across all three fields, while
utilization, debt ratio, and income are markedly long-tailed. Duplicate and
conflicting predictor groups require group-aware splitting to prevent leakage.

**中文：** 数据集包含 15 万名借款人和 10 个预测特征，正类占比仅为 6.684%。
缺失值主要集中在月收入（19.82%）和抚养人数（2.62%）。逾期字段的 96/98
异常码影响 269 行，并在三个逾期字段中同时出现；额度使用率、负债比和收入存在明显长尾。
重复及标签冲突的 predictor 组说明必须使用 group-aware split 以防止数据泄漏。

## Train-only and Validation sensitivity / 仅训练集 EDA 与 Validation 敏感性实验

**English:** The frozen Training partition contains 95,995 rows, with a positive
rate of 6.723%. It contains 173 rows with codes 96/98, and their positive rate is
52.60%. Using the same frozen manifest, Feature Set B, balanced Logistic
Regression, and shared evaluation APIs, `missing_plus_flag` achieves a Validation
PR-AUC of 0.359228 versus 0.304623 for `keep_raw`. Operational precision is
0.175753 versus 0.135163 at approximately 75% recall. No Test model metric is
calculated.

**中文：** 冻结的训练分区包含 95,995 条记录，正类占比为 6.723%。其中 173 行
含有 96/98 异常码，这些记录的正类率为 52.60%。在同一冻结 manifest、
Feature Set B、平衡权重逻辑回归和统一评价 API 下，`missing_plus_flag` 的
Validation PR-AUC 为 0.359228，高于 `keep_raw` 的 0.304623；在约 75% 召回率下，
两者的 operational precision 分别为 0.175753 和 0.135163。未计算任何 Test 模型指标。
