# EXPERIMENT DESIGN

> Stage 1 — Experiment Design
> 状态：Locked for Stage 2 implementation
> 定稿日期：2026-07-21（Asia/Singapore）
> 数据依据：`原始数据集（Give Me Some Credit）/archive/cs-training.csv`
> 范围边界：本阶段未训练模型、未生成正式 split manifest、未读取 independent test 产生任何模型决策。

## 1. Research Objective

最终英文题目：

> **Credit Default Risk Prediction under Class Imbalance: A Comparative Study of Machine Learning Models**

核心研究问题：

> How can we identify high-risk borrowers from highly imbalanced credit data while balancing predictive performance, interpretability, and false-alarm costs?

任务是一个范围明确的 supervised binary classification experiment，不是自动拒贷系统。正类固定为 `SeriousDlqin2yrs = 1`。主比较维度为少数类识别质量、误报代价、可解释性和计算成本，不以 Accuracy 作为主结论。

## 2. Dataset Scope

### 2.1 实际数据事实

以下数字来自对 `原始数据集（Give Me Some Credit）/archive/cs-training.csv` 的直接读取，不是旧报告转录。

| 项目 | 实际值 |
|---|---:|
| 样本数 | 150,000 |
| CSV 列数 | 12 |
| 输入 predictor 数 | 10 |
| 负类 `0` | 139,974 (93.316%) |
| 正类 `1` | 10,026 (6.684%) |
| 正负比 | 约 1:13.96 |
| ID 原列 | `Unnamed: 0` |
| ID 范围 | 1-150,000，150,000 个唯一值 |
| 含 ID 的全列完全重复行 | 0 |
| 去 ID 后 predictor+target 重复的超额行 | 609 |
| 唯一 predictor+target 组合 | 149,391 |

### 2.2 Predictor columns

`feature_hash` 和模型原始输入的基础列固定为：

1. `RevolvingUtilizationOfUnsecuredLines`
2. `age`
3. `NumberOfTime30-59DaysPastDueNotWorse`
4. `DebtRatio`
5. `MonthlyIncome`
6. `NumberOfOpenCreditLinesAndLoans`
7. `NumberOfTimes90DaysLate`
8. `NumberRealEstateLoansOrLines`
9. `NumberOfTime60-89DaysPastDueNotWorse`
10. `NumberOfDependents`

`SeriousDlqin2yrs` 是 target，`Unnamed: 0` 读入后立即重命名为 `row_id`。`row_id` 只用于数据追踪、manifest、重叠检查和预测结果关联，不得进入任何 transformer 或 model feature matrix。

### 2.3 Missing values

| Column | Missing | Rate |
|---|---:|---:|
| `MonthlyIncome` | 29,731 | 19.8207% |
| `NumberOfDependents` | 3,924 | 2.6160% |
| 其他 ID/target/predictor | 0 | 0% |

### 2.4 Labeled and official test datasets

- **正式本地实验数据**：使用 `cs-training.csv` 的全部 150,000 条有标签记录，不再下采样到 100,000。
- **Kaggle official test**：`原始数据集（Give Me Some Credit）/archive/cs-test.csv` 有 101,503 条，target 全空，不用于 AUC、PR-AUC、Precision、Recall 或任何模型选择。它只可在后续作 optional inference demo。
- **Independent internal test**：由本文第 5 节的 group-aware protocol 从 150,000 条有标签数据中生成，与当前 legacy `submission/dataset/test.csv` 不是同一实验定义。

## 3. Data Inclusion and Exclusion

### Included

- `cs-training.csv` 中所有 150,000 条有标签记录。
- 原始 predictor vector 完全相同的记录，但必须按同一 group 划分。
- predictor 相同但 target 不同的记录；不擅自修改标签，标记为 potential label noise。
- 含 96/98 abnormal delinquency code 的 269 条记录；通过预处理协议处理，不删除。

### Excluded from model inputs or evaluation

- `row_id` 不进入模型。
- `cs-test.csv` 不进入有监督评估。
- legacy `submission/dataset/train.csv` 和 `test.csv` 不作为新实验输入。
- 不因极端值、缺失值或 test 表现删除样本。

## 4. Duplicate and Group Strategy

### 4.1 Actual group facts

基于第 2.2 节的 10 个原始 predictors，不含 `row_id` 和 target：

| 统计 | 实际值 |
|---|---:|
| 唯一 predictor vector / `feature_hash` group | 149,354 |
| 样本数 > 1 的 predictor groups | 354 |
| 处于重复 predictor groups 中的总行数 | 1,000 |
| 相对唯一 predictor vector 的超额行 | 646 |
| 最大 group size | 12 |
| predictor 相同但 target 同时含 0/1 的 groups | 37 |
| 上述 conflicting-target groups 中的行 | 145 |
| 其中 target=0 / target=1 | 98 / 47 |

37 个 conflicting-target groups 不必然证明标签错误，因为数据只有 10 个脱敏 predictor，不同客户可能恰好有相同向量。但它们是 potential label noise / irreducible ambiguity，必须作为 limitation 披露。

### 4.2 Locked duplicate policy

1. 主实验保留所有原始记录，不去重。
2. 以 raw predictor vector 的 `feature_hash` 作为 group，同一 group 必须整体进入 train、validation 或 test 之一。
3. conflicting-target group 也不拆分，不修改 target。
4. deduplication 仅作 optional sensitivity analysis，不进入 MVP。

### 4.3 Relation to the legacy 102 cross-split vectors

Stage 0 在已预处理的 legacy 80k/20k 中发现 102 种跨 split 相同向量。本协议可以**确定性解决原始 predictor vector 完全相同的跨 split 问题**。

但 train-fitted imputation 和 clipping 可能使原本不同的 raw vectors 在 transform 后变成相同向量。因此 Stage 2 还要输出 post-transform collision diagnostic。该诊断只用于披露，不反向使用 test 分布重新划分。

## 5. Split Protocol

### 5.1 Feature hash specification

`feature_hash` 使用 SHA-256，不使用 Python 内置 `hash()`，也不把 pandas 内部哈希值当作长期数据 ID。

Canonical payload version 固定为 `feature_hash_v1`：

1. 列顺序严格使用第 2.2 节的 10 列。
2. 先按声明 dtype 读取。
3. missing 统一序列化为 `<NA>`。
4. integer 统一使用无前导零的十进制字符串。
5. float 先转 IEEE-754 binary64，再使用 Python `float.hex()` 的规范形式。
6. payload 包含 schema version、列名和值，字段间使用 ASCII Unit Separator `\x1f`。
7. `feature_hash = sha256(payload.encode("utf-8")).hexdigest()`。

在生成 manifest 前必须验证：相同 raw predictor tuple 得到相同 hash，不同 tuple 未发生实际 hash collision。

### 5.2 Two-stage StratifiedGroupKFold

scikit-learn 的 `StratifiedGroupKFold` 尝试在保持类别比例的同时保证 group 不重叠；官方文档也明确说明，当 group 约束与分层冲突时只能尽可能接近分层。因此本协议定义容差和明确失败策略，不假设精确比例总能自动实现。参考：[scikit-learn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)。

**Stage A — independent test**

- `n_splits=5`
- `shuffle=True`
- `random_state=42`
- `groups=feature_hash`
- 在 5 个 candidate held-out folds 中按预定义的 lexicographic score 选择 test fold：
  1. `abs(fold_sample_fraction - 0.20)` 最小；
  2. 如并列，`abs(fold_positive_rate - full_positive_rate)` 最小；
  3. 如仍并列，fold index 更小。
- 选 fold 不使用任何模型预测或性能指标。

**Stage B — validation**

- 在 Stage A 剩余约 80% 数据上再次运行。
- `n_splits=5`
- `shuffle=True`
- `random_state=43`
- `groups=feature_hash`
- 使用相同 lexicographic rule 选一个 fold 作 validation，占 remaining 约 20%，即全体约 16%。
- 剩余约 64% 作 training。

### 5.3 Acceptance tolerances

| Check | Required tolerance |
|---|---|
| Test fraction | 20% ± 0.5 percentage point |
| Validation fraction of full data | 16% ± 0.5 percentage point |
| Training fraction | 64% ± 0.5 percentage point |
| 每个 split 正类率与 full data | 绝对差 ≤ 0.10 percentage point |
| row_id overlap | 必须为 0 |
| feature_hash overlap | 必须为 0 |
| 三个 split 行数合计 | 必须等于 150,000 |
| 三个 split group 数合计 | 必须等于 149,354 |

如 sklearn `StratifiedGroupKFold` 的实际结果不通过容差，**不得退化为普通 random split**。Stage 2 应终止并输出诊断，然后使用确定性 group-level constrained assignment：以每个 group 的 `(n_negative, n_positive, group_size)` 为分配单元，最小化 split size 偏差和 class-rate 偏差，仍保证 group 完整性。该 fallback 必须作为显式配置和独立测试实现，不得静默切换。

### 5.4 Design-time feasibility preview

当前环境没有 scikit-learn，本 Stage 1 未安装依赖、未写出正式 manifest。使用纯 pandas/numpy 的确定性 group-stratified greedy allocator 进行可行性预演，结果为：

| Split | Rows | Fraction | Groups | Negative | Positive | Positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Training | 96,000 | 64.0000% | 95,587 | 89,583 | 6,417 | 6.684375% |
| Validation | 24,001 | 16.0007% | 23,897 | 22,397 | 1,604 | 6.683055% |
| Independent Test | 29,999 | 19.9993% | 29,870 | 27,994 | 2,005 | 6.683556% |

group overlap 为 0，行数和 group 数均完整覆盖全数据。该结果只证明 64/16/20 group-aware stratification 在本数据上可行，**不是最终 split manifest，也不作为模型结果**。Stage 2 必须用锁定的 scikit-learn 版本重新生成并验证正式 manifest。

### 5.5 Split pseudocode

```python
raw = load_labeled_csv(config.data.labeled_path)
raw = raw.rename(columns={"Unnamed: 0": "row_id"})
assert raw["row_id"].is_unique

predictor_columns = config.data.predictor_columns
raw["feature_hash"] = canonical_sha256(raw[predictor_columns])

outer = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
outer_candidates = list(
    outer.split(raw[predictor_columns], raw[target], groups=raw["feature_hash"])
)
test_fold = choose_fold_by_locked_data_only_rule(outer_candidates, target_fraction=0.20)
development_idx, test_idx = outer_candidates[test_fold]

development = raw.iloc[development_idx].copy()
inner = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=43,
)
inner_candidates = list(
    inner.split(
        development[predictor_columns],
        development[target],
        groups=development["feature_hash"],
    )
)
validation_fold = choose_fold_by_locked_data_only_rule(
    inner_candidates,
    target_fraction=0.20,  # 20% of the 80% development set
)
train_local_idx, validation_local_idx = inner_candidates[validation_fold]

assign_split(raw, test_idx, "test")
assign_split(development, validation_local_idx, "validation")
assign_split(development, train_local_idx, "train")

assert_disjoint_row_ids()
assert_disjoint_feature_hashes()
assert_split_tolerances()
write_manifest_sorted_by_row_id()
```

### 5.6 Split manifest and versioning

`data/processed/split_manifest.csv` 至少包含：

- `row_id`
- `split` (`train`, `validation`, `test`)
- `feature_hash`
- `target`

Stage 2 同时生成 `data/processed/split_metadata.json`，记录：

- raw CSV SHA-256：`1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac`
- manifest SHA-256
- schema/config version
- feature hash version
- sklearn/pandas/numpy/Python 版本
- split seeds、fold indices、行数、group 数、标签数和比例
- conflicting-target group 统计
- 生成时间和代码版本；当前目录不是 Git repo 时记录 `git_commit: null`

一旦正式 manifest 被用于模型实验，不得为改善指标重新生成。如 raw/config/hash 协议改变，必须产生新 dataset version，不覆盖旧 metadata。

## 6. Leakage Prevention Rules

1. 顺序固定为 `raw -> group-aware split -> fit preprocessing on train -> transform validation/test`。
2. median、quantile cap、scaler、encoder、feature selector、sampler 仅在 training data 或 CV training fold 上 `fit` / `fit_resample`。
3. validation/test 只能 `transform` 和 `predict_proba`。
4. CV 使用 `feature_hash` 作 groups，相同 raw predictor vector 不跨 fold。
5. `scale_pos_weight` 和 class-weight ratio 只由当前 training partition 的 target 计算。
6. validation 可用于 model/feature selection、early stopping 和 threshold selection。
7. independent test 不得出现在 `fit`、`eval_set`、CV、Optuna/GridSearch objective、feature ablation 或 threshold selection。
8. 不删除 validation/test 极端样本，不根据 test 重新定义 cap。
9. 预处理 transformer 必须可以记录 fitted medians/caps/columns 和 fit split ID，用于测试。

## 7. Missing and Abnormal Value Handling

### 7.1 Actual 96/98 facts

以下统计来自 raw `cs-training.csv`：

| Value | Affected rows per delinquency field | Unique affected rows | Target 0 | Target 1 | Positive rate |
|---|---:|---:|---:|---:|---:|
| 96 | 5 | 5 | 1 | 4 | 80.00% |
| 98 | 264 | 264 | 121 | 143 | 54.1667% |
| 96 or 98 | 269 | 269 | 122 | 147 | 54.6468% |

对每个受影响行，三个 delinquency fields 同时取 96 或同时取 98。该异常群正类率远高于总体 6.684%，不能直接置 0。本地 `Data Dictionary.xls` 没有解释 96/98 语义。

### 7.2 Locked main strategy

**Missing + Flag（主实验默认）**

1. 在 imputation 前检测三个 delinquency fields 是否出现 96/98。
2. 新建一个行级 `HasAbnormalDelinquencyCode`。
3. 将三个字段中的 96/98 转为 missing。
4. 每个 delinquency field 的 median 只在 training set / CV training fold 上 fit。
5. validation/test 使用 training-fitted median transform。

**Sensitivity A — Keep Raw**

- 保留 96/98 原值，不生成 abnormal flag。
- 用于评估主方案对排序能力和校准的影响。
- 不根据 test 选择方案；方案选择只看 validation PR-AUC 及 operational trade-off。

### 7.3 Field-level preprocessing table

| Column | Data Type | Known Issue | Training-only Fitted Operation | Deterministic Transformation | Missing Indicator | Output Feature |
|---|---|---|---|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | float64 | 极端右尾 | upper 0.99 quantile cap | clip to train cap | No | same name |
| `age` | int64 | 1 条 `age=0`，不符业务常识 | median after invalid-to-missing | `age <= 0 -> missing` | No（仅 1 条） | `age` |
| `NumberOfTime30-59DaysPastDueNotWorse` | int64 | 96/98 abnormal code | median after code-to-missing | `96/98 -> missing` | aggregate abnormal flag | same name + shared flag |
| `DebtRatio` | float64 | 极端右尾 | upper 0.99 quantile cap | clip to train cap | No | same name |
| `MonthlyIncome` | float64 | 29,731 missing；右尾；0 可能是真实无收入 | median + upper 0.99 cap | 0 保留，不再擅自替换 | Yes | same name + `MonthlyIncomeMissingFlag` |
| `NumberOfOpenCreditLinesAndLoans` | int64 | 无已知 missing/code issue | none | identity | No | same name |
| `NumberOfTimes90DaysLate` | int64 | 96/98 abnormal code | median after code-to-missing | `96/98 -> missing` | aggregate abnormal flag | same name + shared flag |
| `NumberRealEstateLoansOrLines` | int64 | 无已知 missing/code issue | none | identity | No | same name |
| `NumberOfTime60-89DaysPastDueNotWorse` | int64 | 96/98 abnormal code | median after code-to-missing | `96/98 -> missing` | aggregate abnormal flag | same name + shared flag |
| `NumberOfDependents` | float64 (conceptually count) | 3,924 missing | median | none before imputation | Yes | same name + `DependentsMissingFlag` |

补充规则：

- missing flags 必须在 imputation 前从原始值生成。
- upper quantile caps 在主 MVP 中固定为 0.99，只存储 training-fitted 值；不设下限 cap。
- Logistic Regression 的 StandardScaler 只 fit 连续/计数型特征；二值 flags 原样 passthrough。
- tree models 使用相同清洗/特征数值，但不使用 StandardScaler。
- 所有步骤必须放在 sklearn `Pipeline` / `ColumnTransformer` 或具有同等 `fit/transform` 语义的 custom transformer 内。

## 8. Feature Sets

### Feature Set A — Raw Baseline

用于检验原始信贷特征的基础区分能力。包含：

- 10 个原始 predictors（经第 7 节的 leakage-safe cleaning）。
- `MonthlyIncomeMissingFlag`
- `DependentsMissingFlag`
- `HasAbnormalDelinquencyCode`

共 13 个模型输入列。三个 flags 被视为数据质量保真信息，而非业务特征工程。

### Feature Set B — Safe Engineered Features

**主模型比较的默认 Feature Set**。包含 Feature Set A，再加：

- `IncomePerDependent = MonthlyIncome / (NumberOfDependents + 1)`
- `LogMonthlyIncome = log1p(MonthlyIncome)`
- `HighDebtFlag = 1[DebtRatio > 1]`

共 16 个输入列，**不包含 `DelinquencyScore`**。

顺序固定为：先使用 training-fitted values 完成 abnormal handling、imputation 和 clipping，再计算上述确定性派生特征。`log1p` 要求收入处理后不小于 0；若将来数据违反该条件，pipeline 应报错而不是静默修改。

### Feature Set C — DelinquencyScore Ablation

`DelinquencyScore = 1*x30_59 + 3*x60_89 + 5*x90_plus`，只在完成 96/98 handling 和 imputation 后计算。

Feature Set C 是 ablation family，不是主模型比较默认输入：

| Variant | Delinquency raw fields | DelinquencyScore | Purpose |
|---|---|---|---|
| C1 / raw-only | 保留 3 个 | 不使用 | 主 Feature Set B 对照 |
| C2 / score-only | 删除 3 个 | 使用 | 检验综合分是否可替代原始逾期信息 |
| C3 / both | 保留 3 个 | 使用 | 展示预测变化和 LR multicollinearity risk |

Logistic Regression 必须完成 C1/C2/C3 三组。C3 中的 coefficient 不得解释为因果，也不得用系数绝对值直接声称业务重要性。

## 9. Class Imbalance Strategy

MVP 不使用 SMOTE。

| Model | Main strategy | Comparison |
|---|---|---|
| Logistic Regression | `class_weight="balanced"` | `None` vs `balanced`，只用 training CV/validation 选择 |
| Random Forest | `class_weight="balanced_subsample"` | `None` vs `balanced_subsample` |
| Balanced Random Forest | per-tree balanced bootstrap sampling | 与相同搜索边界的 Random Forest 比较 |
| XGBoost | `scale_pos_weight = n_negative / n_positive` | 1.0 vs training-derived ratio |

权重比率必须在当前 training partition / CV training fold 上重新计算。SMOTE 仅作 optional 后续消融，且只能通过 imbalanced-learn Pipeline 在 CV training fold 内 `fit_resample`；validation/test 不得过采样。

## 10. Model Scope

模型和接入顺序固定为：

1. **Logistic Regression**：interpretable linear baseline，先打通 Pipeline、Evaluation、Logging 和 Output Protocol。
2. **Random Forest**：Bagging ensemble，检验非线性与交互。
3. **Balanced Random Forest**：imbalance-aware Bagging，每棵树使用类别平衡抽样，与标准 Random Forest 做公平对照。
4. **XGBoost**：Gradient Boosting，小型搜索 + validation-only early stopping。

LightGBM 仅保留为 legacy/optional appendix model，不进入 MVP、核心模型比较或主 presentation。不新增 Deep Learning、Stacking 或 Web 系统。

## 11. Hyperparameter Tuning

### 11.1 Common rules

- Training Set 用于 model fitting、group-aware CV 和 hyperparameter search。
- CV 默认 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=44)`，`groups=feature_hash`。
- primary CV scoring 为 Average Precision / PR-AUC。
- Validation Set 用于最终 feature/model setting 比较、Boosting early stopping 和 operational threshold。
- Test Set 不进入 tuning。
- 搜索预算在 `configs/experiment.yaml` 中锁定，不根据 test 临时扩大。

### 11.2 Bounded search spaces

| Model | MVP bounded search |
|---|---|
| Logistic Regression | `C=[0.03, 0.1, 0.3, 1.0, 3.0]`; penalty=`l2`; class weight 对照；可在后续小型消融加 `l1` |
| Random Forest | `n_estimators=[300, 600]`; `max_depth=[None, 12]`; `min_samples_leaf=[2, 5]`; `max_features=["sqrt"]` |
| Balanced Random Forest | 与 Random Forest 共享可比搜索边界；使用 `BalancedRandomForestClassifier` 并在 metadata 记录 sampling parameters |
| XGBoost | 不超过 12 组候选；`learning_rate=[0.03,0.08]`, `max_depth=[3,5]`, `min_child_weight=[1,5]`，`subsample=0.8`, `colsample_bytree=0.8` |

LightGBM 不进入 MVP 搜索预算；旧版 200 次 Optuna 搜索仅作为 legacy 审计记录。

## 12. Evaluation Metrics

### 12.1 Metric hierarchy

**Primary Metric**

- `pr_auc = average_precision_score(y_true, y_probability)`

**Secondary threshold-independent metrics**

- `roc_auc = roc_auc_score(y_true, y_probability)`
- `ks = max(TPR - FPR)`，等价于正/负类预测概率经验分布的最大差值

**Threshold-dependent metrics**

- Precision
- Recall / Sensitivity
- F1-score
- Specificity = TN / (TN + FP)
- Balanced Accuracy = (Recall + Specificity) / 2
- Confusion Matrix (`TN, FP, FN, TP`)
- Predicted Positive Rate = (TP + FP) / N
- Accuracy（只记录，不作为主结论）

**Optional but presentation-ready recommended**

- Brier Score
- Calibration Curve

所有模型必须调用同一套 metric functions，positive label 固定为 1。`PR-AUC` 在本项目中专指 Average Precision，不与 trapezoidal PR curve area 混用。

### 12.2 Runtime measurement

使用 `time.perf_counter()`，分开记录：

- `search_time_seconds`：CV/hyperparameter search 总时间。
- `final_fit_time_seconds`：锁定参数后最终 pipeline fit 时间，包含 preprocessing fit。
- `validation_inference_time_seconds`。
- `test_inference_time_seconds`。
- `test_inference_microseconds_per_row`。

不得把单次 fit time、全部 tuning time 和端到端 runtime 混写为同一“训练耗时”。

### 12.3 Result schema

`outputs/metrics/model_comparison.csv` 至少包含：

```text
run_id, dataset_version, model, feature_set, imbalance_strategy,
split, threshold_strategy, threshold, pr_auc, roc_auc, ks,
accuracy, precision, recall, f1, specificity, balanced_accuracy,
predicted_positive_rate, tn, fp, fn, tp, brier_score,
search_time_seconds, final_fit_time_seconds,
inference_time_seconds, inference_microseconds_per_row
```

threshold-independent metrics 在两种 threshold rows 中可重复保存，但必须相同；校验器应验证它们不随 threshold 变化。

## 13. Threshold Selection Rule

所有模型报告两种口径。

### A. Default Threshold

- `threshold = 0.5`
- 用于展示默认分类结果。

### B. Operational Threshold

只在 Validation Set 概率上选择：

1. 生成唯一预测概率对应的 candidate thresholds，并包含一个略低于最小概率的候选，以覆盖 all-positive 结果。
2. 对每个 candidate 计算 `prediction = probability >= threshold`。
3. 仅保留 `Recall >= 0.75` 的 candidates。
4. Precision 最大者优先。
5. 如 Precision 并列，Recall 更高者优先。
6. 如仍并列，threshold 更高者优先，以减少不必要风险标记。
7. 选定后冻结 threshold，不根据 test 调整。

对二分类概率，通过足够低的 threshold 总能达到 Recall=1，因此不定义“看 test 后的 fallback”。若实现未找到可行 candidate，应视为 threshold implementation bug 并终止。

## 14. Model Selection Rule

模型和 feature strategy 在 Validation Set 上按以下层次评估：

1. Validation PR-AUC。
2. Validation ROC-AUC。
3. Operational threshold 下 Precision，同时必须满足 Recall ≥ 0.75。
4. Final fit / inference time。
5. Interpretability 和实现复杂度。

这不是把指标简单合成一个伪精确总分。最终 presentation 必须讨论 performance、recall constraint、false-positive cost、interpretability 和 computational cost。ROC-AUC 高 0.001 不足以单独决定“最佳模型”。

Independent Test 只用于报告已选定方案的最终泛化表现，不用于重新排名或返回调参。

## 15. Early Stopping Rule

- Logistic Regression、Random Forest 和 Balanced Random Forest 不使用 early stopping。
- XGBoost `eval_set` 只可包含 Training 和 Validation。
- early-stopping metric 优先使用 Average Precision；若锁定库版本不支持稳定的 AP early stopping，可用 validation AUC，但必须在 metadata 中记录，不得切换为 test metric。
- `best_iteration` 由 Validation 决定。
- MVP 的主比较模型保持“train fit + validation early stopping/threshold + sealed test evaluation”，不为了多用 validation 数据而导致 threshold 口径变动。
- 如后续需要 train+validation 生产化 refit，必须作为独立 artifact，不冒充为主实验 test metrics 对应的同一 fitted object。

## 16. Experiment Matrix

### 16.1 Layered matrix

| ID | Experiment | Feature/strategy | Models | Dependency | Tier | Estimated CPU cost |
|---|---|---|---|---|---|---:|
| D0 | Data/split validation | raw + feature_hash | none | Stage 2 pipeline | MVP | <5 min |
| E1 | End-to-end baseline | Feature Set A + default imbalance | Logistic Regression | D0 | MVP | 5-20 min |
| E2 | Safe feature ablation | A vs B | Logistic Regression | E1 | MVP | 10-30 min |
| E3 | Class weighting | None vs balanced | Logistic Regression | E1 | MVP | 10-30 min |
| E4 | Unified four-model comparison | Feature Set B + each locked imbalance strategy | LR, RF, BRF, XGB | E1-E3/output protocol | MVP | 30-150 min total |
| E5 | 96/98 sensitivity | Keep Raw vs Missing + Flag | LR + validation-selected best tree model | E4 | Presentation-ready | 20-90 min |
| E6 | DelinquencyScore ablation | C1/C2/C3 | LR; B vs C for selected tree optional | E2/E4 | Presentation-ready | 15-60 min |
| E7 | Calibration review | Brier + calibration curve | all four core candidates | E4 | Presentation-ready | <10 min after predictions |
| O1 | Deduplication sensitivity | keep vs defined dedup | LR + selected tree | E4 | Optional | 20-60 min |
| O2 | Fold-internal SMOTE | weight vs SMOTE | LR + selected tree | E4 | Optional | 30-120 min |
| O3 | Age sensitivity/fairness | with vs without age | final candidates | E4 | Optional | 20-60 min |
| O4 | SHAP | bounded sample | selected boosting model | E4 | Optional | 5-30 min |

计算时间是普通个人 CPU 上的粗略规划范围，不是已执行耗时。

### 16.2 MVP deliverable

MVP 必须完成 D0、E1、E2、E3、E4，并为四模型同时输出 default/operational thresholds。SMOTE、deduplication、SHAP 不属于 MVP。

### 16.3 Presentation-ready deliverable

在 MVP 上完成 E5、E6、E7，使 presentation 能回答：

- 96/98 处理是否改变结论；
- `DelinquencyScore` 是否真正增益，以及 LR 共线性风险；
- 在 Recall 要求下的 Precision/误报代价；
- 概率是否有明显校准问题。

## 17. Reproducibility Requirements

- Python 3.10+，具体 minor/package versions 在 Stage 2 `requirements.txt` 锁定。
- `random_seed=42`，test split seed=42，validation split seed=43，CV seed=44。
- 所有 path 相对 project root 解析，使用 `pathlib`。
- 记录 raw/manifest/config hash、feature set、model parameters、library versions、runtime 和输出 hash。
- 原始数据只读；运行前后验证 raw SHA-256。
- 同一 run ID 的 metrics、predictions、model、plots 和 metadata 必须一致。
- 未执行的实验标记 `pending`，不填写结果。

## 18. Test Set Access Policy

### 18.1 Sealed-test lifecycle

1. Stage 2 split builder 可读取 target 以完成 stratification 和 manifest audit，但不做任何模型计算。
2. 开发/调参阶段的 model runner 只加载 train/validation row IDs。
3. 在 feature set、preprocessing、model parameters、early-stopping rule 和 threshold 全部锁定后，生成 immutable evaluation specification hash。
4. sealed evaluator 校验 specification hash 后才加载 test features/labels，调用 `predict_proba` 并写出最终指标。
5. test 结果只用于报告泛化表现，不触发新一轮 model/feature/threshold 选择。
6. 如 test evaluation 暴露后发现纯工程 bug，必须记录 incident、修复内容和重评原因；不得将正常性能不理想当作 bug。

### 18.2 Forbidden test operations

Test Set 不得用于：

- preprocessing `fit`
- feature selection / ablation choice
- hyperparameter search / CV scoring
- early stopping / `eval_set`
- class weight / `scale_pos_weight` calculation
- threshold selection
- calibration fitting
- 选择“最佳模型”或扩大搜索预算

## 19. Risks and Open Questions

1. **SGKF 最终数字待 Stage 2 验证**：本阶段只完成可行性预演，正式 fold indices/manifest 尚未生成。
2. **37 个 conflicting-target groups 语义不可确定**：主方案保留并分组，但它们会形成无法仅根据当前 predictors 消除的歧义。
3. **96/98 真实业务含义未知**：协议只将其当作 abnormal code，不宣称是“无记录”。
4. **上分位 clipping 的精确数值未知**：必须等正式 train split 生成后 fit，本 Stage 1 不报预先看全数据得到的 cap。
5. **库版本/API**：Stage 2 需锁定实际 scikit-learn/imbalanced-learn 版本并验证 SGKF 与 Balanced Random Forest；Stage 3 需实测 XGBoost early-stopping API。
6. **业务成本无真实金额**：Operational threshold 使用 Recall constraint，presentation 只做成本情景讨论，不伪造银行 FP/FN 真实金额。

以上问题不再改变 Stage 2 的核心原则；它们要么是实现期验证项，要么是必须披露的 limitation。

## 20. Stage 2 Acceptance Criteria

Stage 2 只有在以下项全部通过后才能进入模型训练重构：

1. 加载全部 150,000 条并将 `Unnamed: 0` 无损重命名为 `row_id`。
2. 根据 `feature_hash_v1` 生成 149,354 个 groups，无实际 hash collision。
3. 用锁定 sklearn 版本生成 64/16/20 StratifiedGroupKFold manifest，通过第 5.3 节全部容差。
4. train/validation/test `row_id` 和 `feature_hash` 两两无交集。
5. manifest 和 metadata 记录 raw/config/hash/library versions 与 split 统计。
6. raw CSV 运行前后 SHA-256 不变。
7. 96/98 Missing + Flag、income/dependents flags、age invalid handling、train-only imputation/capping 都有单元测试。
8. 测试能证明 fitted medians/caps/scaler 仅来自 training IDs；CV 时仅来自当前 training fold。
9. Feature Sets A/B/C 列顺序固定，transform 后无 NaN/Inf，`row_id/target/feature_hash` 均不进入 model matrix。
10. 输出 post-transform cross-split collision diagnostic，但不用它反向重分 test。
11. 开发 runner 默认拒绝加载 test rows，sealed evaluator 的边界有自动测试。

达成这些标准后，Stage 2 不需再讨论是否使用全量数据、是否 group split、是否把 98 置 0、是否将 SMOTE 纳入 MVP 等核心实验原则。
