# IMPLEMENTATION PLAN

> 基于 `PROJECT_AUDIT.md` 的分阶段整改计划。
> 原则：小步改动、保留 legacy 可追溯性、先修正实验设计再训练，不伪造任何指标。

## 1. Recommended Target Structure

不建议第一步就移动或删除 `submission/` 下的旧脚本。先在项目根目录新建一套并行的、可测试的实验路径；旧代码保留为 legacy evidence。

```text
第九组  Give Me Some Credit  机器学习实践/
├── PROJECT_AUDIT.md
├── EXPERIMENT_DESIGN.md
├── IMPLEMENTATION_PLAN.md
├── README.md                         # Stage 7 更新后的项目主说明
├── submission/                      # 原有材料，先保留，后续标记 legacy
├── 原始数据集（Give Me Some Credit）/ # 只读，不改动
├── configs/
│   └── experiment.yaml
├── src/
│   └── credit_risk/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── preprocessing.py
│       ├── models.py
│       ├── thresholding.py
│       ├── metrics.py
│       ├── plots.py
│       └── experiment.py
├── scripts/
│   ├── audit_data.py
│   ├── run_experiment.py
│   └── generate_presentation_artifacts.py
├── tests/
│   ├── test_data_split.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_test_isolation.py
├── outputs/                        # 全部由代码生成
│   ├── metrics/
│   ├── figures/
│   ├── models/
│   ├── logs/
│   └── metadata/
├── requirements.txt
└── Makefile                        # 或仅保留 scripts/run_experiment.py
```

该结构比 prompt 中的细粒度分层更紧凑，适合当前项目规模。如后续单文件过长，再按 `data/` / `models/` / `evaluation/` 拆分，避免为目录形式过早增加复杂度。

## 2. Locked Experiment Contract (Stage 1 Complete)

Stage 1 已在不训练模型、不查看新 test 指标的前提下锁定以下合同：

1. **数据源**：仅使用 `cs-training.csv` 的全部 150,000 条有标签数据；`cs-test.csv` 只用于未来 Kaggle 无标签推理，不参与本地模型选择或最终有监督评估。
2. **去重与分组**：默认保留全部记录。对 10 个原始预测特征生成版本化 SHA-256 `feature_hash`，相同特征向量必须在同一 split；`row_id`、target 均不进入 hash 或模型特征。
3. **划分**：目标 64% train / 16% validation / 20% independent test。Stage 2 使用两阶段 `StratifiedGroupKFold`（test seed=42，validation seed=43），按预先声明的样本比例、正类率、fold index 字典序选 fold，不使用任何模型指标。
4. **异常与缺失**：三个逾期字段中的 96/98 默认转为缺失并增加聚合标记 `HasAbnormalDelinquencyCode`，不替换为 0；保留 raw 值作为敏感性对照。
5. **处理边界**：imputer、99% 上分位 clipping 与 scaler 只在 train 或当前 CV 训练折 fit；validation/test 只 transform。
6. **特征口径**：A 为清洗后 10 个原始预测特征 + 3 个数据质量标记；B 再加 3 个工程特征，且不含 `DelinquencyScore`；C1/C2/C3 仅用于逾期特征消融。
7. **主次指标**：PR-AUC（`average_precision_score`）为主指标；ROC-AUC、KS 为主要次指标。阈值指标包括 Precision、Recall、F1、Specificity、Balanced Accuracy、混淆矩阵与预测正类率；Accuracy 仅记录。
8. **阈值口径**：同时报告 threshold=0.5 和 validation-selected threshold。后者在 validation recall ≥ 75% 的候选中最大化 precision，平局时依次选 recall 更高、threshold 更高者；候选集必须包含全预测为正的低阈值，因此不设 F1 回退。
9. **不平衡处理**：MVP 仅比较未加权与 class weighting；不使用 SMOTE。
10. **测试规则**：先完成 Logistic Regression 端到端纵向切片，再依次接入 Random Forest、Balanced Random Forest 和 XGBoost。LightGBM 仅作为 legacy/optional appendix。所有参数、特征方案和阈值锁定后，independent test 只做一次统一评估；系数/重要性不做因果解读。

## 3. Phased Remediation Plan

### Stage 1 — Experiment Design

**状态：已完成（本阶段未训练任何模型）**

**目标**

把数据划分、特征组、异常处理、调参、阈值和测试隔离规则固定为可审查的实验协议。

**已新增/修改文件**

- 新增 `configs/experiment.yaml`
- 新增 `EXPERIMENT_DESIGN.md`
- 更新 `IMPLEMENTATION_PLAN.md`
- 暂不修改旧训练脚本

**验收标准**

- 已实测并登记 150,000 行数据的 ID、类别分布、缺失、重复特征组、96/98 异常码和冲突标签组。
- 已固定组隔离划分、test 禁用操作、特征集、预处理、主指标和阈值选择规则。
- 设计期可行性预览为 96,000 / 24,001 / 29,999 行，组重叠为 0；该预览不是 Stage 2 正式 manifest。
- 已将 MVP 限定为 D0–E4，96/98、逾期特征与校准实验放入 presentation-ready 矩阵。

**边界**

Stage 1 只完成设计与可行性检查，未生成正式 split manifest、未 fit 预处理器、未训练或评估模型。

### Stage 2 — Leakage-safe Data Pipeline Refactor

**目标**

建立只在 train fit 的数据流，保留 split ID 并生成可重现的数据审计元数据。

**拟新增/修改文件**

- 新增 `src/credit_risk/config.py`
- 新增 `src/credit_risk/data.py`
- 新增 `src/credit_risk/features.py`
- 新增 `src/credit_risk/preprocessing.py`
- 新增 `scripts/audit_data.py`
- 新增 `tests/test_data_split.py`, `test_preprocessing.py`, `test_features.py`
- 生成 `data/processed/split_manifest.csv` 和带哈希/版本信息的 metadata JSON
- `submission/preprocess_v2.py` 保留不删，仅在 README 中标记 legacy（标记动作留到 Stage 7）

**实现顺序**

1. 实现 YAML 加载、schema 验证和 `feature_hash_v1` 规范化，并锁定 Python/scikit-learn 等依赖版本。
2. 按 seed=42/43 执行两阶段组分层划分，写入包含 `row_id, split, feature_hash, target` 的正式 manifest，先通过行/组隔离与比例容差测试。
3. 以 manifest 为唯一 split 来源，再实现 train-only 预处理、A/B/C 特征集与特征顺序验证。
4. 增加 sealed-test 访问 guard，并检查预处理后新产生的特征碰撞；该诊断不反向改写 test 划分。

**实现要点**

- `feature_hash_v1` 使用固定列顺序、缺失标记、数字规范和 SHA-256，必须通过重跑一致性测试。
- 若 `StratifiedGroupKFold` 无法满足容差，必须显式失败；只能在记录原因后改用确定性组约束分配器，禁止静默退化为普通随机分层。
- 不向磁盘写入伪装成“原始”的清洗数据；所有中间产物必须附 metadata。
- `row_id` 只用于追溯和重叠检查，不进入特征。所有 learned transform 先 split 后 fit，engineered features 不使用 test 统计量。

**验收标准**

- train/validation/test 的 `row_id` 与 `feature_hash` 集合均两两无交集，总数分别回收到 150,000 行和 149,354 个组。
- 样本比例相对 64%/16%/20% 的绝对偏差各不超过 0.005；各 split 正类率与全局差异各不超过 0.001。
- 测试能证明 imputer/cap/scaler 的 `fit` 输入仅包含 train ID。
- 特征列顺序完全一致，输出无 NaN/Inf。
- 原始数据 SHA-256 在运行前后不变。

**风险**

删除重复样本或改变 96/98 处理会使新结果与旧结果不可直接对齐；这是必要的方法修正，不应为保持旧数字而回退。

**成本**：数据处理秒级至数分钟；开发+测试约 2-4 人时。

**Stage 2 并行协作安排**

- 阶段集成分支为 `integration/stage2`，由 Technical Lead 维护；个人分支通过 Pull Request 进入该分支，普通成员不直接 push。
- 王家兴使用 `feat/core-pipeline-lr`，负责配置、`feature_hash_v1`、group-aware split、manifest、train-only 预处理、Feature Sets A/B/C、Logistic Regression 和 Test Set Guard。
- Evaluation 成员可在 `feat/evaluation` 中立即使用合成数据实现纯指标/阈值函数和单元测试，不读取 independent test。
- Data Analysis 成员可在 `feat/data-analysis` 中立即实现 EDA 与数据图生成代码；raw data 仅保留在本地，图表和中间数据不提交。
- Documentation and Presentation 成员可在 `docs/presentation` 中准备复现说明、实验日志模板、references、英文讲稿框架和 Q&A，不填写未生成的指标。
- Forest 和 XGBoost 成员可先审查统一接口或写 synthetic smoke tests，但正式 wrapper 集成、调参和运行必须等待 manifest loader、Feature Set B schema、preprocessing 输出和 estimator contract 锁定。
- 合并门禁依次为：配置合同 → manifest/组隔离 → preprocessing/features → metrics/thresholds → LR 纵向切片 → RF/BRF/XGBoost 集成。
- 更详细的文件责任、Test Set 权限和 Q&A 分工见 `docs/TEAM_WORKFLOW.md`。

### Stage 3 — Unified Evaluation + Logistic Regression Vertical Slice

**目标**

先以 Logistic Regression 打通配置、预处理、训练、阈值、指标、日志和输出的端到端闭环，避免四模型同时开发放大数据口径错误。

**拟新增/修改文件**

- 新增 `src/credit_risk/models.py`
- 新增 `src/credit_risk/metrics.py`
- 新增 `src/credit_risk/thresholding.py`
- 新增 `src/credit_risk/experiment.py`
- 新增 `scripts/run_experiment.py`
- 新增 `tests/test_metrics.py`, `test_test_isolation.py`
- 新增 `requirements.txt`

**实现要点**

- Logistic Regression 的非二值数值特征使用 train-only `StandardScaler`，标记列直接传递；小网格只搜索 YAML 声明的 `C`、penalty 和 class weight。
- 唯一指标实现同时输出 PR-AUC、ROC-AUC、KS 及两种阈值口径，并固定 positive class=1。
- validation-selected threshold 仅接收 validation label/probability；sealed test 访问 guard 覆盖 tuning、early stopping 和 threshold selection。
- 输出先完成 `model_comparison.csv`、run metadata、预测文件和日志的最小合同，再复用给其他模型。

**验收标准**

- 参数仅来自 YAML/命令行，不再分散于 4 个顶层脚本。
- 日志记录时间、数据哈希、split 大小、特征数、参数、seed、fit/inference 耗时。
- 静态/运行时 guard 阻止 independent test 出现在 `fit/eval_set/threshold selection`。
- quick mode 可在数分钟内跑通 Logistic Regression 端到端流程。

**风险**

纵向切片如果绕过 manifest 或复制一套临时指标函数，会让后续树模型继承不一致口径；Stage 3 必须先通过接口和测试验收。

**成本**：开发+快速测试 3-6 人时；不含正式调参。

### Stage 4 — Remaining Model Integration

**目标**

在 Stage 3 已验证的统一合同上，依次接入 Random Forest、Balanced Random Forest 和 XGBoost。

**拟新增/修改文件**

- 扩展 `src/credit_risk/models.py`, `experiment.py`
- 新增 `src/credit_risk/plots.py`

**模型级规则**

- Random Forest：小型参数网格；`balanced_subsample` 作为候选；不硬编码样本数/正类数。
- Balanced Random Forest：使用 `imbalanced-learn` 实现每棵树的类别平衡抽样；与 Random Forest 共享可比搜索边界并记录抽样参数。
- XGBoost：`scale_pos_weight` 只由当前训练分区计算；early stopping 只看 validation，不看 test。

**风险**

Balanced Random Forest 的抽样默认值与 XGBoost early-stopping API 会随库版本变化，必须锁定并记录实际参数，不能只做语法检查。

**验收标准**

- PR-AUC（`average_precision_score`）、ROC-AUC、KS、Specificity、Balanced Accuracy 有单测；Brier Score 与校准曲线允许在 presentation-ready 阶段启用。
- 每个模型都输出 `threshold_0_5` 和相同的 `validation_selected` 策略。
- threshold-independent 指标不在两种阈值下重复冒充不同结果。
- 混淆矩阵明确 positive class=1，各列顺序统一。
- `model_comparison.csv` 字段与 prompt 要求一致。

**成本**：模型接口与快速验证约 2-4 人时；不含正式搜索。

### Stage 5 — Run Experiments

**目标**

执行已锁定实验，产生第一套可追溯最终结果。

**实验优先级**

1. P0 / D0：数据、正式 split manifest 与 leakage guard 验收。
2. P0 / E1：Logistic Regression + feature set A baseline + 两种阈值。
3. P0 / E2：Logistic Regression 的 A vs B 特征对照。
4. P0 / E3：Logistic Regression 类别加权 on/off 对照。
5. P0 / E4：LR、RF、BRF、XGBoost 在 feature set B 上的受控对比。
6. P1 / E5：96/98 异常处理敏感性对照。
7. P1 / E6：Logistic Regression C1/C2/C3 逾期特征消融。
8. P1 / E7：选定模型的 Brier Score/校准曲线；SMOTE、SHAP、age sensitivity 均为可选。

**拟生成文件**

- `outputs/metrics/model_comparison.csv`
- `outputs/metadata/experiment_summary.json`
- `outputs/metadata/best_model_parameters.json`
- `outputs/logs/<run_id>.log`
- `outputs/models/<model_name>.*`
- 每个模型的 validation/test 预测和最小必要的 CV/tuning 记录

**验收标准**

- 每个结果行可追溯到 run ID、数据哈希、代码版本/当前 Git 状态和参数。
- 未运行实验明确为 `pending`，不填充伪数字。
- 重复 seed=42 的 quick/baseline 关键指标在容差范围内一致。
- 测试集的最终评估日志只有一个明确的 sealed-test 阶段。

**预计运行成本（普通个人 CPU，粗略范围）**

| 任务 | 优先级 | 预计成本 |
|---|---|---|
| 四核心模型 quick smoke test | 必须 | 2-10 min |
| Logistic Regression 小网格 | 必须 | 5-20 min |
| Random Forest 4-8 组参数×5 fold | 必须 | 10-40 min |
| Balanced Random Forest 可比小网格 | 必须 | 10-40 min |
| XGBoost 4-12 组/或有限搜索 + early stopping | 必须 | 5-30 min |
| LR 共线性消融 | 必须 | 5-30 min |
| 96/98 消融 | 建议 | 根据重用参数约 15-60 min |
| SMOTE 折内对照 | 可选 | 20-90 min |

总成本预计为约 0.5-3 CPU-hours，具体取决于硬件、并行度和搜索规模。LightGBM 不进入核心实验预算。

### Stage 6 — Generate Presentation Artifacts

**目标**

仅从 Stage 5 的最终 CSV/JSON/预测文件生成英文图表，避免手工抄数。

**拟新增/修改文件**

- 新增 `scripts/generate_presentation_artifacts.py`
- 复用 `src/credit_risk/plots.py`
- 生成 `outputs/figures/*.png`

**最小图表集**

- `class_distribution.png`
- `roc_curves.png`
- `precision_recall_curves.png`
- `model_comparison.png`
- `best_model_confusion_matrix.png`
- `threshold_tradeoff.png`
- `feature_importance.png`
- `calibration_curve.png`（时间允许则保留，建议作为高影响决策的必要补充）

**验收标准**

- 全部英文标题/坐标、高分辨率、无误导性坐标范围。
- 图中数值与 `model_comparison.csv` 程序化比对一致。
- 阈值相关图明确标注 threshold strategy 和 validation selection rule。
- 特征重要性标注 model 与 importance 类型（coefficient/gain/permutation）。

**成本**：生成秒级至数分钟；开发+视觉检查 2-4 人时。

### Stage 7 — Documentation and Final Review

**目标**

用可追溯输出替换旧数字，完成 README、英文 10-12 页 PPT 内容和 Q&A 准备。

**拟新增/修改文件**

- 新增/更新项目根 `README.md`
- 更新 `PROJECT_AUDIT.md` 中已修复项状态
- 新增 `RESULTS_SUMMARY.md`
- 经用户明确授权后，再制作新 PPT/更新报告；不直接覆盖旧文件

**验收标准**

- README、结果 CSV、图表、PPT 的数字和模型名一致。
- PPT 明确区分 threshold-independent 与 threshold-dependent metrics。
- 对 limitations、fairness、false positive/negative cost、human review、calibration/drift 有明确边界，不把模型宣称为自动拒贷系统。
- 最终 15 分钟时间表包括 Q&A 预留和四人发言分工。

**成本**：文档/PPT 整合与排练约 4-8 人时，不含重跑实验。

## 4. Automated Quality Gates

建议 `scripts/run_experiment.py` 在进入正式训练前运行以下 guard，失败即终止：

1. 原始数据哈希与登记值一致。
2. train/validation/test 的 `row_id` 和 `feature_hash` 均两两无交集。
3. split 行数比例与正类率分别满足 0.005 和 0.001 的锁定绝对容差。
4. transformer 的 fit 日志只包含 train ID。
5. 输出特征列顺序与 schema 一致。
6. 任一模型输入无 NaN/Inf。
7. seed=42 的重复 quick run 结果可重现。
8. 评估函数的 positive label 固定为 1。
9. PR-AUC 统一定义为 Average Precision，不与 trapezoidal PR area 混用。
10. independent test 不可出现在 tuning/early stopping/threshold selection 调用链。
11. 所有项目路径由 `pathlib` 和项目根解析，不包含盘符。
12. 指标 CSV、JSON、图表标注与模型参数的 run ID 完全一致。
13. manifest 恰好覆盖 150,000 个 `row_id` 和 149,354 个 `feature_hash`，无重复、无遗漏。
14. 对预处理后特征碰撞只做诊断和记录，禁止因此重分 test。

## 5. Minimum Viable Version (MVP)

MVP 的目标不是做完所有可选实验，而是在最少工程量下得到可用于课程汇报的严谨结果。

**MVP 必须包含**

- D0–E4：组隔离 64/16/20 split、train-only preprocessing、LR 纵向切片、A vs B、class weighting 对照和 LR/RF/BRF/XGBoost 的 B 特征对比。
- 四个核心模型均使用小型、可控搜索空间；LightGBM 仅作为 optional appendix。
- 统一 `model_comparison.csv`。
- threshold=0.5 + 统一 validation-selected threshold。
- 一套可追溯元数据和 6 张核心图。
- 基础 README + 实验结果摘要。

**MVP 可以暂缓**

- SHAP。
- SMOTE。
- C1/C2/C3 `DelinquencyScore` 消融。
- 96/98 keep-raw 敏感性对照。
- 概率校准分析。
- 大规模贝叶斯搜索。
- Stacking/Blending。
- 完整 fairness 优化（但必须有 ethics/fairness limitation 表述）。

## 6. Presentation-ready Full Version

在 MVP 之上增加：

- 96/98 异常处理消融结论。
- 类别加权 on/off 对照；如有时间再做折内 SMOTE。
- 概率校准图/Brier Score。
- 明确的 threshold trade-off 图和业务成本情景。
- 树模型 gain/permutation importance；可选一张简化 SHAP summary。
- Age 作为 sensitive/proxy feature 的限制说明或移除 age 的 sensitivity check。
- 10-12 页英文 PPT、四人讲解分工、时长和 Q&A 备忘。

## 7. Resolved Decisions and Remaining Open Questions

Stage 1 已解决的决策包括：使用全部 150,000 条有标签数据；按特征 hash 组隔离划分；保留重复记录；96/98 默认作缺失+标记；PR-AUC 为主指标；Recall 约束下最大 Precision 的阈值规则；MVP 不使用 SMOTE；保留全部 legacy 材料。

Stage 2 实现时只需要根据环境证据确定以下工程细节，不重开核心实验决策：

1. 锁定的 Python、scikit-learn、pandas、NumPy 及模型库具体版本。
2. 正式 `StratifiedGroupKFold` manifest 的精确行数和类别数（必须满足已锁定容差）。
3. 37 个同特征不同标签组只记为潜在标签噪声/不可约简性，在没有额外业务证据前不删除、不改标。
4. 本地评估完成后是否启动 Kaggle official test 推理演示，留到后续文档/展示阶段确认。
5. 若团队后续提供明确的误拒/漏拒成本，可在不窥视 test 的前提下新增业务成本阈值实验；当前不假设不存在的成本。

## 8. Recommended Immediate Next Three Actions

1. 实现配置/schema 校验与可重现的 `feature_hash_v1`。
2. 生成并测试正式组隔离 split manifest，通过后才开放 train/validation 给后续流程。
3. 实现 train-only 预处理与 A/B/C 特征生成，再以 Logistic Regression 打通阈值、指标、日志和输出协议。

## 9. Stage 2 Implementation Status (2026-07-22)

Stage 2 已在 `feat/core-pipeline-lr` 完成以下项目：

- 严格 YAML schema 校验、仓库相对路径和锁定依赖。
- `feature_hash_v1`、两阶段 `StratifiedGroupKFold`、确定性 fold 选择及 manifest 完整性校验。
- Training-only median imputation、96/98 abnormal code handling、`AgeInvalidFlag` 和 A/B/C1/C2/C3 schema。
- E1 Logistic Regression + Feature Set A、Validation 指标、Validation-only operational threshold 与 Test Set Guard。
- 原始数据、manifest、配置、代码 commit 和运行环境的可追溯 metadata。

未完成且不属于本 Stage 2 纵向切片的项目：Feature Set A/B 正式对照、class-weight 消融、RF/BRF/XGBoost 集成、调参、Independent Test 模型评价和 presentation artifacts。

Stage 3 入口条件：Draft PR 完成对 manifest SHA、Feature Set B 合同、评价 API、Test Guard 和自动化测试的审查；之后其他模型才能共用同一 manifest 和 preprocessing contract。核心模型仍为 LR、RF、BRF、XGBoost，LightGBM 仍仅为 optional/legacy appendix。
