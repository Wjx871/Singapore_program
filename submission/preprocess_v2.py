"""
============================================================
数据预处理 —— 统一版本（四人共用）
============================================================
目的：
  1. 四人用完全相同的训练集和测试集做模型对比
  2. 所有随机种子固定，确保任何人运行都得到相同结果
  3. 数据量控制在 10 万条

输出：
  dataset/train.csv    — 训练集（80%，含标签）
  dataset/test.csv     — 测试集（20%，含标签）
============================================================
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
# 全局设定：所有随机种子在此统一，四人保持一致
# ════════════════════════════════════════════════════════════
SEED = 42
TARGET_N = 100000          # 控制数据量
TEST_SIZE = 0.2            # 20% 作为测试集

# ════════════════════════════════════════════════════════════
# Step 1: 加载原始数据
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: 加载原始数据")
print("=" * 60)

df = pd.read_csv("d:/xxq2/credit/dataset/cs-training.csv")
df.drop(columns=df.columns[0], inplace=True)   # 去掉无名列

print(f"原始数据: {df.shape[0]} 行 x {df.shape[1]} 列")
print(f"违约率: {df['SeriousDlqin2yrs'].mean()*100:.2f}% "
      f"({df['SeriousDlqin2yrs'].sum():.0f}/{len(df)})")

# ════════════════════════════════════════════════════════════
# Step 2: 随机缩减到 10 万条（分层采样，保持违约率不变）
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2: 降采样到 10 万条")
print("=" * 60)

df_0 = df[df['SeriousDlqin2yrs'] == 0]
df_1 = df[df['SeriousDlqin2yrs'] == 1]

# 按原比例分配 10 万条
ratio = len(df_1) / len(df)
n_1 = int(TARGET_N * ratio)
n_0 = TARGET_N - n_1

print(f"原始比例: 1={len(df_1)} ({ratio*100:.2f}%), 0={len(df_0)} ({(1-ratio)*100:.2f}%)")
print(f"目标: {n_0} 正常 + {n_1} 违约 = {TARGET_N} ({n_1/TARGET_N*100:.2f}% 违约)")

df_0_sample = df_0.sample(n=n_0, random_state=SEED)
df_1_sample = df_1.sample(n=n_1, random_state=SEED)

df_sample = pd.concat([df_0_sample, df_1_sample], ignore_index=True)
df_sample = df_sample.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"采样后: {df_sample.shape[0]} 行, 违约率 {df_sample['SeriousDlqin2yrs'].mean()*100:.2f}%")

# ════════════════════════════════════════════════════════════
# Step 3: 缺失值处理
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3: 缺失值处理")
print("=" * 60)

print(f"MonthlyIncome 缺失: {df_sample['MonthlyIncome'].isnull().sum()} ({df_sample['MonthlyIncome'].isnull().mean()*100:.1f}%)")
print(f"NumberOfDependents 缺失: {df_sample['NumberOfDependents'].isnull().sum()} ({df_sample['NumberOfDependents'].isnull().mean()*100:.1f}%)")

# 用中位数填充（对异常值更鲁棒）
df_sample['MonthlyIncome'].fillna(df_sample['MonthlyIncome'].median(), inplace=True)
df_sample['NumberOfDependents'].fillna(df_sample['NumberOfDependents'].median(), inplace=True)

print("已用中位数填充")

# ════════════════════════════════════════════════════════════
# Step 4: 无效 / 错误值修正
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 4: 无效值修正")
print("=" * 60)

# age == 0 → 中位数
age_bad = (df_sample['age'] == 0).sum()
df_sample.loc[df_sample['age'] == 0, 'age'] = df_sample['age'].median()
print(f"age == 0 : {age_bad} 条, 替换为中位数 {df_sample['age'].median():.0f}")

# 逾期次数 == 98 → 大概率是特殊编码「无记录」，置 0
for col in ['NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate']:
    cnt = (df_sample[col] == 98).sum()
    if cnt > 0:
        df_sample.loc[df_sample[col] == 98, col] = 0
        print(f"{col} == 98 : {cnt} 条, 替换为 0")

# MonthlyIncome == 0 → 中位数
income_zero = (df_sample['MonthlyIncome'] == 0).sum()
df_sample.loc[df_sample['MonthlyIncome'] == 0, 'MonthlyIncome'] = df_sample['MonthlyIncome'].median()
print(f"MonthlyIncome == 0 : {income_zero} 条, 替换为中位数 {df_sample['MonthlyIncome'].median():.0f}")

# ════════════════════════════════════════════════════════════
# Step 5: 极端值截断（99 分位数封顶）
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: 极端值截断 (99th percentile)")
print("=" * 60)

cap_cols = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
for col in cap_cols:
    cap_value = df_sample[col].quantile(0.99)
    n_capped = (df_sample[col] > cap_value).sum()
    df_sample[col] = df_sample[col].clip(upper=cap_value)
    print(f"{col}: cap={cap_value:.2f}, 截断 {n_capped} 条 ({n_capped/len(df_sample)*100:.2f}%)")

# ════════════════════════════════════════════════════════════
# Step 6: 特征工程
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 6: 特征工程")
print("=" * 60)

# 综合逾期评分（按严重程度加权）
df_sample['DelinquencyScore'] = (
    df_sample['NumberOfTime30-59DaysPastDueNotWorse'] * 1 +
    df_sample['NumberOfTime60-89DaysPastDueNotWorse'] * 3 +
    df_sample['NumberOfTimes90DaysLate'] * 5
)

# 人均月收入
df_sample['IncomePerDependent'] = (
    df_sample['MonthlyIncome'] / (df_sample['NumberOfDependents'] + 1)
)

# 收入对数（平滑长尾分布）
df_sample['LogMonthlyIncome'] = np.log(df_sample['MonthlyIncome'] + 1)

# 负债率分箱标记（DebtRatio > 1 基本就是入不敷出）
df_sample['HighDebtFlag'] = (df_sample['DebtRatio'] > 1.0).astype(int)

n_new = df_sample.shape[1] - df.shape[1]
print(f"新增 {n_new} 个特征: DelinquencyScore, IncomePerDependent, LogMonthlyIncome, HighDebtFlag")
print(f"当前总列数: {df_sample.shape[1]}")

# ════════════════════════════════════════════════════════════
# Step 7: 划分训练集 / 测试集（固定随机种子，保持标签比例）
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 7: 划分训练集/测试集")
print("=" * 60)

from sklearn.model_selection import train_test_split

X = df_sample.drop(columns=['SeriousDlqin2yrs'])
y = df_sample['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED     # <-- 四个人统一用 42，划分一致
)

print(f"训练集: {X_train.shape[0]} 行, 违约率 {y_train.mean()*100:.2f}%")
print(f"测试集: {X_test.shape[0]} 行, 违约率 {y_test.mean()*100:.2f}%")

# 合并标签回去，方便别人直接用
train_df = X_train.copy()
train_df['SeriousDlqin2yrs'] = y_train.values

test_df = X_test.copy()
test_df['SeriousDlqin2yrs'] = y_test.values

# ════════════════════════════════════════════════════════════
# Step 8: 保存
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 8: 保存数据集")
print("=" * 60)

import os
os.makedirs("d:/xxq2/credit/dataset", exist_ok=True)

train_df.to_csv("d:/xxq2/credit/dataset/train.csv", index=False)
test_df.to_csv("d:/xxq2/credit/dataset/test.csv", index=False)

print(f"已保存: dataset/train.csv  ({train_df.shape[0]} 行 x {train_df.shape[1]} 列)")
print(f"已保存: dataset/test.csv   ({test_df.shape[0]} 行 x {test_df.shape[1]} 列)")

# ════════════════════════════════════════════════════════════
# 输出摘要
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[DONE] 预处理完成 —— 团队共用数据摘要")
print("=" * 60)
print(f"""
数据流向:
  原始 150,000 行
      │ SEED={SEED}, 分层下采样
      ▼
  采样 100,000 行
      │ 缺失值中位数填充, 无效值修正, 异常值截断, 特征工程
      ▼
  清洗后 100,000 行 x 15 列 (11 原始 + 4 衍生)
      │ train_test_split(SEED={SEED}, stratify=y, test_size={TEST_SIZE})
      ▼
  ┌─────────────────────────────────┐
  │ train.csv: {train_df.shape[0]} 行, 违约率 {y_train.mean()*100:.2f}% │
  │ test.csv:  {test_df.shape[0]} 行, 违约率 {y_test.mean()*100:.2f}%  │
  └─────────────────────────────────┘

四个人的模型必须都用这两个文件，严禁各自重新 split！
""")
