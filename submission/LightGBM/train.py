# -*- coding: utf-8 -*-
"""
选题一：银行用户信贷违约风险预测 —— LightGBM 模型
数据集：Give Me Some Credit (Kaggle)
训练集 80,000 条，测试集 20,000 条，14 特征，二分类
"""

import os, pickle, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # 非交互后端，避免弹窗
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay,
)
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

warnings.filterwarnings("ignore")

# ============================================================
# 全局配置
# ============================================================
RANDOM_STATE = 42                # 固定随机种子，保证可复现
TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
OUTPUT_DIR = "results_lightgbm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 14 个特征（10 原始 + 4 衍生）
FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",    # 信用卡额度使用率
    "age",                                      # 年龄
    "NumberOfTime30-59DaysPastDueNotWorse",     # 30-59天逾期次数
    "DebtRatio",                                # 负债率
    "MonthlyIncome",                            # 月收入
    "NumberOfOpenCreditLinesAndLoans",          # 信用账户数
    "NumberOfTimes90DaysLate",                  # 90天+逾期次数
    "NumberRealEstateLoansOrLines",             # 不动产贷款数
    "NumberOfTime60-89DaysPastDueNotWorse",     # 60-89天逾期次数
    "NumberOfDependents",                       # 抚养人数
    "DelinquencyScore",                         # 综合逾期评分（衍生）
    "IncomePerDependent",                       # 人均收入（衍生）
    "LogMonthlyIncome",                         # 收入对数（衍生）
    "HighDebtFlag",                             # 高负债标记（衍生）
]
TARGET_COL = "SeriousDlqin2yrs"

N_TRIALS = 200                   # Optuna 搜索次数
EARLY_STOPPING = 100             # 早停轮数

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg):
    """带时间戳的进度打印"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def compute_metrics(y_true, y_pred, y_proba):
    """
    计算分类核心指标。
    返回 AUC、KS、Accuracy、Precision、Recall、F1 和混淆矩阵四元组。
    KS = max(TPR - FPR)，衡量模型区分正负样本的能力。
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "KS": round(max(tpr - fpr), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


# ============================================================
# 1. 加载数据
# ============================================================
log("[1/7] 加载数据")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"  训练集: {train_df.shape[0]} 条, 违约率 {train_df[TARGET_COL].mean():.4f}")
print(f"  测试集: {test_df.shape[0]} 条, 违约率 {test_df[TARGET_COL].mean():.4f}")

# ============================================================
# 2. 划分训练/验证集
#    从训练集 80K 中按 85:15 分层分出验证集 12K，用于 early stopping 和调参
# ============================================================
log("[2/7] 数据划分")
X_all = train_df[FEATURE_COLS]
y_all = train_df[TARGET_COL]
X_test = test_df[FEATURE_COLS]
y_test = test_df[TARGET_COL]

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_STATE,
)

# 类别不平衡处理：违约率 6.68%，正常/违约 ≈ 14:1
# 给违约样本 14 倍权重，使模型更关注少数类，避免"全猜正常"的退化
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  训练: {X_train.shape[0]}  验证: {X_val.shape[0]}  测试: {X_test.shape[0]}")
print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

# ============================================================
# 3. 基线模型（LightGBM 默认参数，用于对比调参效果）
# ============================================================
log("[3/7] 基线模型（默认参数）")
base = lgb.LGBMClassifier(
    objective="binary", metric="auc", verbosity=-1,
    random_state=RANDOM_STATE, n_jobs=-1,
    scale_pos_weight=scale_pos_weight, n_estimators=2000,
)
base.fit(X_train, y_train,
         eval_set=[(X_val, y_val)], eval_metric="auc",
         callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)])
baseline = compute_metrics(y_test, base.predict(X_test), base.predict_proba(X_test)[:, 1])
print(f"  测试 AUC={baseline['AUC']:.4f}  F1={baseline['F1']:.4f}")

# ============================================================
# 4. Optuna 贝叶斯超参数搜索
#    使用 TPE 采样器 + MedianPruner 剪枝，优化目标为验证集 AUC
#    搜索范围覆盖 11 个关键参数
# ============================================================
log("[4/7] Optuna 调参 (200 次搜索)")


def objective(trial):
    """单次试验：采样一组参数 → 训练 LightGBM → 返回验证集 AUC"""
    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "verbosity": -1, "random_state": RANDOM_STATE, "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        # 树的复杂度
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        # 学习率与迭代次数
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
        # 正则化
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 0.5, log=True),
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], eval_metric="auc",
              callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)])
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_STATE),          # 贝叶斯采样，比随机搜索更高效
    pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=10),  # 提前终止差的试验
)

t0 = datetime.now()


def cb(study, trial):
    """每 20 次试验打印一次进度"""
    if trial.number % 20 == 0:
        e = (datetime.now() - t0).total_seconds()
        eta = (e / (trial.number + 1) * (N_TRIALS - trial.number - 1)) / 60
        print(f"  [{trial.number+1}/{N_TRIALS}] best AUC={study.best_value:.5f}  "
              f"{e/60:.0f}min  eta {eta:.0f}min")


study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb], show_progress_bar=False)
print(f"  最佳 AUC={study.best_value:.5f}  耗时{(datetime.now()-t0).total_seconds()/60:.1f}min")

# ============================================================
# 5. 使用最优参数训练最终模型
# ============================================================
log("[5/7] 最优参数训练")
best_params = study.best_params.copy()
n_est = best_params.pop("n_estimators", 2000)

final = lgb.LGBMClassifier(
    objective="binary", metric="auc", boosting_type="gbdt",
    verbosity=-1, random_state=RANDOM_STATE, n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    n_estimators=n_est, **best_params,
)
final.fit(X_train, y_train,
          eval_set=[(X_val, y_val)], eval_metric="auc",
          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(100)])
print(f"  best_iteration = {final.best_iteration_}")

with open(os.path.join(OUTPUT_DIR, "lightgbm_model.pkl"), "wb") as f:
    pickle.dump(final, f)

# ============================================================
# 6. 模型评估：训练集/验证集/测试集三组指标 + ROC + 混淆矩阵
# ============================================================
log("[6/7] 模型评估")

y_train_proba = final.predict_proba(X_train)[:, 1]
y_val_proba = final.predict_proba(X_val)[:, 1]
y_test_proba = final.predict_proba(X_test)[:, 1]

m_train = compute_metrics(y_train, final.predict(X_train), y_train_proba)
m_val = compute_metrics(y_val, final.predict(X_val), y_val_proba)
m_test = compute_metrics(y_test, final.predict(X_test), y_test_proba)

df_m = pd.DataFrame([m_train, m_val, m_test], index=["训练集", "验证集", "测试集"])
df_m.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), encoding="utf-8-sig")
print(df_m[["AUC", "KS", "Accuracy", "Precision", "Recall", "F1"]].to_string())

gap = m_train["AUC"] - m_test["AUC"]
print(f"  训练-测试 AUC 差距: {gap:.4f} {'(良好)' if gap < 0.02 else '(需关注)'}")

# ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC={m_test['AUC']:.4f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve"); plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
plt.close()

# 混淆矩阵
ConfusionMatrixDisplay.from_predictions(
    y_test, final.predict(X_test), display_labels=["正常", "违约"], cmap="Blues",
).figure_.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 7. 特征重要性 + 结果汇总
#    特征重要性基于 Gain（分裂带来的损失下降总量），越高越重要
# ============================================================
log("[7/7] 特征重要性 + 汇总")

imp = pd.DataFrame({
    "特征": FEATURE_COLS,
    "重要性": final.feature_importances_,
}).sort_values("重要性", ascending=False)
imp["占比"] = (imp["重要性"] / imp["重要性"].sum() * 100).round(1)
imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False, encoding="utf-8-sig")

# 特征重要性柱状图
colors = ["#c0392b" if i == 0 else "#3498db" for i in range(len(imp))]
plt.figure(figsize=(8, 6))
plt.barh(imp["特征"][::-1], imp["重要性"][::-1], color=colors[::-1])
plt.xlabel("Gain Importance"); plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

# 汇总报告
train_time = (datetime.now() - t0).total_seconds() / 60
summary = f"""
======================================================================
  LightGBM 信贷违约预测 —— 结果汇总
======================================================================
数据集: Give Me Some Credit (训练 80K + 测试 20K, 14 特征)
随机种子: {RANDOM_STATE}  训练耗时: {train_time:.1f} min

【最佳超参数】 (Optuna {N_TRIALS}次搜索, 最佳验证AUC={study.best_value:.4f})
{chr(10).join(f'  {k}: {v}' for k, v in study.best_params.items())}

【测试集性能】
  AUC={m_test['AUC']:.4f}  KS={m_test['KS']:.4f}  Accuracy={m_test['Accuracy']:.4f}
  Precision={m_test['Precision']:.4f}  Recall={m_test['Recall']:.4f}  F1={m_test['F1']:.4f}
  TP={m_test['TP']}  FP={m_test['FP']}  TN={m_test['TN']}  FN={m_test['FN']}

【过拟合分析】 训练AUC={m_train['AUC']:.4f}  测试AUC={m_test['AUC']:.4f}  差距={gap:.4f}

【特征重要性 Top 5】
{imp.head(5)[['特征', '占比']].to_string(index=False)}
======================================================================
"""
with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary)
print(summary)
print("\nDone.")
