"""
============================================================
XGBoost 信用违约预测
数据: dataset/train.csv (80000行) / dataset/test.csv (20000行)
============================================================
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import warnings
import os, json
warnings.filterwarnings('ignore')

# 画图
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

SEED = 42

# ============================================================
print("=" * 60)
print("Step 1: 加载统一数据集")
print("=" * 60)

train = pd.read_csv("dataset/train.csv")
test  = pd.read_csv("dataset/test.csv")

X_train_full = train.drop(columns=['SeriousDlqin2yrs'])
y_train_full = train['SeriousDlqin2yrs']
X_test = test.drop(columns=['SeriousDlqin2yrs'])
y_test = test['SeriousDlqin2yrs']

print(f"训练集: {X_train_full.shape[0]} 行, 违约率 {y_train_full.mean()*100:.2f}%")
print(f"测试集: {X_test.shape[0]} 行, 违约率 {y_test.mean()*100:.2f}%")

# ============================================================
print("\n" + "=" * 60)
print("Step 2: 从训练集中再切出验证集（20%，仅调参用）")
print("=" * 60)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, stratify=y_train_full, random_state=SEED
)
print(f"训练: {X_train.shape[0]} 行, 验证: {X_val.shape[0]} 行")

# ============================================================
print("\n" + "=" * 60)
print("Step 3: max_depth 搜索 + Early Stopping")
print("=" * 60)

scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

best_depth, best_auc = None, 0
for depth in [3, 4, 5, 6]:
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=depth,
        learning_rate=0.02,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=1.0,
        scale_pos_weight=scale_weight,
        early_stopping_rounds=50,
        random_state=SEED,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    n_trees = model.get_booster().best_iteration
    print(f"  max_depth={depth}: Val AUC={auc:.4f}, trees={n_trees}")

    if auc > best_auc:
        best_auc = auc
        best_depth = depth

print(f"\n  最优 max_depth = {best_depth} (AUC={best_auc:.4f})")

# ============================================================
print("\n" + "=" * 60)
print("Step 4: 5折交叉验证（验证稳定性）")
print("=" * 60)

final_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': best_depth,
    'learning_rate': 0.02,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'scale_pos_weight': (y_train_full == 0).sum() / (y_train_full == 1).sum(),
    'early_stopping_rounds': 50,
    'random_state': SEED,
    'verbosity': 0,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_aucs = []
for fold, (tr_idx, vl_idx) in enumerate(cv.split(X_train_full, y_train_full)):
    X_tr = X_train_full.iloc[tr_idx]; X_vl = X_train_full.iloc[vl_idx]
    y_tr = y_train_full.iloc[tr_idx]; y_vl = y_train_full.iloc[vl_idx]

    m = xgb.XGBClassifier(**final_params)
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    cv_aucs.append(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1]))
    print(f"  Fold {fold+1}: AUC={cv_aucs[-1]:.4f}")

print(f"\n  5折 CV: mean={np.mean(cv_aucs):.4f}, std={np.std(cv_aucs):.4f}")

# ============================================================
print("\n" + "=" * 60)
print("Step 5: 全量训练（train_val 合并）+ 测试集评估")
print("=" * 60)

final_model = xgb.XGBClassifier(**final_params)
final_model.fit(
    X_train_full, y_train_full,
    eval_set=[(X_test, y_test)],
    verbose=25
)

y_test_pred = final_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_pred)
y_test_label = final_model.predict(X_test)

def calc_ks(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return max(tpr - fpr)

test_ks = calc_ks(y_test, y_test_pred)

print(f"\n  测试集 AUC: {test_auc:.4f}")
print(f"  测试集 KS:  {test_ks:.4f}")
print(f"  实际树数:   {final_model.get_booster().best_iteration}")

# 分类报告
print(f"\n  分类报告:")
print(classification_report(y_test, y_test_label, target_names=['Normal', 'Default']))

cm = confusion_matrix(y_test, y_test_label)
print(f"  混淆矩阵: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

# ============================================================
print("\n" + "=" * 60)
print("Step 6: 特征重要性")
print("=" * 60)

imp = final_model.feature_importances_
idx = np.argsort(imp)[::-1]
for i in range(len(idx)):
    print(f"  {i+1:2d}. {X_train_full.columns[idx[i]]:45s} {imp[idx[i]]:.4f}")

# ============================================================
print("\n" + "=" * 60)
print("Step 7: 保存模型和报告")
print("=" * 60)

os.makedirs("d:/xxq2/credit/output", exist_ok=True)

final_model.save_model("d:/xxq2/credit/output/xgb_model.json")

report = {
    "model": "XGBoost",
    "params": {k: str(v) if isinstance(v, float) else v for k, v in final_params.items()},
    "cv_mean": round(float(np.mean(cv_aucs)), 4),
    "cv_std":  round(float(np.std(cv_aucs)), 4),
    "test_auc": round(test_auc, 4),
    "test_ks":  round(test_ks, 4),
    "n_trees":  final_model.get_booster().best_iteration,
    "top_features": [(X_train_full.columns[idx[i]], float(imp[idx[i]])) for i in range(5)],
}
with open("d:/xxq2/credit/output/results.json", "w") as f:
    json.dump(report, f, indent=2)

with open("d:/xxq2/credit/output/results.txt", "w") as f:
    f.write(f"XGBoost 结果\n{'='*40}\n")
    f.write(f"测试集 AUC: {test_auc:.4f}\n")
    f.write(f"测试集 KS:  {test_ks:.4f}\n")
    f.write(f"5折 CV:     {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}\n")
    f.write(f"特征重要性:\n")
    for i in range(len(idx)):
        f.write(f"  {i+1}. {X_train_full.columns[idx[i]]:45s} {imp[idx[i]]:.4f}\n")

print("[DONE] output/xgb_model.json, output/results.json, output/results.txt")

# ============================================================
if HAS_PLT:
    print("\n" + "=" * 60)
    print("Step 8: 画图")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('XGBoost — Credit Default Prediction', fontsize=14, fontweight='bold')

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    ax = axes[0]
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {test_auc:.4f}')
    ax.plot([0,1], [0,1], 'r--', lw=1, label='Random')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve'); ax.legend(); ax.grid(alpha=0.3)

    # KS
    ax = axes[1]
    ks_curve = tpr - fpr
    ax.plot(fpr, tpr, 'b-', lw=2, label='TPR')
    ax.plot(fpr, fpr, 'r-', lw=2, label='FPR')
    ax.plot(fpr, ks_curve, 'g--', lw=2, label=f'KS = {test_ks:.4f}')
    ax.set_xlabel('FPR'); ax.set_title('KS Curve'); ax.legend(); ax.grid(alpha=0.3)

    # Feature importance
    ax = axes[2]
    top10 = idx[:10][::-1]
    ax.barh(range(10), imp[top10], color='steelblue', edgecolor='white')
    ax.set_yticks(range(10))
    ax.set_yticklabels([X_train_full.columns[i][:30] for i in top10], fontsize=8)
    ax.set_xlabel('Importance'); ax.set_title('Top 10 Features'); ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig("d:/xxq2/credit/output/eval_plots.png", dpi=150, bbox_inches='tight')
    print("[DONE] output/eval_plots.png")

print("\n" + "=" * 60)
print("[DONE] XGBoost 训练完成")
print("=" * 60)
