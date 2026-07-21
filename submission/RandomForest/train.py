"""
金融风险预测：随机森林 5 折分层交叉验证优化版。

流程：
1. 固定使用 d:\\qq\\train.csv 和 d:\\qq\\test.csv，不合并、不重新划分总体数据。
2. 只在 train.csv 内部做 StratifiedKFold 交叉验证选参数和阈值。
3. 用完整 train.csv 训练最终模型。
4. 对 test.csv 先预测，预测完成后才读取测试标签评估。

运行：
python d:\\qq\\credit_risk_random_forest_cv_optimized.py
快速自检：
python d:\\qq\\credit_risk_random_forest_cv_optimized.py --quick
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

SEED = 42
TRAIN_PATH = r"dataset/train.csv"
TEST_PATH = r"dataset/test.csv"
OUTPUT_DIR = r"d:\qq\算法结果\信用风险随机森林模型输出结果_CV优化版"
LABEL = "SeriousDlqin2yrs"
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "DelinquencyScore",
    "IncomePerDependent",
    "LogMonthlyIncome",
    "HighDebtFlag",
]


@dataclass(frozen=True)
class Config:
    train_path: str = TRAIN_PATH
    test_path: str = TEST_PATH
    output_dir: str = OUTPUT_DIR
    seed: int = SEED
    cv_folds: int = 5
    quick: bool = False


def check_csv(path: str, cols: list[str], rows: int, name: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} 不存在：{p}")
    df = pd.read_csv(p)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少列：{missing}")
    if len(df) != rows:
        raise ValueError(f"{name} 行数应为 {rows:,}，实际 {len(df):,}")
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[cols].isna().any().any():
        bad = df[cols].isna().sum()
        raise ValueError(f"{name} 存在缺失/非数值：{bad[bad > 0].to_dict()}")
    return df


def load_train(cfg: Config) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = check_csv(cfg.train_path, FEATURES + [LABEL], 80_000, "train.csv")
    y = df[LABEL].astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("训练标签必须只包含 0/1")
    if int(y.sum()) != 5_347:
        raise ValueError(f"训练集正样本应为 5347，实际 {int(y.sum())}")
    return df, df[FEATURES].to_numpy(float), y


def load_test_features(cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    df = check_csv(cfg.test_path, FEATURES, 20_000, "test.csv 特征")
    feature_df = df[FEATURES].copy()
    return feature_df, feature_df.to_numpy(float)


def load_test_label_after_prediction(cfg: Config) -> np.ndarray:
    df = check_csv(cfg.test_path, [LABEL], 20_000, "test.csv 标签")
    y = df[LABEL].astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("测试标签必须只包含 0/1")
    if int(y.sum()) != 1_337:
        raise ValueError(f"测试集正样本应为 1337，实际 {int(y.sum())}")
    return y


def param_grid(quick: bool) -> list[dict]:
    if quick:
        grid = {
            "n_estimators": [80],
            "max_depth": [None, 12],
            "min_samples_leaf": [2],
            "class_weight": ["balanced_subsample"],
        }
    else:
        grid = {
            "n_estimators": [300],
            "max_depth": [None, 12],
            "min_samples_leaf": [2, 5],
            "class_weight": ["balanced_subsample"],
        }
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def build_model(params: dict, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        **params,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
        bootstrap=True,
        oob_score=False,
    )


def choose_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    precision, recall, thresholds = precision_recall_curve(y, prob)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) != 0)
    idx = int(np.argmax(f1))
    detail = pd.DataFrame({"threshold": thresholds, "precision_cv": p, "recall_cv": r, "f1_cv": f1})
    return float(thresholds[idx]), detail


def ks_score(y: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, prob)
    return float(np.max(tpr - fpr))


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "ks": ks_score(y, prob),
        "true_bad_rate": float(np.mean(y)),
        "pred_bad_rate": float(np.mean(pred)),
    }


def cross_validate_params(x: np.ndarray, y: np.ndarray, cfg: Config) -> tuple[dict, pd.DataFrame, np.ndarray]:
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    rows = []
    best_params = None
    best_auc = -np.inf
    best_oof_prob = None

    for i, params in enumerate(param_grid(cfg.quick), start=1):
        print(f"参数组 {i}: {params}")
        oof_prob = np.zeros(len(y), dtype=float)
        fold_rows = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
            model = build_model(params, cfg.seed + fold)
            model.fit(x[tr_idx], y[tr_idx])
            prob = model.predict_proba(x[va_idx])[:, 1]
            oof_prob[va_idx] = prob
            fold_auc = roc_auc_score(y[va_idx], prob)
            fold_ap = average_precision_score(y[va_idx], prob)
            fold_rows.append({"param_id": i, "fold": fold, **params, "fold_auc": fold_auc, "fold_ap": fold_ap})
            print(f"  fold {fold}: AUC={fold_auc:.4f}, AP={fold_ap:.4f}")
        threshold, _ = choose_threshold(y, oof_prob)
        m = metrics(y, oof_prob, threshold)
        row = {"param_id": i, **params, **m}
        rows.append(row)
        rows.extend(fold_rows)
        print(f"  OOF: AUC={m['roc_auc']:.4f}, AP={m['average_precision']:.4f}, F1={m['f1']:.4f}, threshold={threshold:.6f}")
        if m["roc_auc"] > best_auc:
            best_auc = m["roc_auc"]
            best_params = params
            best_oof_prob = oof_prob.copy()

    if best_params is None or best_oof_prob is None:
        raise RuntimeError("交叉验证未得到有效参数")
    cv_results = pd.DataFrame(rows)
    return best_params, cv_results, best_oof_prob


def self_checks(train_df: pd.DataFrame, test_features: pd.DataFrame, cfg: Config, best_params: dict, threshold: float) -> list[str]:
    checks = []
    assert LABEL in train_df.columns
    checks.append("自查1：训练集含标签，标签仅用于训练/交叉验证")
    assert LABEL not in FEATURES and LABEL not in test_features.columns
    checks.append("自查2：特征列与测试预测特征均不含标签")
    assert len(FEATURES) == 14
    checks.append("自查3：使用 14 个指定特征")
    assert cfg.seed == 42
    checks.append("自查4：随机种子固定为 42")
    assert cfg.cv_folds >= 2
    checks.append(f"自查5：仅在 train.csv 内部做 {cfg.cv_folds} 折分层交叉验证")
    assert isinstance(best_params, dict) and best_params
    checks.append("自查6：最优参数来自训练集交叉验证")
    assert 0 <= threshold <= 1
    checks.append("自查7：分类阈值来自训练集 OOF 概率且合法")
    return checks


def save_outputs(cfg: Config, model, best_params, cv_results, threshold_df, train_cv_m, test_m, test_x, y_test, prob, pred, checks) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "random_forest_cv_optimized_model.joblib")
    config = asdict(cfg) | {"best_params": best_params, "features": FEATURES, "label": LABEL, "leakage_note": "参数和阈值只由 train.csv 内部交叉验证确定；test.csv 只最终评估一次"}
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    cv_results.to_csv(out / "cv_param_search_results.csv", index=False)
    threshold_df.to_csv(out / "cv_threshold_candidates.csv", index=False)
    pd.DataFrame([train_cv_m]).to_csv(out / "train_cv_oof_result_summary.csv", index=False)
    pd.DataFrame([test_m]).to_csv(out / "result_summary.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, pred), index=["实际_未逾期", "实际_逾期"], columns=["预测_未逾期", "预测_逾期"]).to_csv(out / "confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T.to_csv(out / "classification_report.csv")
    pred_df = test_x.copy()
    pred_df["predicted_probability"] = prob
    pred_df["predicted_label"] = pred
    pred_df.to_csv(out / "test_predictions.csv", index=False)
    eval_df = pred_df.copy()
    eval_df[LABEL] = y_test
    eval_df.to_csv(out / "test_evaluation_detail.csv", index=False)
    pd.DataFrame({"self_check": checks}).to_csv(out / "leakage_self_check.csv", index=False, encoding="utf-8-sig")


def run(cfg: Config) -> dict:
    print("=" * 70)
    print("随机森林 CV 优化版：train 内部交叉验证，test 最终评估")
    train_df, x_train, y_train = load_train(cfg)
    print(f"训练集：{len(y_train):,} 行，违约率 {y_train.mean():.4%}")

    print("=" * 70)
    print("[1] 在 train.csv 内部做分层交叉验证选参数")
    best_params, cv_results, best_oof_prob = cross_validate_params(x_train, y_train, cfg)
    threshold, threshold_df = choose_threshold(y_train, best_oof_prob)
    train_cv_m = metrics(y_train, best_oof_prob, threshold)
    print(f"最优参数：{best_params}")
    print(f"CV 阈值：{threshold:.6f}")
    print(f"CV OOF AUC={train_cv_m['roc_auc']:.4f}, AP={train_cv_m['average_precision']:.4f}, F1={train_cv_m['f1']:.4f}")

    print("=" * 70)
    print("[2] 用完整 train.csv 训练最终模型")
    final_model = build_model(best_params, cfg.seed)
    final_model.fit(x_train, y_train)

    print("=" * 70)
    print("[3] 读取测试特征并预测，预测阶段不读取标签")
    test_feature_df, x_test = load_test_features(cfg)
    prob = final_model.predict_proba(x_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    print("=" * 70)
    print("[4] 自查并读取测试标签评估")
    checks = self_checks(train_df, test_feature_df, cfg, best_params, threshold)
    for c in checks:
        print(c)
    y_test = load_test_label_after_prediction(cfg)
    test_m = metrics(y_test, prob, threshold)
    print(f"测试 AUC={test_m['roc_auc']:.4f}, AP={test_m['average_precision']:.4f}, KS={test_m['ks']:.4f}")
    print(f"Accuracy={test_m['accuracy']:.4f}, Precision={test_m['precision']:.4f}, Recall={test_m['recall']:.4f}, F1={test_m['f1']:.4f}")
    print("混淆矩阵：")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=4, zero_division=0))

    print("=" * 70)
    print("[5] 保存优化版输出")
    save_outputs(cfg, final_model, best_params, cv_results, threshold_df, train_cv_m, test_m, test_feature_df, y_test, prob, pred, checks)
    print(f"输出目录：{cfg.output_dir}")
    return {"best_params": best_params, "threshold": threshold, "train_cv": train_cv_m, "test": test_m}


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="随机森林分层交叉验证优化版")
    parser.add_argument("--train", default=TRAIN_PATH)
    parser.add_argument("--test", default=TEST_PATH)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="快速自检模式：小参数网格")
    args = parser.parse_args()
    return Config(args.train, args.test, args.output, SEED, args.cv_folds, args.quick)


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as e:
        print(f"程序运行失败：{e}", file=sys.stderr)
        raise
