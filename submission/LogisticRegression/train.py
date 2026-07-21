from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="金融风控二分类逻辑回归训练与评估")
    parser.add_argument("--train", type=Path, default=project_root / "dataset" / "train.csv")
    parser.add_argument("--test", type=Path, default=project_root / "dataset" / "test.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs")
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--sample-for-tuning", type=int, default=50000, help="调参最多使用的训练样本数，0 表示使用全量")
    return parser.parse_args()


def ensure_binary_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise ValueError(f"未找到目标列: {target}")
    values = sorted(df[target].dropna().unique().tolist())
    if values != [0, 1]:
        raise ValueError(f"目标列必须是 0/1 二分类，当前取值: {values}")


def data_profile(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> dict:
    features = [c for c in train_df.columns if c != target]
    profile = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "target": target,
        "feature_count": len(features),
        "features": features,
        "train_missing_total": int(train_df.isna().sum().sum()),
        "test_missing_total": int(test_df.isna().sum().sum()),
        "train_positive_rate": float(train_df[target].mean()),
        "test_positive_rate": float(test_df[target].mean()) if target in test_df else None,
        "train_target_counts": {str(k): int(v) for k, v in train_df[target].value_counts().sort_index().items()},
        "test_target_counts": {str(k): int(v) for k, v in test_df[target].value_counts().sort_index().items()}
        if target in test_df
        else {},
    }
    return profile


def build_pipeline(feature_names: list[str]) -> Pipeline:
    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_preprocess, feature_names)],
        remainder="drop",
    )
    classifier = LogisticRegression(
        solver="saga",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])


def evaluate(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold_by_f1(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if thresholds.size == 0:
        return 0.5, float(f1_values.max(initial=0.0))
    best_idx = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def save_plots(y_true: pd.Series, y_prob: np.ndarray, metrics: dict, output_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}", color="#1f77b4", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP = {metrics['average_precision']:.4f}", color="#d62728", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=160)
    plt.close()

    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#111111", fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def save_coefficients(model: Pipeline, feature_names: list[str], output_dir: Path) -> pd.DataFrame:
    coefs = model.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(output_dir / "coefficients.csv", index=False, encoding="utf-8-sig")
    return coef_df


def write_report(
    output_dir: Path,
    profile: dict,
    best_params: dict,
    cv_score: float,
    val_metrics: dict,
    test_metrics_default: dict,
    test_metrics_best_threshold: dict,
    threshold_f1: float,
    coef_df: pd.DataFrame,
) -> None:
    top_positive = coef_df[coef_df["coefficient"] > 0].sort_values("coefficient", ascending=False).head(8)
    top_negative = coef_df[coef_df["coefficient"] < 0].sort_values("coefficient", ascending=True).head(8)

    def fmt_metric(metrics: dict) -> str:
        return (
            f"Accuracy={metrics['accuracy']:.4f}, Balanced Accuracy={metrics['balanced_accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, AP={metrics['average_precision']:.4f}"
        )

    report = [
        "# 金融风控分类逻辑回归实践报告",
        "",
        "## 1. 任务说明",
        "",
        "本项目选择金融风控分类方向，使用逻辑回归预测客户在两年内发生严重逾期的概率，目标变量为 `SeriousDlqin2yrs`。",
        "任务 PDF 中未抽取到更细的代码或报告评分条款，因此本项目按机器学习实践的完整流程完成：数据理解、预处理、模型训练、参数调优、测试评估、结果可视化与模型解释。",
        "",
        "## 2. 数据概况",
        "",
        f"- 训练集规模：{profile['train_shape'][0]} 行，{profile['train_shape'][1]} 列。",
        f"- 测试集规模：{profile['test_shape'][0]} 行，{profile['test_shape'][1]} 列。",
        f"- 特征数量：{profile['feature_count']}。",
        f"- 训练集坏样本比例：{profile['train_positive_rate']:.4%}，测试集坏样本比例：{profile['test_positive_rate']:.4%}。",
        f"- 训练集缺失值总数：{profile['train_missing_total']}，测试集缺失值总数：{profile['test_missing_total']}。",
        "",
        "## 3. 方法流程",
        "",
        "1. 读取 `train.csv` 和 `test.csv`，分离特征与标签。",
        "2. 使用中位数填补与标准化构建预处理流水线。",
        "3. 在训练集内划分验证集，并使用 Stratified K-Fold 交叉验证进行参数调优。",
        "4. 逻辑回归调参范围包含正则化强度 `C`、正则项形式 `l1_ratio=0/1`、类别权重 `None/balanced`。",
        "5. 使用 ROC-AUC 作为主调参指标，同时报告 Accuracy、Precision、Recall、F1、AP 等指标。",
        "6. 基于验证集寻找 F1 最优阈值，并在测试集同时报告默认阈值 0.5 与调优阈值结果。",
        "",
        "## 4. 参数调优结果",
        "",
        f"- 最优参数：`{best_params}`。",
        f"- 交叉验证最佳 ROC-AUC：{cv_score:.4f}。",
        f"- 验证集结果：{fmt_metric(val_metrics)}。",
        "",
        "## 5. 测试集结果",
        "",
        f"- 默认阈值 0.5：{fmt_metric(test_metrics_default)}。",
        f"- F1 最优阈值 {test_metrics_best_threshold['threshold']:.4f}：{fmt_metric(test_metrics_best_threshold)}。",
        f"- 验证集阈值搜索得到的最佳 F1：{threshold_f1:.4f}。",
        "",
        "## 6. 模型解释",
        "",
        "逻辑回归系数为标准化特征上的影响方向和强度。正系数越大，表示违约风险越高；负系数越大，表示风险越低。",
        "",
        "### 风险提升最明显的特征",
        "",
        top_positive[["feature", "coefficient", "odds_ratio"]].to_markdown(index=False),
        "",
        "### 风险降低最明显的特征",
        "",
        top_negative[["feature", "coefficient", "odds_ratio"]].to_markdown(index=False),
        "",
        "## 7. 输出文件",
        "",
        "- `best_logistic_regression.joblib`: 训练好的最优逻辑回归模型。",
        "- `metrics.json`: 数据概况、最优参数和评估指标。",
        "- `classification_report.txt`: 测试集分类报告。",
        "- `cv_results.csv`: 网格搜索交叉验证结果。",
        "- `coefficients.csv`: 模型系数与 odds ratio。",
        "- `test_predictions.csv`: 测试集预测概率与类别。",
        "- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`: 评估图表。",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    ensure_binary_target(train_df, args.target)
    ensure_binary_target(test_df, args.target)

    feature_names = [c for c in train_df.columns if c != args.target]
    X = train_df[feature_names]
    y = train_df[args.target].astype(int)
    X_test = test_df[feature_names]
    y_test = test_df[args.target].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    tune_X, tune_y = X_train, y_train
    if args.sample_for_tuning and args.sample_for_tuning < len(X_train):
        tune_X, _, tune_y, _ = train_test_split(
            X_train,
            y_train,
            train_size=args.sample_for_tuning,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

    pipeline = build_pipeline(feature_names)
    param_grid = {
        "model__l1_ratio": [0.0, 1.0],
        "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "model__class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(tune_X, tune_y)

    best_model: Pipeline = grid.best_estimator_
    best_model.fit(X_train, y_train)

    val_prob = best_model.predict_proba(X_val)[:, 1]
    best_threshold, threshold_f1 = best_threshold_by_f1(y_val, val_prob)
    val_metrics = evaluate(y_val, val_prob, threshold=best_threshold)

    final_model = build_pipeline(feature_names)
    final_model.set_params(**grid.best_params_)
    final_model.fit(X, y)

    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics_default = evaluate(y_test, test_prob, threshold=0.5)
    test_metrics_best_threshold = evaluate(y_test, test_prob, threshold=best_threshold)
    profile = data_profile(train_df, test_df, args.target)

    save_plots(y_test, test_prob, test_metrics_best_threshold, output_dir)
    coef_df = save_coefficients(final_model, feature_names, output_dir)
    joblib.dump(final_model, output_dir / "best_logistic_regression.joblib")

    predictions = test_df.copy()
    predictions["predicted_probability"] = test_prob
    predictions["predicted_label_threshold_0_5"] = (test_prob >= 0.5).astype(int)
    predictions["predicted_label_best_threshold"] = (test_prob >= best_threshold).astype(int)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    cv_results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(output_dir / "cv_results.csv", index=False, encoding="utf-8-sig")

    test_pred_best = (test_prob >= best_threshold).astype(int)
    report_text = classification_report(y_test, test_pred_best, digits=4, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    metrics = {
        "profile": profile,
        "best_params": grid.best_params_,
        "best_cv_roc_auc": float(grid.best_score_),
        "validation_metrics_best_threshold": val_metrics,
        "test_metrics_threshold_0_5": test_metrics_default,
        "test_metrics_best_threshold": test_metrics_best_threshold,
        "best_threshold_from_validation": float(best_threshold),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        output_dir=output_dir,
        profile=profile,
        best_params=grid.best_params_,
        cv_score=float(grid.best_score_),
        val_metrics=val_metrics,
        test_metrics_default=test_metrics_default,
        test_metrics_best_threshold=test_metrics_best_threshold,
        threshold_f1=threshold_f1,
        coef_df=coef_df,
    )

    print("训练完成")
    print(f"最优参数: {grid.best_params_}")
    print(f"CV ROC-AUC: {grid.best_score_:.4f}")
    print(f"验证集最佳阈值: {best_threshold:.4f}")
    print(f"测试集 ROC-AUC: {test_metrics_best_threshold['roc_auc']:.4f}")
    print(f"测试集 F1: {test_metrics_best_threshold['f1']:.4f}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
