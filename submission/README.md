# 信用违约预测 — 四模型对比实验

## 项目简介

基于 Kaggle "Give Me Some Credit" 数据集，四人分别采用 XGBoost、LightGBM、随机森林、逻辑回归四种模型，在统一划分的训练集（80,000 条）和测试集（20,000 条）上进行对比实验。

## 文件结构

```
submission/
├── README.md
├── preprocess_v2.py          # 数据预处理（共用，运行一次）
├── dataset/
│   ├── train.csv             # 训练集 80,000 条
│   └── test.csv              # 测试集 20,000 条
├── XGBoost/
│   └── train.py
├── LightGBM/
│   └── train.py
├── RandomForest/
│   └── train.py
└── LogisticRegression/
    └── train.py
```

## 环境依赖

```
Python >= 3.8
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
```

安装：

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
```

## 运行顺序

### 第一步：数据预处理

```bash
python preprocess_v2.py
```

生成 `dataset/train.csv` 和 `dataset/test.csv`。如数据集已存在可跳过。

### 第二步：各模型独立训练

四个模型互不依赖，可任意顺序运行：

```bash
python XGBoost/train.py
python LightGBM/train.py
python RandomForest/train.py
python LogisticRegression/train.py
```

每个脚本从 `dataset/` 读取数据，在各自目录下输出结果。

## 模型信息

| 模型 | 负责人 | 训练脚本 | 不平衡处理 | AUC | KS |
|------|--------|----------|-----------|------|------|
| XGBoost | 杨家懿 | XGBoost/train.py | scale_pos_weight | 0.8673 | 0.5827 |
| LightGBM | 陈博涵 | LightGBM/train.py | is_unbalance + Optuna | 0.8658 | 0.5834 |
| 随机森林 | 邓文涛 | RandomForest/train.py | class_weight | 0.8570 | 0.5627 |
| 逻辑回归 | 谢育民 | LogisticRegression/train.py | class_weight | 0.8436 | 0.5230 |

## 注意事项

- 四个模型使用完全相同的训练集和测试集，严禁各自重新划分
- 随机种子统一为 random_state=42
- 评估指标均以测试集上的 AUC 和 KS 值为准
- 团队已对 SMOTE 过采样进行跨模型验证，三个树模型均为负优化，最终采用原始数据分布方案
