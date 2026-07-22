# Data Analysis Handoff

Stage 3 provides a read-only Training interface for label-conditioned EDA. It validates the frozen manifest SHA before returning any rows and has no Validation/Test selector.

```python
from src.analysis.train_only import load_train_partition, transform_train_with_strategy

train = load_train_partition("configs/experiment.yaml")
raw_predictors = train.predictors
target = train.target

features, metadata = transform_train_with_strategy(
    "configs/experiment.yaml",
    strategy="missing_plus_flag",
    feature_set="A",
)
```

Use `train.row_id` and `train.feature_hash_v1` only for traceability and grouping. Do not add them to plots as model features. Generated tables and figures remain local under ignored output directories. Do not read Validation/Test labels, create a new split, or duplicate the 96/98 handling logic in EDA code.
