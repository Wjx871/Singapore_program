"""Safe analysis interfaces that cannot expose sealed partitions."""

from src.analysis.train_only import TrainPartition, load_train_partition, transform_train_with_strategy

__all__ = ["TrainPartition", "load_train_partition", "transform_train_with_strategy"]
