"""RF/BRF handoff contracts; no estimators are implemented in Stage 3.5."""

RF_DEFAULT_CONTRACT = {
    "feature_set": "B",
    "preprocessing_strategy": "missing_plus_flag",
    "requires_scaled_features": False,
    "random_state": 42,
    "class_weight": None,
    "smote_allowed": False,
    "probability_output_required": True,
}

BRF_DEFAULT_CONTRACT = {
    "feature_set": "B",
    "preprocessing_strategy": "missing_plus_flag",
    "requires_scaled_features": False,
    "random_state": 42,
    "implementation": "imblearn.ensemble.BalancedRandomForestClassifier",
    "imbalance_strategy": "internal_balanced_sampling",
    "smote_allowed": False,
    "probability_output_required": True,
}
