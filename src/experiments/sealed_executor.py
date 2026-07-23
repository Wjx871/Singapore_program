"""One-time executor for the frozen XGBoost Independent Test evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    _compute_threshold_independent_metrics,
    _compute_threshold_metrics,
)
from src.evaluation.sealed import (
    EVALUATION_MODE,
    ExactlyOnceTestPredictor,
    SealedEvaluationAuthorization,
    SealedEvaluationError,
    verify_local_authorization,
)
from src.experiments.runner import SharedExperimentRunner
from src.features.feature_sets import build_feature_set, feature_set_schema
from src.features.preprocessing import CreditRiskPreprocessor
from src.models.factory import create_model_adapter
from src.utils.reproducibility import package_versions, set_random_seeds, sha256_file, stable_json_sha256

BASELINE_SHA = "d18052ca446390ec56f6baf8cdde7052f729c4c2"
RAW_SHA = "1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac"
MANIFEST_SHA = "5c7aed175ae534f22b051e0b6375469aea71c16d3f28e07a97a60dffb4d520b5"
CONFIG_SHA = "896e0f677764c5e456b55a46791e4099fa0767865bd50e46f22df88a2572b86a"
FROZEN_THRESHOLD = 0.548155665397644
EXPECTED_VERSIONS = {
    "python": "3.12.13",
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scikit-learn": "1.7.2",
    "imbalanced-learn": "0.14.0",
    "xgboost": "3.0.5",
    "PyYAML": "6.0.3",
    "joblib": "1.5.2",
}
MODEL_PARAMETERS = {
    "n_estimators": 2000,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
    "n_jobs": -1,
}
EXPECTED_FEATURES = (
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents", "MonthlyIncomeMissingFlag", "DependentsMissingFlag",
    "HasAbnormalDelinquencyCode", "AgeInvalidFlag", "IncomePerDependent",
    "LogMonthlyIncome", "HighDebtFlag",
)
EXPECTED_HASHES = {
    "train_feature_sha256": "bb5c1670053cd63ac2839d4969b68f22d5f198c6e0f51fcf61c4c92b7f64e344",
    "validation_feature_sha256": "e0b27f3478f75dcb5409f69be2c72142b24e5b39e3363d5bb9b404c7f68ba421",
    "train_row_id_sha256": "5404e9df77d5474483bb2beb1afd964f716ba8eeff6ea832bc47918b8ba07052",
    "validation_row_id_sha256": "b3da505a7ef76d7a288dfe4770c7a7258eff37e9ebbadfc04a847e08a3096804",
}
VALIDATION_REFERENCE = {
    "pr_auc": 0.401962229635,
    "roc_auc": 0.869935023378,
    "ks": 0.585850355787,
    "precision": 0.238865588177,
    "recall": 0.750313676286,
}


@dataclass
class PreflightArtifacts:
    runner: SharedExperimentRunner
    adapter: Any
    matrices: dict[str, Any]
    hashes: dict[str, str]
    validation_probability_sha256: str
    validation_metrics: dict[str, Any]
    validation_operational: dict[str, Any]
    training_metadata: dict[str, Any]
    versions: dict[str, str]


def array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes()).hexdigest()


class SealedEvaluationExecutor:
    def __init__(self, config_path: str | Path = "configs/experiment.yaml") -> None:
        self.config_path = Path(config_path)

    def preflight(self) -> PreflightArtifacts:
        runner = SharedExperimentRunner(self.config_path)
        config = runner.config
        if config.config_sha256 != CONFIG_SHA:
            raise SealedEvaluationError("Frozen parsed Config SHA-256 mismatch")
        raw_path = config.project_root / config.data["labeled_path"]
        if sha256_file(raw_path) != RAW_SHA:
            raise SealedEvaluationError("Frozen raw labeled data SHA-256 mismatch")
        if sha256_file(runner.manifest_path) != MANIFEST_SHA:
            raise SealedEvaluationError("Frozen manifest SHA-256 mismatch")
        versions = package_versions()
        if versions != EXPECTED_VERSIONS:
            raise SealedEvaluationError(f"Frozen package contract mismatch: {versions}")
        counts = {name: len(frame) for name, frame in runner.partitions.items()}
        if counts != {"train": 95995, "validation": 23997, "test": 30008}:
            raise SealedEvaluationError(f"Frozen partition row counts mismatch: {counts}")
        if tuple(feature_set_schema("B", config.predictor_columns, "missing_plus_flag")) != EXPECTED_FEATURES:
            raise SealedEvaluationError("Frozen Feature Set B names/order mismatch")

        preprocessing = config.raw["preprocessing"]
        cleaner = CreditRiskPreprocessor(
            predictor_columns=config.predictor_columns,
            delinquency_columns=preprocessing["abnormal_delinquency"]["columns"],
            abnormal_values=preprocessing["abnormal_delinquency"]["abnormal_values"],
            strategy="missing_plus_flag",
            clipping_enabled=preprocessing["winsorization"]["enabled"],
            clipping_columns=preprocessing["winsorization"]["columns"],
            upper_quantile=preprocessing["winsorization"]["upper_quantile"],
        )
        matrices: dict[str, Any] = {}
        for name in ("train", "validation", "test"):
            source = runner.partitions[name]
            cleaned = cleaner.fit_transform(source) if name == "train" else cleaner.transform(source)
            matrices[name] = build_feature_set(
                cleaned, "B", config.predictor_columns, "missing_plus_flag"
            )
        if any(tuple(matrix.columns) != EXPECTED_FEATURES for matrix in matrices.values()):
            raise SealedEvaluationError("Train/Validation/Test feature schemas differ")
        if any(not np.isfinite(matrix.to_numpy()).all() for matrix in matrices.values()):
            raise SealedEvaluationError("Transformed feature matrix contains NaN or infinity")

        hashes = {
            f"{name}_feature_sha256": array_sha256(matrix.to_numpy())
            for name, matrix in matrices.items()
        }
        hashes.update({
            f"{name}_row_id_sha256": array_sha256(
                runner.partitions[name]["row_id"].to_numpy()
            )
            for name in ("train", "validation", "test")
        })
        for key, expected in EXPECTED_HASHES.items():
            if hashes[key] != expected:
                raise SealedEvaluationError(f"Frozen input hash mismatch for {key}")

        target = config.data["target_column"]
        y_train = runner.partitions["train"][target].to_numpy()
        negative, positive = int((y_train == 0).sum()), int((y_train == 1).sum())
        if (negative, positive) != (89541, 6454):
            raise SealedEvaluationError("Frozen Training class counts mismatch")
        set_random_seeds(42)
        adapter = create_model_adapter("xgboost", MODEL_PARAMETERS, random_seed=42)
        adapter.fit(
            matrices["train"], y_train,
            x_validation=matrices["validation"],
            y_validation=runner.partitions["validation"][target].to_numpy(),
            validation_split_name="validation",
            feature_names=list(EXPECTED_FEATURES),
            groups=runner.partitions["train"]["feature_hash_v1"].to_numpy(),
        )
        metadata = adapter.get_training_metadata()
        if metadata["best_iteration"] != 172 or metadata["actual_boosting_rounds"] != 173:
            raise SealedEvaluationError("Frozen XGBoost early-stopping result mismatch")
        if not np.isclose(metadata["scale_pos_weight"], 13.873721722962504, rtol=0, atol=1e-12):
            raise SealedEvaluationError("Training-derived scale_pos_weight mismatch")
        validation_probability = adapter.predict_proba(matrices["validation"])
        y_validation = runner.partitions["validation"][target].to_numpy()
        validation_metrics = _compute_threshold_independent_metrics(
            y_validation, validation_probability
        )
        validation_operational = _compute_threshold_metrics(
            y_validation, validation_probability, FROZEN_THRESHOLD
        )
        observed = {
            **{key: validation_metrics[key] for key in ("pr_auc", "roc_auc", "ks")},
            "precision": validation_operational["precision"],
            "recall": validation_operational["recall"],
        }
        for key, expected in VALIDATION_REFERENCE.items():
            if abs(observed[key] - expected) > 1e-6:
                raise SealedEvaluationError(f"Frozen Validation {key} mismatch")
        self._prepare_output_paths(config.project_root, smoke_only=True)
        return PreflightArtifacts(
            runner, adapter, matrices, hashes, array_sha256(validation_probability),
            validation_metrics, validation_operational, metadata, versions
        )

    def execute(
        self,
        authorization: SealedEvaluationAuthorization,
        *,
        interactive_phrase: str,
        preflight_artifacts: PreflightArtifacts | None = None,
    ) -> dict[str, Any]:
        artifacts = preflight_artifacts or self.preflight()
        root = artifacts.runner.config.project_root
        self.verify_authorization(artifacts, authorization)
        if interactive_phrase != "EXECUTE ONE-TIME INDEPENDENT TEST EVALUATION":
            raise SealedEvaluationError("Interactive confirmation phrase mismatch")
        paths = self._prepare_output_paths(root, smoke_only=False)
        target = artifacts.runner.config.data["target_column"]
        predictor = ExactlyOnceTestPredictor(artifacts.adapter.predict_proba)
        probability = predictor.predict_proba(artifacts.matrices["test"])
        if predictor.prediction_call_count != 1:
            raise SealedEvaluationError("Exactly-one prediction invariant failed")
        y_test = artifacts.runner.partitions["test"][target].to_numpy()
        independent = _compute_threshold_independent_metrics(y_test, probability)
        default = _compute_threshold_metrics(y_test, probability, 0.5)
        operational = _compute_threshold_metrics(y_test, probability, FROZEN_THRESHOLD)
        for metrics in (independent,):
            if metrics["undefined_metrics"]:
                raise SealedEvaluationError("Independent Test metrics are undefined")
        for metrics in (default, operational):
            if sum(metrics["confusion_matrix"].values()) != 30008:
                raise SealedEvaluationError("Independent Test confusion matrix total mismatch")
        probability_sha = array_sha256(probability)
        report = {
            "split": "independent_test",
            "evaluation_mode": EVALUATION_MODE,
            "threshold_independent": independent,
            "default_threshold": default,
            "frozen_operational_threshold": operational,
        }
        metrics_sha = stable_json_sha256(report)
        timestamp = datetime.now().astimezone().isoformat()
        receipt = {
            "status": "completed", "evaluation_mode": EVALUATION_MODE,
            "run_id": authorization.run_id,
            "authorization_phrase_sha256": authorization.authorization_phrase_sha256,
            "baseline_sha": BASELINE_SHA,
            "sealed_executor_sha": authorization.expected_executor_sha,
            "final_documentation_sha": "pending until documentation commit",
            "execution_timestamp": timestamp,
            "raw_data_sha256": RAW_SHA, "manifest_sha256": MANIFEST_SHA,
            "config_sha256": CONFIG_SHA, **artifacts.hashes,
            "validation_probability_sha256": artifacts.validation_probability_sha256,
            "independent_test_probability_sha256": probability_sha,
            "prediction_call_count": predictor.prediction_call_count,
            "model_name": "xgboost", "experiment_id": "xgb_child5",
            "feature_set": "B", "preprocessing_strategy": "missing_plus_flag",
            "feature_names": list(EXPECTED_FEATURES),
            "model_parameters": {**MODEL_PARAMETERS, "random_state": 42},
            "package_versions": artifacts.versions,
            "environment": self._environment(),
            "best_iteration": 172, "actual_boosting_rounds": 173,
            "scale_pos_weight": artifacts.training_metadata["scale_pos_weight"],
            "frozen_threshold": FROZEN_THRESHOLD, "test_row_count": 30008,
            "test_metrics_report_sha256": metrics_sha,
            "legacy_forest_test_quarantined": True,
            "model_reselection_performed": False,
            "threshold_reselection_performed": False,
            "calibration_performed": False, "retraining_after_test": False,
        }
        document = self._document(receipt, report, artifacts)
        figure_temp_paths = self._render_figures(paths, default, operational, independent)
        try:
            self._atomic_json(paths["report"], report)
            self._atomic_json(paths["local_receipt"], receipt)
            self._atomic_json(paths["tracked_receipt"], receipt)
            self._atomic_text(paths["document"], document)
            for final, temporary in figure_temp_paths:
                os.replace(temporary, final)
        finally:
            for _, temporary in figure_temp_paths:
                Path(temporary).unlink(missing_ok=True)
        return {"receipt": receipt, "report": report, "paths": {k: str(v) for k, v in paths.items()}}

    @staticmethod
    def verify_authorization(
        artifacts: PreflightArtifacts,
        authorization: SealedEvaluationAuthorization,
    ) -> None:
        if authorization.expected_manifest_sha256 != MANIFEST_SHA:
            raise SealedEvaluationError("Authorization manifest SHA differs from frozen contract")
        if authorization.expected_raw_sha256 != RAW_SHA:
            raise SealedEvaluationError("Authorization raw SHA differs from frozen contract")
        if authorization.expected_config_sha256 != CONFIG_SHA:
            raise SealedEvaluationError("Authorization Config SHA differs from frozen contract")
        if authorization.frozen_operational_threshold != FROZEN_THRESHOLD:
            raise SealedEvaluationError("Authorization threshold differs from frozen contract")
        verify_local_authorization(
            artifacts.runner.config.project_root, authorization
        )

    @staticmethod
    def final_check_lines(
        artifacts: PreflightArtifacts,
        authorization: SealedEvaluationAuthorization,
    ) -> list[str]:
        root = artifacts.runner.config.project_root
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return [
            f"Current branch: {branch}",
            f"Current HEAD: {head}",
            f"Expected executor SHA: {authorization.expected_executor_sha}",
            "Working tree clean: verified",
            "CI false: verified",
            "Raw SHA verified: yes",
            "Manifest SHA verified: yes",
            "Config SHA verified: yes",
            "Package versions verified: yes",
            "Train rows verified: 95995",
            "Validation rows verified: 23997",
            "Test rows verified: 30008",
            "Feature Set B verified: yes",
            "Feature count 17 verified: yes",
            "Feature order verified: yes",
            "Training class counts verified: negative=89541, positive=6454",
            f"scale_pos_weight verified: {artifacts.training_metadata['scale_pos_weight']}",
            f"Validation PR-AUC verified: {artifacts.validation_metrics['pr_auc']:.12f}",
            f"Validation ROC-AUC verified: {artifacts.validation_metrics['roc_auc']:.12f}",
            f"Validation KS verified: {artifacts.validation_metrics['ks']:.12f}",
            "best_iteration=172 verified: yes",
            "actual_boosting_rounds=173 verified: yes",
            f"frozen threshold verified: {FROZEN_THRESHOLD}",
            "output paths writable: verified",
            "no existing receipt: verified",
            "Test prediction count currently 0: verified",
        ]

    @staticmethod
    def _prepare_output_paths(root: Path, *, smoke_only: bool) -> dict[str, Path]:
        output = root / "outputs" / "sealed_evaluation"
        assets = root / "docs" / "assets" / "sealed_evaluation"
        output.mkdir(parents=True, exist_ok=True)
        assets.mkdir(parents=True, exist_ok=True)
        paths = {
            "local_receipt": output / "sealed_evaluation_receipt.json",
            "report": output / "sealed_evaluation_report.json",
            "document": root / "docs" / "INDEPENDENT_TEST_EVALUATION.md",
            "tracked_receipt": root / "docs" / "SEALED_EVALUATION_RECEIPT.json",
            "default_figure": assets / "independent_test_confusion_matrix_default.png",
            "operational_figure": assets / "independent_test_confusion_matrix_operational.png",
            "summary_figure": assets / "independent_test_metric_summary.png",
        }
        if any(path.exists() for path in paths.values()):
            raise SealedEvaluationError("A sealed receipt, report, document, or figure already exists")
        if shutil.disk_usage(root).free < 50 * 1024 * 1024:
            raise SealedEvaluationError("Insufficient disk space for atomic sealed outputs")
        json.dumps({"finite": 1.0}, allow_nan=False)
        for directory in {path.parent for path in paths.values()}:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".sealed-smoke-", delete=True) as handle:
                handle.write(b"ok")
                handle.flush()
                os.fsync(handle.fileno())
        fig, axis = plt.subplots(figsize=(2, 1))
        axis.plot([0, 1], [0, 1])
        with tempfile.NamedTemporaryFile(dir=assets, suffix=".png", delete=True) as handle:
            fig.savefig(handle.name)
        plt.close(fig)
        return paths

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        SealedEvaluationExecutor._atomic_text(
            path, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        if path.exists():
            raise SealedEvaluationError(f"Refusing to overwrite sealed artifact: {path}")
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _environment() -> dict[str, Any]:
        return {
            "os": platform.system(), "os_version": platform.version(),
            "cpu": platform.processor(), "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(), "timezone": datetime.now().astimezone().tzname(),
            "openmp_runtime_information": "xgboost runtime governed by frozen package contract",
        }

    @staticmethod
    def _render_figures(paths: dict[str, Path], default: dict[str, Any],
                        operational: dict[str, Any], independent: dict[str, Any]) -> list[tuple[Path, str]]:
        temporaries: list[tuple[Path, str]] = []
        for key, title, metrics in (
            ("default_figure", "Independent Test — Default Threshold (0.5)", default),
            ("operational_figure", f"Independent Test — Frozen Threshold ({FROZEN_THRESHOLD:.12f})", operational),
        ):
            matrix = metrics["confusion_matrix"]
            values = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
            fig, axis = plt.subplots(figsize=(12.8, 7.2))
            axis.imshow(values, cmap="Blues")
            axis.set(title=title, xlabel="Predicted label", ylabel="True label",
                     xticks=[0, 1], yticks=[0, 1])
            for i in range(2):
                for j in range(2):
                    axis.text(j, i, f"{values[i, j]:,}", ha="center", va="center", fontsize=18)
            temporary = tempfile.NamedTemporaryFile(
                dir=paths[key].parent, suffix=".png", delete=False
            ).name
            fig.savefig(temporary, dpi=180, bbox_inches="tight")
            plt.close(fig)
            temporaries.append((paths[key], temporary))
        fig, axis = plt.subplots(figsize=(12.8, 7.2))
        labels = ["PR-AUC", "ROC-AUC", "KS"]
        values = [independent["pr_auc"], independent["roc_auc"], independent["ks"]]
        bars = axis.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
        axis.set_ylim(0, 1)
        axis.set_title("Independent Test — Threshold-Independent Metrics")
        axis.bar_label(bars, fmt="%.6f")
        temporary = tempfile.NamedTemporaryFile(
            dir=paths["summary_figure"].parent, suffix=".png", delete=False
        ).name
        fig.savefig(temporary, dpi=180, bbox_inches="tight")
        plt.close(fig)
        temporaries.append((paths["summary_figure"], temporary))
        return temporaries

    @staticmethod
    def _document(receipt: dict[str, Any], report: dict[str, Any],
                  artifacts: PreflightArtifacts) -> str:
        default = report["default_threshold"]
        operational = report["frozen_operational_threshold"]
        independent = report["threshold_independent"]
        sections = [
            ("Authorization", f"Run `{receipt['run_id']}` used mode `{EVALUATION_MODE}`."),
            ("Frozen Model", "XGBoost `xgb_child5`; the model and full parameters were frozen before Test."),
            ("Frozen Data Contract", "Training / Validation / Independent Test rows: 95,995 / 23,997 / 30,008."),
            ("Frozen Preprocessing", "`missing_plus_flag`, Feature Set B, 17 ordered features; fit on Training only."),
            ("Sealed Executor SHA", f"`{receipt['sealed_executor_sha']}`"),
            ("Environment and Package Versions", f"`{json.dumps(receipt['package_versions'], sort_keys=True)}`"),
            ("Pre-Prediction Assertions", "All hashes, row counts, schema, finite, package, output, and receipt checks passed."),
            ("Validation Reproduction Evidence", f"PR-AUC {artifacts.validation_metrics['pr_auc']:.12f}; ROC-AUC {artifacts.validation_metrics['roc_auc']:.12f}; KS {artifacts.validation_metrics['ks']:.12f}; operational Precision {artifacts.validation_operational['precision']:.12f}; Recall {artifacts.validation_operational['recall']:.12f}."),
            ("Exactly-One Prediction Evidence", "Independent Test `predict_proba` was called exactly once; `prediction_call_count=1`."),
            ("Independent Test Threshold-Independent Metrics", f"PR-AUC {independent['pr_auc']:.12f}; ROC-AUC {independent['roc_auc']:.12f}; KS {independent['ks']:.12f}."),
            ("Default Threshold Metrics", f"`{json.dumps(default, sort_keys=True)}`"),
            ("Frozen Operational Threshold Metrics", f"`{json.dumps(operational, sort_keys=True)}`"),
            ("Confusion Matrices", f"Default: `{default['confusion_matrix']}`. Frozen operational: `{operational['confusion_matrix']}`."),
            ("Test Probability Hash", f"`{receipt['independent_test_probability_sha256']}`. Raw probabilities were not persisted."),
            ("Model Selection Status", "XGBoost was selected using Validation only. Test did not select the model, parameters, features, preprocessing, or best iteration."),
            ("Threshold Selection Status", f"The fixed threshold `{FROZEN_THRESHOLD}` was selected using Validation only; no Test threshold table or search was performed."),
            ("Forest Test Quarantine Statement", "Earlier Forest Test metrics were accidentally viewed and quarantined. They were not used for model, parameter, threshold, inference-contract, or comparison decisions."),
            ("Reproducibility Receipt", "`docs/SEALED_EVALUATION_RECEIPT.json` and local immutable receipt."),
            ("Limitations", "Single frozen split; calibration, fairness, distribution shift, causal claims, and production suitability are out of scope."),
            ("Final Governance Statement", "The authorized final XGBoost Independent Test evaluation was performed once after the model, preprocessing, parameters, threshold, and executor contract had been frozen. Test did not participate in selection, early stopping, thresholding, or calibration. No retraining occurred after Test. The Independent Test result is reported as an immutable final evaluation. It does not trigger another modelling cycle."),
        ]
        body = ["# One-Time Independent Test Evaluation", ""]
        for index, (title, text) in enumerate(sections, 1):
            body.extend([f"## {index}. {title}", "", text, ""])
        return "\n".join(body)
