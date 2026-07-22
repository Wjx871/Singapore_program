"""Consistent file and console logging for Stage 2 scripts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path


def configure_experiment_logging(root: Path, stage: int) -> tuple[logging.Logger, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / "logs" / f"stage{stage}_{timestamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"stage{stage}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger, path


def configure_stage2_logging(root: Path) -> tuple[logging.Logger, Path]:
    return configure_experiment_logging(root, stage=2)
