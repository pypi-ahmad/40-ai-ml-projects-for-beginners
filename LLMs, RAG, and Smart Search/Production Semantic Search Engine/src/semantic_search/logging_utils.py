"""Structured logging setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from semantic_search.config import AppConfig


LOGGER_NAME = "semantic_search"


def configure_logging(config: AppConfig) -> None:
    """Configure console and file logging sinks."""
    logs_dir = Path(config.paths["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / config.logging.file_name

    logger.remove()
    logger.add(
        sys.stdout,
        level=config.logging.level,
        serialize=config.logging.json_logs,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_file,
        level=config.logging.level,
        serialize=config.logging.json_logs,
        enqueue=True,
        rotation="20 MB",
        retention=5,
        backtrace=False,
        diagnose=False,
    )


def get_logger():
    """Return shared logger instance."""
    return logger
