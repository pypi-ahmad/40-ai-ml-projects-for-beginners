"""Structured logging setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logging(log_dir: Path) -> None:
    """Configure stdout + rotating file logger."""

    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", enqueue=True, backtrace=False, diagnose=False)
    logger.add(
        log_dir / "assistant.log",
        level="INFO",
        rotation="10 MB",
        retention="10 days",
        compression="gz",
        serialize=True,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
