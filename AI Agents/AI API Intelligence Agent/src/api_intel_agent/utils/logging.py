"""Structured logging setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logging(path: str = "artifacts/logs/agent.log"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(path, serialize=True, enqueue=False, backtrace=False, diagnose=False)
    logger.add(sys.stderr, level="INFO")
    return logger
