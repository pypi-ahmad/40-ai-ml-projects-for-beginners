"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure structured logging sinks for console and file output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "framework.log"

    logger.remove()
    logger.add(
        sink=sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        log_file,
        level=level,
        serialize=True,
        rotation="50 MB",
        retention="10 days",
    )

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logger_opt = logger.bind(module=record.name)
            logger_opt.log(record.levelname, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=level)
