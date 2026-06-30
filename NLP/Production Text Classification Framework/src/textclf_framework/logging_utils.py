"""Logging setup utilities."""

from __future__ import annotations

import sys
from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure global structured logging.

    Args:
        level: Minimum log level.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        enqueue=True,
    )
