"""Logging setup helpers."""

from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure application logger.

    Args:
        level: Minimum log level.
    """

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    )
