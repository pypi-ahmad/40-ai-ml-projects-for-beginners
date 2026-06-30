"""Logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Final

from loguru import logger

DEFAULT_FORMAT: Final[str] = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib + loguru logging.

    Args:
        level: Desired minimum log level.
    """
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), format=DEFAULT_FORMAT, backtrace=False, diagnose=False)
