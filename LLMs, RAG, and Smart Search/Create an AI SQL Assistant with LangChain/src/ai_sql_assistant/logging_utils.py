"""Centralized logging configuration."""

from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru logger once for the whole app."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
    )


__all__ = ["configure_logging", "logger"]
