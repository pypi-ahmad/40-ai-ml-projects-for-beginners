"""Logging utilities with optional Rich integration."""

from __future__ import annotations

import logging
from logging import Logger


def configure_logging(level: str = "INFO") -> Logger:
    """Configure application logger.

    Args:
        level: Logging level string.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger("peft_platform")
    if logger.handlers:
        return logger

    logger.setLevel(level.upper())

    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        formatter = logging.Formatter("%(message)s")
    except Exception:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
