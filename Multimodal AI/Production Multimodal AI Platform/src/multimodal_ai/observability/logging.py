"""Structured logger setup."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging with rich handler."""

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    logger = logging.getLogger("multimodal_ai")
    logger.setLevel(level)
    return logger


def get_logger(name: str = "multimodal_ai") -> logging.Logger:
    """Get configured logger."""

    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
