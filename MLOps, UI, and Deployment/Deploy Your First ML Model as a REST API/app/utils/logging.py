"""Loguru configuration for console + rotating file logs."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from app.config import settings


def setup_logging():
    """Configure log sinks once and return bound logger."""
    logger.remove()

    if settings.log_format == "json":
        logger.add(
            sys.stderr,
            level=settings.log_level,
            serialize=True,
            colorize=False,
        )
    else:
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <7}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            colorize=True,
        )

    log_path = Path("logs") / "api.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        enqueue=False,
    )

    return logger
