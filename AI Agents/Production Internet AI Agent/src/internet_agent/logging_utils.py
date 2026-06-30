"""Structured logging configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from internet_agent.config import Settings


def _serialize(record: dict[str, Any]) -> str:
    payload = {
        "time": record["time"].isoformat(),
        "level": record["level"].name,
        "name": record["name"],
        "message": record["message"],
        "extra": record["extra"],
    }
    return json.dumps(payload, ensure_ascii=True)


def configure_logging(settings: Settings) -> None:
    """Configure console and file logging sinks."""

    logger.remove()
    level = settings.logging.level.upper()

    logger.add(sys.stderr, level=level, serialize=settings.logging.json_logs, backtrace=False)

    log_path = Path(settings.logging.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if settings.logging.json_logs:
        logger.add(log_path, level=level, format="{message}", serialize=True)
    else:
        logger.add(log_path, level=level, format="{time} {level} {message}")


def get_logger(name: str):
    """Return named logger binder for consistent structured fields."""

    return logger.bind(component=name)
