"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(log_dir: str, level: str = "INFO", use_json: bool = True) -> None:
    """Configure root logger once."""

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    file_handler = logging.FileHandler(Path(log_dir) / "agent.log", encoding="utf-8")
    handlers.append(file_handler)

    formatter: logging.Formatter
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers = handlers


def get_logger(name: str) -> logging.Logger:
    """Get namespaced logger."""

    return logging.getLogger(name)


def log_event(logger: logging.Logger, message: str, **extra: Any) -> None:
    """Log event with attached metadata."""

    logger.info(message, extra={"extra": extra})
