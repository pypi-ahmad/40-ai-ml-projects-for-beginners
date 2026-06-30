"""Structured logging setup with Rich-friendly console rendering."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "trace_id"):
            payload["trace_id"] = record.trace_id
        if hasattr(record, "context"):
            payload["context"] = record.context
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger once."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    if root.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get named logger."""
    return logging.getLogger(name)
