"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.logging import RichHandler


def configure_logging(
    log_path: str = "artifacts/logs/platform.jsonl", level: int = logging.INFO
) -> logging.Logger:
    """Configure rich console logger with JSONL file sink."""

    logger = logging.getLogger("langgraph_platform")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    logger.addHandler(RichHandler(rich_tracebacks=True))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    class JsonlFileHandler(logging.Handler):
        def __init__(self, file_path: str) -> None:
            super().__init__(level=level)
            self.file_path = Path(file_path)

        def emit(self, record: logging.LogRecord) -> None:
            payload: dict[str, Any] = {
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "time": self.formatter.formatTime(record),
            }
            if record.exc_info:
                payload["exc_info"] = self.format(record)
            with self.file_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload) + "\n")

    logger.addHandler(JsonlFileHandler(log_path))
    return logger
