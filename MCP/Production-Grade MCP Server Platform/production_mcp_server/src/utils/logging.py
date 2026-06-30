from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config.settings import LoggingConfig


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(config: LoggingConfig) -> None:
    log_path = Path(config.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")

    if config.json_logs:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)
