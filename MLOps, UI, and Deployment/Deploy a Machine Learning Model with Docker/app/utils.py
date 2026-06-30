"""Structured JSON logging utilities used by API and scripts."""

import json
import logging
import os
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Render logs as JSON objects for machine-friendly ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log.update(record.extra)
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)
        return json.dumps(log, default=str)


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Return singleton logger configured with JSON output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        level_name = os.getenv("LOG_LEVEL", "").upper()
        resolved_level = logging.getLevelName(level_name) if level_name else level
        logger.setLevel(resolved_level if isinstance(resolved_level, int) else level)
    return logger


logger = get_logger(__name__)
