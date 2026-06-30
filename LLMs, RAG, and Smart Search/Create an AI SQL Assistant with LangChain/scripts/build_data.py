#!/usr/bin/env python3
"""Build Northwind raw and scaled SQLite datasets."""

from __future__ import annotations

import json

from ai_sql_assistant.config import get_settings
from ai_sql_assistant.data.northwind import build_northwind_databases, sqlite_md5
from ai_sql_assistant.logging_utils import configure_logging, logger


def main() -> None:
    configure_logging()
    settings = get_settings()
    result = build_northwind_databases(
        raw_db_path=settings.database.raw_db_path,
        scaled_db_path=settings.database.scaled_db_path,
        scale_factor=8,
    )
    payload = {
        "source": result.source,
        "raw_orders": result.raw_orders,
        "scaled_orders": result.scaled_orders,
        "raw_md5": sqlite_md5(result.raw_db_path),
        "scaled_md5": sqlite_md5(result.scaled_db_path),
    }
    logger.info("Data build summary:\n{}", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
