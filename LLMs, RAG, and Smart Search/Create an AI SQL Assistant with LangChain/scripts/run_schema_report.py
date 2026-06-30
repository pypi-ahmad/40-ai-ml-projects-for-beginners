#!/usr/bin/env python3
"""Generate schema reports and ERD from active SQLite DB."""

from __future__ import annotations

from ai_sql_assistant.config import get_settings
from ai_sql_assistant.logging_utils import configure_logging, logger
from ai_sql_assistant.schema.introspector import generate_erd, inspect_database, save_schema_report


def main() -> None:
    configure_logging()
    settings = get_settings()
    report = inspect_database(settings.database.active_db_path)
    md_path, json_path = save_schema_report(report)
    erd_path = generate_erd(report)
    logger.info("schema markdown: {}", md_path)
    logger.info("schema json: {}", json_path)
    logger.info("erd: {}", erd_path)


if __name__ == "__main__":
    main()
