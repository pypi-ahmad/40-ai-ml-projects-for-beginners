#!/usr/bin/env python3
"""Run sample end-to-end queries for smoke validation."""

from __future__ import annotations

from ai_sql_assistant.config import get_settings
from ai_sql_assistant.logging_utils import configure_logging, logger
from ai_sql_assistant.pipeline.assistant import AISQLAssistant
from ai_sql_assistant.types import QueryRequest


QUESTIONS = [
    "Show monthly net revenue for Europe in 2024.",
    "Top 10 customers by revenue in Germany.",
    "Which product categories have highest revenue for Enterprise segment in 2023?",
]


def main() -> None:
    configure_logging()
    settings = get_settings()
    assistant = AISQLAssistant(settings)
    try:
        for idx, question in enumerate(QUESTIONS, start=1):
            req = QueryRequest(
                question=question,
                user_id="demo",
                conversation_id="demo-conv",
                persona="Business Analyst",
            )
            response = assistant.ask(req, approach="langchain", model=settings.models.generator_model)
            logger.info("[{}] status={} rows={}", idx, response.execution.status, response.execution.row_count)
            logger.info("{}", response.execution.sql)
            logger.info("{}", "-" * 40)
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
