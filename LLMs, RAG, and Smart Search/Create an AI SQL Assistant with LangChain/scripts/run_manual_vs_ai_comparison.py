#!/usr/bin/env python3
"""Compare manual SQL answers against AI-generated SQL for selected prompts."""

from __future__ import annotations

import json
from pathlib import Path

from ai_sql_assistant.config import get_settings
from ai_sql_assistant.execution.executor import QueryExecutor
from ai_sql_assistant.logging_utils import configure_logging, logger
from ai_sql_assistant.pipeline.assistant import AISQLAssistant
from ai_sql_assistant.types import QueryRequest
from ai_sql_assistant.utils.sql_utils import normalize_sql


CASES = [
    {
        "question": "Show monthly net revenue for Europe in 2024.",
        "manual_sql": """
            SELECT strftime('%Y-%m', o.order_date) AS month,
                   ROUND(SUM(od.quantity * od.unit_price * (1 - od.discount)), 2) AS revenue
            FROM orders o
            JOIN order_details od ON o.order_id = od.order_id
            WHERE o.market = 'Europe' AND strftime('%Y', o.order_date) = '2024'
            GROUP BY strftime('%Y-%m', o.order_date)
            ORDER BY month
        """.strip(),
    },
    {
        "question": "Top 10 customers by revenue in Germany.",
        "manual_sql": """
            SELECT c.customer_id,
                   c.company_name,
                   ROUND(SUM(od.quantity * od.unit_price * (1 - od.discount)), 2) AS revenue
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN order_details od ON o.order_id = od.order_id
            WHERE c.country = 'Germany'
            GROUP BY c.customer_id, c.company_name
            ORDER BY revenue DESC
            LIMIT 10
        """.strip(),
    },
]


def main() -> None:
    configure_logging()
    settings = get_settings()
    assistant = AISQLAssistant(settings)
    executor = QueryExecutor(settings.database.active_db_path)

    rows = []
    try:
        for idx, case in enumerate(CASES, start=1):
            manual = executor.execute(case["manual_sql"])
            req = QueryRequest(
                question=case["question"],
                user_id="comparison",
                conversation_id=f"manual-vs-ai-{idx}",
                persona="Business Analyst",
            )
            ai = assistant.ask(req, approach="langchain", model=settings.models.generator_model)

            rows.append(
                {
                    "question": case["question"],
                    "manual_status": manual.status,
                    "manual_rows": manual.row_count,
                    "ai_status": ai.execution.status,
                    "ai_rows": ai.execution.row_count,
                    "sql_exact_match": normalize_sql(case["manual_sql"]) == normalize_sql(ai.execution.sql),
                    "manual_sql": case["manual_sql"],
                    "ai_sql": ai.execution.sql,
                    "manual_latency_ms": manual.execution_time_ms,
                    "ai_generation_latency_ms": ai.generation.latency_ms,
                    "ai_execution_latency_ms": ai.execution.execution_time_ms,
                }
            )

        out = Path("artifacts/reports/manual_vs_ai_comparison.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        logger.info("Manual vs AI report: {}", out)
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
