#!/usr/bin/env python3
"""End-to-end local pipeline runner for AI SQL assistant."""

from __future__ import annotations

import socket
import os
from pathlib import Path

from ai_sql_assistant.benchmarking.benchmark import BenchmarkRunner
from ai_sql_assistant.config import get_settings
from ai_sql_assistant.data.northwind import build_northwind_databases
from ai_sql_assistant.logging_utils import configure_logging, logger
from ai_sql_assistant.pipeline.assistant import AISQLAssistant
from ai_sql_assistant.schema.introspector import generate_erd, inspect_database, save_schema_report
from ai_sql_assistant.types import QueryRequest
from ai_sql_assistant.utils.runtime import ensure_python_version


def ollama_available(host: str = "127.0.0.1", port: int = 11434) -> bool:
    """Check if Ollama host is reachable."""
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        if sock is not None:
            sock.close()


def main() -> None:
    configure_logging()

    py_check = ensure_python_version()
    if not py_check.python_ok:
        raise RuntimeError(py_check.message)

    settings = get_settings()
    build = build_northwind_databases(
        raw_db_path=settings.database.raw_db_path,
        scaled_db_path=settings.database.scaled_db_path,
        scale_factor=8,
    )
    logger.info("Built datasets: source={}, raw_orders={}, scaled_orders={}", build.source, build.raw_orders, build.scaled_orders)

    report = inspect_database(settings.database.active_db_path)
    md_path, json_path = save_schema_report(report)
    erd_path = generate_erd(report)
    logger.info("Schema report generated: {}, {}, {}", md_path, json_path, erd_path)

    assistant = AISQLAssistant(settings)
    try:
        request = QueryRequest(
            question="Show monthly net revenue for Europe in 2024.",
            user_id="e2e",
            conversation_id="e2e-session",
            persona="Business Analyst",
        )

        if not ollama_available():
            logger.warning("Ollama unreachable on localhost:11434. Skipping LLM generation/evaluation steps.")
            return

        response = assistant.ask(request, approach="langchain", model=settings.models.generator_model)
        logger.info("Sample query status={} rows={}", response.execution.status, response.execution.row_count)

        cases_path = Path(os.getenv("AI_SQL_BENCHMARK_CASES", "benchmarks/benchmark_cases_sample1.json"))
        if cases_path.exists():
            runner = BenchmarkRunner(assistant, settings)
            try:
                cases = runner.load_cases(cases_path)
                run = runner.run(cases)
                logger.info("Benchmark runs={} cases={}", run.metrics.get("total_runs"), run.metrics.get("total_cases"))
            finally:
                runner.close()
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
