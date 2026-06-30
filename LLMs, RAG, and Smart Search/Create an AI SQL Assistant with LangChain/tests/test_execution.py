from __future__ import annotations

from pathlib import Path

from ai_sql_assistant.execution.executor import QueryExecutor


def test_execute_success(sqlite_paths: dict[str, Path]) -> None:
    executor = QueryExecutor(sqlite_paths["scaled"], max_rows=100)
    result = executor.execute("SELECT customer_id FROM orders LIMIT 20")

    assert result.status == "success"
    assert result.row_count <= 20
    assert "customer_id" in result.columns


def test_executor_applies_limit(sqlite_paths: dict[str, Path]) -> None:
    executor = QueryExecutor(sqlite_paths["scaled"], max_rows=50)
    result = executor.execute("SELECT * FROM orders")

    assert result.status == "success"
    assert result.row_count <= 50
