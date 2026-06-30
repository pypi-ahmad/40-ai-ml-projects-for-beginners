"""Safe read-only SQL execution with metrics and plan estimation."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd
import sqlglot
from sqlglot import exp

from ai_sql_assistant.types import ExecutionResult


class QueryExecutor:
    """Execute validated SQL against SQLite in read-only mode."""

    def __init__(self, db_path: Path, max_rows: int = 5000, max_query_seconds: float = 30.0) -> None:
        self.db_path = db_path
        self.max_rows = max_rows
        self.max_query_seconds = max_query_seconds

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, sql: str) -> ExecutionResult:
        """Execute SQL and collect outputs, explain plan, and complexity score."""
        limited_sql = self._apply_limit(sql)

        try:
            explain_rows = self._explain(limited_sql)
            complexity = self._complexity_score(limited_sql, explain_rows)
        except Exception:
            explain_rows = []
            complexity = 0.0

        try:
            conn = self._connect()
            started = time.perf_counter()

            timeout_at = started + self.max_query_seconds

            def progress_handler() -> int:
                return 1 if time.perf_counter() > timeout_at else 0

            conn.set_progress_handler(progress_handler, 2_000)
            frame = pd.read_sql_query(limited_sql, conn)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            records = frame.to_dict(orient="records")
            columns = list(frame.columns)
            conn.close()

            return ExecutionResult(
                status="success",
                sql=limited_sql,
                columns=columns,
                rows=records,
                row_count=len(records),
                execution_time_ms=elapsed_ms,
                explain_plan=explain_rows,
                complexity_score=complexity,
            )
        except Exception as exc:
            return ExecutionResult(
                status="error",
                sql=limited_sql,
                error_message=str(exc),
                explain_plan=explain_rows,
                complexity_score=complexity,
            )

    def _explain(self, sql: str) -> list[dict[str, Any]]:
        conn = self._connect()
        rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def _apply_limit(self, sql: str) -> str:
        """Inject LIMIT when absent to protect UI/app memory."""
        try:
            expression = sqlglot.parse_one(sql, read="sqlite")
            if isinstance(expression, exp.Select) and expression.args.get("limit") is None:
                expression.set("limit", exp.Limit(expression=exp.Literal.number(self.max_rows)))
                return expression.sql(dialect="sqlite")
            if isinstance(expression, exp.With) and expression.this and isinstance(expression.this, exp.Select):
                if expression.this.args.get("limit") is None:
                    expression.this.set("limit", exp.Limit(expression=exp.Literal.number(self.max_rows)))
                return expression.sql(dialect="sqlite")
            return sql
        except Exception:
            lowered = sql.lower()
            if "limit" in lowered:
                return sql
            return f"SELECT * FROM ({sql}) AS limited_query LIMIT {self.max_rows}"

    def _complexity_score(self, sql: str, explain_rows: list[dict[str, Any]]) -> float:
        """Heuristic complexity score in [0, 1]."""
        score = 0.0
        text = sql.lower()

        score += min(text.count(" join ") * 0.08, 0.32)
        score += min(text.count(" group by ") * 0.08, 0.16)
        score += min(text.count(" over ") * 0.12, 0.24)
        score += min(text.count(" with ") * 0.08, 0.16)
        score += min(text.count("select") * 0.03, 0.12)

        explain_text = " ".join(str(item.get("detail", "")).lower() for item in explain_rows)
        if "scan" in explain_text:
            score += 0.1
        if "temp b-tree" in explain_text:
            score += 0.1

        return round(min(score, 1.0), 3)
