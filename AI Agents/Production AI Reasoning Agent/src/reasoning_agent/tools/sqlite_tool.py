"""Optional SQLite query tool (read-only)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel

from reasoning_agent.tools.base import BaseTool


class SQLiteInput(BaseModel):
    database_path: str
    query: str
    limit: int = 100


class SQLiteOutput(BaseModel):
    columns: list[str]
    rows: list[list[str | int | float | None]]


class SQLiteQueryTool(BaseTool[SQLiteInput, SQLiteOutput]):
    name = "sqlite_query"
    description = "Executes read-only SQL query against sqlite database"
    input_model = SQLiteInput
    output_model = SQLiteOutput

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    async def run(self, payload: SQLiteInput) -> SQLiteOutput:
        sql = payload.query.strip().lower()
        if not sql.startswith("select"):
            raise ValueError("Only SELECT queries are allowed")
        path = (self.workspace_root / payload.database_path).resolve()
        if self.workspace_root not in path.parents and path != self.workspace_root:
            raise ValueError("Path escapes workspace")

        with sqlite3.connect(path) as conn:
            cursor = conn.execute(payload.query)
            columns = [col[0] for col in cursor.description or []]
            rows = [list(row) for row in cursor.fetchmany(payload.limit)]
        return SQLiteOutput(columns=columns, rows=rows)
