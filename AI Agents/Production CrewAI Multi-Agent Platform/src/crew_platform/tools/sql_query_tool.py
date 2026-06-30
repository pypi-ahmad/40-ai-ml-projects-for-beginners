"""SQL query alias tool."""

from __future__ import annotations

from crew_platform.tools.sqlite_tool import SQLiteQueryTool


class SQLQueryTool(SQLiteQueryTool):
    name = "sql_query_tool"
    description = "Read-only SQL query tool"
