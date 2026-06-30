"""SQL formatting, normalization, and utility helpers."""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp


def clean_sql_text(text: str) -> str:
    """Extract SQL-only content from model output."""
    cleaned = text.strip()
    cleaned = cleaned.replace("```sql", "").replace("```", "").strip()
    cleaned = re.sub(r"(?im)^\s*sqlquery\s*:\s*", "", cleaned).strip()
    cleaned = re.sub(r"(?im)^\s*query\s*:\s*", "", cleaned).strip()

    # Keep from first SELECT/WITH onward when model adds preamble.
    match = re.search(r"(?is)\b(select|with)\b.*", cleaned)
    if match:
        cleaned = match.group(0).strip()

    return cleaned


def fix_common_sqlite_patterns(sql: str) -> str:
    """Apply deterministic safe rewrites for common SQLite generation mistakes."""
    fixed = sql
    # STRFTIME returns text in SQLite. Quote 4-digit years in equality predicates.
    fixed = re.sub(
        r"(STRFTIME\s*\(\s*'%Y'\s*,\s*[^)]+\)\s*=\s*)(\d{4})\b",
        r"\1'\2'",
        fixed,
        flags=re.IGNORECASE,
    )
    fixed = re.sub(
        r"(STRFTIME\s*\(\s*'%Y'\s*,\s*[^)]+\)\s+IN\s*\()([0-9,\s]+)(\))",
        lambda m: m.group(1)
        + ", ".join(f"'{x.strip()}'" for x in m.group(2).split(",") if x.strip())
        + m.group(3),
        fixed,
        flags=re.IGNORECASE,
    )
    fixed = re.sub(
        r"\bMONTH\s*\(\s*([^)]+)\s*\)",
        r"CAST(STRFTIME('%m', \1) AS INTEGER)",
        fixed,
        flags=re.IGNORECASE,
    )
    fixed = re.sub(
        r"\bYEAR\s*\(\s*([^)]+)\s*\)",
        r"CAST(STRFTIME('%Y', \1) AS INTEGER)",
        fixed,
        flags=re.IGNORECASE,
    )
    return fixed


def format_sql(sql: str) -> str:
    """Format SQL using sqlglot canonical pretty output."""
    try:
        expression = sqlglot.parse_one(sql, read="sqlite")
        return expression.sql(dialect="sqlite", pretty=True)
    except Exception:
        return sql.strip()


def normalize_sql(sql: str) -> str:
    """Create normalization for exact-match and regression tests."""
    try:
        expression = sqlglot.parse_one(sql, read="sqlite")
        return expression.sql(dialect="sqlite", normalize=True)
    except Exception:
        collapsed = re.sub(r"\s+", " ", sql.strip().lower())
        return collapsed


def extract_identifiers(sql: str) -> tuple[set[str], set[str]]:
    """Extract table and column identifiers from parsed SQL.

    Returns:
        tuple[set[str], set[str]]: Table names and column names.
    """
    tables: set[str] = set()
    columns: set[str] = set()
    try:
        expression = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return tables, columns

    for table in expression.find_all(exp.Table):
        if table.name:
            tables.add(table.name.lower())

    for col in expression.find_all(exp.Column):
        if col.name:
            columns.add(col.name.lower())

    return tables, columns
