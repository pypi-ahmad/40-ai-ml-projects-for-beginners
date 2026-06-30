"""Schema context rendering for prompt grounding."""

from __future__ import annotations

from typing import Any


def schema_context_text(report: dict[str, Any], max_tables: int = 30) -> str:
    """Render compact schema text for LLM prompts."""
    chunks: list[str] = []
    for idx, (table, meta) in enumerate(report.get("tables", {}).items()):
        if idx >= max_tables:
            break
        column_defs = ", ".join(f"{col['name']} {col['type']}" for col in meta["columns"])
        chunks.append(f"TABLE {table} ({column_defs})")

    if report.get("relationships"):
        chunks.append("RELATIONSHIPS:")
        chunks.extend(
            f"{r['from_table']}.{r['from_column']} -> {r['to_table']}.{r['to_column']}"
            for r in report["relationships"]
        )

    return "\n".join(chunks)
