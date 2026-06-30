"""Schema summarization and business-friendly descriptions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ai_sql_assistant.constants import REPORTS_DIR


def schema_signature(schema_report: dict[str, Any]) -> str:
    """Compute signature for schema cache invalidation."""
    payload = json.dumps(schema_report, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_schema(schema_report: dict[str, Any]) -> dict[str, Any]:
    """Create business-friendly summaries for each table and relationship."""
    summaries: dict[str, Any] = {
        "tables": {},
        "relationships": [],
    }
    for table, meta in schema_report["tables"].items():
        columns = meta["columns"]
        dimension_guess = [c["name"] for c in columns if c["name"].endswith("_id")]

        summaries["tables"][table] = {
            "description": f"Table `{table}` stores {meta['row_count']} records for business analytics.",
            "column_descriptions": {
                col["name"]: f"{col['name']} ({col['type']}) used in `{table}` analysis."
                for col in columns
            },
            "business_notes": [
                "Use for trend analysis with date filters." if any("date" in c["name"] for c in columns) else "",
                "Likely dimension keys: " + ", ".join(dimension_guess) if dimension_guess else "",
            ],
        }

    for rel in schema_report["relationships"]:
        summaries["relationships"].append(
            {
                "summary": (
                    f"`{rel['from_table']}` links to `{rel['to_table']}` via "
                    f"`{rel['from_column']}` -> `{rel['to_column']}`"
                )
            }
        )

    return summaries


def load_cached_summary(cache_path: Path) -> dict[str, Any] | None:
    """Load schema summary cache if file exists."""
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_schema_summary(summary: dict[str, Any], signature: str) -> Path:
    """Persist schema summary cache keyed by schema signature."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"schema_summary_{signature[:12]}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
