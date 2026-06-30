"""Database introspection and schema report generation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from ai_sql_assistant.constants import DIAGRAMS_DIR, REPORTS_DIR
from ai_sql_assistant.logging_utils import logger

matplotlib.use("Agg")


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    """List user tables in SQLite database."""
    rows = conn.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [row[0] for row in rows]


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    """Get column metadata for a table."""
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "not_null": bool(row[3]),
            "default_value": row[4],
            "is_primary_key": bool(row[5]),
        }
        for row in rows
    ]


def get_foreign_keys(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    """Get foreign key metadata for a table."""
    rows = conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
    return [
        {
            "id": row[0],
            "seq": row[1],
            "ref_table": row[2],
            "from": row[3],
            "to": row[4],
            "on_update": row[5],
            "on_delete": row[6],
        }
        for row in rows
    ]


def get_indexes(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    """Get index metadata for a table."""
    rows = conn.execute(f"PRAGMA index_list('{table}')").fetchall()
    indexes: list[dict[str, Any]] = []
    for row in rows:
        idx_name = row[1]
        columns = conn.execute(f"PRAGMA index_info('{idx_name}')").fetchall()
        indexes.append(
            {
                "name": idx_name,
                "unique": bool(row[2]),
                "columns": [col[2] for col in columns],
            }
        )
    return indexes


def get_null_stats(conn: sqlite3.Connection, table: str, columns: list[str]) -> dict[str, int]:
    """Compute null counts for table columns."""
    stats: dict[str, int] = {}
    for column in columns:
        row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL").fetchone()
        stats[column] = int(row[0]) if row else 0
    return stats


def sample_rows(conn: sqlite3.Connection, table: str, limit: int = 5) -> list[dict[str, Any]]:
    """Get sample rows for a table."""
    frame = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {limit}", conn)
    return frame.to_dict(orient="records")


def inspect_database(db_path: Path) -> dict[str, Any]:
    """Generate full schema inspection report for SQLite DB."""
    conn = _connect_ro(db_path)
    report: dict[str, Any] = {
        "database": str(db_path),
        "tables": {},
        "relationships": [],
    }

    for table in list_tables(conn):
        columns_meta = get_table_columns(conn, table)
        columns = [col["name"] for col in columns_meta]

        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        row_count = int(row[0]) if row else 0

        foreign_keys = get_foreign_keys(conn, table)
        report["relationships"].extend(
            {
                "from_table": table,
                "from_column": fk["from"],
                "to_table": fk["ref_table"],
                "to_column": fk["to"],
            }
            for fk in foreign_keys
        )

        report["tables"][table] = {
            "row_count": row_count,
            "columns": columns_meta,
            "foreign_keys": foreign_keys,
            "indexes": get_indexes(conn, table),
            "null_stats": get_null_stats(conn, table, columns),
            "sample_rows": sample_rows(conn, table, limit=5),
        }

    conn.close()
    return report


def report_to_markdown(report: dict[str, Any]) -> str:
    """Render schema report as markdown."""
    lines = [
        f"# Schema Report: `{report['database']}`",
        "",
        "## Relationships",
    ]
    if not report["relationships"]:
        lines.append("No foreign keys detected.")
    else:
        for rel in report["relationships"]:
            lines.append(
                f"- `{rel['from_table']}.{rel['from_column']}` -> `{rel['to_table']}.{rel['to_column']}`"
            )

    lines.append("\n## Tables")
    for table, meta in report["tables"].items():
        lines.append(f"\n### `{table}`")
        lines.append(f"- Rows: **{meta['row_count']}**")

        lines.append("- Columns:")
        for col in meta["columns"]:
            pk = " PK" if col["is_primary_key"] else ""
            nn = " NOT NULL" if col["not_null"] else ""
            lines.append(f"  - `{col['name']}` `{col['type']}`{pk}{nn}")

        lines.append("- Indexes:")
        if meta["indexes"]:
            for idx in meta["indexes"]:
                lines.append(f"  - `{idx['name']}` ({', '.join(idx['columns'])})")
        else:
            lines.append("  - none")

        lines.append("- Null Stats:")
        for column, nulls in meta["null_stats"].items():
            lines.append(f"  - `{column}`: {nulls}")

    return "\n".join(lines)


def save_schema_report(
    report: dict[str, Any],
    markdown_path: Path | None = None,
    json_path: Path | None = None,
) -> tuple[Path, Path]:
    """Save schema report to markdown and JSON files."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    markdown_path = markdown_path or REPORTS_DIR / "schema_report.md"
    json_path = json_path or REPORTS_DIR / "schema_report.json"

    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    logger.info("Saved schema markdown report to {}", markdown_path)
    logger.info("Saved schema JSON report to {}", json_path)
    return markdown_path, json_path


def generate_erd(report: dict[str, Any], output_path: Path | None = None) -> Path:
    """Generate ERD graph as PNG."""
    output_path = output_path or DIAGRAMS_DIR / "schema_erd.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = nx.DiGraph()
    for table in report["tables"].keys():
        graph.add_node(table)

    for rel in report["relationships"]:
        graph.add_edge(rel["from_table"], rel["to_table"], label=rel["from_column"])

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(graph, seed=42, k=1.2)
    nx.draw_networkx_nodes(graph, pos, node_size=2500, node_color="#87CEEB", alpha=0.9)
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(graph, pos, arrowstyle="-|>", arrowsize=18, edge_color="#444444")
    edge_labels = {(u, v): data["label"] for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Northwind ERD (Auto-generated)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    logger.info("Saved ERD diagram to {}", output_path)
    return output_path
