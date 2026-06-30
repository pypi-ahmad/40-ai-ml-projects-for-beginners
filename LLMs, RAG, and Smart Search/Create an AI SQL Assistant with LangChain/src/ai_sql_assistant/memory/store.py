"""Persistent query history, favorites, and conversation memory."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class AppStateStore:
    """SQLite-backed app state for history, favorites, and conversation memory."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                sql TEXT NOT NULL,
                approach TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                row_count INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                question TEXT NOT NULL,
                sql TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                sql TEXT NOT NULL,
                explanation TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_query_history_conv ON query_history(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_turns_conv ON conversation_turns(conversation_id, created_at);
            """
        )
        conn.commit()
        conn.close()

    def add_history(
        self,
        conversation_id: str,
        user_id: str,
        question: str,
        sql: str,
        approach: str,
        model: str,
        status: str,
        latency_ms: float,
        row_count: int,
    ) -> None:
        """Persist query execution record."""
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO query_history(
                conversation_id, user_id, question, sql, approach, model,
                status, latency_ms, row_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                user_id,
                question,
                sql,
                approach,
                model,
                status,
                latency_ms,
                row_count,
            ),
        )
        conn.commit()
        conn.close()

    def add_turn(
        self,
        conversation_id: str,
        user_id: str,
        question: str,
        sql: str,
        explanation: str,
    ) -> None:
        """Persist conversation turn for context-aware follow-ups."""
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO conversation_turns(conversation_id, user_id, question, sql, explanation)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, user_id, question, sql, explanation),
        )
        conn.commit()
        conn.close()

    def conversation_context(self, conversation_id: str, limit: int = 4) -> str:
        """Render recent conversation context text for prompting."""
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT question, sql
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        conn.close()

        if not rows:
            return "No prior conversation context."

        lines = ["Recent context:"]
        for question, sql in reversed(rows):
            lines.append(f"Q: {question}")
            lines.append(f"SQL: {sql}")
        return "\n".join(lines)

    def last_sql(self, conversation_id: str) -> str | None:
        """Return most recent SQL for conversation, if present."""
        conn = self._connect()
        row = conn.execute(
            """
            SELECT sql
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        conn.close()
        return str(row[0]) if row else None

    def add_favorite(self, label: str, question: str, sql: str) -> None:
        """Save favorite query."""
        conn = self._connect()
        conn.execute(
            "INSERT INTO favorites(label, question, sql) VALUES (?, ?, ?)",
            (label, question, sql),
        )
        conn.commit()
        conn.close()

    def list_favorites(self) -> pd.DataFrame:
        """Get favorites as dataframe."""
        conn = self._connect()
        frame = pd.read_sql_query("SELECT * FROM favorites ORDER BY id DESC", conn)
        conn.close()
        return frame

    def history(self, limit: int = 50) -> pd.DataFrame:
        """Get query history for dashboard."""
        conn = self._connect()
        frame = pd.read_sql_query(
            "SELECT * FROM query_history ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
        conn.close()
        return frame

    def dashboard_stats(self) -> dict[str, Any]:
        """Aggregate key metrics for dashboard panels."""
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]
        success = conn.execute("SELECT COUNT(*) FROM query_history WHERE status = 'success'").fetchone()[0]
        avg_latency = conn.execute("SELECT COALESCE(AVG(latency_ms), 0) FROM query_history").fetchone()[0]
        sql_rows = conn.execute("SELECT sql FROM query_history").fetchall()
        conn.close()

        table_counts: dict[str, int] = {}
        for (sql_text,) in sql_rows:
            for table in re.findall(r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql_text.lower()):
                table_counts[table] = table_counts.get(table, 0) + 1
        top_tables = [key for key, _ in sorted(table_counts.items(), key=lambda item: item[1], reverse=True)[:5]]

        return {
            "total_queries": int(total),
            "success_rate": float(success / total) if total else 0.0,
            "avg_latency_ms": float(avg_latency),
            "top_tables": top_tables,
        }
