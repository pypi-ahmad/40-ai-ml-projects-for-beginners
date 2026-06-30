"""Persistent SQLite memory for requests, results, and reports."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api_intel_agent.config import load_settings
from api_intel_agent.core.schemas import AnalyzeResponse, QueryHistoryItem, RunStatus


class SQLiteMemoryStore:
    def __init__(self, path: str | None = None) -> None:
        settings = load_settings()
        self.path = path or settings.memory.sqlite_path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS queries (
                    run_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS responses (
                    run_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS api_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def save_response(self, query: str, response: AnalyzeResponse) -> None:
        created_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO queries(run_id, query, status, created_at) VALUES (?, ?, ?, ?)",
                (response.run_id, query, response.status.value, created_at),
            )
            conn.execute(
                "INSERT OR REPLACE INTO responses(run_id, payload, created_at) VALUES (?, ?, ?)",
                (response.run_id, response.model_dump_json(), created_at),
            )

    def save_api_summary(self, run_id: str, provider: str, summary: str) -> None:
        created_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO api_summaries(run_id, provider, summary, created_at) VALUES (?, ?, ?, ?)",
                (run_id, provider, summary, created_at),
            )

    def history(self, limit: int = 20) -> list[QueryHistoryItem]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT run_id, query, status, created_at FROM queries ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            QueryHistoryItem(
                run_id=row[0],
                query=row[1],
                status=RunStatus(row[2]),
                created_at=datetime.fromisoformat(row[3]),
            )
            for row in rows
        ]

    def get_response(self, run_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT payload FROM responses WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return json.loads(row[0])
