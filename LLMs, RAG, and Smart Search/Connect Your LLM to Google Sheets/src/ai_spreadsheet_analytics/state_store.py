"""SQLite state and artifact store."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import orjson


class SQLiteStateStore:
    """Lightweight persistence for cache metadata, chat, and benchmark/eval artifacts."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sheet_cache_manifest (
                    cache_key TEXT PRIMARY KEY,
                    spreadsheet_id TEXT NOT NULL,
                    worksheet_title TEXT NOT NULL,
                    row_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS judge_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    model_evaluated TEXT NOT NULL,
                    judge_model TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def upsert_cache_manifest(
        self,
        cache_key: str,
        spreadsheet_id: str,
        worksheet_title: str,
        row_hash: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sheet_cache_manifest (cache_key, spreadsheet_id, worksheet_title, row_hash, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    row_hash=excluded.row_hash,
                    updated_at=excluded.updated_at
                """,
                (cache_key, spreadsheet_id, worksheet_title, row_hash, now),
            )

    def get_cache_manifest(self, cache_key: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sheet_cache_manifest WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            return dict(row) if row else None

    def add_chat_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        evidence: dict[str, Any],
        model: str,
        latency_ms: float,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO query_history
                (session_id, question, answer, evidence_json, model, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    question,
                    answer,
                    orjson.dumps(evidence).decode("utf-8"),
                    model,
                    latency_ms,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def get_chat_history(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT session_id, question, answer, evidence_json, model, latency_ms, created_at
                FROM query_history
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        history: list[dict[str, Any]] = []
        for row in reversed(rows):
            payload = dict(row)
            payload["evidence"] = orjson.loads(payload.pop("evidence_json"))
            history.append(payload)
        return history

    def add_benchmark_result(self, run_id: str, case_id: str, model: str, metrics: dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_results (run_id, case_id, model, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    case_id,
                    model,
                    orjson.dumps(metrics).decode("utf-8"),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def add_judge_report(
        self,
        run_id: str,
        model_evaluated: str,
        judge_model: str,
        report: dict[str, Any],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO judge_reports (run_id, model_evaluated, judge_model, report_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_evaluated,
                    judge_model,
                    orjson.dumps(report).decode("utf-8"),
                    datetime.now(UTC).isoformat(),
                ),
            )
