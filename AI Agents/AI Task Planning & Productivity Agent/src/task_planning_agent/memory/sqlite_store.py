"""SQLite persistence layer for tasks, plans, preferences, and reflections."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from task_planning_agent.schemas import PlanSession, ReflectionRecord, Task, UserPreference


class SQLiteMemoryStore:
    """Operational persistent store using SQLite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    raw_input TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reflections (
                    plan_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def upsert_user(self, user_id: str, password_hash: str, role: str = "user") -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO users(user_id, password_hash, role, created_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    password_hash = excluded.password_hash,
                    role = excluded.role
                """,
                (user_id, password_hash, role, datetime.utcnow().isoformat()),
            )

    def get_user(self, user_id: str) -> dict[str, str] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row is not None else None

    def save_tasks(self, user_id: str, tasks: list[Task]) -> None:
        with self._conn() as conn:
            for task in tasks:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tasks(task_id, user_id, payload, created_at)
                    VALUES(?, ?, ?, ?)
                    """,
                    (task.id, user_id, task.model_dump_json(), datetime.utcnow().isoformat()),
                )

    def list_tasks(self, user_id: str) -> list[Task]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM tasks WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
            ).fetchall()
        return [Task.model_validate_json(row["payload"]) for row in rows]

    def save_plan(self, session: PlanSession) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO plans(plan_id, user_id, raw_input, payload, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    session.plan_id,
                    session.user_id,
                    session.raw_input,
                    session.model_dump_json(),
                    datetime.utcnow().isoformat(),
                ),
            )

    def list_plan_sessions(self, user_id: str, limit: int = 30) -> list[PlanSession]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM plans WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [PlanSession.model_validate_json(row["payload"]) for row in rows]

    def save_reflection(self, reflection: ReflectionRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reflections(plan_id, payload, created_at)
                VALUES(?, ?, ?)
                """,
                (reflection.plan_id, reflection.model_dump_json(), datetime.utcnow().isoformat()),
            )

    def upsert_preferences(self, preferences: UserPreference) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO preferences(user_id, payload, updated_at)
                VALUES(?, ?, ?)
                """,
                (
                    preferences.user_id,
                    preferences.model_dump_json(),
                    datetime.utcnow().isoformat(),
                ),
            )

    def get_preferences(self, user_id: str) -> UserPreference | None:
        with self._conn() as conn:
            row = conn.execute("SELECT payload FROM preferences WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return UserPreference.model_validate_json(row["payload"])

    def export_user_snapshot(self, user_id: str) -> dict[str, object]:
        return {
            "tasks": [task.model_dump() for task in self.list_tasks(user_id)],
            "plans": [session.model_dump() for session in self.list_plan_sessions(user_id)],
            "preferences": (
                self.get_preferences(user_id).model_dump() if self.get_preferences(user_id) else None
            ),
        }

    def search_tasks_text(self, user_id: str, query: str, limit: int = 20) -> list[Task]:
        query_low = query.lower()
        matches: list[Task] = []
        for task in self.list_tasks(user_id):
            blob = json.dumps(task.model_dump(), ensure_ascii=False).lower()
            if query_low in blob:
                matches.append(task)
            if len(matches) >= limit:
                break
        return matches
