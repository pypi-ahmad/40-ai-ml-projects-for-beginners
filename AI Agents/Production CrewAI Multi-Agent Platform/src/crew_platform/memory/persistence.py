"""SQLite persistence store for plans, tasks, tool calls, memory, and reports."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import JSON, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """Base ORM model."""


class ConversationORM(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class PlanORM(Base):
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    objective: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class TaskORM(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str] = mapped_column(String(128), index=True)
    agent_role: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ToolCallORM(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ReportORM(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ApprovalORM(Base):
    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    reviewer: Mapped[str] = mapped_column(String(128))
    approved: Mapped[int] = mapped_column(Integer)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class PersistenceStore:
    """Repository facade for SQLite persistence."""

    def __init__(self, sqlite_path: str) -> None:
        engine_url = f"sqlite+pysqlite:///{sqlite_path}"
        self.engine = create_engine(engine_url, future=True)
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    def save_plan(self, run_id: str, session_id: str, objective: str, payload: dict[str, Any]) -> None:
        with self._session() as db:
            db.add(PlanORM(run_id=run_id, session_id=session_id, objective=objective, payload=payload))

    def save_task(
        self,
        run_id: str,
        task_id: str,
        agent_role: str,
        status: str,
        payload: dict[str, Any],
        error: str | None,
        attempt: int,
    ) -> None:
        with self._session() as db:
            db.add(
                TaskORM(
                    run_id=run_id,
                    task_id=task_id,
                    agent_role=agent_role,
                    status=status,
                    payload=payload,
                    error=error,
                    attempt=attempt,
                )
            )

    def save_report(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._session() as db:
            existing = db.scalar(select(ReportORM).where(ReportORM.run_id == run_id))
            if existing is None:
                db.add(ReportORM(run_id=run_id, payload=payload))
            else:
                existing.payload = payload

    def save_approval(self, run_id: str, reviewer: str, approved: bool, feedback: str | None) -> None:
        with self._session() as db:
            db.add(
                ApprovalORM(
                    run_id=run_id,
                    reviewer=reviewer,
                    approved=1 if approved else 0,
                    feedback=feedback,
                )
            )

    def save_conversation(self, run_id: str, session_id: str, role: str, content: str) -> None:
        with self._session() as db:
            db.add(
                ConversationORM(
                    run_id=run_id,
                    session_id=session_id,
                    role=role,
                    content=content,
                )
            )

    def save_tool_call(self, run_id: str, tool_name: str, payload: dict[str, Any], status: str) -> None:
        with self._session() as db:
            db.add(ToolCallORM(run_id=run_id, tool_name=tool_name, payload=payload, status=status))

    def fetch_tasks(self, run_id: str) -> list[dict[str, Any]]:
        with self._session() as db:
            rows = db.scalars(select(TaskORM).where(TaskORM.run_id == run_id)).all()
            return [
                {
                    "task_id": row.task_id,
                    "agent_role": row.agent_role,
                    "status": row.status,
                    "payload": row.payload,
                    "error": row.error,
                    "attempt": row.attempt,
                }
                for row in rows
            ]

    def fetch_report(self, run_id: str) -> dict[str, Any] | None:
        with self._session() as db:
            row = db.scalar(select(ReportORM).where(ReportORM.run_id == run_id))
            return row.payload if row else None

    def fetch_memory(self, limit: int = 50) -> dict[str, list[dict[str, Any]]]:
        with self._session() as db:
            conversations = db.scalars(
                select(ConversationORM).order_by(ConversationORM.created_at.desc()).limit(limit)
            ).all()
            tasks = db.scalars(select(TaskORM).order_by(TaskORM.created_at.desc()).limit(limit)).all()
            tool_calls = db.scalars(
                select(ToolCallORM).order_by(ToolCallORM.created_at.desc()).limit(limit)
            ).all()

        return {
            "conversations": [
                {
                    "run_id": row.run_id,
                    "session_id": row.session_id,
                    "role": row.role,
                    "content": row.content,
                    "created_at": row.created_at.isoformat(),
                }
                for row in conversations
            ],
            "tasks": [
                {
                    "run_id": row.run_id,
                    "task_id": row.task_id,
                    "agent_role": row.agent_role,
                    "status": row.status,
                    "payload": row.payload,
                    "error": row.error,
                    "attempt": row.attempt,
                    "created_at": row.created_at.isoformat(),
                }
                for row in tasks
            ],
            "tool_calls": [
                {
                    "run_id": row.run_id,
                    "tool_name": row.tool_name,
                    "payload": row.payload,
                    "status": row.status,
                    "created_at": row.created_at.isoformat(),
                }
                for row in tool_calls
            ],
        }
