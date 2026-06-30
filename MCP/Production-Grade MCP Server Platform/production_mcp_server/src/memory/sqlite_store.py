from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    response_payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class PromptRun(Base):
    __tablename__ = "prompt_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt_name: Mapped[str] = mapped_column(String(128), index=True)
    variables: Mapped[dict[str, Any]] = mapped_column(JSON)
    rendered_prompt: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class ResponseRecord(Base):
    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    response_type: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class MetadataRecord(Base):
    __tablename__ = "metadata"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(128), index=True)
    action: Mapped[str] = mapped_column(String(128), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(128), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class CacheEntry(Base):
    __tablename__ = "cache_entries"

    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    payload: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class MetricRow(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric_name: Mapped[str] = mapped_column(String(128), index=True)
    metric_value: Mapped[float] = mapped_column()
    labels: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))


class SQLiteStore:
    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{sqlite_path}", future=True)
        Base.metadata.create_all(self.engine)

    def _session(self) -> Session:
        return Session(self.engine)

    def log_conversation(self, session_id: str, role: str, content: str) -> None:
        with self._session() as session:
            session.add(Conversation(session_id=session_id, role=role, content=content))
            session.commit()

    def fetch_recent_conversations(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._session() as session:
            stmt = (
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(Conversation.id.desc())
                .limit(limit)
            )
            rows = list(session.scalars(stmt))
        rows.reverse()
        return [
            {
                "id": row.id,
                "session_id": row.session_id,
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        latency_ms: int,
    ) -> None:
        with self._session() as session:
            session.add(
                ToolCall(
                    session_id=session_id,
                    tool_name=tool_name,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    latency_ms=latency_ms,
                )
            )
            session.commit()

    def log_prompt(self, prompt_name: str, variables: dict[str, Any], rendered_prompt: str) -> None:
        with self._session() as session:
            session.add(PromptRun(prompt_name=prompt_name, variables=variables, rendered_prompt=rendered_prompt))
            session.commit()

    def log_response(self, session_id: str, response_type: str, payload: dict[str, Any]) -> None:
        with self._session() as session:
            session.add(ResponseRecord(session_id=session_id, response_type=response_type, payload=payload))
            session.commit()

    def log_audit(self, actor: str, action: str, payload: dict[str, Any]) -> None:
        with self._session() as session:
            session.add(AuditLog(actor=actor, action=action, payload=payload))
            session.commit()

    def log_usage(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._session() as session:
            session.add(UsageEvent(event_type=event_type, payload=payload))
            session.commit()

    def log_metric(self, metric_name: str, metric_value: float, labels: dict[str, Any]) -> None:
        with self._session() as session:
            session.add(MetricRow(metric_name=metric_name, metric_value=metric_value, labels=labels))
            session.commit()

    def recent_metrics(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._session() as session:
            stmt = select(MetricRow).order_by(MetricRow.id.desc()).limit(limit)
            rows = list(session.scalars(stmt))
        rows.reverse()
        return [
            {
                "metric_name": row.metric_name,
                "metric_value": row.metric_value,
                "labels": row.labels,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def cache_set(self, key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        expires_at = datetime.now(UTC).timestamp() + ttl_seconds
        expiry = datetime.fromtimestamp(expires_at, UTC).replace(tzinfo=None)
        raw_payload = json.dumps(payload)

        with self._session() as session:
            row = session.get(CacheEntry, key)
            if row is None:
                row = CacheEntry(key=key, payload=raw_payload, expires_at=expiry)
                session.add(row)
            else:
                row.payload = raw_payload
                row.expires_at = expiry
            session.commit()

    def cache_get(self, key: str) -> dict[str, Any] | None:
        with self._session() as session:
            row = session.get(CacheEntry, key)
            if row is None:
                return None
            if row.expires_at <= datetime.now(UTC).replace(tzinfo=None):
                session.delete(row)
                session.commit()
                return None
            return json.loads(row.payload)

    def cache_cleanup(self) -> int:
        now = datetime.now(UTC).replace(tzinfo=None)
        removed = 0
        with self._session() as session:
            stmt = select(CacheEntry).where(CacheEntry.expires_at <= now)
            rows = list(session.scalars(stmt))
            for row in rows:
                session.delete(row)
                removed += 1
            session.commit()
        return removed

    def top_tool_usage(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._session() as session:
            stmt = select(ToolCall).order_by(ToolCall.id.desc()).limit(limit)
            rows = list(session.scalars(stmt))
        return [
            {
                "tool_name": row.tool_name,
                "latency_ms": row.latency_ms,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
