"""Repository for SQLite-backed long-term memory and cache."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, select

from internet_agent.config import Settings
from internet_agent.memory.db import Database
from internet_agent.memory.models import (
    CacheEntry,
    ConversationMessage,
    ReportRecord,
    RetrievedDocument,
    SearchRecord,
    SummaryRecord,
    ToolHistory,
    VisitedURL,
)


class MemoryRepository:
    """Persistence service for conversations, retrieval artifacts, and cache."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings)
        self.db.create_all()

    @staticmethod
    def _utcnow_naive() -> datetime:
        return datetime.now(UTC).replace(tzinfo=None)

    @staticmethod
    def make_cache_key(namespace: str, payload: str) -> str:
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self.db.session() as session:
            session.add(ConversationMessage(session_id=session_id, role=role, content=content))

    def get_messages(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.db.session() as session:
            stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.session_id == session_id)
                .order_by(ConversationMessage.created_at.desc())
                .limit(limit)
            )
            rows = list(session.scalars(stmt))
        rows.reverse()
        return [
            {
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def add_search_record(self, session_id: str, query: str, provider: str, results: dict[str, Any]) -> None:
        with self.db.session() as session:
            session.add(
                SearchRecord(session_id=session_id, query=query, provider=provider, results_json=results)
            )

    def add_visited_url(self, session_id: str, url: str, title: str, status_code: int) -> None:
        with self.db.session() as session:
            session.add(
                VisitedURL(session_id=session_id, url=url, title=title, status_code=status_code)
            )

    def add_document(
        self,
        session_id: str,
        url: str,
        title: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        with self.db.session() as session:
            session.add(
                RetrievedDocument(
                    session_id=session_id,
                    url=url,
                    title=title,
                    content=content,
                    metadata_json=metadata,
                )
            )

    def get_recent_documents(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.db.session() as session:
            stmt = (
                select(RetrievedDocument)
                .where(RetrievedDocument.session_id == session_id)
                .order_by(RetrievedDocument.retrieved_at.desc())
                .limit(limit)
            )
            rows = list(session.scalars(stmt))
        return [
            {
                "url": row.url,
                "title": row.title,
                "content": row.content,
                "metadata": row.metadata_json,
                "retrieved_at": row.retrieved_at.isoformat(),
            }
            for row in rows
        ]

    def add_summary(self, session_id: str, query: str, summary: str, confidence: float) -> None:
        with self.db.session() as session:
            session.add(
                SummaryRecord(
                    session_id=session_id,
                    query=query,
                    summary=summary,
                    confidence=confidence,
                )
            )

    def add_tool_history(
        self,
        session_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any],
        status: str,
        latency_ms: float,
    ) -> None:
        with self.db.session() as session:
            session.add(
                ToolHistory(
                    session_id=session_id,
                    tool_name=tool_name,
                    tool_input_json=tool_input,
                    tool_output_json=tool_output,
                    status=status,
                    latency_ms=latency_ms,
                )
            )

    def get_tool_history(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.db.session() as session:
            stmt = (
                select(ToolHistory)
                .where(ToolHistory.session_id == session_id)
                .order_by(ToolHistory.created_at.desc())
                .limit(limit)
            )
            rows = list(session.scalars(stmt))
        rows.reverse()
        return [
            {
                "tool_name": row.tool_name,
                "input": row.tool_input_json,
                "output": row.tool_output_json,
                "status": row.status,
                "latency_ms": row.latency_ms,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def add_report(self, session_id: str, fmt: str, path: str, payload: dict[str, Any]) -> None:
        with self.db.session() as session:
            session.add(
                ReportRecord(session_id=session_id, format=fmt, path=path, payload_json=payload)
            )

    def get_reports(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.db.session() as session:
            stmt = (
                select(ReportRecord)
                .where(ReportRecord.session_id == session_id)
                .order_by(ReportRecord.created_at.desc())
                .limit(limit)
            )
            rows = list(session.scalars(stmt))
        return [
            {
                "format": row.format,
                "path": row.path,
                "payload": row.payload_json,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def cache_get(self, key: str) -> dict[str, Any] | None:
        with self.db.session() as session:
            row = session.get(CacheEntry, key)
            if row is None:
                return None
            if row.expires_at <= self._utcnow_naive():
                session.delete(row)
                return None
            return row.value_json

    def cache_set(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int,
        namespace: str = "default",
    ) -> None:
        expires_at = self._utcnow_naive() + timedelta(seconds=ttl_seconds)
        with self.db.session() as session:
            existing = session.get(CacheEntry, key)
            if existing:
                existing.value_json = value
                existing.expires_at = expires_at
                existing.namespace = namespace
            else:
                session.add(
                    CacheEntry(
                        key=key,
                        value_json=value,
                        expires_at=expires_at,
                        namespace=namespace,
                    )
                )

    def cache_purge_namespace(self, namespace: str) -> int:
        with self.db.session() as session:
            stmt = delete(CacheEntry).where(CacheEntry.namespace == namespace)
            result = session.execute(stmt)
            return int(result.rowcount or 0)
