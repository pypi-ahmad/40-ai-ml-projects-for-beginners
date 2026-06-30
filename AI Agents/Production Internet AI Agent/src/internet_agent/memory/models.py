"""SQLAlchemy ORM models for persistent memory."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from internet_agent.memory.db import Base


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class SearchRecord(Base):
    __tablename__ = "search_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    query: Mapped[str] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(String(64))
    results_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class VisitedURL(Base):
    __tablename__ = "visited_urls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    url: Mapped[str] = mapped_column(Text, index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    status_code: Mapped[int] = mapped_column(Integer, default=0)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class RetrievedDocument(Base):
    __tablename__ = "retrieved_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text, default="")
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class SummaryRecord(Base):
    __tablename__ = "summary_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    query: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ToolHistory(Base):
    __tablename__ = "tool_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    tool_input_json: Mapped[dict] = mapped_column(JSON)
    tool_output_json: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="ok")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ReportRecord(Base):
    __tablename__ = "report_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    format: Mapped[str] = mapped_column(String(16), index=True)
    path: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class CacheEntry(Base):
    __tablename__ = "cache_entries"

    key: Mapped[str] = mapped_column(String(512), primary_key=True)
    value_json: Mapped[dict] = mapped_column(JSON)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    namespace: Mapped[str] = mapped_column(String(64), default="default", index=True)
