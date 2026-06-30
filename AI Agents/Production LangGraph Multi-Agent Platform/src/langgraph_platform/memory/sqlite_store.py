"""SQLite persistence layer for conversations, workflows, and traces."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from langgraph_platform.state.models import WorkflowState


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class WorkflowRun(Base):
    """Top-level workflow run record."""

    __tablename__ = "workflow_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workflow_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    user_request: Mapped[str] = mapped_column(Text)
    final_report: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    verification_status: Mapped[str] = mapped_column(String(32), default="unknown")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class AgentOutputRecord(Base):
    """Persisted agent output snapshots."""

    __tablename__ = "agent_outputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workflow_id: Mapped[str] = mapped_column(String(128), index=True)
    agent_name: Mapped[str] = mapped_column(String(128), index=True)
    content: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class ToolCallRecord(Base):
    """Persisted tool call trace."""

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workflow_id: Mapped[str] = mapped_column(String(128), index=True)
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    args: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    success: Mapped[bool] = mapped_column(default=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class GraphStateRecord(Base):
    """Persisted graph state snapshots."""

    __tablename__ = "graph_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workflow_id: Mapped[str] = mapped_column(String(128), index=True)
    node_name: Mapped[str] = mapped_column(String(128), index=True)
    state_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class SQLiteStore:
    """Repository for SQLite persistence operations."""

    def __init__(self, sqlite_url: str) -> None:
        path = sqlite_url.replace("sqlite:///", "")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(sqlite_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True)

    def init(self) -> None:
        """Create required tables."""

        Base.metadata.create_all(self.engine)

    def _session(self) -> Session:
        return self.session_factory()

    def save_workflow_state(self, workflow_id: str, node_name: str, state: WorkflowState) -> None:
        """Persist graph state snapshot."""

        with self._session() as session:
            session.add(
                GraphStateRecord(
                    workflow_id=workflow_id,
                    node_name=node_name,
                    state_payload=state.model_dump(mode="json"),
                )
            )
            session.commit()

    def save_agent_output(
        self,
        workflow_id: str,
        agent_name: str,
        content: str,
        confidence: float,
        payload: dict[str, Any],
    ) -> None:
        """Persist one agent output."""

        with self._session() as session:
            session.add(
                AgentOutputRecord(
                    workflow_id=workflow_id,
                    agent_name=agent_name,
                    content=content,
                    confidence=confidence,
                    payload=payload,
                )
            )
            session.commit()

    def save_tool_call(
        self,
        workflow_id: str,
        tool_name: str,
        args: dict[str, Any],
        success: bool,
        latency_ms: float,
        error: str | None,
    ) -> None:
        """Persist one tool call trace."""

        with self._session() as session:
            session.add(
                ToolCallRecord(
                    workflow_id=workflow_id,
                    tool_name=tool_name,
                    args=args,
                    success=success,
                    latency_ms=latency_ms,
                    error=error,
                )
            )
            session.commit()

    def finalize_workflow(self, state: WorkflowState, final_report: str) -> None:
        """Persist final workflow summary."""

        with self._session() as session:
            session.add(
                WorkflowRun(
                    workflow_id=state.execution_metadata.workflow_id,
                    session_id=state.execution_metadata.session_id,
                    user_request=state.user_request,
                    final_report=final_report,
                    confidence=state.confidence_score,
                    verification_status=state.verification_status.value,
                )
            )
            session.commit()

    def list_recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent workflow run summaries."""

        with self._session() as session:
            runs = (
                session.query(WorkflowRun)
                .order_by(WorkflowRun.created_at.desc())
                .limit(limit)
                .all()
            )
        return [
            {
                "workflow_id": run.workflow_id,
                "session_id": run.session_id,
                "user_request": run.user_request,
                "confidence": run.confidence,
                "verification_status": run.verification_status,
                "created_at": run.created_at.isoformat(),
            }
            for run in runs
        ]

    def close(self) -> None:
        """Release SQLAlchemy engine resources."""

        self.engine.dispose()
