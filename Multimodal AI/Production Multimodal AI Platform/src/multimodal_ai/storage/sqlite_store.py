"""SQLite repository utilities."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from multimodal_ai.storage.sqlite_models import (
    Asset,
    Base,
    Caption,
    DetectionRecord,
    ModelUsageMetric,
    OCRRecord,
    ProcessingEvent,
)


class SQLiteStore:
    """SQLite persistence layer."""

    def __init__(self, sqlite_url: str) -> None:
        self._engine = create_engine(sqlite_url, future=True)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    def create_tables(self) -> None:
        """Create DB tables if absent."""

        Base.metadata.create_all(self._engine)

    def add_asset(self, asset_id: str, path: str, media_type: str) -> int:
        """Insert asset and return row id."""

        with self._session_factory() as session:
            item = Asset(asset_id=asset_id, path=path, media_type=media_type)
            session.add(item)
            session.commit()
            return item.id

    def get_asset_by_asset_id(self, asset_id: str) -> Asset | None:
        """Fetch asset row by external id."""

        with self._session_factory() as session:
            stmt = select(Asset).where(Asset.asset_id == asset_id)
            return session.scalar(stmt)

    def add_caption(self, asset_row_id: int, style: str, content: str, confidence: float) -> None:
        """Persist caption record."""

        with self._session_factory() as session:
            session.add(
                Caption(asset_id=asset_row_id, style=style, content=content, confidence=confidence)
            )
            session.commit()

    def add_ocr(self, asset_row_id: int, engine: str, text: str) -> None:
        """Persist OCR record."""

        with self._session_factory() as session:
            session.add(OCRRecord(asset_id=asset_row_id, engine=engine, text=text))
            session.commit()

    def add_detections(self, asset_row_id: int, detections: list[dict[str, Any]]) -> None:
        """Persist detection records."""

        if not detections:
            return
        with self._session_factory() as session:
            for item in detections:
                session.add(
                    DetectionRecord(
                        asset_id=asset_row_id,
                        label=item.get("label", "unknown"),
                        confidence=float(item.get("confidence", 0.0)),
                        bbox=json.dumps(item.get("bbox", [])),
                    )
                )
            session.commit()

    def add_event(
        self,
        trace_id: str,
        action: str,
        latency_ms: float,
        status: str = "ok",
        details: str = "",
    ) -> None:
        """Persist processing event."""

        with self._session_factory() as session:
            session.add(
                ProcessingEvent(
                    trace_id=trace_id,
                    action=action,
                    latency_ms=latency_ms,
                    status=status,
                    details=details,
                )
            )
            session.commit()

    def bump_model_usage(self, model_name: str, capability: str, latency_ms: float) -> None:
        """Update model usage counters."""

        with self._session_factory() as session:
            stmt = select(ModelUsageMetric).where(
                ModelUsageMetric.model_name == model_name,
                ModelUsageMetric.capability == capability,
            )
            metric = session.scalar(stmt)
            if metric is None:
                metric = ModelUsageMetric(
                    model_name=model_name,
                    capability=capability,
                    calls=1,
                    avg_latency_ms=latency_ms,
                )
                session.add(metric)
            else:
                new_calls = metric.calls + 1
                metric.avg_latency_ms = (
                    metric.avg_latency_ms * metric.calls + latency_ms
                ) / new_calls
                metric.calls = new_calls
                metric.updated_at = datetime.utcnow()
            session.commit()

    def list_model_usage(self) -> list[dict[str, Any]]:
        """List model usage metrics."""

        with self._session_factory() as session:
            rows = session.scalars(select(ModelUsageMetric)).all()
            return [
                {
                    "model_name": row.model_name,
                    "capability": row.capability,
                    "calls": row.calls,
                    "avg_latency_ms": row.avg_latency_ms,
                }
                for row in rows
            ]
