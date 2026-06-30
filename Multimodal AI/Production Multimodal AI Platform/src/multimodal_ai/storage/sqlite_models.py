"""SQLAlchemy ORM models for platform metadata."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class."""


class Asset(Base):
    """Uploaded asset metadata."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    path: Mapped[str] = mapped_column(Text)
    media_type: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    captions: Mapped[list[Caption]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )
    ocr_results: Mapped[list[OCRRecord]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )
    detections: Mapped[list[DetectionRecord]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )


class Caption(Base):
    """Caption records per asset."""

    __tablename__ = "captions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), index=True)
    style: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    asset: Mapped[Asset] = relationship(back_populates="captions")


class OCRRecord(Base):
    """OCR output records."""

    __tablename__ = "ocr_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), index=True)
    engine: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    asset: Mapped[Asset] = relationship(back_populates="ocr_results")


class DetectionRecord(Base):
    """Object detection records."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), index=True)
    label: Mapped[str] = mapped_column(String(128))
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    asset: Mapped[Asset] = relationship(back_populates="detections")


class ProcessingEvent(Base):
    """Processing history and latency."""

    __tablename__ = "processing_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    details: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelUsageMetric(Base):
    """Aggregated model usage counters."""

    __tablename__ = "model_usage_metrics"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    capability: Mapped[str] = mapped_column(String(64), index=True)
    calls: Mapped[int] = mapped_column(Integer, default=0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
