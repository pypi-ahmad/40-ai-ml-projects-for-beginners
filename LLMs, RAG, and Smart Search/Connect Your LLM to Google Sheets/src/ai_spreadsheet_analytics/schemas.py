"""Core dataclasses and pydantic models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


@dataclass(slots=True)
class SheetMetadata:
    """Google Sheet worksheet metadata."""

    spreadsheet_id: str
    spreadsheet_title: str
    worksheet_title: str
    rows: int
    columns: int
    dtypes: dict[str, str]
    missing_by_column: dict[str, int]
    sample_records: list[dict[str, Any]]


@dataclass(slots=True)
class DatasetFrame:
    """Single sheet dataset."""

    key: str
    dataframe: pd.DataFrame
    metadata: SheetMetadata


@dataclass(slots=True)
class DatasetBundle:
    """Collection of dataframes from multiple worksheets/spreadsheets."""

    frames: dict[str, DatasetFrame] = field(default_factory=dict)

    def combined(self, source_column: str = "__source") -> pd.DataFrame:
        """Return combined analytical dataframe across all frames."""
        if not self.frames:
            return pd.DataFrame()
        merged: list[pd.DataFrame] = []
        for key, frame in self.frames.items():
            part = frame.dataframe.copy()
            part[source_column] = key
            merged.append(part)
        return pd.concat(merged, ignore_index=True, sort=False)


@dataclass(slots=True)
class QualityIssue:
    """Single quality issue summary."""

    check_name: str
    severity: str
    message: str
    details: dict[str, Any]


@dataclass(slots=True)
class QualityReport:
    """Data quality report for one dataframe."""

    dataset_key: str
    row_count: int
    column_count: int
    issues: list[QualityIssue]
    metrics: dict[str, Any]
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class CleaningStrategy:
    """Cleaning strategy selected by user."""

    missing_value_strategy: str = "median"
    drop_duplicate_rows: bool = True
    parse_currency: bool = True
    parse_percentage: bool = True
    normalize_dates: bool = True
    drop_empty_columns: bool = True
    drop_constant_columns: bool = False


@dataclass(slots=True)
class CleaningResult:
    """Cleaning output for one dataframe."""

    dataset_key: str
    cleaned: pd.DataFrame
    actions: list[str]


@dataclass(slots=True)
class InsightPacket:
    """LLM generated insights with deterministic evidence."""

    prompt_role: str
    model: str
    summary: str
    findings: list[str]
    recommendations: list[str]
    deterministic_evidence: dict[str, Any]
    latency_ms: float
    token_estimate: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class ChatTurn:
    """Stored chat turn."""

    session_id: str
    question: str
    answer: str
    evidence: dict[str, Any]
    model: str
    latency_ms: float
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class BenchmarkCase:
    """Single benchmark question."""

    case_id: str
    question: str
    expected_type: str
    expected_keywords: list[str]
    domain: str


@dataclass(slots=True)
class BenchmarkResult:
    """Benchmark result for one case/model."""

    run_id: str
    case_id: str
    model: str
    latency_ms: float
    passed_keywords: int
    total_keywords: int
    hallucination_flag: bool
    consistency_score: float
    usefulness_score: float


class JudgeScores(BaseModel):
    """Structured judge scores."""

    insight_quality: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    business_relevance: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    rationale: str


class EvaluationReport(BaseModel):
    """Persisted LLM-as-judge result."""

    run_id: str
    model_evaluated: str
    judge_model: str
    scores: JudgeScores
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReportArtifacts(BaseModel):
    """Generated report paths."""

    markdown: Path | None = None
    html: Path | None = None
    pdf: Path | None = None
    excel: Path | None = None
    powerpoint: Path | None = None
