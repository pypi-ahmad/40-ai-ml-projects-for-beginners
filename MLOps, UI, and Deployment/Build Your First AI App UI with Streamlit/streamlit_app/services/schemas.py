"""Typed result objects for model outputs and benchmark records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC


@dataclass(slots=True)
class SentimentResult:
    sentiment: str
    confidence: float
    explanation: str

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(slots=True)
class ClassificationResult:
    category: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkRecord:
    model: str
    run_id: int
    latency_seconds: float
    output_length: int
    output_word_count: int
    throughput_words_per_second: float
    process_memory_mb: float
    lexical_diversity: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkSummary:
    model: str
    runs: int
    mean_latency: float
    median_latency: float
    min_latency: float
    max_latency: float
    std_latency: float
    mean_output_len: float
    mean_output_words: float
    mean_throughput_wps: float
    mean_memory_mb: float
    mean_quality_score: float
    created_at_utc: str

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).isoformat()
