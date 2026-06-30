"""Typed data schemas for dataset profiling and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class LabelStats:
    label_id: int
    label_name: str
    count: int
    proportion: float


@dataclass(slots=True)
class DatasetProfile:
    dataset_name: str
    split_sizes: dict[str, int]
    num_classes: int
    class_distribution: list[LabelStats]
    missing_text_count: int
    duplicate_text_count: int
    outlier_count: int
    avg_char_length: float
    median_char_length: float
    p95_char_length: float
    token_mean: float
    token_p95: float
    vocabulary_size: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class DatasetVersionManifest:
    dataset_name: str
    source: str
    revision: str | None
    train_fingerprint: str
    validation_fingerprint: str
    test_fingerprint: str
    split_seed: int
    preprocessing_hash: str
    label_names: list[str]
    created_at_utc: str
