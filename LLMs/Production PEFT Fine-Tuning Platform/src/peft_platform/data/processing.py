"""Dataset cleaning, deduplication, split, and stats."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from hashlib import sha256
import random
from typing import Any

from peft_platform.data.schemas import DatasetStats, Sample


def clean_samples(samples: Iterable[Sample]) -> list[Sample]:
    cleaned: list[Sample] = []
    for sample in samples:
        if not (sample.output or sample.label is not None):
            continue
        sample.instruction = sample.instruction.strip()
        sample.input = sample.input.strip()
        sample.output = sample.output.strip()
        cleaned.append(sample)
    return cleaned


def deduplicate_samples(samples: Iterable[Sample]) -> list[Sample]:
    seen: set[str] = set()
    deduped: list[Sample] = []
    for sample in samples:
        payload = f"{sample.task_type}|{sample.instruction}|{sample.input}|{sample.output}|{sample.label}"
        digest = sha256(payload.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        deduped.append(sample)
    return deduped


def split_samples(
    samples: list[Sample],
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[Sample]]:
    if not samples:
        return {"train": [], "validation": [], "test": []}

    holdout_ratio = max(min(val_size + test_size, 0.9), 0.0)
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    holdout_size = int(len(shuffled) * holdout_ratio)
    holdout = shuffled[:holdout_size]
    train = shuffled[holdout_size:]

    if not holdout:
        return {"train": train, "validation": [], "test": []}

    test_ratio = (test_size / (val_size + test_size)) if (val_size + test_size) > 0 else 0.5
    test_size_abs = int(len(holdout) * test_ratio)
    test = holdout[:test_size_abs]
    validation = holdout[test_size_abs:]
    return {"train": train, "validation": validation, "test": test}


def compute_stats(samples: list[Sample]) -> DatasetStats:
    if not samples:
        return DatasetStats(size=0, avg_input_chars=0.0, avg_output_chars=0.0, unique_ratio=0.0)

    inputs = [len(item.input) for item in samples]
    outputs = [len(item.output) for item in samples]

    unique = len(
        {
            sha256(str(asdict(item)).encode("utf-8")).hexdigest()
            for item in samples
        }
    )

    return DatasetStats(
        size=len(samples),
        avg_input_chars=sum(inputs) / len(inputs),
        avg_output_chars=sum(outputs) / len(outputs),
        unique_ratio=unique / len(samples),
    )


def to_records(samples: list[Sample]) -> list[dict[str, Any]]:
    return [asdict(sample) for sample in samples]
