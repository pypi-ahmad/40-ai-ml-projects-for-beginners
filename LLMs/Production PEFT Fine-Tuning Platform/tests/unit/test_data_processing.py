from __future__ import annotations

from peft_platform.data.processing import clean_samples, compute_stats, deduplicate_samples, split_samples
from peft_platform.data.schemas import Sample


def test_processing_pipeline() -> None:
    samples = [
        Sample(task_type="instruction", instruction="  inst ", input=" in ", output=" out "),
        Sample(task_type="instruction", instruction="inst", input="in", output="out"),
        Sample(task_type="instruction", instruction="x", input="y", output=""),
    ]

    cleaned = clean_samples(samples)
    assert len(cleaned) == 2

    deduped = deduplicate_samples(cleaned)
    assert len(deduped) == 1

    stats = compute_stats(deduped)
    assert stats.size == 1
    assert 0.0 <= stats.unique_ratio <= 1.0

    splits = split_samples(deduped * 6, val_size=0.2, test_size=0.2, seed=42)
    assert len(splits["train"]) > 0
