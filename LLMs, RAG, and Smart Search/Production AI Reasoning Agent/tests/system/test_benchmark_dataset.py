from __future__ import annotations

from reasoning_agent.evaluation.datasets import BenchmarkDataset


def test_dataset_has_100_plus_prompts() -> None:
    ds = BenchmarkDataset.load_jsonl("benchmarks/prompts.jsonl")
    assert len(ds.records) >= 100
