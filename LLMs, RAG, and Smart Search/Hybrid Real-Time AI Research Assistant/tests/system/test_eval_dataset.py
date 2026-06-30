from __future__ import annotations

from collections import Counter

from hybrid_research_assistant.utils import read_jsonl


def test_eval_dataset_has_100_rows_and_categories() -> None:
    rows = read_jsonl("data/eval/benchmark_questions.jsonl")  # type: ignore[arg-type]
    assert len(rows) == 100

    counts = Counter(row["category"] for row in rows)
    assert counts["factual"] == 20
    assert counts["reasoning"] == 20
    assert counts["comparison"] == 20
    assert counts["summarization"] == 15
    assert counts["multi_document"] == 15
    assert counts["fresh_knowledge"] == 10
