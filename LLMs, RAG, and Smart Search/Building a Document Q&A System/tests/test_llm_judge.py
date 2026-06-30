from __future__ import annotations

from local_rag.llm_judge import LLMJudge
from local_rag.types import JudgeScore


def test_judge_aggregate_means() -> None:
    rows = [
        JudgeScore(
            correctness=3,
            groundedness=4,
            completeness=5,
            faithfulness=4,
            conciseness=3,
            citation_quality=5,
            rationale="a",
        ),
        JudgeScore(
            correctness=5,
            groundedness=2,
            completeness=3,
            faithfulness=4,
            conciseness=5,
            citation_quality=3,
            rationale="b",
        ),
    ]
    agg = LLMJudge.aggregate(rows)
    assert agg.count == 2
    assert agg.correctness == 4.0
    assert agg.groundedness == 3.0
    assert agg.citation_quality == 4.0
