"""Failure-case scenarios for RAG quality analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from local_rag.rag import RAGPipeline


@dataclass(slots=True)
class FailureCase:
    """Configurable failure scenario."""

    name: str
    query: str
    top_k: int
    description: str


@dataclass(slots=True)
class FailureResult:
    """Observed behavior for one scenario."""

    name: str
    description: str
    answer: str
    citation_count: int
    top_score: float
    error: str | None = None


DEFAULT_FAILURE_CASES = [
    FailureCase(
        name="poor_retrieval_low_k",
        query="How do kernel boot parameters affect memory acceptance?",
        top_k=1,
        description="Low top-k may miss relevant context.",
    ),
    FailureCase(
        name="poor_chunk_coverage",
        query="What does ACPI chapter say about power states and kernel behavior?",
        top_k=2,
        description="Small retrieval window can miss multi-part explanation.",
    ),
    FailureCase(
        name="moderate_retrieval",
        query="How do kernel boot parameters affect memory acceptance?",
        top_k=5,
        description="Higher top-k usually improves recall.",
    ),
    FailureCase(
        name="missing_docs",
        query="What does this corpus say about quantum compiler internals?",
        top_k=5,
        description="Missing domain content should trigger unavailable response.",
    ),
    FailureCase(
        name="unrelated_question",
        query="Who won the latest FIFA world cup final?",
        top_k=5,
        description="Out-of-domain consumer question should avoid hallucination.",
    ),
    FailureCase(
        name="ambiguous_question",
        query="Explain system behavior when it fails unexpectedly.",
        top_k=5,
        description="Ambiguous prompt should stay grounded to retrieved context.",
    ),
]


def run_failure_case_analysis(
    pipeline: RAGPipeline,
    cases: list[FailureCase] | None = None,
) -> list[FailureResult]:
    """Run canned failure cases through pipeline."""

    results: list[FailureResult] = []
    for case in cases or DEFAULT_FAILURE_CASES:
        try:
            response = pipeline.ask(case.query, top_k=case.top_k)
            top_score = response.retrieved[0].score if response.retrieved else 0.0
            results.append(
                FailureResult(
                    name=case.name,
                    description=case.description,
                    answer=response.answer,
                    citation_count=len(response.citations),
                    top_score=top_score,
                    error=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                FailureResult(
                    name=case.name,
                    description=case.description,
                    answer="",
                    citation_count=0,
                    top_score=0.0,
                    error=str(exc),
                )
            )
    return results


def compare_failure_results(
    baseline: list[FailureResult],
    improved: list[FailureResult],
    key_fn: Callable[[FailureResult], str] = lambda row: row.name,
) -> list[dict[str, str | float]]:
    """Compare before/after scenario outputs."""

    improved_map = {key_fn(row): row for row in improved}
    rows: list[dict[str, str | float]] = []
    for row in baseline:
        new_row = improved_map.get(key_fn(row))
        if not new_row:
            continue
        rows.append(
            {
                "name": row.name,
                "baseline_citations": float(row.citation_count),
                "improved_citations": float(new_row.citation_count),
                "baseline_top_score": row.top_score,
                "improved_top_score": new_row.top_score,
            }
        )
    return rows
