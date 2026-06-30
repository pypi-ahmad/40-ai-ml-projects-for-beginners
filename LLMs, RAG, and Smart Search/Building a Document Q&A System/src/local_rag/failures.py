"""Failure-case scenarios for RAG quality analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from local_rag.rag import RAGPipeline
from local_rag.retriever import RetrievalStrategy


@dataclass(slots=True)
class FailureCase:
    """Configurable failure scenario."""

    name: str
    query: str
    top_k: int
    strategy: RetrievalStrategy
    prompt_template: str
    description: str


@dataclass(slots=True)
class FailureResult:
    """Observed behavior for one scenario."""

    name: str
    description: str
    answer: str
    citation_count: int
    top_score: float
    retrieval_strategy: str


DEFAULT_FAILURE_CASES = [
    FailureCase(
        name="poor_chunking_low_context",
        query="What changed between encryption policy versions?",
        top_k=1,
        strategy="vector",
        prompt_template="unknown_safe",
        description="Low top-k and strict mode can miss cross-document context.",
    ),
    FailureCase(
        name="bad_prompt_template",
        query="Which policy mentions data retention and encryption together?",
        top_k=5,
        strategy="vector",
        prompt_template="technical_qa",
        description="Template mismatch can reduce citation quality for policy questions.",
    ),
    FailureCase(
        name="wrong_retrieval_strategy",
        query="Find exact clause for password expiration in policy handbook.",
        top_k=3,
        strategy="vector",
        prompt_template="legal_qa",
        description="Dense-only retrieval may miss exact keyword clauses.",
    ),
    FailureCase(
        name="improved_hybrid_retrieval",
        query="Find exact clause for password expiration in policy handbook.",
        top_k=5,
        strategy="hybrid",
        prompt_template="legal_qa",
        description="Hybrid retrieval recovers lexical clause and improves grounding.",
    ),
]


def run_failure_case_analysis(
    pipeline: RAGPipeline,
    cases: list[FailureCase] | None = None,
) -> list[FailureResult]:
    """Run canned failure cases through pipeline."""

    results: list[FailureResult] = []
    for case in cases or DEFAULT_FAILURE_CASES:
        response = pipeline.ask(
            case.query,
            top_k=case.top_k,
            strategy=case.strategy,
            prompt_template=case.prompt_template,
        )
        top_score = response.retrieved[0].score if response.retrieved else 0.0
        results.append(
            FailureResult(
                name=case.name,
                description=case.description,
                answer=response.answer,
                citation_count=len(response.citations),
                top_score=top_score,
                retrieval_strategy=response.retrieval_strategy,
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
                "baseline_strategy": row.retrieval_strategy,
                "improved_strategy": new_row.retrieval_strategy,
                "baseline_citations": float(row.citation_count),
                "improved_citations": float(new_row.citation_count),
                "baseline_top_score": row.top_score,
                "improved_top_score": new_row.top_score,
            }
        )
    return rows
