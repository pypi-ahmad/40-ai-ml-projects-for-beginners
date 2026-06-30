"""Failure-case probes and recovery verification."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_research_assistant.schemas import RetrievalMode
from hybrid_research_assistant.workflow import WorkflowRuntime


@dataclass(slots=True)
class FailureCaseResult:
    name: str
    success: bool
    details: str


def run_failure_analysis(runtime: WorkflowRuntime, *, include_web: bool = False) -> list[FailureCaseResult]:
    """Execute failure scenarios and capture recovery behavior."""

    rows: list[FailureCaseResult] = []

    response = runtime.ask("Non-existent synthetic query with no matching evidence", mode=RetrievalMode.LOCAL)
    rows.append(
        FailureCaseResult(
            name="no_matching_documents",
            success="I don't know based on the retrieved information." in response.answer,
            details="fallback_expected_for_empty_context",
        )
    )

    if include_web:
        response = runtime.ask("Latest breaking AI news today", mode=RetrievalMode.WEB)
        rows.append(
            FailureCaseResult(
                name="web_unavailable_or_empty",
                success=bool(response.answer),
                details="response_non_empty_even_when_web_sparse",
            )
        )

        response = runtime.ask("Compare contradictory policies in local docs", mode=RetrievalMode.HYBRID)
        rows.append(
            FailureCaseResult(
                name="conflicting_sources",
                success=bool(response.citations),
                details="citations_present_for_comparison",
            )
        )

    return rows
