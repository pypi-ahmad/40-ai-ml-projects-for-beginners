"""State contract for multi-agent LangGraph workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from api_intel_agent.core.schemas import AnalyzeRequest, AnalyzeResponse, ConnectorResult, ErrorRecord, RunStatus


class PlanStep(BaseModel):
    provider: str
    purpose: str
    params: dict[str, Any] = Field(default_factory=dict)
    parallel_group: int = 0


class GraphState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    request: AnalyzeRequest
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    selected_apis: list[str] = Field(default_factory=list)
    plan_steps: list[PlanStep] = Field(default_factory=list)

    auth_status: dict[str, str] = Field(default_factory=dict)
    connector_results: list[ConnectorResult] = Field(default_factory=list)
    validated_records: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    reasoning_summary: str = ""
    insights: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    charts: list[dict[str, Any]] = Field(default_factory=list)
    report_paths: dict[str, str] = Field(default_factory=dict)

    retries: int = 0
    errors: list[ErrorRecord] = Field(default_factory=list)
    done: bool = False

    telemetry: dict[str, Any] = Field(default_factory=dict)

    def to_response(self) -> AnalyzeResponse:
        status = RunStatus.SUCCESS
        if self.errors and self.connector_results:
            status = RunStatus.PARTIAL
        if self.errors and not self.connector_results:
            status = RunStatus.FAILED

        sources = []
        for result in self.connector_results:
            if result.status in {"ok", "empty"}:
                sources.append(
                    {
                        "provider": result.provider,
                        "endpoint": result.endpoint,
                    }
                )

        return AnalyzeResponse(
            run_id=self.run_id,
            status=status,
            summary=self.reasoning_summary or "No summary generated.",
            insights=self.insights,
            recommendations=self.recommendations,
            sources=sources,
            charts=self.charts,
            artifacts=self.report_paths,
            telemetry=self.telemetry,
            errors=self.errors,
        )
