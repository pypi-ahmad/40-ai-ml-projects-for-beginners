"""Workflow runtime wrapper for LangGraph invocation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hybrid_research_assistant.graph import GraphComponents, build_workflow
from hybrid_research_assistant.schemas import (
    AssistantResponse,
    Citation,
    RetrievalMode,
    RetrievedContext,
    TimingBreakdown,
)
from hybrid_research_assistant.utils import json_dump


class WorkflowRuntime:
    """Run and introspect hybrid LangGraph workflow."""

    def __init__(self, components: GraphComponents) -> None:
        self.components = components
        self.graph = build_workflow(components)

    def ask(
        self,
        query: str,
        *,
        mode: RetrievalMode = RetrievalMode.AUTO,
        prompt_name: str = "research_assistant",
        provider: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AssistantResponse:
        """Invoke full workflow and return typed response."""

        state = {
            "query": query,
            "requested_mode": mode.value,
            "prompt_name": prompt_name,
            "provider": provider,
            "metadata_filter": metadata_filter,
        }
        final_state = self.graph.invoke(state)

        retrieved = [RetrievedContext(**row) for row in final_state.get("retrieved", [])]
        citations = [Citation(**row) for row in final_state.get("citations", [])]
        timings_map = final_state.get("timings", {})

        return AssistantResponse(
            query=query,
            answer=str(final_state.get("answer", "")),
            mode=RetrievalMode(final_state.get("route", {}).get("mode", mode.value)),
            route_reason=str(final_state.get("route", {}).get("reason", "")),
            citations=citations,
            retrieved=retrieved,
            timings=TimingBreakdown(
                intent_ms=float(timings_map.get("intent_ms", 0.0)),
                retrieval_ms=float(timings_map.get("retrieval_ms", 0.0)),
                rerank_ms=float(timings_map.get("rerank_ms", 0.0)),
                context_ms=float(timings_map.get("context_ms", 0.0)),
                generation_ms=float(timings_map.get("generation_ms", 0.0)),
                judge_ms=float(timings_map.get("judge_ms", 0.0)),
                total_ms=float(timings_map.get("total_ms", 0.0)),
            ),
            prompt_name=prompt_name,
            judge=dict(final_state.get("judge", {})),
        )

    def export_diagram(self, output_path: Path) -> Path:
        """Export graph diagram to markdown text artifact."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagram = (
            "flowchart TD\n"
            "  A[User Question] --> B[Intent Detection]\n"
            "  B --> C{Router}\n"
            "  C -->|Local| D[Local Retrieve]\n"
            "  C -->|Web| E[Web Retrieve]\n"
            "  C -->|Hybrid| F[Hybrid Retrieve]\n"
            "  D --> G[Rerank]\n"
            "  E --> G\n"
            "  F --> G\n"
            "  G --> H[Context Builder]\n"
            "  H --> I[Generation]\n"
            "  I --> J[Judge]\n"
            "  J --> K[Response]\n"
            "  I --> L[Error Recovery]\n"
            "  L --> K\n"
        )
        try:
            graph_obj = self.graph.get_graph()  # type: ignore[attr-defined]
            diagram = graph_obj.draw_mermaid()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        output_path.write_text(diagram, encoding="utf-8")
        return output_path

    def dump_last_state(self, output_path: Path, state: dict[str, Any]) -> Path:
        """Write raw workflow state for debugging."""

        json_dump(output_path, json.loads(json.dumps(state, default=str)))
        return output_path
