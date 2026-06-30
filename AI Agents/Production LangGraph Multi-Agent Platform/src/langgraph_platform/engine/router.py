"""Routing logic for conditional graph execution."""

from __future__ import annotations

from langgraph_platform.config.settings import RoutingConfig
from langgraph_platform.state.models import RoutingDecision, WorkflowState


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def decide_routing(state: WorkflowState, config: RoutingConfig) -> RoutingDecision:
    """Decide dynamic execution path based on request and prior state."""

    request = state.user_request
    return RoutingDecision(
        require_web_search=_contains_any(request, config.web_search_keywords),
        require_rag=_contains_any(request, config.rag_keywords) or bool(state.retrieved_documents),
        require_memory=_contains_any(request, config.memory_keywords) or bool(state.memory),
        require_code_execution=("python" in request.lower() or "code" in request.lower()),
        require_verification=True,
    )


def should_retry_reflection(
    state: WorkflowState, confidence_threshold: float, max_retries: int
) -> bool:
    """Determine whether reflection loop should rerun."""

    retries = state.execution_metadata.retries.get("reflection_loop", 0)
    return state.confidence_score < confidence_threshold and retries < max_retries
