from langgraph_platform.config.settings import RoutingConfig
from langgraph_platform.engine.router import decide_routing
from langgraph_platform.state.models import ExecutionMetadata, WorkflowState


def _state(request: str) -> WorkflowState:
    return WorkflowState(
        user_request=request,
        execution_metadata=ExecutionMetadata(workflow_id="wf_test", session_id="session_test"),
    )


def test_routing_flags_from_keywords() -> None:
    config = RoutingConfig(
        web_search_keywords=["latest"],
        rag_keywords=["document"],
        memory_keywords=["previous"],
    )

    state = _state("Use latest document and previous report")
    decision = decide_routing(state, config)

    assert decision.require_web_search is True
    assert decision.require_rag is True
    assert decision.require_memory is True
