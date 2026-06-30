from api_intel_agent.agents.state import GraphState
from api_intel_agent.core.schemas import AnalyzeRequest, ConnectorResult, RunStatus


def test_graph_state_to_response_partial_status():
    state = GraphState(request=AnalyzeRequest(query='test'))
    state.connector_results = [
        ConnectorResult(
            provider='jsonplaceholder',
            endpoint='/posts',
            status='ok',
            records=[{'id': 1}],
            latency_ms=1.0,
        )
    ]
    state.reasoning_summary = 'summary'
    response = state.to_response()
    assert response.status in {RunStatus.SUCCESS, RunStatus.PARTIAL}
    assert response.summary == 'summary'
    assert response.sources
