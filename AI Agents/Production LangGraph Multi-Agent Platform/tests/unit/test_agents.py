from langgraph_platform.agents.registry import AGENT_REGISTRY


def test_required_agent_count() -> None:
    assert len(AGENT_REGISTRY) == 20


def test_planner_contract() -> None:
    planner = AGENT_REGISTRY["planner"]
    assert planner.role == "Planner Agent"
    assert "routing" in planner.output_schema
