"""LangGraph workflow builder."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from task_planning_agent.agent.nodes import WorkflowNodes
from task_planning_agent.agent.state import AgentState


def build_graph(nodes: WorkflowNodes):
    """Build and compile production planning graph."""

    graph = StateGraph(AgentState)
    graph.add_node("planner", nodes.planner)
    graph.add_node("scheduler", nodes.scheduler_node)
    graph.add_node("validator", nodes.validator)
    graph.add_node("reflection", nodes.reflection_node)
    graph.add_node("memory", nodes.memory_node)
    graph.add_node("reporter", nodes.reporter_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "scheduler")
    graph.add_edge("scheduler", "validator")
    graph.add_edge("validator", "reflection")
    graph.add_edge("reflection", "memory")
    graph.add_edge("memory", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()
