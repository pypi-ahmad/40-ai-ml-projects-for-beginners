"""LangGraph workflow for high-level run state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph


@dataclass(slots=True)
class WorkflowNodes:
    plan_fn: Any
    approve_fn: Any
    execute_fn: Any
    verify_fn: Any
    reflect_fn: Any
    report_fn: Any


class CollaborationGraph:
    """State graph for planner -> approval -> execution -> verification -> report."""

    def __init__(self, nodes: WorkflowNodes) -> None:
        self.nodes = nodes

    def compile(self):
        graph = StateGraph(dict)
        graph.add_node("planner", self.nodes.plan_fn)
        graph.add_node("approval", self.nodes.approve_fn)
        graph.add_node("execute", self.nodes.execute_fn)
        graph.add_node("verify", self.nodes.verify_fn)
        graph.add_node("reflect", self.nodes.reflect_fn)
        graph.add_node("report", self.nodes.report_fn)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "approval")
        graph.add_conditional_edges(
            "approval",
            lambda s: "execute" if s.get("approved", False) else "report",
            {"execute": "execute", "report": "report"},
        )
        graph.add_edge("execute", "verify")
        graph.add_edge("verify", "reflect")
        graph.add_edge("reflect", "report")
        graph.add_edge("report", END)
        return graph.compile()
