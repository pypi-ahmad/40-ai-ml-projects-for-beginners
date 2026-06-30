"""LangGraph assembly for multi-agent API intelligence workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from api_intel_agent.agents.nodes import AgentRuntime
from api_intel_agent.agents.state import GraphState


class IntelligenceGraph:
    def __init__(self, runtime: AgentRuntime | None = None) -> None:
        self.runtime = runtime or AgentRuntime()

    def compile(self):
        graph = StateGraph(dict)
        graph.add_node("request_planner", self._request_planner)
        graph.add_node("api_router", self._api_router)
        graph.add_node("authentication", self._authentication)
        graph.add_node("data_fetch", self._data_fetch)
        graph.add_node("validation", self._validation)
        graph.add_node("reasoning", self._reasoning)
        graph.add_node("report_generator", self._report_generator)
        graph.add_node("memory", self._memory)
        graph.add_node("reflection", self._reflection)

        graph.set_entry_point("request_planner")
        graph.add_edge("request_planner", "api_router")
        graph.add_edge("api_router", "authentication")
        graph.add_edge("authentication", "data_fetch")
        graph.add_edge("data_fetch", "validation")
        graph.add_edge("validation", "reasoning")
        graph.add_edge("reasoning", "report_generator")
        graph.add_edge("report_generator", "memory")
        graph.add_edge("memory", "reflection")
        graph.add_conditional_edges(
            "reflection",
            self._after_reflection,
            {"retry": "data_fetch", "end": END},
        )
        return graph.compile()

    async def _request_planner(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = GraphState.model_validate(state_dict["state"])
        state = await self.runtime.request_planner.run(state)
        return {"state": state.model_dump(mode="json")}

    async def _api_router(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = GraphState.model_validate(state_dict["state"])
        state = await self.runtime.api_router.run(state)
        return {"state": state.model_dump(mode="json")}

    async def _authentication(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = GraphState.model_validate(state_dict["state"])
        state = await self.runtime.auth.run(state)
        return {"state": state.model_dump(mode="json")}

    async def _data_fetch(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = GraphState.model_validate(state_dict["state"])
        state = await self.runtime.fetch.run(state)
        return {"state": state.model_dump(mode="json")}

    async def _validation(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = GraphState.model_validate(state_dict["state"])
        state = await self.runtime.validation.run(state)
        return {"state": state.model_dump(mode="json")}

    async def _reasoning(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = GraphState.model_validate(state_dict["state"])
        state = await self.runtime.reasoning.run(state)
        return {"state": state.model_dump(mode="json")}

    async def _report_generator(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = GraphState.model_validate(state_dict["state"])
        state = await self.runtime.report_generation.run(state)
        return {"state": state.model_dump(mode="json")}

    async def _memory(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = GraphState.model_validate(state_dict["state"])
        state = await self.runtime.memory_agent.run(state)
        return {"state": state.model_dump(mode="json")}

    async def _reflection(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        state = GraphState.model_validate(state_dict["state"])
        state = await self.runtime.reflection.run(state)
        return {"state": state.model_dump(mode="json")}

    def _after_reflection(self, state_dict: dict[str, Any]) -> str:
        state = GraphState.model_validate(state_dict["state"])
        if state.done:
            return "end"
        return "retry"
