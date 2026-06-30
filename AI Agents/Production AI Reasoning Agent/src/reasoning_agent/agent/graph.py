"""LangGraph orchestration for the reasoning agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from reasoning_agent.agent.error_handler import ErrorHandler
from reasoning_agent.agent.execution_logger import ExecutionLogger
from reasoning_agent.agent.executor import Executor
from reasoning_agent.agent.observation_processor import ObservationProcessor
from reasoning_agent.agent.planner import Planner
from reasoning_agent.agent.reflection import Reflector
from reasoning_agent.agent.response_generator import ResponseGenerator
from reasoning_agent.agent.state import AgentState
from reasoning_agent.agent.tool_router import ToolRouter


@dataclass(slots=True)
class AgentGraph:
    planner: Planner
    router: ToolRouter
    executor: Executor
    observer: ObservationProcessor
    reflector: Reflector
    responder: ResponseGenerator
    errors: ErrorHandler
    logger: ExecutionLogger

    def compile(self):
        graph = StateGraph(dict)
        graph.add_node("planner", self._planner_node)
        graph.add_node("tool_router", self._tool_router_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("observation", self._observation_node)
        graph.add_node("reflection", self._reflection_node)
        graph.add_node("response", self._response_node)
        graph.add_node("error_handler", self._error_node)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "tool_router")
        graph.add_edge("tool_router", "executor")
        graph.add_edge("executor", "observation")
        graph.add_edge("observation", "reflection")

        graph.add_conditional_edges(
            "reflection",
            self._after_reflection,
            {"continue": "tool_router", "respond": "response", "error": "error_handler"},
        )
        graph.add_conditional_edges(
            "error_handler",
            self._after_error,
            {"continue": "tool_router", "respond": "response"},
        )
        graph.add_edge("response", END)

        return graph.compile()

    async def _planner_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        run_id = state_dict.get("run_id", "run")
        state = AgentState.model_validate(state_dict["state"])
        state = await self.planner.build_plan(state, available_tools=state_dict["available_tools"])
        self.logger.log_state(run_id, "planner", state)
        return {"state": state.model_dump(mode="json")}

    async def _tool_router_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        run_id = state_dict.get("run_id", "run")
        state = AgentState.model_validate(state_dict["state"])
        state = self.router.route(state)
        self.logger.log_state(run_id, "tool_router", state)
        return {"state": state.model_dump(mode="json")}

    async def _executor_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        run_id = state_dict.get("run_id", "run")
        state = AgentState.model_validate(state_dict["state"])
        state = await self.executor.execute_current_step(state, run_id=run_id)
        self.logger.log_state(run_id, "executor", state)
        return {"state": state.model_dump(mode="json")}

    async def _observation_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        run_id = state_dict.get("run_id", "run")
        state = AgentState.model_validate(state_dict["state"])
        state = self.observer.process(state)
        self.logger.log_state(run_id, "observation", state)
        return {"state": state.model_dump(mode="json")}

    async def _reflection_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        run_id = state_dict.get("run_id", "run")
        state = AgentState.model_validate(state_dict["state"])
        state = await self.reflector.reflect(state)
        self.logger.log_state(run_id, "reflection", state)
        return {"state": state.model_dump(mode="json")}

    async def _error_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        run_id = state_dict.get("run_id", "run")
        state = AgentState.model_validate(state_dict["state"])
        state = self.errors.apply(state)
        self.logger.log_state(run_id, "error_handler", state)
        return {"state": state.model_dump(mode="json")}

    async def _response_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        run_id = state_dict.get("run_id", "run")
        state = AgentState.model_validate(state_dict["state"])
        state = await self.responder.generate(state)
        state.done = True
        self.logger.log_state(run_id, "response", state)
        return {"state": state.model_dump(mode="json")}

    def _after_reflection(self, state_dict: dict[str, Any]) -> str:
        state = AgentState.model_validate(state_dict["state"])
        if state.error:
            return "error"
        if state.should_continue():
            return "continue"
        return "respond"

    def _after_error(self, state_dict: dict[str, Any]) -> str:
        state = AgentState.model_validate(state_dict["state"])
        return "continue" if state.should_continue() else "respond"
