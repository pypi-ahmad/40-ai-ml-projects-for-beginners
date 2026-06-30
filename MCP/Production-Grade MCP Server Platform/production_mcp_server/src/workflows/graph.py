from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from memory.service import MemoryService
from prompts.library import PromptLibrary
from tools.registry import ToolRegistry
from workflows.state import WorkflowState

logger = logging.getLogger(__name__)


class WorkflowEngine:
    def __init__(self, tools: ToolRegistry, memory: MemoryService, prompts: PromptLibrary) -> None:
        self.tools = tools
        self.memory = memory
        self.prompts = prompts
        self.graph = self._build()

    def _build(self):
        graph = StateGraph(dict)
        graph.add_node("planner", self._planner)
        graph.add_node("tool_selector", self._tool_selector)
        graph.add_node("mcp_router", self._mcp_router)
        graph.add_node("execution_agent", self._execution_agent)
        graph.add_node("reflection_agent", self._reflection_agent)
        graph.add_node("memory_agent", self._memory_agent)
        graph.add_node("report_agent", self._report_agent)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "tool_selector")
        graph.add_edge("tool_selector", "mcp_router")
        graph.add_edge("mcp_router", "execution_agent")
        graph.add_edge("execution_agent", "reflection_agent")
        graph.add_conditional_edges(
            "reflection_agent",
            self._after_reflection,
            {"continue": "memory_agent", "report": "report_agent", "fail": END},
        )
        graph.add_edge("memory_agent", "report_agent")
        graph.add_edge("report_agent", END)

        return graph.compile()

    async def run(self, query: str) -> dict[str, Any]:
        # LangGraph object is built for MCP client inspection, but runtime uses
        # deterministic fallback flow to avoid event-loop deadlocks across hosts.
        state = WorkflowState(query=query)

        for step in [
            self._planner,
            self._tool_selector,
            self._mcp_router,
            self._execution_agent,
            self._reflection_agent,
        ]:
            payload = await step({"state": state.model_dump(mode="json")})
            state = WorkflowState.model_validate(payload["state"])

        branch = self._after_reflection({"state": state.model_dump(mode="json")})
        if branch == "fail":
            state.status = "failed"
            return state.model_dump(mode="json")

        if branch == "continue":
            payload = await self._memory_agent({"state": state.model_dump(mode="json")})
            state = WorkflowState.model_validate(payload["state"])

        payload = await self._report_agent({"state": state.model_dump(mode="json")})
        state = WorkflowState.model_validate(payload["state"])
        return state.model_dump(mode="json")

    async def _planner(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = WorkflowState.model_validate(payload["state"])
        state.plan = [
            "Search memory and docs",
            "Select tools",
            "Execute tools",
            "Reflect and generate report",
        ]
        state.steps.append("planner")
        return {"state": state.model_dump(mode="json")}

    async def _tool_selector(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = WorkflowState.model_validate(payload["state"])
        tools = ["chroma_search", "calculator", "report_generator"]
        if "weather" in state.query.lower():
            tools.insert(0, "weather")
        if "code" in state.query.lower():
            tools.insert(0, "code_search")
        state.selected_tools = [tool for tool in tools if tool in self.tools.names()]
        state.steps.append("tool_selector")
        return {"state": state.model_dump(mode="json")}

    async def _mcp_router(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = WorkflowState.model_validate(payload["state"])
        state.steps.append("mcp_router")
        return {"state": state.model_dump(mode="json")}

    async def _execution_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = WorkflowState.model_validate(payload["state"])
        outputs: list[dict[str, Any]] = []

        for tool_name in state.selected_tools:
            if tool_name == "chroma_search":
                args = {"query": state.query, "top_k": 3}
            elif tool_name == "calculator":
                args = {"expression": "2 + 2"}
            elif tool_name == "report_generator":
                continue
            elif tool_name == "weather":
                args = {"location": "Bengaluru"}
            elif tool_name == "code_search":
                args = {"pattern": "def ", "path": "src"}
            else:
                args = {}

            result = await self.tools.call(tool_name, args)
            outputs.append({"tool": tool_name, "result": result})

        state.tool_outputs = outputs
        state.steps.append("execution_agent")
        return {"state": state.model_dump(mode="json")}

    async def _reflection_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = WorkflowState.model_validate(payload["state"])
        failures = [item for item in state.tool_outputs if not item.get("result", {}).get("ok", False)]
        if failures:
            state.reflection = f"{len(failures)} tool failures detected"
            state.status = "degraded"
        else:
            state.reflection = "All selected tools executed successfully"
            state.status = "running"
        state.steps.append("reflection_agent")
        return {"state": state.model_dump(mode="json")}

    def _after_reflection(self, payload: dict[str, Any]) -> Literal["continue", "report", "fail"]:
        state = WorkflowState.model_validate(payload["state"])
        if not state.tool_outputs:
            return "fail"
        if state.status == "running":
            return "continue"
        return "report"

    async def _memory_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = WorkflowState.model_validate(payload["state"])
        state.memory_context = self.memory.semantic_search(state.query, top_k=3)
        state.steps.append("memory_agent")
        return {"state": state.model_dump(mode="json")}

    async def _report_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = WorkflowState.model_validate(payload["state"])
        summary_lines = [
            f"Query: {state.query}",
            f"Plan: {', '.join(state.plan)}",
            f"Executed tools: {', '.join(state.selected_tools)}",
            f"Reflection: {state.reflection}",
        ]
        for output in state.tool_outputs:
            summary_lines.append(f"- {output['tool']}: ok={output['result'].get('ok')}")

        report_body = "\n".join(summary_lines)
        report_result = await self.tools.call(
            "report_generator",
            {"title": "workflow_report", "body": report_body, "save": True},
        )
        state.report = report_result.get("path", report_result.get("report", ""))
        if state.status == "running":
            state.status = "completed"
        state.steps.append("report_agent")

        self.memory.log_response(
            session_id="workflow",
            response_type="workflow_report",
            payload={"query": state.query, "status": state.status, "report": state.report},
        )
        return {"state": state.model_dump(mode="json")}
