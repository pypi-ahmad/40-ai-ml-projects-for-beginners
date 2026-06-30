"""Runtime runner for LangGraph or deterministic fallback flow."""

from __future__ import annotations

import asyncio
from typing import Any

from api_intel_agent.agents.graph import IntelligenceGraph
from api_intel_agent.agents.nodes import AgentRuntime
from api_intel_agent.agents.state import GraphState
from api_intel_agent.core.schemas import AnalyzeRequest, AnalyzeResponse


class AgentRunner:
    def __init__(self) -> None:
        self.runtime = AgentRuntime()
        self.graph = IntelligenceGraph(self.runtime).compile()

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        state = GraphState(request=request)
        try:
            result = await self.graph.ainvoke({"state": state.model_dump(mode="json")})
            final_state = GraphState.model_validate(result["state"])
            return final_state.to_response()
        except Exception:
            # Deterministic fallback keeps system responsive.
            state = await self.runtime.run_once(state)
            while not state.done and state.retries < 3:
                state = await self.runtime.run_once(state)
            return state.to_response()

    def analyze_sync(self, request: AnalyzeRequest) -> AnalyzeResponse:
        return asyncio.run(self.analyze(request))

    def query_sync(self, query: str, **kwargs: Any) -> AnalyzeResponse:
        request = AnalyzeRequest(query=query, **kwargs)
        return self.analyze_sync(request)
