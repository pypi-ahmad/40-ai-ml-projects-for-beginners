"""Public runner for the production reasoning agent."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from reasoning_agent.agent.error_handler import ErrorHandler
from reasoning_agent.agent.execution_logger import ExecutionLogger
from reasoning_agent.agent.executor import Executor
from reasoning_agent.agent.graph import AgentGraph
from reasoning_agent.agent.models import AgentRequest, AgentRunResult
from reasoning_agent.agent.observation_processor import ObservationProcessor
from reasoning_agent.agent.planner import Planner
from reasoning_agent.agent.reflection import Reflector
from reasoning_agent.agent.response_generator import ResponseGenerator
from reasoning_agent.agent.state import AgentState
from reasoning_agent.agent.tool_router import ToolRouter
from reasoning_agent.config import Settings
from reasoning_agent.llm.ollama import OllamaProvider
from reasoning_agent.memory import ChromaSemanticStore, MemoryManager, SessionMemoryStore
from reasoning_agent.observability.metrics import MetricsStore
from reasoning_agent.observability.tracer import JsonlTracer
from reasoning_agent.tools.factory import create_default_registry


class AgentRunner:
    """Top-level orchestrator for end-to-end execution."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = OllamaProvider(
            base_url=settings.llm.base_url,
            timeout_seconds=settings.llm.request_timeout_seconds,
        )

        tracer = JsonlTracer(Path(settings.logging.run_log_path)) if settings.logging.json_logs else None
        self.metrics = MetricsStore()

        semantic = (
            ChromaSemanticStore(path=settings.memory.chroma_path)
            if settings.memory.chroma_enabled
            else None
        )
        self.memory = MemoryManager(
            session=SessionMemoryStore(window_size=settings.memory.conversation_window),
            semantic=semantic,
        )
        self.registry = create_default_registry(
            workspace_root=Path(settings.tools.workspace_root),
            memory_store=semantic,
            tracer=tracer,
            metrics=self.metrics,
            python_timeout=settings.tools.python_timeout_seconds,
            python_memory_mb=settings.tools.python_memory_limit_mb,
            optional_tools=settings.tools.optional_tools,
            enabled_tools=settings.tools.enabled_tools,
            enable_python_tool=settings.tools.enable_python_tool,
        )

        self.workflow = AgentGraph(
            planner=Planner(
                llm=self.llm,
                model=settings.llm.primary_model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                use_llm=settings.agent.use_llm_for_planning,
            ),
            router=ToolRouter(registry=self.registry),
            executor=Executor(registry=self.registry),
            observer=ObservationProcessor(),
            reflector=Reflector(
                max_retries=settings.retries.max_retries,
                llm=self.llm,
                model=settings.llm.primary_model,
                temperature=settings.llm.temperature,
                max_tokens=min(256, settings.llm.max_tokens),
                use_llm=settings.agent.use_llm_for_planning,
                retry_backoff_seconds=settings.retries.backoff_seconds,
            ),
            responder=ResponseGenerator(
                llm=self.llm,
                model=settings.llm.primary_model,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                use_llm=settings.agent.use_llm_for_response,
            ),
            errors=ErrorHandler(),
            logger=ExecutionLogger(tracer=tracer),
        )
        self.graph = self.workflow.compile()

    async def run(self, request: AgentRequest) -> AgentRunResult:
        run_id = f"{request.session_id}-{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()
        runtime_mode = self._resolve_runtime_mode()

        state = AgentState(
            query=request.query,
            reasoning_mode=request.reasoning_mode or self.settings.agent.reasoning_mode,
            max_iterations=self.settings.agent.max_iterations,
        )
        self.memory.append("user", request.query, run_id=run_id)
        memory_context = self.memory.context_for_query(
            request.query,
            top_k=self.settings.memory.memory_top_k,
        )
        semantic_hits = memory_context.get("semantic", [])
        if semantic_hits:
            top_hit = semantic_hits[0]
            state.observations.append(f"memory_recall: {top_hit.get('text', '')}")
            self.metrics.inc("memory.semantic_hit_count", len(semantic_hits))

        graph_input = {
            "run_id": run_id,
            "available_tools": [d.name for d in self.registry.discover()],
            "state": state.model_dump(mode="json"),
        }

        if runtime_mode == "fallback":
            final_state = await self._fallback_execute(state=state, run_id=run_id)
            final_state.thoughts.append("Fallback executor used (runtime mode: fallback)")
        elif runtime_mode == "graph":
            final_state = await self._execute_graph_with_optional_fallback(
                graph_input=graph_input,
                state=state,
                run_id=run_id,
                allow_fallback=self.settings.agent.graph_fallback_on_error,
            )
        else:
            final_state = await self._execute_graph_with_optional_fallback(
                graph_input=graph_input,
                state=state,
                run_id=run_id,
                allow_fallback=True,
            )

        self.memory.append("assistant", final_state.final_answer, run_id=run_id)
        latency = (time.perf_counter() - started) * 1000
        self.metrics.observe_ms("run.total_latency_ms", latency)

        metrics_snapshot = self.metrics.snapshot()
        final_state.metrics["total_latency_ms"] = latency

        return AgentRunResult(
            answer=final_state.final_answer,
            success=final_state.error is None,
            plan=[step.model_dump(mode="json") for step in final_state.plan],
            tool_calls=[call.model_dump(mode="json") for call in final_state.tool_calls],
            observations=final_state.observations,
            reflection=final_state.reflection,
            error=final_state.error,
            metrics={
                "total_latency_ms": latency,
                "tool_calls": float(len(final_state.tool_calls)),
                "iterations": float(final_state.iteration),
            }
            | {k: float(v) for k, v in metrics_snapshot["counters"].items()},
            citations=final_state.citations,
        )

    async def _fallback_execute(self, state: AgentState, run_id: str) -> AgentState:
        """Deterministic fallback execution path when graph runtime stalls."""

        state = await self.workflow.planner.build_plan(
            state,
            available_tools=[d.name for d in self.registry.discover()],
        )
        for _ in range(state.max_iterations):
            state = self.workflow.router.route(state)
            state = await self.workflow.executor.execute_current_step(state, run_id=run_id)
            state = self.workflow.observer.process(state)
            state = await self.workflow.reflector.reflect(state)
            state = self.workflow.errors.apply(state)
            if not state.should_continue():
                break
        state = await self.workflow.responder.generate(state)
        state.done = True
        return state

    def _resolve_runtime_mode(self) -> str:
        mode = self.settings.agent.runtime_mode
        if mode in {"graph", "fallback", "auto"}:
            return mode
        return "auto"

    async def _execute_graph_with_optional_fallback(
        self,
        graph_input: dict[str, object],
        state: AgentState,
        run_id: str,
        allow_fallback: bool,
    ) -> AgentState:
        try:
            output = await asyncio.wait_for(
                self.graph.ainvoke(graph_input),
                timeout=self.settings.agent.graph_timeout_seconds,
            )
            return AgentState.model_validate(output["state"])
        except TimeoutError:
            self.metrics.inc("graph.timeout")
            if not allow_fallback:
                state.error = "LangGraph execution timed out"
                state.done = True
                return state
            fallback = await self._fallback_execute(state=state, run_id=run_id)
            fallback.thoughts.append("LangGraph timeout; fallback executor used")
            return fallback
        except Exception as exc:  # noqa: BLE001
            self.metrics.inc("graph.error")
            if not allow_fallback:
                state.error = f"LangGraph execution failed: {exc}"
                state.done = True
                return state
            fallback = await self._fallback_execute(state=state, run_id=run_id)
            fallback.thoughts.append(f"LangGraph error; fallback executor used: {exc}")
            return fallback
