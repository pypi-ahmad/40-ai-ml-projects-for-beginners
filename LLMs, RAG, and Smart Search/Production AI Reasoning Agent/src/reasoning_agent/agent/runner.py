"""Agent runner entrypoint."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from reasoning_agent.executor.executor import Executor
from reasoning_agent.graph.workflow import AgentComponents, build_workflow
from reasoning_agent.llm.ollama_client import OllamaClient
from reasoning_agent.memory import (
    ChromaMemoryStore,
    EmbeddingProvider,
    MemoryEvent,
    MemoryScope,
    MemoryService,
    SimpleMemoryStore,
)
from reasoning_agent.observability.logger import configure_logging
from reasoning_agent.observability.metrics import MetricCollector
from reasoning_agent.observability.trace_store import TraceStore
from reasoning_agent.observation.processor import ObservationProcessor
from reasoning_agent.planner.planner import Planner
from reasoning_agent.planner.reflection import Reflector
from reasoning_agent.recovery.error_handler import ErrorHandler
from reasoning_agent.response.generator import ResponseGenerator
from reasoning_agent.routing.tool_router import ToolRouter
from reasoning_agent.schemas import AgentResponse, ReasoningMode
from reasoning_agent.settings import Settings, load_settings
from reasoning_agent.tooling import ToolRegistry
from reasoning_agent.tooling.tools import register_default_tools


@dataclass
class RunnerArtifacts:
    """Runner supporting objects for diagnostics."""

    llm: OllamaClient
    registry: ToolRegistry
    trace_store: TraceStore
    memory: MemoryService


class AgentRunner:
    """Production-grade orchestrated reasoning agent."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        configure_logging(self.settings.log_dir, self.settings.log_level, self.settings.log_json)

        llm = OllamaClient(self.settings.ollama_base_url, self.settings.request_timeout_seconds)
        registry = ToolRegistry(workspace_root=".")

        embedding = EmbeddingProvider(llm=llm, embedding_model=self.settings.embedding_model)
        semantic = ChromaMemoryStore(
            chroma_dir=self.settings.chroma_dir,
            embedding_provider=embedding,
        )
        short_term = SimpleMemoryStore()
        memory = MemoryService(short_term=short_term, semantic=semantic)

        register_default_tools(registry=registry, settings=self.settings, memory=memory)

        self.artifacts = RunnerArtifacts(
            llm=llm,
            registry=registry,
            trace_store=TraceStore(base_dir=f"{self.settings.log_dir}/traces"),
            memory=memory,
        )

    def run(self, session_id: str, user_input: str, mode: ReasoningMode = ReasoningMode.REACT) -> AgentResponse:
        """Execute one full reasoning run."""

        run_id = str(uuid.uuid4())
        metrics = MetricCollector()

        components = AgentComponents(
            planner=Planner(
                llm=self.artifacts.llm,
                model=self.settings.primary_model,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            ),
            router=ToolRouter(
                llm=self.artifacts.llm,
                model=self.settings.primary_model,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            ),
            executor=Executor(self.artifacts.registry),
            observer=ObservationProcessor(self.artifacts.memory),
            reflector=Reflector(
                llm=self.artifacts.llm,
                model=self.settings.primary_model,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            ),
            responder=ResponseGenerator(
                llm=self.artifacts.llm,
                model=self.settings.primary_model,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            ),
            errors=ErrorHandler(),
            metrics=metrics,
        )

        workflow = build_workflow(components, self.artifacts.registry.metadata())

        initial_state = {
            "session_id": session_id,
            "run_id": run_id,
            "mode": mode,
            "user_input": user_input,
            "plan_steps": [],
            "current_step": 0,
            "thought_summary": "",
            "selected_tool": "",
            "tool_args": {},
            "observations": [],
            "trace": [],
            "answer": "",
            "citations": [],
            "retries": 0,
            "max_retries": self.settings.max_retries,
            "iteration": 0,
            "max_iterations": self.settings.max_iterations,
            "done": False,
            "termination_reason": "",
            "errors": [],
            "metrics": {},
        }

        final_state = workflow.invoke(initial_state)

        self.artifacts.memory.write(
            MemoryEvent(
                session_id=session_id,
                run_id=run_id,
                scope=MemoryScope.CONVERSATION,
                text=f"User: {user_input}\nAssistant: {final_state.get('answer', '')}",
                metadata={"mode": mode.value, "termination": final_state.get("termination_reason", "")},
            )
        )

        trace_payload = {
            "session_id": session_id,
            "run_id": run_id,
            "input": user_input,
            "mode": mode.value,
            "termination_reason": final_state.get("termination_reason", ""),
            "trace": [item.model_dump() for item in final_state.get("trace", [])],
            "metrics": final_state.get("metrics", {}),
        }
        self.artifacts.trace_store.save(run_id, trace_payload)

        return AgentResponse(
            session_id=session_id,
            run_id=run_id,
            answer=final_state.get("answer", ""),
            mode=mode,
            success=final_state.get("termination_reason", "") in {"completed", "max_iterations"},
            iterations=int(final_state.get("iteration", 0)),
            termination_reason=final_state.get("termination_reason", "error"),
            trace=final_state.get("trace", []),
            citations=final_state.get("citations", []),
            metrics=final_state.get("metrics", {}),
        )

    def close(self) -> None:
        """Close long-lived resources."""

        self.artifacts.llm.close()
