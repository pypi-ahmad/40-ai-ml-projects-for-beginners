"""LangGraph workflow with specialized internet-agent nodes."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from internet_agent.agent.state import AgentState
from internet_agent.config import Settings
from internet_agent.llm.client import OllamaClient
from internet_agent.memory.chroma_store import ChromaMemoryStore
from internet_agent.memory.repository import MemoryRepository
from internet_agent.metrics import METRICS
from internet_agent.retrieval.pipeline import RetrievalPipeline
from internet_agent.tools.registry import ToolRegistry


def _load_prompt(name: str) -> str:
    path = Path("src/internet_agent/prompts") / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


class InternetAgentWorkflow:
    """Multi-agent LangGraph orchestration runtime."""

    def __init__(
        self,
        settings: Settings,
        llm: OllamaClient,
        memory_repo: MemoryRepository,
        semantic_store: ChromaMemoryStore,
        tool_registry: ToolRegistry,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.memory_repo = memory_repo
        self.semantic_store = semantic_store
        self.tool_registry = tool_registry
        self.retrieval = RetrievalPipeline(settings, memory_repo, semantic_store)
        self.graph = self._compile()

    def _compile(self):
        graph = StateGraph(dict)
        graph.add_node("user_intent_agent", self.user_intent_agent)
        graph.add_node("planner_agent", self.planner_agent)
        graph.add_node("search_decision_agent", self.search_decision_agent)
        graph.add_node("search_agent", self.search_agent)
        graph.add_node("web_extraction_agent", self.web_extraction_agent)
        graph.add_node("summarization_agent", self.summarization_agent)
        graph.add_node("verification_agent", self.verification_agent)
        graph.add_node("memory_agent", self.memory_agent)
        graph.add_node("reflection_agent", self.reflection_agent)
        graph.add_node("report_agent", self.report_agent)

        graph.set_entry_point("user_intent_agent")
        graph.add_edge("user_intent_agent", "planner_agent")
        graph.add_edge("planner_agent", "search_decision_agent")

        graph.add_conditional_edges(
            "search_decision_agent",
            self._route_after_search_decision,
            {
                "search": "search_agent",
                "summarize": "summarization_agent",
            },
        )
        graph.add_edge("search_agent", "web_extraction_agent")
        graph.add_edge("web_extraction_agent", "summarization_agent")
        graph.add_edge("summarization_agent", "verification_agent")

        graph.add_conditional_edges(
            "verification_agent",
            self._route_after_verification,
            {
                "retry_search": "search_agent",
                "memory": "memory_agent",
            },
        )

        graph.add_edge("memory_agent", "reflection_agent")
        graph.add_edge("reflection_agent", "report_agent")
        graph.add_edge("report_agent", END)

        checkpointer = None
        if self.settings.agent.checkpointer_enabled:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

                checkpointer = SqliteSaver.from_conn_string(
                    "sqlite:///./artifacts/graph_checkpoints.db"
                )
            except Exception:
                checkpointer = None

        return graph.compile(checkpointer=checkpointer)

    async def run(self, session_id: str, query: str) -> AgentState:
        state = AgentState(session_id=session_id, user_query=query)
        payload = {"state": state.model_dump(mode="json")}

        payload = await self.user_intent_agent(payload)
        payload = await self.planner_agent(payload)
        payload = await self.search_decision_agent(payload)
        current = AgentState.model_validate(payload["state"])

        if current.need_internet:
            while True:
                payload = await self.search_agent(payload)
                payload = await self.web_extraction_agent(payload)
                payload = await self.summarization_agent(payload)
                payload = await self.verification_agent(payload)
                current = AgentState.model_validate(payload["state"])
                if not current.can_retry_search(self.settings.agent.max_verification_loops):
                    break
        else:
            payload = await self.summarization_agent(payload)
            payload = await self.verification_agent(payload)

        payload = await self.memory_agent(payload)
        payload = await self.reflection_agent(payload)
        payload = await self.report_agent(payload)
        return AgentState.model_validate(payload["state"])

    def _compile_without_checkpointer(self):
        graph = StateGraph(dict)
        graph.add_node("user_intent_agent", self.user_intent_agent)
        graph.add_node("planner_agent", self.planner_agent)
        graph.add_node("search_decision_agent", self.search_decision_agent)
        graph.add_node("search_agent", self.search_agent)
        graph.add_node("web_extraction_agent", self.web_extraction_agent)
        graph.add_node("summarization_agent", self.summarization_agent)
        graph.add_node("verification_agent", self.verification_agent)
        graph.add_node("memory_agent", self.memory_agent)
        graph.add_node("reflection_agent", self.reflection_agent)
        graph.add_node("report_agent", self.report_agent)

        graph.set_entry_point("user_intent_agent")
        graph.add_edge("user_intent_agent", "planner_agent")
        graph.add_edge("planner_agent", "search_decision_agent")
        graph.add_conditional_edges(
            "search_decision_agent",
            self._route_after_search_decision,
            {"search": "search_agent", "summarize": "summarization_agent"},
        )
        graph.add_edge("search_agent", "web_extraction_agent")
        graph.add_edge("web_extraction_agent", "summarization_agent")
        graph.add_edge("summarization_agent", "verification_agent")
        graph.add_conditional_edges(
            "verification_agent",
            self._route_after_verification,
            {"retry_search": "search_agent", "memory": "memory_agent"},
        )
        graph.add_edge("memory_agent", "reflection_agent")
        graph.add_edge("reflection_agent", "report_agent")
        graph.add_edge("report_agent", END)
        return graph.compile()

    async def user_intent_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.model_validate(payload["state"])
        q = state.user_query.lower()
        if any(token in q for token in ("weather", "temperature", "forecast")):
            state.intent = "weather_lookup"
        elif any(token in q for token in ("convert", "currency", "exchange")):
            state.intent = "currency_or_unit"
        elif any(token in q for token in ("latest", "today", "news", "current", "yesterday")):
            state.intent = "time_sensitive_search"
        else:
            state.intent = "general_qa"
        state.reasoning_trace.append({"agent": "user_intent_agent", "intent": state.intent})
        return {"state": state.model_dump(mode="json")}

    async def planner_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.model_validate(payload["state"])
        query = state.user_query.lower()
        plan_steps = ["understand query", "decide search", "generate answer", "verify"]
        planned_tools: list[dict[str, Any]] = []

        if "weather" in query:
            plan_steps.insert(2, "call weather tool")
            planned_tools.append({"tool": "weather", "input": {"location": state.user_query}})
        if "currency" in query and "to" in query:
            plan_steps.insert(2, "call currency_exchange tool")
        if "calculate" in query:
            plan_steps.insert(2, "call calculator tool")

        state.plan_steps = plan_steps
        state.planned_tools = planned_tools
        state.reasoning_trace.append({"agent": "planner_agent", "plan_steps": plan_steps})
        return {"state": state.model_dump(mode="json")}

    async def search_decision_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.model_validate(payload["state"])
        cached_hits = self.semantic_store.query(state.user_query, top_k=2)
        has_useful_cache = any(hit.get("content") for hit in cached_hits)

        time_sensitive = any(
            token in state.user_query.lower()
            for token in ("latest", "today", "current", "yesterday", "news", "this week")
        )

        if has_useful_cache and not time_sensitive:
            state.need_internet = False
            state.selected_providers = []
            state.semantic_hits = cached_hits
            state.reasoning_trace.append(
                {
                    "agent": "search_decision_agent",
                    "need_internet": False,
                    "reason": "semantic cache sufficient",
                }
            )
            return {"state": state.model_dump(mode="json")}

        prompt = _load_prompt("search_planning.md")
        fallback = {
            "need_internet": state.intent in {"time_sensitive_search"},
            "reason": "heuristic",
            "providers": ["duckduckgo", "news"],
        }
        decision = await self.llm.ask_json(
            task_model=self.settings.llm.planning_model,
            system_prompt=prompt,
            user_prompt=f"User query: {state.user_query}",
            fallback=fallback,
        )

        state.need_internet = bool(decision.get("need_internet", fallback["need_internet"]))
        if state.need_internet:
            providers = decision.get("providers") or self.settings.search.providers
            state.selected_providers = [str(p) for p in providers]
        else:
            state.selected_providers = []

        state.reasoning_trace.append(
            {
                "agent": "search_decision_agent",
                "need_internet": state.need_internet,
                "providers": state.selected_providers,
                "reason": decision.get("reason", ""),
            }
        )
        return {"state": state.model_dump(mode="json")}

    async def search_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.model_validate(payload["state"])

        tool_outputs: list[dict[str, Any]] = []
        for tool_call in state.planned_tools:
            name = tool_call.get("tool", "")
            tool_input = tool_call.get("input", {})
            if self.tool_registry.has(name):
                out = await self.tool_registry.invoke(state.session_id, name, tool_input)
                tool_outputs.append(out)

        retrieval = await self.retrieval.run(
            session_id=state.session_id,
            query=state.user_query,
            providers=state.selected_providers,
        )

        state.search_results = retrieval.get("results", [])
        state.semantic_hits = retrieval.get("semantic_hits", [])
        state.retrieved_documents = retrieval.get("documents", [])
        state.retrieved_chunks = retrieval.get("chunks", [])
        state.tool_outputs.extend(tool_outputs)

        for provider, rows in zip(
            retrieval.get("providers", []),
            [retrieval.get("results", [])],
            strict=False,
        ):
            self.memory_repo.add_search_record(
                session_id=state.session_id,
                query=state.user_query,
                provider=provider,
                results={"rows": rows},
            )

        state.reasoning_trace.append(
            {
                "agent": "search_agent",
                "num_results": len(state.search_results),
                "num_documents": len(state.retrieved_documents),
            }
        )
        return {"state": state.model_dump(mode="json")}

    async def web_extraction_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.model_validate(payload["state"])
        docs = state.retrieved_documents
        citations = []
        for row in docs[: self.settings.agent.top_k_sources]:
            citations.append(
                {
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "source": row.get("source", "web"),
                    "score": next(
                        (
                            r.get("rank_score", 0.0)
                            for r in state.search_results
                            if r.get("url") == row.get("url")
                        ),
                        0.0,
                    ),
                }
            )
        state.citations = citations
        state.reasoning_trace.append({"agent": "web_extraction_agent", "citations": len(citations)})
        return {"state": state.model_dump(mode="json")}

    async def summarization_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.model_validate(payload["state"])
        prompt = _load_prompt("summarization.md")

        if state.need_internet and state.retrieved_chunks:
            context = "\n\n".join(chunk["content"] for chunk in state.retrieved_chunks[:8])
            memory_context = "\n\n".join(hit.get("content", "") for hit in state.semantic_hits[:3])
            user_prompt = (
                f"Question: {state.user_query}\n\n"
                f"Context:\n{context[:10000]}\n\n"
                f"Semantic memory:\n{memory_context[:2000]}\n\n"
                "Write concise answer with key findings and explicit uncertainty when needed."
            )
            answer = await self.llm.ask(
                task_model=self.settings.llm.summarization_model,
                system_prompt=prompt,
                user_prompt=user_prompt,
            )
        else:
            memory_context = "\n\n".join(hit.get("content", "") for hit in state.semantic_hits[:3])
            answer = await self.llm.ask(
                task_model=self.settings.llm.reasoning_model,
                system_prompt="You are a reliable AI assistant. Answer accurately and clearly.",
                user_prompt=(
                    f"Question: {state.user_query}\n\n"
                    f"Relevant memory:\n{memory_context[:2000]}"
                ),
            )

        state.draft_answer = answer
        state.reasoning_trace.append(
            {
                "agent": "summarization_agent",
                "answer_chars": len(state.draft_answer),
            }
        )
        return {"state": state.model_dump(mode="json")}

    async def verification_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.model_validate(payload["state"])
        prompt = _load_prompt("verification.md")
        context = "\n\n".join(chunk["content"] for chunk in state.retrieved_chunks[:6])

        fallback = {
            "confidence": 0.55 if state.need_internet else 0.8,
            "hallucination_risk": "medium" if state.need_internet else "low",
            "missing_info": [],
            "conflicts": [],
            "retry_search": state.need_internet and len(state.retrieved_chunks) < 2,
        }

        verification = await self.llm.ask_json(
            task_model=self.settings.llm.verification_model,
            system_prompt=prompt,
            user_prompt=(
                f"Question: {state.user_query}\n\n"
                f"Draft answer:\n{state.draft_answer}\n\n"
                f"Evidence:\n{context[:9000]}"
            ),
            fallback=fallback,
        )

        state.confidence = float(verification.get("confidence", fallback["confidence"]))
        state.hallucination_risk = str(
            verification.get("hallucination_risk", fallback["hallucination_risk"])
        )
        state.missing_info = [str(x) for x in verification.get("missing_info", [])]
        state.conflicts = [str(x) for x in verification.get("conflicts", [])]

        retry_search = bool(verification.get("retry_search", fallback["retry_search"]))
        if state.confidence < self.settings.agent.verification_confidence_threshold:
            retry_search = True

        state.retry_search = retry_search
        if state.retry_search:
            state.verification_loops += 1

        state.reasoning_trace.append(
            {
                "agent": "verification_agent",
                "confidence": state.confidence,
                "risk": state.hallucination_risk,
                "retry_search": state.retry_search,
                "loops": state.verification_loops,
            }
        )
        return {"state": state.model_dump(mode="json")}

    async def memory_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.model_validate(payload["state"])

        self.memory_repo.add_message(state.session_id, "user", state.user_query)
        self.memory_repo.add_message(state.session_id, "assistant", state.draft_answer)
        self.memory_repo.add_summary(
            session_id=state.session_id,
            query=state.user_query,
            summary=state.draft_answer,
            confidence=state.confidence,
        )

        if state.draft_answer.strip():
            item_id = hashlib.sha256(
                f"{state.session_id}:{state.user_query}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()
            self.semantic_store.upsert(
                ids=[item_id],
                texts=[state.draft_answer],
                metadatas=[{"type": "answer", "session_id": state.session_id}],
            )

        state.reasoning_trace.append({"agent": "memory_agent", "persisted": True})
        return {"state": state.model_dump(mode="json")}

    async def reflection_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.model_validate(payload["state"])
        prompt = _load_prompt("reflection.md")

        reflection = await self.llm.ask(
            task_model=self.settings.llm.reflection_model,
            system_prompt=prompt,
            user_prompt=(
                f"Question: {state.user_query}\n"
                f"Answer: {state.draft_answer}\n"
                f"Confidence: {state.confidence}\n"
                f"Missing: {state.missing_info}\n"
                f"Conflicts: {state.conflicts}"
            ),
            temperature=0.1,
        )

        state.reasoning_trace.append({"agent": "reflection_agent", "reflection": reflection})
        state.final_answer = state.draft_answer
        return {"state": state.model_dump(mode="json")}

    async def report_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.model_validate(payload["state"])
        report = {
            "session_id": state.session_id,
            "query": state.user_query,
            "answer": state.final_answer,
            "confidence": state.confidence,
            "hallucination_risk": state.hallucination_risk,
            "missing_info": state.missing_info,
            "conflicts": state.conflicts,
            "citations": state.citations,
            "tool_outputs": state.tool_outputs,
            "reasoning_trace": state.reasoning_trace,
            "timestamp": datetime.utcnow().isoformat(),
        }
        state.report_payload = report
        state.done = True
        METRICS.inc("agent.completed_runs")
        return {"state": state.model_dump(mode="json")}

    def _route_after_search_decision(self, payload: dict[str, Any]) -> str:
        state = AgentState.model_validate(payload["state"])
        return "search" if state.need_internet else "summarize"

    def _route_after_verification(self, payload: dict[str, Any]) -> str:
        state = AgentState.model_validate(payload["state"])
        if state.can_retry_search(self.settings.agent.max_verification_loops):
            return "retry_search"
        return "memory"
