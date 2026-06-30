"""LangGraph workflow for hybrid RAG orchestration."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from hybrid_research_assistant.llm import LLMJudge, OllamaLLM
from hybrid_research_assistant.prompts import build_context_block, build_messages
from hybrid_research_assistant.rerank import Reranker
from hybrid_research_assistant.retrieval import IntentRouter, RetrievalService
from hybrid_research_assistant.schemas import Citation, RetrievalMode

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None


@dataclass(slots=True)
class GraphComponents:
    """Dependencies injected into LangGraph nodes."""

    intent_router: IntentRouter
    retrieval: RetrievalService
    reranker: Reranker
    llm: OllamaLLM
    judge: LLMJudge
    fallback_text: str
    retrieval_top_k: int
    candidate_k: int


class FallbackWorkflow:
    """Pure-Python fallback workflow if LangGraph is unavailable."""

    def __init__(self, components: GraphComponents) -> None:
        self.components = components

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        state = intent_detection_node(state, self.components)
        route = state["route"]["mode"]
        if route == RetrievalMode.LOCAL.value:
            state = local_retrieve_node(state, self.components)
        elif route == RetrievalMode.WEB.value:
            state = web_retrieve_node(state, self.components)
        else:
            state = hybrid_retrieve_node(state, self.components)
        state = rerank_node(state, self.components)
        state = context_builder_node(state)
        state = generation_node(state, self.components)
        state = judge_node(state, self.components)
        state = response_node(state)
        return state


def _run_async_sync(coro):  # type: ignore[no-untyped-def]
    """Run async coroutine from sync code, including notebook event-loop contexts."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    holder: dict[str, Any] = {}

    def _runner() -> None:
        holder["value"] = asyncio.run(coro)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    return holder["value"]


def intent_detection_node(state: dict[str, Any], components: GraphComponents) -> dict[str, Any]:
    started = time.perf_counter()
    decision = components.intent_router.route(
        query=state["query"],
        requested_mode=RetrievalMode(state.get("requested_mode", RetrievalMode.AUTO.value)),
    )
    state["route"] = {
        "mode": decision.mode.value,
        "reason": decision.reason,
        "confidence": decision.confidence,
    }
    state.setdefault("timings", {})["intent_ms"] = (time.perf_counter() - started) * 1000
    return state


def local_retrieve_node(state: dict[str, Any], components: GraphComponents) -> dict[str, Any]:
    rows, latency_ms = components.retrieval.retrieve_local(
        state["query"],
        top_k=components.candidate_k,
        metadata_filter=state.get("metadata_filter"),
    )
    state["retrieved"] = [asdict(row) for row in rows]
    state.setdefault("timings", {})["retrieval_ms"] = latency_ms
    return state


def web_retrieve_node(state: dict[str, Any], components: GraphComponents) -> dict[str, Any]:
    rows, latency_ms = _run_async_sync(
        components.retrieval.retrieve_web(
            state["query"],
            top_k=components.candidate_k,
            provider=state.get("provider"),
        )
    )
    state["retrieved"] = [asdict(row) for row in rows]
    state.setdefault("timings", {})["retrieval_ms"] = latency_ms
    return state


def hybrid_retrieve_node(state: dict[str, Any], components: GraphComponents) -> dict[str, Any]:
    rows, latency_ms = _run_async_sync(
        components.retrieval.retrieve_hybrid(
            state["query"],
            local_k=components.candidate_k,
            web_k=components.candidate_k,
            metadata_filter=state.get("metadata_filter"),
            provider=state.get("provider"),
        )
    )
    state["retrieved"] = [asdict(row) for row in rows]
    state.setdefault("timings", {})["retrieval_ms"] = latency_ms
    return state


def rerank_node(state: dict[str, Any], components: GraphComponents) -> dict[str, Any]:
    started = time.perf_counter()
    retrieved = state.get("retrieved", [])
    if not retrieved:
        state["retrieved"] = []
        state.setdefault("timings", {})["rerank_ms"] = (time.perf_counter() - started) * 1000
        return state

    from hybrid_research_assistant.schemas import RetrievedContext  # noqa: PLC0415

    rows = [RetrievedContext(**row) for row in retrieved]
    ranked, report = components.reranker.rerank(
        query=state["query"],
        rows=rows,
        top_k=components.retrieval_top_k,
    )
    state["retrieved"] = [asdict(row) for row in ranked]
    state["rerank_report"] = {
        "before": report.before_scores,
        "after": report.after_scores,
    }
    state.setdefault("timings", {})["rerank_ms"] = report.latency_ms
    return state


def context_builder_node(state: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    retrieved = state.get("retrieved", [])
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    from hybrid_research_assistant.schemas import RetrievedContext  # noqa: PLC0415

    rows = [RetrievedContext(**row) for row in retrieved]
    for row in rows:
        source_file = str(row.metadata.get("source", ""))
        key = (source_file, row.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        citation = Citation(
            source_file=source_file,
            page_number=row.metadata.get("page_number"),
            url=row.metadata.get("url"),
            chunk_id=row.chunk_id,
            confidence=max(0.0, min(1.0, row.score)),
            title=row.metadata.get("document_title"),
        )
        citations.append(asdict(citation))

    state["citations"] = citations
    state["context_text"] = build_context_block(rows)
    state.setdefault("timings", {})["context_ms"] = (time.perf_counter() - started) * 1000
    return state


def generation_node(state: dict[str, Any], components: GraphComponents) -> dict[str, Any]:
    started = time.perf_counter()
    if not state.get("retrieved"):
        state["answer"] = components.fallback_text
        state.setdefault("timings", {})["generation_ms"] = (time.perf_counter() - started) * 1000
        return state

    from hybrid_research_assistant.schemas import RetrievedContext  # noqa: PLC0415

    rows = [RetrievedContext(**row) for row in state.get("retrieved", [])]
    messages = build_messages(
        query=state["query"],
        rows=rows,
        prompt_name=state.get("prompt_name", "research_assistant"),
        fallback=components.fallback_text,
    )
    try:
        answer, _gen_ms = components.llm.generate(messages)
        state["answer"] = answer
    except Exception as err:  # noqa: BLE001
        state["answer"] = components.fallback_text
        state.setdefault("errors", []).append(f"generation_failed: {err}")
    state.setdefault("timings", {})["generation_ms"] = (time.perf_counter() - started) * 1000
    return state


def judge_node(state: dict[str, Any], components: GraphComponents) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        judge, _judge_ms = components.judge.evaluate(
            query=state["query"],
            answer=state.get("answer", ""),
            context=state.get("context_text", ""),
        )
        state["judge"] = judge
    except Exception as err:  # noqa: BLE001
        state["judge"] = {"error": str(err)}
    state.setdefault("timings", {})["judge_ms"] = (time.perf_counter() - started) * 1000
    return state


def error_recovery_node(state: dict[str, Any], components: GraphComponents) -> dict[str, Any]:
    state["answer"] = components.fallback_text
    state.setdefault("errors", []).append("pipeline_recovery_triggered")
    return state


def response_node(state: dict[str, Any]) -> dict[str, Any]:
    timings = state.setdefault("timings", {})
    total_ms = sum(
        float(timings.get(key, 0.0))
        for key in ["intent_ms", "retrieval_ms", "rerank_ms", "context_ms", "generation_ms", "judge_ms"]
    )
    timings["total_ms"] = total_ms
    return state


def build_workflow(components: GraphComponents):
    """Build LangGraph workflow or local fallback."""

    if StateGraph is None:
        return FallbackWorkflow(components)

    graph = StateGraph(dict)
    graph.add_node("intent_detection", lambda state: intent_detection_node(state, components))
    graph.add_node("local_retrieve", lambda state: local_retrieve_node(state, components))
    graph.add_node("web_retrieve", lambda state: web_retrieve_node(state, components))
    graph.add_node("hybrid_retrieve", lambda state: hybrid_retrieve_node(state, components))
    graph.add_node("rerank", lambda state: rerank_node(state, components))
    graph.add_node("context_builder", context_builder_node)
    graph.add_node("generation", lambda state: generation_node(state, components))
    graph.add_node("judge", lambda state: judge_node(state, components))
    graph.add_node("error_recovery", lambda state: error_recovery_node(state, components))
    graph.add_node("response", response_node)

    graph.add_edge(START, "intent_detection")

    def route_to_retriever(state: dict[str, Any]) -> str:
        mode = state.get("route", {}).get("mode", RetrievalMode.LOCAL.value)
        if mode == RetrievalMode.WEB.value:
            return "web"
        if mode == RetrievalMode.HYBRID.value:
            return "hybrid"
        return "local"

    graph.add_conditional_edges(
        "intent_detection",
        route_to_retriever,
        {"local": "local_retrieve", "web": "web_retrieve", "hybrid": "hybrid_retrieve"},
    )

    graph.add_edge("local_retrieve", "rerank")
    graph.add_edge("web_retrieve", "rerank")
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "context_builder")
    graph.add_edge("context_builder", "generation")
    graph.add_edge("generation", "judge")
    graph.add_edge("judge", "response")
    graph.add_edge("error_recovery", "response")
    graph.add_edge("response", END)

    return graph.compile()
