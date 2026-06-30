"""LangGraph node implementations."""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import UTC, datetime
from typing import Any

from langgraph_platform.agents.registry import get_agent
from langgraph_platform.config.settings import AppConfig
from langgraph_platform.engine.llm import OllamaClient
from langgraph_platform.engine.router import decide_routing
from langgraph_platform.memory.sqlite_store import SQLiteStore
from langgraph_platform.monitoring.system import SystemMonitor
from langgraph_platform.rag.pipeline import RAGPipeline
from langgraph_platform.state.models import (
    AgentOutput,
    Citation,
    NodeStatus,
    RetrievedDocument,
    VerificationStatus,
    WorkflowState,
)
from langgraph_platform.tools.base import ToolRegistry
from langgraph_platform.tools.builtin import build_default_registry


class NodeRuntime:
    """Holds dependencies used by node handlers."""

    def __init__(
        self,
        config: AppConfig,
        sqlite_store: SQLiteStore,
        rag_pipeline: RAGPipeline,
        tool_registry: ToolRegistry | None = None,
        llm_client: OllamaClient | None = None,
        monitor: SystemMonitor | None = None,
    ) -> None:
        self.config = config
        self.sqlite_store = sqlite_store
        self.rag_pipeline = rag_pipeline
        self.tool_registry = tool_registry or build_default_registry(
            chroma_path=config.memory.chroma_path,
            db_path=config.memory.sqlite_path,
        )
        self.llm_client = llm_client or OllamaClient()
        self.monitor = monitor or SystemMonitor(
            enable_gpu_metrics=config.monitoring.enable_gpu_metrics
        )

    def close(self) -> None:
        self.llm_client.close()
        close_fn = getattr(self.tool_registry, "close", None)
        if callable(close_fn):
            close_fn()


def _deserialize(state: dict[str, Any]) -> WorkflowState:
    return WorkflowState.model_validate(state)


def _serialize(state: WorkflowState) -> dict[str, Any]:
    state.execution_metadata.updated_at = datetime.now(UTC)
    return state.model_dump(mode="json")


def _persist_state(runtime: NodeRuntime, state: WorkflowState, node_name: str) -> None:
    runtime.sqlite_store.save_workflow_state(
        workflow_id=state.execution_metadata.workflow_id,
        node_name=node_name,
        state=state,
    )


def _mark_node_start(runtime: NodeRuntime, state: WorkflowState, node_name: str) -> None:
    runtime.monitor.start_node(node_name)
    state.execution_metadata.active_node = node_name
    state.execution_metadata.node_status[node_name] = NodeStatus.RUNNING


def _mark_node_end(
    runtime: NodeRuntime, state: WorkflowState, node_name: str, failed: bool = False
) -> None:
    duration_ms = runtime.monitor.stop_node(node_name)
    state.execution_metadata.node_durations_ms[node_name] = duration_ms
    state.execution_metadata.node_status[node_name] = (
        NodeStatus.FAILED if failed else NodeStatus.SUCCESS
    )


def _safe_json(data: str) -> dict[str, Any]:
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}


def planner_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Planner: task decomposition + dynamic routing."""

    state = _deserialize(state_dict)
    node = "planner"
    _mark_node_start(runtime, state, node)

    agent = get_agent("planner")
    prompt = f"User request:\n{state.user_request}\n\nReturn JSON matching schema."
    fallback_models = [runtime.config.model.planner] + runtime.config.model.fallback_chain
    parsed, response = runtime.llm_client.json_with_fallback(
        prompt=prompt,
        model_chain=fallback_models,
        system=agent.system_prompt,
    )

    state.execution_plan = str(
        parsed.get("plan", "Create enterprise report with citations and QA.")
    )
    state.subtasks = parsed.get("subtasks", []) or [
        "Collect evidence",
        "Run analysis",
        "Write report",
        "Verify output",
    ]
    state.routing = decide_routing(state, runtime.config.routing)
    if isinstance(parsed.get("routing"), dict):
        routing = parsed["routing"]
        state.routing.require_web_search = bool(
            routing.get("web", state.routing.require_web_search)
        )
        state.routing.require_rag = bool(routing.get("rag", state.routing.require_rag))
        state.routing.require_memory = bool(routing.get("memory", state.routing.require_memory))
        state.routing.require_code_execution = bool(
            routing.get("code", state.routing.require_code_execution)
        )
        state.routing.require_verification = bool(
            routing.get("verification", state.routing.require_verification)
        )

    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content=state.execution_plan,
        structured=parsed,
        confidence=float(parsed.get("confidence", 0.7)),
    )
    state.token_usage.input_tokens += response.prompt_tokens
    state.token_usage.output_tokens += response.completion_tokens
    state.token_usage.total_tokens += response.prompt_tokens + response.completion_tokens
    state.token_usage.by_model[response.model] = (
        state.token_usage.by_model.get(response.model, 0)
        + response.prompt_tokens
        + response.completion_tokens
    )

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def _parallel_research_sync(
    state: WorkflowState, runtime: NodeRuntime
) -> tuple[list[dict[str, Any]], list[RetrievedDocument], list[Citation]]:
    callables: list[tuple[str, Any]] = []
    if state.routing.require_web_search:
        callables.append(
            (
                "duckduckgo_search",
                lambda: runtime.tool_registry.run(
                    "duckduckgo_search",
                    {"query": state.user_request, "max_results": 6},
                ),
            )
        )
    callables.append(
        (
            "github_search",
            lambda: runtime.tool_registry.run(
                "github_search", {"query": state.user_request, "max_results": 4}
            ),
        )
    )
    callables.append(
        (
            "documentation_search",
            lambda: runtime.tool_registry.run(
                "documentation_search",
                {"query": state.user_request, "path": "docs", "limit": 4},
            ),
        )
    )
    if state.routing.require_memory:
        callables.append(
            (
                "memory_search",
                lambda: runtime.tool_registry.run(
                    "memory_search",
                    {"query": state.user_request, "limit": runtime.config.memory.top_k},
                ),
            )
        )

    tool_results: list[Any] = []
    if callables:
        result_queue: queue.Queue[Any] = queue.Queue()

        def _worker(name: str, fn: Any) -> None:
            try:
                result_queue.put(fn())
            except Exception as exc:
                result_queue.put(exc)

        threads: list[threading.Thread] = []
        for name, fn in callables:
            thread = threading.Thread(target=_worker, args=(name, fn), daemon=True)
            threads.append(thread)
            thread.start()

        deadline = time.monotonic() + 15
        for thread in threads:
            remaining = deadline - time.monotonic()
            thread.join(max(0.0, remaining))

        while not result_queue.empty():
            tool_results.append(result_queue.get())
        if len(tool_results) < len(callables):
            tool_results.append(TimeoutError("Parallel research timed out for one or more tools"))

    collected_results: list[dict[str, Any]] = []
    citations: list[Citation] = []

    for item in tool_results:
        if item is None:
            continue
        if isinstance(item, Exception):
            collected_results.append({"error": str(item)})
            continue

        if item.ok:
            if isinstance(item.output, list):
                collected_results.extend(item.output)
            else:
                collected_results.append(item.output)

            if isinstance(item.output, list):
                for entry in item.output[:4]:
                    title = str(entry.get("title", entry.get("name", "source")))
                    url = entry.get("href") or entry.get("url")
                    snippet = str(entry.get("body", entry.get("snippet", "")))
                    citations.append(
                        Citation(
                            source_id=f"{item.source}:{title[:24]}",
                            title=title,
                            url=str(url) if url else None,
                            snippet=snippet[:300],
                            confidence=0.6,
                        )
                    )

    retrieved_docs: list[RetrievedDocument] = []
    if state.routing.require_rag:
        retrieved_docs = runtime.rag_pipeline.retrieve(
            state.user_request, top_k=runtime.config.memory.top_k
        )

    return collected_results, retrieved_docs, citations


def parallel_research_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Parallel knowledge gathering: web/github/docs/memory + optional RAG."""

    state = _deserialize(state_dict)
    node = "parallel_research"
    _mark_node_start(runtime, state, node)

    results, rag_docs, citations = _parallel_research_sync(state, runtime)

    state.search_results.extend(results)
    state.retrieved_documents.extend(rag_docs)
    state.citations.extend(citations)
    state.memory.extend(
        [result for result in results if isinstance(result, dict) and "workflow_id" in result]
    )

    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content=f"Collected {len(results)} search items and {len(rag_docs)} rag docs",
        structured={"search_count": len(results), "rag_count": len(rag_docs)},
        confidence=0.68,
    )

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def knowledge_merge_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Merge evidence and deduplicate."""

    state = _deserialize(state_dict)
    node = "knowledge_merge"
    _mark_node_start(runtime, state, node)

    seen_keys: set[str] = set()
    merged_lines: list[str] = []

    for item in state.search_results:
        if not isinstance(item, dict):
            continue
        key = str(item.get("href") or item.get("url") or item.get("title") or item)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        title = str(item.get("title", "result"))
        body = str(item.get("body", item.get("snippet", "")))
        merged_lines.append(f"- {title}: {body[:240]}")

    for doc in state.retrieved_documents:
        merged_lines.append(f"- [RAG] {doc.source}: {doc.content[:240]}")

    merged_content = "\n".join(merged_lines[:120])
    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content=merged_content,
        structured={"merged_items": len(merged_lines)},
        confidence=0.72,
    )

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def consensus_reasoning_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Consensus reasoning with three independent drafts."""

    state = _deserialize(state_dict)
    node = "consensus_reasoning"
    _mark_node_start(runtime, state, node)

    context = state.intermediate_outputs.get(
        "knowledge_merge", AgentOutput(agent_name="x", content="")
    ).content
    candidates: list[str] = []
    models = [
        runtime.config.model.writer,
        runtime.config.model.researcher,
        runtime.config.model.verifier,
    ]

    for model in models:
        parsed, _ = runtime.llm_client.json_with_fallback(
            prompt=f"Context:\n{context}\n\nGenerate concise reasoning draft with key claims.",
            model_chain=[model] + runtime.config.model.fallback_chain,
            system='Return JSON: {"draft": str, "confidence": float}',
        )
        candidates.append(str(parsed.get("draft", parsed.get("content", ""))))

    best = max(candidates, key=len) if candidates else ""
    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content=best,
        structured={"candidates": candidates},
        confidence=0.74,
    )

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def writer_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Generate report draft."""

    state = _deserialize(state_dict)
    node = "writer"
    _mark_node_start(runtime, state, node)

    consensus = state.intermediate_outputs.get(
        "consensus_reasoning", AgentOutput(agent_name="x", content="")
    ).content
    context = state.intermediate_outputs.get(
        "knowledge_merge", AgentOutput(agent_name="x", content="")
    ).content
    prompt = (
        f"User request:\n{state.user_request}\n\n"
        f"Plan:\n{state.execution_plan}\n\n"
        f"Context:\n{context}\n\nConsensus reasoning:\n{consensus}\n\n"
        "Write enterprise report in markdown with sections: Executive Summary, Findings, Risks, Recommendations, Citations."
    )

    response = runtime.llm_client.generate_with_fallback(
        prompt=prompt,
        model_chain=[runtime.config.model.writer] + runtime.config.model.fallback_chain,
        system="You are report writer. Use factual, cited style.",
    )

    report = response.text.strip() or "# Report\n\nNo report generated."
    state.reports.append(report)
    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content=report,
        structured={"length": len(report)},
        confidence=0.78,
    )
    state.confidence_score = max(state.confidence_score, 0.78)

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def fact_checker_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Fact checking against available citations/documents."""

    state = _deserialize(state_dict)
    node = "fact_checker"
    _mark_node_start(runtime, state, node)

    report = state.reports[-1] if state.reports else ""
    has_citations = len(state.citations) > 0 or "http" in report
    if has_citations and len(report) > 200:
        status = VerificationStatus.PASSED
        issues: list[str] = []
        confidence = 0.81
    else:
        status = VerificationStatus.NEEDS_REVIEW
        issues = ["Insufficient citation evidence"]
        confidence = 0.55

    state.verification_status = status
    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content="\n".join(issues) if issues else "Verification passed.",
        structured={"status": status.value, "issues": issues},
        confidence=confidence,
    )
    state.confidence_score = (state.confidence_score + confidence) / 2

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def reflection_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Reflection loop node to improve quality."""

    state = _deserialize(state_dict)
    node = "reflection"
    _mark_node_start(runtime, state, node)

    retries = state.execution_metadata.retries.get("reflection_loop", 0)
    report = state.reports[-1] if state.reports else ""

    parsed, _ = runtime.llm_client.json_with_fallback(
        prompt=f"Review report and suggest concise improvements:\n{report[:6000]}",
        model_chain=[runtime.config.model.verifier] + runtime.config.model.fallback_chain,
        system='Return JSON {"improvements": list[str], "confidence": float}',
    )
    improvements = parsed.get("improvements", [])
    confidence = float(parsed.get("confidence", 0.7))

    if improvements and state.reports:
        state.reports[-1] = (
            state.reports[-1]
            + "\n\n## Reflection Improvements\n"
            + "\n".join(f"- {item}" for item in improvements[:8])
        )

    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content="\n".join(improvements) if improvements else "No improvements.",
        structured={"improvements": improvements},
        confidence=confidence,
        retries=retries,
    )
    state.execution_metadata.retries["reflection_loop"] = retries + 1
    state.confidence_score = max(state.confidence_score, confidence)

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def critic_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Critic node for red-team checks."""

    state = _deserialize(state_dict)
    node = "critic"
    _mark_node_start(runtime, state, node)

    report = state.reports[-1] if state.reports else ""
    risks = []
    if "assumption" not in report.lower():
        risks.append("Report missing explicit assumptions section.")
    if len(state.citations) < 2:
        risks.append("Low citation count.")

    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content="\n".join(risks) if risks else "No major risks.",
        structured={"risks": risks},
        confidence=0.65 if risks else 0.8,
    )

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def qa_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """QA gate over report and workflow state."""

    state = _deserialize(state_dict)
    node = "qa"
    _mark_node_start(runtime, state, node)

    issues = []
    if not state.reports:
        issues.append("No report generated")
    if state.verification_status == VerificationStatus.FAILED:
        issues.append("Fact checker failed")
    if state.confidence_score < runtime.config.retry.confidence_threshold:
        issues.append("Confidence below threshold")

    qa_status = "passed" if not issues else "needs_review"
    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content="\n".join(issues) if issues else "QA passed.",
        structured={"qa_status": qa_status, "issues": issues},
        confidence=0.82 if not issues else 0.58,
    )

    if issues:
        state.verification_status = VerificationStatus.NEEDS_REVIEW

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def citation_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Normalize citation list and append to report."""

    state = _deserialize(state_dict)
    node = "citation"
    _mark_node_start(runtime, state, node)

    dedup: dict[str, Citation] = {}
    for citation in state.citations:
        dedup[citation.source_id] = citation
    state.citations = list(dedup.values())

    if state.reports and state.citations:
        citation_lines = [
            f"- [{idx + 1}] {c.title} ({c.url or c.source_id})"
            for idx, c in enumerate(state.citations)
        ]
        state.reports[-1] = state.reports[-1] + "\n\n## Citations\n" + "\n".join(citation_lines)

    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content=f"Normalized {len(state.citations)} citations",
        structured={"citation_count": len(state.citations)},
        confidence=0.83,
    )

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def supervisor_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Supervisor final gate with confidence + verification checks."""

    state = _deserialize(state_dict)
    node = "supervisor"
    _mark_node_start(runtime, state, node)

    qa_output = state.intermediate_outputs.get("qa")
    qa_status = qa_output.structured.get("qa_status") if qa_output else "needs_review"
    approve = (
        qa_status == "passed"
        and state.verification_status in {VerificationStatus.PASSED, VerificationStatus.UNKNOWN}
        and state.confidence_score >= runtime.config.retry.confidence_threshold
    )

    if approve:
        state.verification_status = VerificationStatus.PASSED
    elif state.verification_status == VerificationStatus.UNKNOWN:
        state.verification_status = VerificationStatus.NEEDS_REVIEW

    state.intermediate_outputs[node] = AgentOutput(
        agent_name=node,
        content="Approved" if approve else "Needs retry",
        structured={"approve": approve, "qa_status": qa_status},
        confidence=state.confidence_score,
    )

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def finalize_node(state_dict: dict[str, Any], runtime: NodeRuntime) -> dict[str, Any]:
    """Finalize workflow run in persistence layer."""

    state = _deserialize(state_dict)
    node = "finalize"
    _mark_node_start(runtime, state, node)

    final_report = state.reports[-1] if state.reports else "No report generated."
    state.execution_metadata.finished_at = datetime.now(UTC)
    runtime.sqlite_store.finalize_workflow(state=state, final_report=final_report)

    _mark_node_end(runtime, state, node)
    _persist_state(runtime, state, node)
    return _serialize(state)


def parse_node_result(raw_result: dict[str, Any]) -> WorkflowState:
    """Convert graph output dict to typed state."""

    return WorkflowState.model_validate(raw_result)
