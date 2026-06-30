"""Node implementations for LangGraph multi-agent workflow."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from api_intel_agent.analytics import AnalyticsEngine
from api_intel_agent.auth import AuthManager
from api_intel_agent.cache import CacheManager
from api_intel_agent.config import load_settings
from api_intel_agent.connectors import ConnectorRegistry
from api_intel_agent.core.schemas import ErrorRecord, MemorySearchRequest, OutputFormat
from api_intel_agent.llm import OllamaProvider
from api_intel_agent.memory import MemoryManager
from api_intel_agent.monitoring import MetricsCollector, ProgressTracker
from api_intel_agent.reporting import ReportGenerator
from api_intel_agent.tools import build_tool_registry
from api_intel_agent.utils.logging import configure_logging

from .state import GraphState, PlanStep


class RequestPlannerAgent:
    def __init__(self, connectors: ConnectorRegistry) -> None:
        self.connectors = connectors

    async def run(self, state: GraphState) -> GraphState:
        if state.request.apis:
            selected = [api for api in state.request.apis if api in self.connectors.list_names()]
        else:
            selected = self.connectors.infer_from_query(state.request.query)

        state.selected_apis = selected
        state.plan_steps = [
            PlanStep(provider=provider, purpose=f"Query {provider} for relevant data", parallel_group=0)
            for provider in selected
        ]
        return state


class APIRouterAgent:
    def __init__(self, connectors: ConnectorRegistry) -> None:
        self.connectors = connectors

    async def run(self, state: GraphState) -> GraphState:
        if not state.plan_steps:
            state.plan_steps = [
                PlanStep(provider=provider, purpose="Fallback routed API") for provider in state.selected_apis
            ]
        return state


class AuthenticationAgent:
    def __init__(self) -> None:
        self.auth = AuthManager()

    async def run(self, state: GraphState) -> GraphState:
        for provider in state.selected_apis:
            auth_headers = self.auth.auth_headers(provider)
            state.auth_status[provider] = auth_headers.status
        return state


class DataFetchAgent:
    def __init__(self, connectors: ConnectorRegistry, cache: CacheManager, metrics: MetricsCollector) -> None:
        self.connectors = connectors
        self.cache = cache
        self.metrics = metrics
        self.settings = load_settings()

    async def _run_connector(self, step: PlanStep, query: str, use_cache: bool) -> tuple[str, Any]:
        cache_key = f"{step.provider}:{query}:{step.params}"
        if use_cache:
            cached = self.cache.get("api_response", cache_key)
            if cached:
                self.metrics.record_cache(True)
                return step.provider, cached
            self.metrics.record_cache(False)

        connector = self.connectors.get(step.provider)
        result = await connector.execute(query=query, params=step.params)
        result_payload = result.model_dump(mode="json")
        if use_cache and result.status in {"ok", "empty"}:
            self.cache.set("api_response", cache_key, result_payload)
        if result.latency_ms is not None:
            self.metrics.record_api_latency(step.provider, result.latency_ms)
        return step.provider, result_payload

    async def run(self, state: GraphState) -> GraphState:
        grouped: dict[int, list[PlanStep]] = defaultdict(list)
        for step in state.plan_steps:
            grouped[step.parallel_group].append(step)

        for group in sorted(grouped):
            tasks = [self._run_connector(step, state.request.query, state.request.use_cache) for step in grouped[group]]
            for provider, payload in await asyncio.gather(*tasks):
                result = payload
                if isinstance(result, dict) and result.get("provider"):
                    from api_intel_agent.core.schemas import ConnectorResult

                    connector_result = ConnectorResult.model_validate(result)
                    state.connector_results.append(connector_result)
                else:
                    state.errors.append(
                        ErrorRecord(
                            code="BAD_CONNECTOR_RESULT",
                            message=f"Invalid response from {provider}",
                            provider=provider,
                            retryable=True,
                        )
                    )
        return state


class ValidationAgent:
    async def run(self, state: GraphState) -> GraphState:
        for result in state.connector_results:
            if result.status in {"failed", "skipped_missing_credentials"} and result.error:
                state.errors.append(result.error)
                continue
            valid = [item for item in result.records if isinstance(item, dict)]
            state.validated_records[result.provider] = valid
        return state


class ReasoningAgent:
    def __init__(self, llm: OllamaProvider) -> None:
        self.llm = llm

    async def run(self, state: GraphState) -> GraphState:
        summaries = []
        for provider, records in state.validated_records.items():
            summaries.append(f"{provider}: {len(records)} records")

        prompt = (
            "You are enterprise AI analyst. Produce concise summary, key insights, and recommendations.\n"
            f"User query: {state.request.query}\n"
            f"Data summaries: {'; '.join(summaries)}"
        )

        try:
            llm_response = await self.llm.generate(prompt, model=state.request.model)
            text = llm_response.get("response", "")
        except Exception:
            text = ""
        if not text:
            text = "Data collected. Some connectors may have partial, missing auth, or offline LLM analysis."

        state.reasoning_summary = text.strip()
        state.insights = [
            f"{provider} returned {len(records)} records"
            for provider, records in state.validated_records.items()
        ]
        if not state.insights:
            state.insights = ["No records available from selected APIs"]

        state.recommendations = [
            "Review top-ranked sources and validate trend consistency across APIs",
            "Enable credentials for skipped providers to increase confidence",
            "Schedule recurring report for longitudinal trend tracking",
        ]
        return state


class ReportGeneratorAgent:
    def __init__(self, analytics: AnalyticsEngine, reporter: ReportGenerator) -> None:
        self.analytics = analytics
        self.reporter = reporter

    async def run(self, state: GraphState) -> GraphState:
        github_records = state.validated_records.get("github", [])
        repo_rankings = self.analytics.repository_rankings(github_records)
        language_dist = self.analytics.language_distribution(github_records) if github_records else {}
        latency_summary = self.analytics.api_latency_summary(state.connector_results)
        charts = self.analytics.generate_charts(repo_rankings, language_dist, latency_summary)
        state.charts = [chart.model_dump() for chart in charts]

        response = state.to_response()
        for fmt in OutputFormat:
            path = self.reporter.generate(response, fmt)
            state.report_paths[fmt.value] = path
        return state


class MemoryAgent:
    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory

    async def run(self, state: GraphState) -> GraphState:
        response = state.to_response()
        self.memory.store_analysis(query=state.request.query, response=response)

        if state.request.use_memory:
            hits = self.memory.search(MemorySearchRequest(query=state.request.query, top_k=3))
            state.telemetry["memory_hits"] = [hit.model_dump(mode="json") for hit in hits]
        return state


class ReflectionAgent:
    def __init__(self) -> None:
        self.settings = load_settings()

    async def run(self, state: GraphState) -> GraphState:
        state.telemetry["completed_at"] = datetime.now(UTC).isoformat()
        retryable = any(err.retryable for err in state.errors)
        if retryable and state.retries < self.settings.agent.retry_max_attempts:
            state.retries += 1
            # Trim retryable errors before second pass.
            state.errors = [err for err in state.errors if not err.retryable]
        else:
            state.done = True
        return state


class AgentRuntime:
    """Orchestrates node classes in deterministic sequence (LangGraph wrapper uses same nodes)."""

    def __init__(self) -> None:
        self.log = configure_logging()
        self.connectors = ConnectorRegistry()
        self.cache = CacheManager()
        self.metrics = MetricsCollector()
        self.tracker = ProgressTracker()
        self.llm = OllamaProvider()
        self.analytics = AnalyticsEngine()
        self.reporter = ReportGenerator()
        self.memory = MemoryManager()

        self.request_planner = RequestPlannerAgent(self.connectors)
        self.api_router = APIRouterAgent(self.connectors)
        self.auth = AuthenticationAgent()
        self.fetch = DataFetchAgent(self.connectors, self.cache, self.metrics)
        self.validation = ValidationAgent()
        self.reasoning = ReasoningAgent(self.llm)
        self.report_generation = ReportGeneratorAgent(self.analytics, self.reporter)
        self.memory_agent = MemoryAgent(self.memory)
        self.reflection = ReflectionAgent()

        self.tool_registry = build_tool_registry()

    async def run_once(self, state: GraphState) -> GraphState:
        state = await self.request_planner.run(state)
        state = await self.api_router.run(state)
        state = await self.auth.run(state)
        state = await self.fetch.run(state)
        state = await self.validation.run(state)
        state = await self.reasoning.run(state)
        state = await self.report_generation.run(state)
        state = await self.memory_agent.run(state)
        state = await self.reflection.run(state)

        snapshot = self.metrics.snapshot()
        state.telemetry.update(
            {
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "gpu_name": snapshot.gpu_name,
                "gpu_vram_used_mb": snapshot.gpu_vram_used_mb,
                "gpu_vram_total_mb": snapshot.gpu_vram_total_mb,
                "cache_hit_rate": snapshot.cache_hit_rate,
                "api_latency_ms": snapshot.api_latency_ms,
                "elapsed_seconds": self.tracker.elapsed_seconds(),
                "pid": self.tracker.pid(),
                "tool_count": len(self.tool_registry.list_tools()),
            }
        )

        return state
