"""Top-level runtime service for API, CLI, Streamlit, and MCP surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from internet_agent.agent.workflow import InternetAgentWorkflow
from internet_agent.config import Settings, get_settings
from internet_agent.llm.client import OllamaClient
from internet_agent.logging_utils import configure_logging, get_logger
from internet_agent.memory.chroma_store import ChromaMemoryStore
from internet_agent.memory.repository import MemoryRepository
from internet_agent.metrics import METRICS
from internet_agent.retrieval.extract import clean_html, extract_markdown
from internet_agent.retrieval.fetch import WebsiteFetcher
from internet_agent.retrieval.pipeline import RetrievalPipeline
from internet_agent.services.analytics import build_analytics
from internet_agent.services.monitoring import system_snapshot
from internet_agent.services.report_service import ReportService
from internet_agent.tools.factory import build_default_registry
from internet_agent.tools.plugins import ToolPluginLoader


class InternetAgentService:
    """Runtime orchestrator exposing unified operations for all frontends."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings)
        self.logger = get_logger("internet_agent")
        self._mlflow_enabled = self.settings.monitoring.mlflow_enabled

        if self._mlflow_enabled:
            try:
                tracking_uri = self.settings.monitoring.mlflow_tracking_uri.strip()
                if not tracking_uri:
                    default_dir = Path("artifacts/mlruns")
                    default_dir.mkdir(parents=True, exist_ok=True)
                    tracking_uri = "sqlite:///artifacts/mlruns/mlflow.db"
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(self.settings.monitoring.mlflow_experiment)
            except Exception as exc:  # noqa: BLE001
                self._mlflow_enabled = False
                self.logger.warning("MLflow disabled due to initialization error: {}", exc)

        self.memory_repo = MemoryRepository(self.settings)
        self.semantic_store = ChromaMemoryStore(self.settings)
        self.tool_registry = build_default_registry(self.memory_repo)
        if self.settings.plugins.tool_factories:
            loaded = ToolPluginLoader.load_into_registry(
                self.tool_registry, self.settings.plugins.tool_factories
            )
            self.logger.info("Loaded plugin tools: {}", loaded)
        self.llm = OllamaClient(self.settings)
        self.workflow = InternetAgentWorkflow(
            settings=self.settings,
            llm=self.llm,
            memory_repo=self.memory_repo,
            semantic_store=self.semantic_store,
            tool_registry=self.tool_registry,
        )
        self.retrieval = RetrievalPipeline(
            settings=self.settings,
            memory_repo=self.memory_repo,
            semantic_store=self.semantic_store,
        )
        self.fetcher = WebsiteFetcher(timeout_seconds=self.settings.search.request_timeout_seconds)
        self.reports = ReportService(self.settings)

    async def chat(self, session_id: str, message: str) -> dict[str, Any]:
        with mlflow.start_run(run_name="chat", nested=True) if self._mlflow_enabled else NoopContext():
            state = await self.workflow.run(session_id=session_id, query=message)
            mlflow.log_metric("confidence", state.confidence) if self._mlflow_enabled else None

        return {
            "session_id": state.session_id,
            "query": state.user_query,
            "answer": state.final_answer,
            "confidence": state.confidence,
            "hallucination_risk": state.hallucination_risk,
            "citations": state.citations,
            "reasoning_trace": state.reasoning_trace,
            "tool_outputs": state.tool_outputs,
            "report": state.report_payload,
        }

    async def search(self, session_id: str, query: str, providers: list[str] | None = None) -> dict[str, Any]:
        return await self.retrieval.run(session_id=session_id, query=query, providers=providers)

    async def browse(self, session_id: str, url: str) -> dict[str, Any]:
        try:
            payload = await self.fetcher.fetch(url)
        except Exception as exc:  # noqa: BLE001
            return {
                "url": url,
                "status_code": 0,
                "content_type": "",
                "content": "",
                "markdown": "",
                "error": str(exc),
            }
        if "pdf" in payload.get("content_type", "").lower() or url.endswith(".pdf"):
            content = "PDF fetched. Use /search or pdf_reader tool for extraction."
            markdown = content
        else:
            content = clean_html(payload.get("text", ""))
            markdown = extract_markdown(payload.get("text", ""))

        self.memory_repo.add_visited_url(
            session_id=session_id,
            url=url,
            title=url,
            status_code=payload.get("status_code", 0),
        )
        return {
            "url": payload.get("url", url),
            "status_code": payload.get("status_code", 0),
            "content_type": payload.get("content_type", ""),
            "content": content,
            "markdown": markdown,
        }

    def history(self, session_id: str) -> dict[str, Any]:
        return {
            "messages": self.memory_repo.get_messages(session_id, limit=200),
            "tool_history": self.memory_repo.get_tool_history(session_id, limit=200),
            "reports": self.memory_repo.get_reports(session_id, limit=30),
        }

    def memory_search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        k = top_k or self.settings.memory.memory_top_k
        hits = self.semantic_store.query(query, top_k=k)
        return {"query": query, "hits": hits}

    def export_report(self, session_id: str, payload: dict[str, Any], fmt: str) -> dict[str, Any]:
        path = self.reports.generate(session_id=session_id, payload=payload, fmt=fmt)
        self.memory_repo.add_report(session_id=session_id, fmt=fmt, path=str(path), payload=payload)
        return {"format": fmt, "path": str(path)}

    def metrics(self) -> dict[str, Any]:
        return METRICS.snapshot()

    def analytics(self, session_id: str | None = None) -> dict[str, Any]:
        return build_analytics(self.memory_repo, session_id=session_id)

    def monitor(self) -> dict[str, Any]:
        return system_snapshot()


class NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
