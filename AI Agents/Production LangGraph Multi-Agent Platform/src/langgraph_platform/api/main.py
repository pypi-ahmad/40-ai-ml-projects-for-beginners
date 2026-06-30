"""FastAPI application for multi-agent workflow platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from langgraph_platform.agents.registry import list_agents
from langgraph_platform.analytics.service import AnalyticsService
from langgraph_platform.api.schemas import (
    ChatRequest,
    HITLRequest,
    KnowledgeIngestRequest,
    ReportExportRequest,
    SearchRequest,
    WorkflowRequest,
)
from langgraph_platform.config.loader import load_config
from langgraph_platform.engine.workflow import LangGraphWorkflowEngine
from langgraph_platform.exporters.report_exporter import ReportExporter
from langgraph_platform.hitl.service import HITLService
from langgraph_platform.mcp.server import MCPServerAdapter
from langgraph_platform.monitoring.system import SystemMonitor


def create_app(
    engine: LangGraphWorkflowEngine | None = None,
    hitl: HITLService | None = None,
) -> FastAPI:
    """Build and configure FastAPI application."""

    config = load_config()
    engine = engine or LangGraphWorkflowEngine(config)
    hitl = hitl or HITLService()
    analytics = AnalyticsService(engine.sqlite_store)
    monitor = SystemMonitor(enable_gpu_metrics=config.monitoring.enable_gpu_metrics)
    exporter = ReportExporter(output_dir="artifacts/reports")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        engine.close()

    app = FastAPI(
        title="LangGraph Multi-Agent Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    mcp_server = MCPServerAdapter(
        registry=engine.runtime.tool_registry,
        exposed_tools=[
            "duckduckgo_search",
            "calculator",
            "memory_search",
            "chroma_search",
            "file_reader",
            "markdown_reader",
        ],
    )
    app.include_router(mcp_server.router())

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "langgraph-platform", "env": config.env}

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        result = engine.run(user_request=request.message, session_id=request.session_id)
        return {
            "workflow_id": result.workflow_id,
            "response": result.final_report,
            "confidence": result.confidence,
            "verification_status": result.verification_status.value,
            "citations": [citation.model_dump(mode="json") for citation in result.citations],
        }

    @app.post("/workflow")
    def workflow(request: WorkflowRequest) -> dict[str, Any]:
        result = engine.run(user_request=request.user_request, session_id=request.session_id)
        return {
            "workflow_id": result.workflow_id,
            "final_report": result.final_report,
            "confidence": result.confidence,
            "verification_status": result.verification_status.value,
            "metadata": result.metadata.model_dump(mode="json"),
        }

    @app.get("/graph")
    def graph() -> dict[str, Any]:
        return engine.inspect_graph()

    @app.get("/agents")
    def agents() -> list[dict[str, Any]]:
        return [
            {
                "name": agent.name,
                "role": agent.role,
                "objective": agent.objective,
                "tools": agent.tools,
                "constraints": agent.constraints,
                "output_schema": agent.output_schema,
                "retry_strategy": agent.retry_strategy,
            }
            for agent in list_agents()
        ]

    @app.post("/tasks")
    def tasks(request: HITLRequest) -> dict[str, Any]:
        state = hitl.apply(request.workflow_id, request.action, request.note)
        return {
            "workflow_id": request.workflow_id,
            "action": request.action.value,
            "state": {
                "approved": state.approved,
                "paused": state.paused,
                "rejected": state.rejected,
                "overrides": state.overrides,
            },
        }

    @app.get("/memory")
    def memory(query: str | None = None, limit: int = 10) -> dict[str, Any]:
        if query:
            result = engine.runtime.tool_registry.run(
                "memory_search", {"query": query, "limit": limit}
            )
            return {"items": result.output}
        return {"items": engine.sqlite_store.list_recent_runs(limit=limit)}

    @app.post("/reports")
    def reports(request: ReportExportRequest) -> dict[str, Any]:
        try:
            paths = exporter.export_all(
                request.workflow_id, request.markdown_report, request.payload
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"workflow_id": request.workflow_id, "exports": paths}

    @app.post("/knowledge")
    def knowledge(request: KnowledgeIngestRequest) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        if request.paths:
            outputs["paths"] = engine.rag_pipeline.ingest_paths(request.paths).__dict__
        if request.urls:
            outputs["urls"] = engine.rag_pipeline.ingest_urls(request.urls).__dict__
        return outputs

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        result = engine.runtime.tool_registry.run(
            "duckduckgo_search",
            {"query": request.query, "max_results": request.max_results},
        )
        if not result.ok:
            raise HTTPException(status_code=400, detail=result.error)
        return {"results": result.output}

    @app.get("/analytics")
    def analytics_endpoint() -> dict[str, Any]:
        return analytics.summary()

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return monitor.capture()

    return app


app = create_app()
