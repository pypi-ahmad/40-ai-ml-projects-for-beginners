"""FastAPI application for crew platform."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from crew_platform.analytics import AnalyticsService
from crew_platform.config import load_settings
from crew_platform.mcp import ExternalMCPClient, InternalMCPServer, MCPCallRequest
from crew_platform.monitoring import MonitoringService
from crew_platform.orchestration import CollaborationService, CrewRunRequest, PlanApproval, create_service
from crew_platform.rag import RAGIngestionService, RAGRetriever


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


class KnowledgeIngestRequest(BaseModel):
    path: str | None = None
    url: str | None = None


class TaskRerunRequest(BaseModel):
    task_id: str
    feedback: str | None = None


class ReportFormatRequest(BaseModel):
    format: str = Field(default="html")


@lru_cache(maxsize=1)
def _service() -> CollaborationService:
    settings = load_settings()
    return create_service(settings)


@lru_cache(maxsize=1)
def _analytics() -> AnalyticsService:
    return AnalyticsService()


@lru_cache(maxsize=1)
def _monitoring() -> MonitoringService:
    return MonitoringService()


@lru_cache(maxsize=1)
def _mcp_server() -> InternalMCPServer:
    return InternalMCPServer(_service().tool_registry)


@lru_cache(maxsize=1)
def _rag_ingestion() -> RAGIngestionService:
    svc = _service()
    return RAGIngestionService(svc.settings, svc.runtime_memory)


@lru_cache(maxsize=1)
def _rag_retriever() -> RAGRetriever:
    svc = _service()
    return RAGRetriever(svc.settings, svc.runtime_memory)


app = FastAPI(title="Production CrewAI Multi-Agent Platform", version="0.1.0")


@app.post("/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    return await _service().chat(query=request.query, session_id=request.session_id)


@app.post("/crew")
async def crew(request: CrewRunRequest) -> dict[str, Any]:
    result = await _service().create_plan(request)
    return result.model_dump(mode="json")


@app.post("/crew/{run_id}/approve")
async def crew_approve(run_id: str, approval: PlanApproval) -> dict[str, Any]:
    try:
        result = await _service().apply_approval(run_id, approval)
        return result.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/crew/{run_id}/execute")
async def crew_execute(run_id: str, force_consensus: bool = False) -> dict[str, Any]:
    try:
        result = await _service().execute_run(run_id, force_consensus=force_consensus)
        return result.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/crew/{run_id}/pause")
def crew_pause(run_id: str) -> dict[str, Any]:
    try:
        result = _service().pause_run(run_id)
        return result.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/crew/{run_id}/resume")
def crew_resume(run_id: str) -> dict[str, Any]:
    try:
        result = _service().resume_run(run_id)
        return result.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/agents")
def agents() -> dict[str, Any]:
    return {"agents": _service().list_agents()}


@app.get("/tasks")
def tasks(run_id: str = Query(...)) -> dict[str, Any]:
    run = _service().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")
    return {"run_id": run_id, "tasks": [task.model_dump(mode="json") for task in run.tasks]}


@app.post("/tasks/{run_id}/rerun")
async def rerun_task(run_id: str, request: TaskRerunRequest) -> dict[str, Any]:
    try:
        result = await _service().rerun_task(run_id, task_id=request.task_id, feedback=request.feedback)
        return result.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/memory")
def memory(limit: int = 50, query: str | None = None) -> dict[str, Any]:
    snapshot = _service().persistence.fetch_memory(limit=limit)
    semantic = _service().runtime_memory.search(query, top_k=10) if query else []
    return {"snapshot": snapshot, "semantic": semantic}


@app.get("/reports")
def reports(run_id: str | None = None) -> dict[str, Any]:
    if run_id:
        report = _service().persistence.fetch_report(run_id)
        if report is None:
            raise HTTPException(status_code=404, detail=f"No report for run_id {run_id}")
        return {"report": report}

    runs = _service().list_runs()
    return {
        "reports": [
            run.report.model_dump(mode="json")
            for run in runs
            if run.report is not None
        ]
    }


@app.post("/reports/{run_id}/export")
def export_report(run_id: str, request: ReportFormatRequest) -> dict[str, Any]:
    run = _service().get_run(run_id)
    if run is None or run.report is None:
        raise HTTPException(status_code=404, detail=f"No report for run_id {run_id}")

    path = _service().report_generator.generate_on_demand(run.report, request.format)
    return {"run_id": run_id, "format": request.format, "path": str(path)}


@app.get("/search")
async def search(q: str = Query(..., min_length=2)) -> dict[str, Any]:
    return await _service().search(q)


@app.post("/knowledge")
async def knowledge_ingest(request: KnowledgeIngestRequest) -> dict[str, Any]:
    if not request.path and not request.url:
        raise HTTPException(status_code=400, detail="Provide path or url")

    if request.path:
        try:
            result = _rag_ingestion().ingest_path(request.path)
            return {"ingestion": result}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = await _rag_ingestion().ingest_url(request.url or "")
    return {"ingestion": result}


@app.get("/knowledge")
def knowledge_search(q: str = Query(...), top_k: int = 5) -> dict[str, Any]:
    return {"matches": _rag_retriever().retrieve(q, top_k=top_k)}


@app.get("/analytics")
def analytics(run_id: str | None = None) -> dict[str, Any]:
    runs = _service().list_runs()
    selected = None
    if run_id:
        selected = _service().get_run(run_id)
        if selected is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}")

    tasks = selected.tasks if selected else [task for run in runs for task in run.tasks]
    summary = _analytics().summarize(tasks)

    if tasks:
        graph_path = Path("artifacts/workflow") / f"{run_id or 'aggregate'}.html"
        _analytics().save_workflow_html(tasks, str(graph_path))
        summary["workflow_graph_path"] = str(graph_path)
    return summary


@app.get("/health")
def health() -> dict[str, Any]:
    status = _service().health()
    status["monitoring"] = _monitoring().collect()
    return status


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return {
        "runtime": _monitoring().collect(),
        "service": _service().metrics_snapshot(),
    }


@app.get("/mcp/tools")
def mcp_tools() -> dict[str, Any]:
    return {"tools": [tool.model_dump(mode="json") for tool in _mcp_server().list_tools()]}


@app.post("/mcp/call")
async def mcp_call(request: MCPCallRequest) -> dict[str, Any]:
    response = await _mcp_server().call_tool(
        tool_name=request.tool_name,
        arguments=request.arguments,
        run_id="mcp-call",
    )
    return response.model_dump(mode="json")


@app.get("/mcp/external/tools")
async def mcp_external_tools(endpoint: str) -> dict[str, Any]:
    client = ExternalMCPClient(endpoint=endpoint)
    tools = await client.list_tools()
    return {"endpoint": endpoint, "tools": tools}


@app.post("/mcp/external/call")
async def mcp_external_call(endpoint: str, request: MCPCallRequest) -> dict[str, Any]:
    client = ExternalMCPClient(endpoint=endpoint)
    data = await client.call_tool(tool_name=request.tool_name, arguments=request.arguments)
    return {"endpoint": endpoint, "response": data}


def run() -> None:
    settings = load_settings()
    uvicorn.run("crew_platform.api.main:app", host=settings.api.host, port=settings.api.port, reload=False)
