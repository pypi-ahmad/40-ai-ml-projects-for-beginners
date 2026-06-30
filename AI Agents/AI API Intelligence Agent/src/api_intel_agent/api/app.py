"""FastAPI application exposing agent APIs."""

from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from api_intel_agent.agents import AgentRunner
from api_intel_agent.api.db import User, init_db
from api_intel_agent.api.security import get_current_user, get_db, require_role
from api_intel_agent.auth import AuthManager
from api_intel_agent.config import load_settings
from api_intel_agent.connectors import ConnectorRegistry
from api_intel_agent.core.schemas import AnalyzeRequest, MemorySearchRequest
from api_intel_agent.memory import MemoryManager
from api_intel_agent.monitoring import MetricsCollector
from api_intel_agent.scheduler import SchedulerService
from api_intel_agent.tools import register_openapi_tools
from api_intel_agent.utils.notifications import send_email_notification, send_slack_notification

settings = load_settings()
app = FastAPI(title="AI API Intelligence Agent", version="0.1.0")

runner = AgentRunner()
connectors = ConnectorRegistry()
memory = MemoryManager()
metrics = MetricsCollector()
auth = AuthManager()
scheduler = SchedulerService()


@app.on_event("startup")
async def startup_event() -> None:
    init_db()

    def _scheduled_report() -> str:
        response = runner.query_sync("scheduled ai ecosystem summary", output_format="markdown")
        return response.artifacts.get("markdown", "")

    if os.getenv("AGENT_DISABLE_SCHEDULER", "0") != "1":
        scheduler.start(_scheduled_report)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    scheduler.shutdown()


@app.post("/auth/register")
def register(username: str, password: str, role: str = "viewer", db: Session = Depends(get_db)):
    existing = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="username already exists")
    user = User(username=username, password_hash=auth.hash_password(password), role=role)
    db.add(user)
    db.commit()
    return {"status": "ok", "username": username, "role": role}


@app.post("/auth/token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.username == form.username)).scalar_one_or_none()
    if not user or not auth.verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid credentials")

    return {
        "access_token": auth.create_access_token(form.username, user.role),
        "refresh_token": auth.create_refresh_token(form.username),
        "token_type": "bearer",
    }


@app.post("/analyze")
async def analyze(
    request: AnalyzeRequest,
    _user: User = Depends(get_current_user),
):
    response = await runner.analyze(request)
    return response.model_dump(mode="json")


@app.post("/query")
async def query(
    query_text: str,
    model: str | None = None,
    _user: User = Depends(get_current_user),
):
    request = AnalyzeRequest(query=query_text, model=model)
    response = await runner.analyze(request)
    return response.model_dump(mode="json")


@app.get("/github")
async def github(q: str = Query("llm"), _user: User = Depends(get_current_user)):
    connector = connectors.get("github")
    result = await connector.execute(query=q)
    return result.model_dump(mode="json")


@app.get("/news")
async def news(q: str = Query("ai"), _user: User = Depends(get_current_user)):
    connector = connectors.get("news")
    result = await connector.execute(query=q)
    return result.model_dump(mode="json")


@app.get("/weather")
async def weather(lat: float = 51.5, lon: float = -0.12, _user: User = Depends(get_current_user)):
    connector = connectors.get("weather")
    result = await connector.execute(query="weather", params={"latitude": lat, "longitude": lon})
    return result.model_dump(mode="json")


@app.get("/repos")
async def repos(q: str = Query("open source llm"), _user: User = Depends(get_current_user)):
    connector = connectors.get("github")
    result = await connector.execute(query=q)
    return {"count": len(result.records), "items": result.records}


@app.get("/memory")
def memory_search(
    q: str = Query(...),
    top_k: int = 5,
    _user: User = Depends(get_current_user),
):
    hits = memory.search(MemorySearchRequest(query=q, top_k=top_k))
    return {"hits": [hit.model_dump(mode="json") for hit in hits]}


@app.get("/search")
async def search(q: str = Query(...), _user: User = Depends(get_current_user)):
    request = AnalyzeRequest(query=q)
    response = await runner.analyze(request)
    return {
        "run_id": response.run_id,
        "summary": response.summary,
        "sources": [src.model_dump(mode="json") for src in response.sources],
    }


@app.get("/history")
def history(limit: int = 20, _user: User = Depends(get_current_user)):
    items = memory.history(limit=limit)
    return {"items": [item.model_dump(mode="json") for item in items]}


@app.get("/report/{run_id}")
def report(run_id: str, _user: User = Depends(get_current_user)):
    payload = memory.get_response(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="report not found")
    return payload


@app.get("/report")
def report_by_query(run_id: str, _user: User = Depends(get_current_user)):
    payload = memory.get_response(run_id)
    if not payload:
        raise HTTPException(status_code=404, detail="report not found")
    return payload


@app.get("/health")
def health():
    ollama_status = "up" if runner.runtime.llm else "unknown"
    return {
        "status": "ok",
        "time": datetime.now(UTC).isoformat(),
        "ollama": ollama_status,
        "connectors": connectors.list_names(),
    }


@app.get("/metrics")
def get_metrics(_user: User = Depends(require_role("analyst"))):
    snapshot = metrics.snapshot()
    return {
        "cpu_percent": snapshot.cpu_percent,
        "memory_percent": snapshot.memory_percent,
        "gpu_name": snapshot.gpu_name,
        "gpu_vram_used_mb": snapshot.gpu_vram_used_mb,
        "gpu_vram_total_mb": snapshot.gpu_vram_total_mb,
        "cache_hit_rate": snapshot.cache_hit_rate,
        "api_latency_ms": snapshot.api_latency_ms,
    }


@app.post("/notify/slack")
def notify_slack(message: str, _user: User = Depends(require_role("analyst"))):
    ok = send_slack_notification(message)
    return {"sent": ok}


@app.post("/notify/email")
def notify_email(to_email: str, subject: str, body: str, _user: User = Depends(require_role("analyst"))):
    ok = send_email_notification(subject=subject, body=body, to_email=to_email)
    return {"sent": ok}


@app.post("/tools/openapi")
def ingest_openapi(
    path_or_url: str,
    service_name: str = "external",
    _user: User = Depends(require_role("admin")),
):
    count = register_openapi_tools(runner.runtime.tool_registry, path_or_url=path_or_url, service_name=service_name)
    return {"registered_tools": count}
