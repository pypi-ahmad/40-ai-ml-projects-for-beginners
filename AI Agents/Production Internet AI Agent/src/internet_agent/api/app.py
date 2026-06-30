"""FastAPI application entrypoint."""

from __future__ import annotations

from functools import lru_cache

import uvicorn
from fastapi import Depends, FastAPI

from internet_agent.api.auth import verify_api_key
from internet_agent.api.schemas import (
    BrowseRequest,
    ChatRequest,
    ChatResponse,
    MemoryRequest,
    ReportRequest,
    SearchRequest,
    SearchResponse,
)
from internet_agent.config import get_settings
from internet_agent.services.agent_service import InternetAgentService

app = FastAPI(title="Production Internet AI Agent", version="0.1.0")


@lru_cache(maxsize=1)
def get_service() -> InternetAgentService:
    return InternetAgentService(settings=get_settings())


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app.name,
        "environment": settings.app.environment,
    }


@app.get("/metrics", dependencies=[Depends(verify_api_key)])
async def metrics() -> dict:
    return get_service().metrics()


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest) -> ChatResponse:
    payload = await get_service().chat(session_id=request.session_id, message=request.message)
    return ChatResponse(**payload)


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
async def search(request: SearchRequest) -> SearchResponse:
    payload = await get_service().search(
        session_id=request.session_id,
        query=request.query,
        providers=request.providers,
    )
    return SearchResponse(**payload)


@app.post("/browse", dependencies=[Depends(verify_api_key)])
async def browse(request: BrowseRequest) -> dict:
    return await get_service().browse(session_id=request.session_id, url=request.url)


@app.get("/history", dependencies=[Depends(verify_api_key)])
async def history(session_id: str) -> dict:
    return get_service().history(session_id)


@app.post("/memory", dependencies=[Depends(verify_api_key)])
async def memory(request: MemoryRequest) -> dict:
    return get_service().memory_search(request.query, top_k=request.top_k)


@app.post("/report", dependencies=[Depends(verify_api_key)])
async def report(request: ReportRequest) -> dict:
    return get_service().export_report(
        session_id=request.session_id,
        payload=request.payload,
        fmt=request.format,
    )


@app.get("/monitor", dependencies=[Depends(verify_api_key)])
async def monitor() -> dict:
    return get_service().monitor()


@app.get("/analytics", dependencies=[Depends(verify_api_key)])
async def analytics(session_id: str | None = None) -> dict:
    return get_service().analytics(session_id)


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "internet_agent.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
        log_level=settings.logging.level.lower(),
    )


if __name__ == "__main__":
    run()
