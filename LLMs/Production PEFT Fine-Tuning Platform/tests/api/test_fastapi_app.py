from __future__ import annotations

import pytest

httpx = pytest.importorskip("httpx")

from peft_platform.api.app import create_app


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_generate_endpoint() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/generate", json={"prompt": "What is LoRA?"})
    assert response.status_code == 200
    body = response.json()
    assert "text" in body


@pytest.mark.asyncio
async def test_generate_stream_endpoint() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/generate/stream", json={"prompt": "stream this please"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
