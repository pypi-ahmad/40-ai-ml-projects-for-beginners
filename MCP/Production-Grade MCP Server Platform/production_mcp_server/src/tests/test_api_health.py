from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from api.app import create_api_app
from server.platform import Platform


@pytest.mark.asyncio
async def test_api_health() -> None:
    platform = Platform.from_config("configs/default.yaml")
    app = create_api_app(platform)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
