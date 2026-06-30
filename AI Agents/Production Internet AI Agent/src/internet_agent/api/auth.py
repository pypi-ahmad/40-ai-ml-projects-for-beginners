"""Optional API key guard."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status

from internet_agent.config import get_settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.api.require_api_key:
        return

    expected = os.getenv(settings.api.api_key_env, "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"API key enabled but env var {settings.api.api_key_env} is empty",
        )
    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
