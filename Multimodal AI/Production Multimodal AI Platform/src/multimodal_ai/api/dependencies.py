"""Dependency providers for API layer."""

from __future__ import annotations

from functools import lru_cache

from multimodal_ai.services.bootstrap import build_platform_service
from multimodal_ai.services.platform_service import PlatformService


@lru_cache(maxsize=1)
def get_platform_service() -> PlatformService:
    """Singleton platform service for API process."""

    return build_platform_service()
