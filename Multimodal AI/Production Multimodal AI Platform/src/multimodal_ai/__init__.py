"""Multimodal AI Platform package."""

from multimodal_ai.api.app import create_app
from multimodal_ai.services.platform_service import PlatformService

__all__ = ["create_app", "PlatformService"]
