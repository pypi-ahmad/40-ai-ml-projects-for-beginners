"""FastAPI application factory."""

from __future__ import annotations

from fastapi import Depends, FastAPI

from multimodal_ai.api.dependencies import get_platform_service
from multimodal_ai.api.schemas import APIRequest
from multimodal_ai.services.platform_service import PlatformService


def create_app() -> FastAPI:
    """Create and configure FastAPI app."""

    app = FastAPI(
        title="Multimodal AI Platform",
        version="0.1.0",
        description="Production-grade local multimodal platform API",
    )

    @app.get("/health")
    async def health(service: PlatformService = Depends(get_platform_service)) -> dict:
        return service.health().model_dump()

    @app.post("/caption")
    async def caption(
        payload: APIRequest, service: PlatformService = Depends(get_platform_service)
    ) -> dict:
        return service.caption(payload.to_envelope()).model_dump()

    @app.post("/search")
    async def search(
        payload: APIRequest, service: PlatformService = Depends(get_platform_service)
    ) -> dict:
        return service.search(payload.to_envelope()).model_dump()

    @app.post("/retrieve")
    async def retrieve(
        payload: APIRequest, service: PlatformService = Depends(get_platform_service)
    ) -> dict:
        return service.retrieve(payload.to_envelope()).model_dump()

    @app.post("/vqa")
    async def vqa(
        payload: APIRequest, service: PlatformService = Depends(get_platform_service)
    ) -> dict:
        return service.vqa(payload.to_envelope()).model_dump()

    @app.post("/ocr")
    async def ocr(
        payload: APIRequest, service: PlatformService = Depends(get_platform_service)
    ) -> dict:
        return service.ocr(payload.to_envelope()).model_dump()

    @app.post("/compare")
    async def compare(
        payload: APIRequest, service: PlatformService = Depends(get_platform_service)
    ) -> dict:
        return service.compare(payload.to_envelope()).model_dump()

    @app.post("/analyze")
    async def analyze(
        payload: APIRequest, service: PlatformService = Depends(get_platform_service)
    ) -> dict:
        return service.analyze(payload.to_envelope()).model_dump()

    @app.post("/documents")
    async def documents(
        payload: APIRequest, service: PlatformService = Depends(get_platform_service)
    ) -> dict:
        return service.documents(payload.to_envelope()).model_dump()

    @app.post("/embeddings")
    async def embeddings(
        payload: APIRequest, service: PlatformService = Depends(get_platform_service)
    ) -> dict:
        return service.embeddings(payload.to_envelope()).model_dump()

    @app.post("/analytics")
    async def analytics(
        payload: APIRequest, service: PlatformService = Depends(get_platform_service)
    ) -> dict:
        return service.analytics(payload.to_envelope()).model_dump()

    return app


app = create_app()
