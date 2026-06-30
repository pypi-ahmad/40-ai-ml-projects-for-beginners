"""FastAPI service."""

from __future__ import annotations

from pathlib import Path
import json
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from peft_platform.api.schemas import (
    BatchRequest,
    ChatRequest,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from peft_platform.inference.engine import InferenceEngine
from peft_platform.model_registry import list_models
from peft_platform.peft.adapters import AdapterManager
from peft_platform.utils.logging import configure_logging
from peft_platform.version import __version__

logger = configure_logging()


def create_app() -> FastAPI:
    app = FastAPI(title="Production PEFT Platform API", version=__version__)

    manager = AdapterManager(registry_path=Path("artifacts/adapter_registry.json"))
    default_engine = InferenceEngine()
    default_engine.load()

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/models")
    async def models() -> list[dict[str, str]]:
        return [{"id": item.id, "family": item.family, "tier": item.tier.value} for item in list_models()]

    @app.get("/adapters")
    async def adapters() -> list[dict[str, str | bool]]:
        return [adapter.__dict__ for adapter in manager.list_adapters()]

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest) -> GenerateResponse:
        engine = default_engine if payload.model_id is None else InferenceEngine(payload.model_id)
        if payload.model_id is not None:
            engine.load()
        output = engine.generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
        )
        return GenerateResponse(
            text=output.text,
            latency_ms=output.latency_ms,
            tokens_generated=output.tokens_generated,
        )

    @app.post("/generate/stream")
    async def generate_stream(payload: GenerateRequest) -> StreamingResponse:
        engine = default_engine if payload.model_id is None else InferenceEngine(payload.model_id)
        if payload.model_id is not None:
            engine.load()

        output = engine.generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
        )
        words = output.text.split()

        async def event_stream() -> AsyncIterator[str]:
            for idx, word in enumerate(words):
                chunk = {"index": idx, "token": word}
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/chat", response_model=GenerateResponse)
    async def chat(payload: ChatRequest) -> GenerateResponse:
        engine = default_engine if payload.model_id is None else InferenceEngine(payload.model_id)
        if payload.model_id is not None:
            engine.load()
        prompt = "\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in payload.messages)
        output = engine.generate(
            prompt=prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
        )
        return GenerateResponse(
            text=output.text,
            latency_ms=output.latency_ms,
            tokens_generated=output.tokens_generated,
        )

    @app.post("/batch")
    async def batch(payload: BatchRequest) -> list[GenerateResponse]:
        engine = default_engine if payload.model_id is None else InferenceEngine(payload.model_id)
        if payload.model_id is not None:
            engine.load()
        responses: list[GenerateResponse] = []
        for prompt in payload.prompts:
            output = engine.generate(
                prompt=prompt,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
            )
            responses.append(
                GenerateResponse(
                    text=output.text,
                    latency_ms=output.latency_ms,
                    tokens_generated=output.tokens_generated,
                )
            )
        return responses

    return app
