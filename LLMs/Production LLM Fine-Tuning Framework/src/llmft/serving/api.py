"""FastAPI app factory for inference endpoints."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from llmft.inference.backends import InferenceRouter


@dataclass(slots=True)
class ChatRequest:
    """Fallback request model when FastAPI/Pydantic not installed."""

    prompt: str


def create_app(router: InferenceRouter):
    """Create FastAPI app.

    Raises:
        RuntimeError: FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("FastAPI not installed. Install extra: llmft-framework[serve]") from exc

    app = FastAPI(title="LLMFT API", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(payload: dict[str, str]) -> dict[str, str]:
        prompt = payload.get("prompt", "")
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is required")
        response = await router.generate(prompt)
        return {"response": response}

    @app.post("/batch")
    async def batch(payload: dict[str, list[str]]) -> dict[str, list[str]]:
        prompts = payload.get("prompts", [])
        responses = await router.generate_batch(prompts)
        return {"responses": responses}

    @app.get("/bench")
    async def bench() -> dict[str, float | str | int]:
        prompts = ["hello"] * 4
        result = await router.benchmark(prompts)
        return asdict(result)

    @app.get("/stream")
    async def stream(prompt: str) -> dict[str, str]:
        response = await router.generate(prompt)
        # Placeholder stream behavior: callers should use SSE middleware in production.
        await asyncio.sleep(0)
        return {"response": response}

    return app
