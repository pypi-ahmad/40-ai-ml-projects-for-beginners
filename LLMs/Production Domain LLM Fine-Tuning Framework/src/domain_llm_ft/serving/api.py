"""FastAPI serving app."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

import torch
from fastapi import FastAPI
from loguru import logger
from starlette.requests import Request

from domain_llm_ft.inference.engine import InferenceEngine
from domain_llm_ft.serving.schemas import (
    BatchPredictRequest,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(title="Domain LLM Fine-Tuning API", version="0.1.0")


@lru_cache(maxsize=8)
def _engine(model_name: str) -> InferenceEngine:
    return InferenceEngine(model_name)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    latency_ms = (perf_counter() - start) * 1000
    logger.info(
        "request path={} status={} latency_ms={:.2f}",
        request.url.path,
        response.status_code,
        latency_ms,
    )
    response.headers["x-latency-ms"] = f"{latency_ms:.2f}"
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health probe endpoint."""
    return HealthResponse(status="ok", torch_cuda=torch.cuda.is_available())


@app.post("/predict", response_model=PredictResponse)
@app.post("/classify", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Single-text classification endpoint."""
    pred = await _engine(request.model_name).predict_async(request.text)
    return PredictResponse(
        label=pred.label,
        score=pred.score,
        probabilities=pred.probabilities,
        model_name=request.model_name,
    )


@app.post("/batch", response_model=list[PredictResponse])
async def batch_predict(request: BatchPredictRequest) -> list[PredictResponse]:
    """Batch classification endpoint."""
    preds = _engine(request.model_name).predict_batch(request.texts)
    return [
        PredictResponse(
            label=pred.label,
            score=pred.score,
            probabilities=pred.probabilities,
            model_name=request.model_name,
        )
        for pred in preds
    ]
