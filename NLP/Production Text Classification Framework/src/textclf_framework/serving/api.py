"""FastAPI serving app factory."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from textclf_framework.serving.inference import InferenceEngine
from textclf_framework.serving.monitoring import InferenceMetrics


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class PredictionItem(BaseModel):
    label_id: int
    label_name: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    latency_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[list[PredictionItem]]
    latency_ms: float


@dataclass(slots=True)
class APIConfig:
    model_path: str
    label_names: list[str]
    cache_size: int = 2048


def _to_prediction_item(prediction: object) -> PredictionItem:
    return PredictionItem(
        label_id=int(getattr(prediction, "label_id")),
        label_name=str(getattr(prediction, "label_name")),
        confidence=float(getattr(prediction, "confidence")),
    )


def create_app(config: APIConfig, engine: InferenceEngine | None = None) -> FastAPI:
    """Create FastAPI app with model inference endpoints."""
    app = FastAPI(title="Project21 Text Classification API", version="0.1.0")
    infer_engine = engine or InferenceEngine(
        model_path=config.model_path,
        label_names=config.label_names,
        cache_size=config.cache_size,
    )
    metrics = InferenceMetrics()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def get_metrics() -> dict[str, float]:
        return metrics.snapshot()

    @app.post("/predict", response_model=PredictResponse)
    async def predict(payload: PredictRequest) -> PredictResponse:
        try:
            preds, latency = await infer_engine.predict(payload.text, top_k=payload.top_k)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            metrics.record_error()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        top_conf = preds[0].confidence if preds else 0.0
        metrics.record_success(latency_ms=latency, input_len=len(payload.text), confidence=top_conf)
        return PredictResponse(
            predictions=[_to_prediction_item(p) for p in preds],
            latency_ms=latency,
        )

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
        try:
            batch_preds, latency = await infer_engine.predict_batch(payload.texts, top_k=payload.top_k)
        except Exception as exc:  # pragma: no cover
            metrics.record_error()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        for text, preds in zip(payload.texts, batch_preds, strict=True):
            top_conf = preds[0].confidence if preds else 0.0
            metrics.record_success(latency_ms=latency / max(1, len(payload.texts)), input_len=len(text), confidence=top_conf)

        return BatchPredictResponse(
            predictions=[[_to_prediction_item(p) for p in group] for group in batch_preds],
            latency_ms=latency,
        )

    return app
