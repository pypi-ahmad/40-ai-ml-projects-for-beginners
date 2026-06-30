"""FastAPI request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    model_name: str = "distilbert-base-uncased"
    backend: str = "torch"


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    model_name: str = "distilbert-base-uncased"
    backend: str = "torch"


class PredictResponse(BaseModel):
    label: str
    score: float
    probabilities: list[float]
    model_name: str


class HealthResponse(BaseModel):
    status: str
    torch_cuda: bool
