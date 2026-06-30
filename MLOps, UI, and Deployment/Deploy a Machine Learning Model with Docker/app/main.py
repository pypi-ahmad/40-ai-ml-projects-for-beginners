from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from app.schemas import (
    BatchPredictionInput,
    BatchPredictionOutput,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionInput,
    PredictionOutput,
)
from app.model import predict, predict_batch, explain, load_metadata, load_model
from app.metrics import MetricsMiddleware, metrics_endpoint
from app.rate_limit import configure_rate_limiting
from app.utils import logger

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def _input_to_list(inp: PredictionInput) -> list[float]:
    return [getattr(inp, f) for f in FEATURE_NAMES]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info("Model loaded and cached on startup")
    yield


app = FastAPI(title="California Housing ML API", version="1.0.0", lifespan=lifespan)
app.add_middleware(MetricsMiddleware)

DEFAULT_LIMIT = os.getenv("RATE_LIMIT_DEFAULT", "120/minute")
PREDICT_LIMIT = os.getenv("RATE_LIMIT_PREDICT", "60/minute")
BATCH_LIMIT = os.getenv("RATE_LIMIT_BATCH", "30/minute")
EXPLAIN_LIMIT = os.getenv("RATE_LIMIT_EXPLAIN", "20/minute")

limiter = configure_rate_limiting(app, default_limit=DEFAULT_LIMIT)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(_: Request, exc: RateLimitExceeded) -> JSONResponse:
    logger.warning("Rate limit exceeded: %s", exc)
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


@app.get("/health", response_model=HealthResponse)
async def health():
    logger.info("Health check")
    return HealthResponse(status="healthy")


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    meta = load_metadata()
    return ModelInfoResponse(
        model_name=meta.get("best_model", "unknown"),
        best_model_source=meta.get("best_model_source"),
        features=meta.get("features", FEATURE_NAMES),
        metrics=meta.get("metrics", {}),
        profile=meta.get("profile"),
        training_timestamp=meta.get("timestamp"),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionOutput, responses={422: {"model": ErrorResponse}})
@limiter.limit(PREDICT_LIMIT)
async def single_predict(request: Request, input: PredictionInput):
    del request
    try:
        value = predict(_input_to_list(input))
        return PredictionOutput(predicted_value=round(value, 4))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))


def _batch_response(input_payload: BatchPredictionInput) -> BatchPredictionOutput:
    """Shared implementation for canonical and legacy batch routes."""
    vectors = [_input_to_list(i) for i in input_payload.instances]
    values = predict_batch(vectors)
    return BatchPredictionOutput(predictions=[round(v, 4) for v in values])


@app.post("/predict-batch", response_model=BatchPredictionOutput, responses={422: {"model": ErrorResponse}})
@limiter.limit(BATCH_LIMIT)
async def predict_batch_canonical(request: Request, input: BatchPredictionInput):
    del request
    try:
        return _batch_response(input)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionOutput, responses={422: {"model": ErrorResponse}})
@limiter.limit(BATCH_LIMIT)
async def batch_predict_legacy(request: Request, input: BatchPredictionInput):
    """Backward-compatible alias for older notebook callers."""
    del request
    try:
        return _batch_response(input)
    except Exception as e:
        logger.error(f"Legacy batch prediction failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/explain", response_model=ExplainResponse, responses={422: {"model": ErrorResponse}})
@limiter.limit(EXPLAIN_LIMIT)
async def explain_prediction(request: Request, payload: ExplainRequest):
    del request
    try:
        result = explain(_input_to_list(payload.input))
        return ExplainResponse(**result)
    except Exception as e:
        logger.error(f"Explain failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))


@app.api_route("/metrics", methods=["GET", "HEAD"], include_in_schema=False)
async def metrics_route(request: Request):
    return metrics_endpoint(request)
