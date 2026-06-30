from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from ml_package import (
    ModelExplainer,
    ModelLoader,
    PackageSettings,
    PredictionEngine,
    VersionRegistry,
    setup_logging,
)
from ml_package.exceptions import ArtifactVerificationError, UnsafeDeserializationError
from ml_package.logging_config import PredictionLogger
from ml_package.validation import IrisFeatures

from api.schemas import (
    BatchPredictResponse,
    ExplainRequest,
    ExplainResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictBatchRequest,
    PredictRequest,
    PredictResponse,
)

settings = PackageSettings.from_env()
logger = setup_logging("iris_api", log_dir=str(settings.log_dir))


class ModelService:
    """Singleton model service — loaded once at startup."""

    def __init__(self, runtime_settings: PackageSettings):
        self.settings = runtime_settings
        self.engine: PredictionEngine | None = None
        self.registry: VersionRegistry | None = None
        self.explainer: ModelExplainer | None = None
        self.background_data: np.ndarray | None = None
        self.load_error: str | None = None
        self.prediction_logger = PredictionLogger("iris_api.predictions")

    def load(self, model_path: str | Path | None = None) -> None:
        target_model_path = Path(model_path) if model_path else self.settings.model_path
        logger.info(f"Loading model from {target_model_path}")

        loader = ModelLoader(
            target_model_path,
            verify_integrity=self.settings.verify_artifacts,
            require_manifest=True,
            trusted_digests=self.settings.resolved_trusted_digests(target_model_path),
            allow_unsafe_deserialization=self.settings.allow_unsafe_deserialization,
        )
        model = loader.load()
        self.registry = VersionRegistry(str(self.settings.registry_path))
        self.engine = PredictionEngine(model, model_name="iris_classifier")
        self.load_error = None

        active = self.registry.get_active()
        if active:
            self.engine.model_version = active.version_id

        if self.settings.background_data_path.exists():
            self.background_data = np.load(self.settings.background_data_path)
            logger.info(
                "Background data loaded for lazy explainability initialization"
            )

        logger.info("Model loaded successfully")

    def load_explainer(self, background_data: np.ndarray):
        if self.engine:
            self.background_data = background_data
            self.explainer = ModelExplainer(self.engine.model, background_data)
            logger.info("Explainer initialized")

    def predict(self, features: np.ndarray, raw_features: dict[str, Any]) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("Model not loaded")
        try:
            result = self.engine.predict(features)
            self.prediction_logger.log_prediction(
                features=raw_features,
                prediction=result["prediction"],
                confidence=result["confidence"] if result["confidence"] is not None else 0.0,
                latency_ms=result["latency_ms"],
                model_version=result["model_version"],
            )
            return result
        except Exception as exc:
            self.prediction_logger.log_error(
                input_data=[
                    raw_features.get("sepal_length"),
                    raw_features.get("sepal_width"),
                    raw_features.get("petal_length"),
                    raw_features.get("petal_width"),
                ],
                error=str(exc),
            )
            raise


service = ModelService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.model_path.exists():
        try:
            service.load(settings.model_path)
        except Exception as exc:  # pragma: no cover - startup guard
            service.engine = None
            service.explainer = None
            service.load_error = str(exc)
            logger.warning(
                "Model startup load failed. API continues in setup mode. error=%s",
                exc,
            )
    else:
        logger.warning(
            f"Model not found at {settings.model_path}. API starts in setup mode."
        )
    yield


app = FastAPI(
    title="Iris Classifier API",
    description="Production-grade ML model serving for Iris species classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    tags=["System"],
    response_model=HealthResponse,
    summary="Health check",
    description="Checks API liveness and model load status.",
)
async def health():
    return {
        "status": "healthy" if service.engine is not None else "degraded",
        "model_loaded": service.engine is not None,
        "load_error": service.load_error,
    }


@app.get(
    "/model-info",
    tags=["Model"],
    response_model=ModelInfoResponse,
    summary="Model metadata",
    description="Returns model runtime metadata and active registry version details.",
)
async def model_info():
    if service.engine is None:
        raise HTTPException(503, detail="Model not loaded")
    info = service.engine.get_model_info()
    info["active_version"] = (
        service.registry.get_active().to_dict()
        if service.registry and service.registry.get_active()
        else None
    )
    return info


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict one sample",
    description="Runs one Iris prediction with strict Pydantic payload validation.",
)
async def predict(request: PredictRequest):
    if service.engine is None:
        raise HTTPException(503, detail="Model not loaded")

    features_model = IrisFeatures(
        sepal_length=request.sepal_length,
        sepal_width=request.sepal_width,
        petal_length=request.petal_length,
        petal_width=request.petal_width,
    )

    features = features_model.to_array()

    result = service.predict(
        features,
        raw_features={
            "sepal_length": request.sepal_length,
            "sepal_width": request.sepal_width,
            "petal_length": request.petal_length,
            "petal_width": request.petal_width,
        },
    )
    return PredictResponse(**result)


@app.post(
    "/predict-batch",
    tags=["Prediction"],
    response_model=BatchPredictResponse,
    summary="Predict batch",
    description="Runs batch prediction for up to 10,000 Iris samples per request.",
)
async def predict_batch(request: PredictBatchRequest):
    if service.engine is None:
        raise HTTPException(503, detail="Model not loaded")

    features = np.array([
        [s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
        for s in request.samples
    ])

    results = service.engine.predict_batch(features)
    return {"predictions": results, "count": len(results)}


@app.get(
    "/metrics",
    tags=["System"],
    response_model=MetricsResponse,
    summary="Operational metrics",
    description="Returns runtime prediction counters, latency stats, and version metadata.",
)
async def metrics():
    if service.engine is None:
        raise HTTPException(503, detail="Model not loaded")
    return {
        "model_name": service.engine.model_name,
        "model_version": service.engine.model_version,
        "versions": (
            service.registry.list_versions()
            if service.registry else []
        ),
        "prediction_stats": service.prediction_logger.get_stats(),
    }


def _format_prometheus_metrics(payload: dict[str, Any]) -> str:
    stats = payload["prediction_stats"]
    lines = [
        "# HELP iris_model_loaded Whether model is loaded (1 loaded, 0 unloaded)",
        "# TYPE iris_model_loaded gauge",
        f"iris_model_loaded {1 if service.engine is not None else 0}",
        "# HELP iris_predictions_total Total successful predictions",
        "# TYPE iris_predictions_total counter",
        f"iris_predictions_total {stats['total_predictions']}",
        "# HELP iris_prediction_errors_total Total prediction errors",
        "# TYPE iris_prediction_errors_total counter",
        f"iris_prediction_errors_total {stats['total_errors']}",
        "# HELP iris_prediction_average_latency_ms Average prediction latency in milliseconds",
        "# TYPE iris_prediction_average_latency_ms gauge",
        f"iris_prediction_average_latency_ms {stats['average_latency_ms']}",
    ]
    return "\n".join(lines) + "\n"


@app.get(
    "/metrics/prometheus",
    tags=["System"],
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
    description="Exports minimal Prometheus-compatible metrics for monitoring systems.",
)
async def metrics_prometheus():
    payload = await metrics()
    return PlainTextResponse(
        _format_prometheus_metrics(payload),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post(
    "/explain",
    tags=["Explainability"],
    response_model=ExplainResponse,
    summary="Generate explanations",
    description="Returns SHAP local explanation for one sample or global feature importance.",
)
async def explain(request: ExplainRequest):
    if service.engine is None:
        raise HTTPException(503, detail="Model not loaded")
    if service.explainer is None:
        if service.background_data is None:
            raise HTTPException(503, detail="Explainer not initialized")
        service.load_explainer(service.background_data)

    if request.mode == "global":
        if service.background_data is None:
            raise HTTPException(503, detail="Background data not available for global explain")
        result = service.explainer.get_global_importance(service.background_data)
    else:
        assert request.sample is not None  # validated by ExplainRequest
        features = np.array([
            [
                request.sample.sepal_length,
                request.sample.sepal_width,
                request.sample.petal_length,
                request.sample.petal_width,
            ]
        ])
        result = service.explainer.explain_single(features)

    return ExplainResponse(**result)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="Unexpected failure in server runtime",
        ).model_dump(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation error",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(UnsafeDeserializationError)
async def unsafe_deserialization_handler(
    request: Request,
    exc: UnsafeDeserializationError,
):
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="Model security policy violation",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(ArtifactVerificationError)
async def artifact_verification_handler(
    request: Request,
    exc: ArtifactVerificationError,
):
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="Model artifact verification failed",
            detail=str(exc),
        ).model_dump(),
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
