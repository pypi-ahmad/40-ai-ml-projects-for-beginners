from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictRequest(StrictSchema):
    """Single prediction request payload."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    )

    sepal_length: float = Field(..., gt=0, lt=20, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, lt=20, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, lt=20, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, lt=20, description="Petal width in cm")


class PredictBatchRequest(BaseModel):
    """Batch prediction request."""
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "samples": [
                        {
                            "sepal_length": 5.1,
                            "sepal_width": 3.5,
                            "petal_length": 1.4,
                            "petal_width": 0.2,
                        },
                        {
                            "sepal_length": 6.3,
                            "sepal_width": 3.3,
                            "petal_length": 6.0,
                            "petal_width": 2.5,
                        },
                    ]
                }
            ]
        },
    )

    samples: list[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of Iris samples",
    )


class PredictResponse(StrictSchema):
    """Prediction response for one sample."""

    prediction: int
    species: str
    confidence: float | None = None
    probabilities: list[float] | None = None
    latency_ms: float
    model_name: str
    model_version: str


class ErrorResponse(StrictSchema):
    """Standardized API error response."""

    error: str
    detail: str | list[str] | None = None


class HealthResponse(StrictSchema):
    status: str
    model_loaded: bool
    load_error: str | None = None


class BatchPredictResponse(StrictSchema):
    predictions: list[dict[str, Any]]
    count: int


class MetricsResponse(StrictSchema):
    model_name: str
    model_version: str
    versions: list[dict[str, Any]]
    prediction_stats: dict[str, Any]


class ModelInfoResponse(StrictSchema):
    model_name: str
    model_version: str
    model_type: str
    target_classes: list[str]
    has_preprocessor: bool
    supports_proba: bool
    n_features_in_: int | None
    feature_names: list[str] | None
    active_version: dict[str, Any] | None = None


class ExplainResponse(StrictSchema):
    """Explain endpoint response.

    Supports success payload and compatibility error payload from SHAP wrapper.
    """

    base_value: float | list[float] | None = None
    feature_names: list[str] | None = None
    shap_values: dict[str, list[float]] | list[float] | None = None
    feature_importance: dict[str, float] | None = None
    global_importance: list[dict[str, float | str]] | None = None
    error: str | None = None


class ExplainRequest(StrictSchema):
    """Explainability request for local/global modes."""

    mode: Literal["local", "global"] = Field(
        default="local",
        description="Explanation mode: local sample-level or global dataset-level",
    )
    sample: PredictRequest | None = Field(
        default=None,
        description="Required for local mode; ignored for global mode",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "mode": "local",
                    "sample": {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    },
                },
                {"mode": "global"},
            ]
        },
    )

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "ExplainRequest":
        if self.mode == "local" and self.sample is None:
            raise ValueError("sample is required when mode='local'")
        return self
