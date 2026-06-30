"""Pydantic contracts for API request and response payloads."""

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Service liveness response."""

    status: str


class ModelInfoResponse(BaseModel):
    """Model metadata exposed for deployment introspection."""

    model_name: str
    best_model_source: str | None = None
    features: list[str]
    metrics: dict[str, float] = Field(default_factory=dict)
    profile: str | None = None
    training_timestamp: str | None = None
    version: str


class PredictionInput(BaseModel):
    """Single California housing feature vector."""

    model_config = ConfigDict(extra="forbid")

    MedInc: float = Field(..., ge=0, le=20, description="Median income in block group")
    HouseAge: float = Field(..., ge=0, le=100, description="Median house age")
    AveRooms: float = Field(..., ge=0, le=100, description="Average number of rooms")
    AveBedrms: float = Field(..., ge=0, le=20, description="Average number of bedrooms")
    Population: float = Field(..., ge=0, le=100_000, description="Block population")
    AveOccup: float = Field(..., ge=0, le=100, description="Average household occupancy")
    Latitude: float = Field(..., ge=32, le=42, description="Latitude in California range")
    Longitude: float = Field(..., ge=-125, le=-114, description="Longitude in California range")


class BatchPredictionInput(BaseModel):
    """Batch prediction payload."""

    model_config = ConfigDict(extra="forbid")

    instances: list[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=1_000,
        description="Collection of instances for batch inference",
    )


class PredictionOutput(BaseModel):
    """Single-value model prediction response."""

    predicted_value: float


class BatchPredictionOutput(BaseModel):
    """Batch prediction output aligned to input ordering."""

    predictions: list[float]


class ErrorResponse(BaseModel):
    """Validation or inference error payload."""

    detail: str


class ExplainRequest(BaseModel):
    """Single instance explanation request."""

    model_config = ConfigDict(extra="forbid")

    input: PredictionInput


class ExplainResponse(BaseModel):
    """SHAP-like explanation response for one prediction."""

    shap_values: dict[str, float]
    base_value: float
    prediction: float
