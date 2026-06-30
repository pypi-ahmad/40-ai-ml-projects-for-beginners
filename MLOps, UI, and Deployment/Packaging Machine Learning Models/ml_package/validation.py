from typing import List, Tuple

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator


class IrisFeatures(BaseModel):
    """Pydantic model for validating Iris flower measurements.

    Each field corresponds to a sepal/petal measurement in centimeters.
    Validation constraints reflect realistic Iris flower ranges.

    Design: Pydantic v2 model with field validators for input sanitization.
    This prevents malformed or out-of-range data from reaching the model.
    """
    sepal_length: float = Field(
        ..., gt=0.0, lt=20.0, description="Sepal length in cm"
    )
    sepal_width: float = Field(
        ..., gt=0.0, lt=20.0, description="Sepal width in cm"
    )
    petal_length: float = Field(
        ..., gt=0.0, lt=20.0, description="Petal length in cm"
    )
    petal_width: float = Field(
        ..., gt=0.0, lt=20.0, description="Petal width in cm"
    )

    @field_validator("sepal_length", "sepal_width", "petal_length", "petal_width")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Measurement must be finite: {v}")
        return v

    def to_array(self) -> np.ndarray:
        """Convert validated features to numpy array for model input."""
        return np.array([
            [self.sepal_length, self.sepal_width,
             self.petal_length, self.petal_width]
        ])

    def to_list(self) -> list:
        """Convert to plain list for serialization."""
        return [self.sepal_length, self.sepal_width,
                self.petal_length, self.petal_width]


class IrisBatchFeatures(BaseModel):
    """Pydantic model for batch prediction requests.

    Accepts a list of Iris feature sets with optional sample IDs.
    """
    samples: List[IrisFeatures] = Field(
        ..., min_length=1, max_length=10000,
        description="List of Iris flower measurements for batch prediction"
    )
    sample_ids: List[str] | None = Field(
        None, description="Optional identifiers for each sample"
    )

    @field_validator("sample_ids")
    @classmethod
    def check_ids_match(cls, v: List[str] | None, info) -> List[str] | None:
        if v is not None and len(v) != len(info.data.get("samples", [])):
            raise ValueError(
                "Number of sample_ids must match number of samples"
            )
        return v


class IrisValidator:
    """Production input validation for Iris model predictions.

    Provides both Pydantic-based validation and traditional constraint checks.
    Acts as a defense-in-depth layer before model inference.
    """

    FEATURE_NAMES = [
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    ]

    TARGET_NAMES = ["setosa", "versicolor", "virginica"]

    def __init__(self):
        self.expected_features = 4

    def validate_single(self, features: IrisFeatures) -> IrisFeatures:
        """Validate a single Iris feature set using Pydantic.

        Args:
            features: Validated IrisFeatures instance

        Returns:
            The validated instance (pass-through for API integration)
        """
        return features

    def validate_batch(
        self, batch: IrisBatchFeatures
    ) -> Tuple[List[IrisFeatures], List[str] | None]:
        """Validate a batch of Iris feature sets.

        Args:
            batch: Validated IrisBatchFeatures instance

        Returns:
            Tuple of (validated_features, optional_sample_ids)
        """
        return batch.samples, batch.sample_ids

    @staticmethod
    def validate_request(
        sepal_length: float, sepal_width: float,
        petal_length: float, petal_width: float
    ) -> list[str]:
        errors: list[str] = []
        try:
            IrisFeatures(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
            )
        except ValidationError as e:
            for err in e.errors():
                field = ".".join(str(p) for p in err["loc"])
                errors.append(f"{field}: {err['msg']}")
        return errors
