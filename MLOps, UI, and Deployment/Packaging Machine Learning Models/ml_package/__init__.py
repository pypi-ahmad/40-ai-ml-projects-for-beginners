from ml_package.model_loader import ModelLoader
from ml_package.prediction_engine import PredictionEngine
from ml_package.validation import IrisValidator
from ml_package.logging_config import setup_logging
from ml_package.versioning import ModelVersion, VersionRegistry
from ml_package.settings import PackageSettings
from ml_package.exceptions import (
    ArtifactVerificationError,
    MLPackageError,
    ModelNotLoadedError,
    UnsafeDeserializationError,
)

__all__ = [
    "ModelLoader",
    "PredictionEngine",
    "IrisValidator",
    "setup_logging",
    "ModelVersion",
    "VersionRegistry",
    "ModelExplainer",
    "PackageSettings",
    "MLPackageError",
    "ModelNotLoadedError",
    "ArtifactVerificationError",
    "UnsafeDeserializationError",
]


def __getattr__(name: str):
    if name == "ModelExplainer":
        from ml_package.explainability import ModelExplainer

        return ModelExplainer
    raise AttributeError(f"module 'ml_package' has no attribute {name!r}")
