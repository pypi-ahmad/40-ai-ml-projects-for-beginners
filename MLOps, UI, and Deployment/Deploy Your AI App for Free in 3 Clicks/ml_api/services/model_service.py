"""Model loading, prediction, and explainability service."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from ml_api.core.config import Settings
from ml_api.core.errors import ModelNotReadyError
from ml_api.schemas.prediction import HouseFeatures
from ml_api.training.feature_spec import ALL_FEATURES
from ml_api.training.persistence import load_metadata, load_model
from ml_api.training.pipeline import train_and_serialize

logger = logging.getLogger(__name__)


class ModelService:
    """Service that owns the active model artifacts in process memory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline: Pipeline | None = None
        self.metadata: dict[str, Any] | None = None

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None and self.metadata is not None

    def load(self) -> None:
        """Load serialized model and metadata from disk."""
        model_path = self.settings.model_dir / self.settings.joblib_artifact
        metadata_path = self.settings.model_dir / self.settings.metadata_artifact

        if not model_path.exists() or not metadata_path.exists():
            if self.settings.auto_train_on_startup:
                logger.info("Model artifacts missing. Running startup training.")
                train_and_serialize(self.settings)
            else:
                logger.warning("Model artifacts missing: %s / %s", model_path, metadata_path)
                return

        self.pipeline = load_model(model_path)
        self.metadata = load_metadata(metadata_path)
        logger.info("Model loaded: %s", self.metadata.get("model_name", "unknown"))

    def ensure_ready(self) -> None:
        if not self.is_ready:
            raise ModelNotReadyError()

    def predict_one(self, record: HouseFeatures) -> float:
        """Predict price for a single validated record."""
        self.ensure_ready()
        frame = self._to_frame([record])
        prediction = float(self.pipeline.predict(frame)[0])  # type: ignore[union-attr]
        return round(prediction, 4)

    def predict_batch(self, records: list[HouseFeatures]) -> list[float]:
        """Predict prices for multiple validated records using vectorized inference."""
        self.ensure_ready()
        frame = self._to_frame(records)
        preds = self.pipeline.predict(frame)  # type: ignore[union-attr]
        return [round(float(item), 4) for item in preds]

    def explain(self, record: HouseFeatures, top_k: int) -> dict[str, Any]:
        """Return explanation using SHAP when available, else perturbation fallback."""
        self.ensure_ready()
        frame = self._to_frame([record])
        prediction = float(self.pipeline.predict(frame)[0])  # type: ignore[union-attr]

        shap_payload = self._try_shap(frame, top_k=top_k)
        if shap_payload is not None:
            return {
                "prediction": round(prediction, 4),
                "base_value": shap_payload["base_value"],
                "method": "shap",
                "contributions": shap_payload["contributions"],
            }

        baseline_row = self.metadata.get("baseline_row", {}) if self.metadata else {}
        baseline_df = pd.DataFrame([baseline_row], columns=ALL_FEATURES)
        baseline_pred = float(self.pipeline.predict(baseline_df)[0])  # type: ignore[union-attr]

        contributions: list[dict[str, Any]] = []
        record_dict = record.model_dump()
        for feature in ALL_FEATURES:
            shifted = dict(baseline_row)
            shifted[feature] = record_dict[feature]
            shifted_df = pd.DataFrame([shifted], columns=ALL_FEATURES)
            shifted_pred = float(self.pipeline.predict(shifted_df)[0])  # type: ignore[union-attr]
            contributions.append(
                {
                    "feature": feature,
                    "value": record_dict[feature],
                    "contribution": round(shifted_pred - baseline_pred, 4),
                }
            )

        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return {
            "prediction": round(prediction, 4),
            "base_value": round(baseline_pred, 4),
            "method": "perturbation",
            "contributions": contributions[:top_k],
        }

    def model_info(self) -> dict[str, Any]:
        """Return metadata for `/model-info` and `/metrics`."""
        self.ensure_ready()
        return {
            "model_name": self.metadata.get("model_name", "unknown"),
            "model_type": self.metadata.get("model_type", "unknown"),
            "model_version": self.metadata.get("model_version", "v0"),
            "training_rows": int(self.metadata.get("training_rows", 0)),
            "feature_count": len(self.metadata.get("feature_columns", ALL_FEATURES)),
            "target_column": self.metadata.get("target_column", "sale_price"),
            "dataset_hash": self.metadata.get("dataset_hash", ""),
            "validation_rmse": float(self.metadata.get("validation_rmse", 0.0)),
            "test_rmse": float(self.metadata.get("test_rmse", 0.0)),
            "serialization_formats": ["joblib", "pickle"],
        }

    def reload_stability_check(self, cycles: int = 3) -> dict[str, float | bool]:
        """Reload artifact multiple times and check deterministic inference output."""
        self.ensure_ready()
        model_path = self.settings.model_dir / self.settings.joblib_artifact
        baseline_row = self.metadata.get("baseline_row", {})
        baseline_df = pd.DataFrame([baseline_row], columns=ALL_FEATURES)

        values: list[float] = []
        for _ in range(cycles):
            loaded = load_model(model_path)
            values.append(float(loaded.predict(baseline_df)[0]))

        drift = max(values) - min(values)
        return {
            "stable": drift <= 1e-9,
            "drift": round(float(drift), 12),
            "cycles": cycles,
        }

    def _to_frame(self, records: list[HouseFeatures]) -> pd.DataFrame:
        rows = [record.model_dump() for record in records]
        return pd.DataFrame(rows, columns=ALL_FEATURES)

    def _try_shap(self, frame: pd.DataFrame, top_k: int) -> dict[str, Any] | None:
        shap = _try_import("shap")
        if shap is None:
            return None

        try:
            model = self.pipeline.named_steps["model"]  # type: ignore[union-attr]
            preprocessor = self.pipeline.named_steps["preprocessor"]  # type: ignore[union-attr]
            transformed = preprocessor.transform(frame)
            explainer = shap.Explainer(model)
            values = explainer(transformed)
            contrib = values.values[0]
            base_value = float(values.base_values[0])

            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feature_names = [f"feature_{idx}" for idx in range(len(contrib))]

            rows: list[dict[str, Any]] = []
            for idx, contribution in enumerate(contrib):
                rows.append(
                    {
                        "feature": feature_names[idx],
                        "value": frame.iloc[0].to_dict().get(feature_names[idx], ""),
                        "contribution": round(float(contribution), 4),
                    }
                )
            rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
            return {"base_value": round(base_value, 4), "contributions": rows[:top_k]}
        except Exception as exc:  # pragma: no cover - optional path
            logger.warning("SHAP explanation fallback triggered: %s", exc)
            return None


def _try_import(module: str):
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        return None
