"""ML model loader and inference wrapper for California Housing regression."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from app.config import settings
from src.constants import FEATURE_NAMES


class Predictor:
    """Thread-safe predictor singleton used by API routers."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._metadata: dict[str, Any] = {}
        self._loaded = False
        self._lock = threading.Lock()
        self._explainer: Any | None = None
        self._explainer_type: str = "none"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def model(self) -> Any:
        return self._model

    @property
    def feature_names(self) -> list[str]:
        names = self._metadata.get("feature_names")
        if isinstance(names, list) and names:
            return [str(n) for n in names]
        return FEATURE_NAMES

    def _validate_loaded_artifacts(self) -> None:
        """Validate model + metadata compatibility before serving requests."""
        if self._model is None:
            raise RuntimeError("Model object is missing after load.")
        if not hasattr(self._model, "predict"):
            raise RuntimeError("Loaded model does not implement predict().")

        feature_names = self.feature_names
        if len(feature_names) != len(FEATURE_NAMES):
            raise RuntimeError(
                "Loaded metadata feature_names length does not match expected schema "
                f"({len(FEATURE_NAMES)})."
            )

        n_features_in = getattr(self._model, "n_features_in_", None)
        if isinstance(n_features_in, int) and n_features_in != len(feature_names):
            raise RuntimeError(
                f"Model expects {n_features_in} features, but metadata defines "
                f"{len(feature_names)} features."
            )

        metadata_schema_version = self._metadata.get("feature_schema_version")
        if metadata_schema_version and metadata_schema_version != settings.feature_schema_version:
            raise RuntimeError(
                "Feature schema version mismatch: "
                f"model={metadata_schema_version} runtime={settings.feature_schema_version}"
            )

    def load(self) -> None:
        """Load model and metadata from disk once."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            model_path: Path = settings.model_path
            meta_path: Path = settings.metadata_path

            if not model_path.is_file():
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not meta_path.is_file():
                raise FileNotFoundError(f"Metadata not found at {meta_path}")

            self._model = joblib.load(str(model_path))
            with meta_path.open("r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            self._validate_loaded_artifacts()

            self._loaded = True
            self._explainer = None
            self._explainer_type = "none"

        logger.info(
            "Loaded model {name} v{version}",
            name=self._metadata.get("model_name", "unknown"),
            version=self._metadata.get("model_version", "?"),
        )

    def unload(self) -> None:
        """Unload cached model artifacts."""
        with self._lock:
            self._model = None
            self._metadata = {}
            self._loaded = False
            self._explainer = None
            self._explainer_type = "none"
        logger.info("Model unloaded")

    def reload(self) -> None:
        """Force reload model and metadata."""
        self.unload()
        self.load()

    def _to_frame(self, records: list[dict[str, float]]) -> pd.DataFrame:
        """Convert records to canonical dataframe with fixed column ordering."""
        df = pd.DataFrame(records)
        missing = [name for name in self.feature_names if name not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return df[self.feature_names]

    def predict_one(self, record: dict[str, float]) -> float:
        """Predict single record value."""
        if not self._loaded:
            self.load()
        X = self._to_frame([record])
        pred = self._model.predict(X)
        return float(pred[0])

    def predict_batch(self, records: list[dict[str, float]]) -> list[float]:
        """Predict list of records preserving input order."""
        if not self._loaded:
            self.load()
        X = self._to_frame(records)
        preds = self._model.predict(X)
        return [float(v) for v in preds]

    def _get_explainer(self, X_background: pd.DataFrame) -> tuple[Any, str, str | None]:
        """Create and cache SHAP explainer with model-compatible strategy."""
        if self._explainer is not None:
            return self._explainer, self._explainer_type, None

        import shap

        try:
            self._explainer = shap.TreeExplainer(self._model)
            self._explainer_type = "TreeExplainer"
            return self._explainer, self._explainer_type, None
        except Exception:
            pass

        try:
            background = X_background.head(settings.explain_max_background_rows)
            self._explainer = shap.LinearExplainer(self._model, background)
            self._explainer_type = "LinearExplainer"
            return self._explainer, self._explainer_type, "Linear fallback used for non-tree model."
        except Exception:
            pass

        background = X_background.head(min(32, len(X_background)))
        self._explainer = shap.KernelExplainer(self._model.predict, background)
        self._explainer_type = "KernelExplainer"
        return self._explainer, self._explainer_type, "Kernel fallback used. This path is slower."

    def explain_one(
        self,
        record: dict[str, float],
    ) -> tuple[float, float, list[float], dict[str, float], str, str | None]:
        """Return prediction + SHAP decomposition for one record."""
        if not self._loaded:
            self.load()

        X = self._to_frame([record])
        prediction = float(self._model.predict(X)[0])

        explainer, explainer_type, explanation_note = self._get_explainer(X)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        values = np.asarray(shap_values).reshape(-1)
        if values.size != len(self.feature_names):
            values = values[: len(self.feature_names)]

        base_value = explainer.expected_value
        if isinstance(base_value, (list, tuple, np.ndarray)):
            base_value = np.asarray(base_value).reshape(-1)[0]

        contributions = {
            name: float(val)
            for name, val in zip(self.feature_names, values, strict=False)
        }
        return (
            prediction,
            float(base_value),
            values.astype(float).tolist(),
            contributions,
            explainer_type,
            explanation_note,
        )


predictor = Predictor()
