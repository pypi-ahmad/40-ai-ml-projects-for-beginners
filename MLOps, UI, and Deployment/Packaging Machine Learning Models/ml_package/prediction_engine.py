import time
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline


class PredictionEngine:
    """High-level prediction orchestrator with pre/post processing.

    Wraps a trained model with preprocessing, prediction, confidence scoring,
    and result formatting. Designed for reuse across API, CLI, and batch jobs.

    Architecture:
    - Accepts any sklearn-compatible estimator or Pipeline
    - Optionally wraps with preprocessing Pipeline
    - Returns structured dict with prediction + metadata
    """

    TARGET_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}

    def __init__(
        self,
        model: Any,
        preprocessor: Pipeline | None = None,
        model_name: str = "iris_classifier",
        model_version: str = "1.0.0",
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.model_name = model_name
        self.model_version = model_version

    @staticmethod
    def _validate_features_array(features: np.ndarray, *, expected_rows: int | None = None) -> np.ndarray:
        arr = np.asarray(features, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"features must be 2D array, got shape {arr.shape}")
        if expected_rows is not None and arr.shape[0] != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} row(s), got {arr.shape[0]} rows"
            )
        if arr.shape[1] != 4:
            raise ValueError(f"Expected 4 feature columns, got {arr.shape[1]}")
        if not np.isfinite(arr).all():
            raise ValueError("features contain non-finite values")
        return arr

    def predict(self, features: np.ndarray) -> dict:
        """Run single prediction with timing and metadata.

        Args:
            features: 2D numpy array of shape (1, n_features)

        Returns:
            Dict with prediction, probability, species name, metadata
        """
        start = time.perf_counter()

        features = self._validate_features_array(features, expected_rows=1)

        X = self.preprocessor.transform(features) if self.preprocessor else features
        pred_class = int(self.model.predict(X)[0])
        pred_species = self.TARGET_MAP.get(pred_class, "unknown")

        proba = None
        if hasattr(self.model, "predict_proba"):
            proba_raw = self.model.predict_proba(X)[0]
            confidence = float(max(proba_raw))
            proba = proba_raw.tolist()
        else:
            confidence = None

        latency = (time.perf_counter() - start) * 1000

        return {
            "prediction": pred_class,
            "species": pred_species,
            "confidence": confidence,
            "probabilities": proba,
            "latency_ms": round(latency, 3),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }

    def predict_batch(self, features: np.ndarray) -> list[dict]:
        """Run batch predictions efficiently.

        For large batches, uses vectorized model inference.
        Returns list of per-sample prediction dicts.

        Args:
            features: 2D numpy array of shape (n_samples, n_features)

        Returns:
            List of prediction result dicts
        """
        start = time.perf_counter()

        features = self._validate_features_array(features)

        X = self.preprocessor.transform(features) if self.preprocessor else features
        pred_classes = self.model.predict(X)
        pred_species = [self.TARGET_MAP.get(int(c), "unknown") for c in pred_classes]

        proba_list = None
        if hasattr(self.model, "predict_proba"):
            proba_list = self.model.predict_proba(X).tolist()
            confidences = [max(p) for p in proba_list]
        else:
            confidences = [None] * len(pred_classes)

        batch_latency = (time.perf_counter() - start) * 1000

        results = []
        for i in range(len(pred_classes)):
            results.append({
                "sample_id": i,
                "prediction": int(pred_classes[i]),
                "species": pred_species[i],
                "confidence": confidences[i],
                "probabilities": proba_list[i] if proba_list else None,
                "batch_size": len(pred_classes),
                "model_name": self.model_name,
                "model_version": self.model_version,
            })

        return results

    def get_model_info(self) -> dict:
        """Return model metadata for introspection endpoints."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": type(self.model).__name__,
            "target_classes": list(self.TARGET_MAP.values()),
            "has_preprocessor": self.preprocessor is not None,
            "supports_proba": hasattr(self.model, "predict_proba"),
            "n_features_in_": getattr(self.model, "n_features_in_", None),
            "feature_names": (
                list(self.preprocessor.feature_names_in_)
                if self.preprocessor and hasattr(self.preprocessor, "feature_names_in_")
                else None
            ),
        }
