"""Model loading and inference helpers used by FastAPI routes."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
META_PATH = MODEL_PATH.with_name("model_metadata.json")
BACKGROUND_PATH = MODEL_PATH.with_name("background_sample.npy")


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load and cache model artifact from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, Any]:
    """Load model metadata; provide safe defaults when file missing."""
    if not META_PATH.exists():
        return {
            "best_model": "unknown",
            "best_model_source": "unknown",
            "features": [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
            ],
            "metrics": {},
            "profile": "unknown",
            "timestamp": "",
        }
    return json.loads(META_PATH.read_text())


def _to_array(features: list[float]) -> np.ndarray:
    """Convert single input vector to expected 2D float array."""
    return np.asarray(features, dtype=np.float32).reshape(1, -1)


def predict(features: list[float]) -> float:
    """Predict one housing value from ordered feature list."""
    model = load_model()
    return float(model.predict(_to_array(features))[0])


def predict_batch(features_batch: list[list[float]]) -> list[float]:
    """Predict multiple instances with one model call."""
    model = load_model()
    batch_array = np.asarray(features_batch, dtype=np.float32)
    preds = model.predict(batch_array)
    return [float(value) for value in np.asarray(preds).tolist()]


@lru_cache(maxsize=1)
def _load_explainer() -> Any:
    """Create and cache SHAP explainer."""
    import shap

    model = load_model()
    if hasattr(model, "feature_importances_"):
        try:
            return shap.TreeExplainer(model)
        except Exception:
            pass

    if BACKGROUND_PATH.exists():
        background = np.load(BACKGROUND_PATH)
    else:
        background = np.zeros((128, len(load_metadata().get("features", [])) or 8), dtype=np.float32)

    try:
        return shap.Explainer(model.predict, background, algorithm="permutation")
    except Exception:
        return shap.Explainer(model.predict, background)


def explain(features: list[float]) -> dict[str, Any]:
    """Return SHAP values and baseline for one instance."""
    model = load_model()
    meta = load_metadata()
    feature_names = meta.get("features", [f"f{i}" for i in range(len(features))])
    arr = _to_array(features)

    explainer = _load_explainer()
    if hasattr(explainer, "shap_values"):
        raw_values = np.asarray(explainer.shap_values(arr))
        if raw_values.ndim == 3:
            raw_values = raw_values[0]
        values = np.asarray(raw_values).reshape(-1)
        base = float(np.asarray(getattr(explainer, "expected_value", 0.0)).reshape(-1)[0])
    else:
        shap_values = explainer(arr, max_evals=(2 * len(feature_names) + 1))[0]
        values = np.asarray(shap_values.values).reshape(-1)
        base = float(np.asarray(shap_values.base_values).reshape(-1)[0])
    prediction = float(model.predict(arr)[0])

    return {
        "shap_values": {name: float(value) for name, value in zip(feature_names, values, strict=False)},
        "base_value": base,
        "prediction": prediction,
    }
