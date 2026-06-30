from typing import Any

import numpy as np

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]


class ModelExplainer:
    """SHAP-based model explainability wrapper.

    Provides local and global feature importance explanations
    for any sklearn-compatible model. Enables debugging, monitoring,
    and regulatory compliance for production ML systems.

    Usage:
        explainer = ModelExplainer(model, X_train)
        explanation = explainer.explain_single(X_sample)
        global_importance = explainer.get_global_importance()
    """

    def __init__(self, model: Any, background_data: np.ndarray | None = None):
        self.model = model

        if background_data is not None:
            self._build_explainer(background_data)
        else:
            self.explainer = None

    def _build_explainer(self, background_data: np.ndarray) -> None:
        """Build SHAP explainer tailored to model type."""
        if shap is None:
            self.explainer = None
            return

        if hasattr(self.model, "predict_proba"):
            self.explainer = shap.TreeExplainer(
                self.model, background_data[:100]
            ) if "XGB" in type(self.model).__name__ or hasattr(
                self.model, "get_booster"
            ) else shap.KernelExplainer(
                self.model.predict_proba, background_data[:50]
            )
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict, background_data[:50]
            )

    def explain_single(self, sample: np.ndarray) -> dict:
        """Generate SHAP explanation for a single prediction.

        Args:
            sample: 2D numpy array of shape (1, n_features)

        Returns:
            Dict with shap_values, base_value, feature_importance
        """
        if shap is None:
            return {"error": "SHAP not installed. Install with: uv add shap"}
        if self.explainer is None:
            return {"error": "Explainer not initialized. Provide background_data."}

        sample = np.asarray(sample, dtype=float)
        if sample.ndim != 2 or sample.shape[0] != 1:
            return {"error": f"Sample must be 2D with single row, got shape {sample.shape}"}

        shap_values = self.explainer.shap_values(sample)

        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (int, float)):
            base_value: float | list[float] = float(expected_value)
        else:
            expected_arr = np.asarray(expected_value, dtype=float).flatten()
            base_value = (
                float(expected_arr[0])
                if expected_arr.size == 1
                else expected_arr.tolist()
            )

        result = {"base_value": base_value, "feature_names": feature_names}

        if isinstance(shap_values, list):
            result["shap_values"] = {
                f"class_{i}": sv[0].tolist()
                for i, sv in enumerate(shap_values)
            }
            result["feature_importance"] = {
                name: float(abs(shap_values[0][0][idx]))
                for idx, name in enumerate(feature_names)
            }
        else:
            shap_array = np.asarray(shap_values)
            if shap_array.ndim == 3:
                # shape: (n_samples, n_features, n_classes)
                class_map = {}
                for class_idx in range(shap_array.shape[2]):
                    class_map[f"class_{class_idx}"] = shap_array[0, :, class_idx].tolist()
                result["shap_values"] = class_map
                result["feature_importance"] = {
                    name: float(np.abs(shap_array[0, idx, :]).mean())
                    for idx, name in enumerate(feature_names)
                }
            else:
                result["shap_values"] = shap_array[0].tolist()
                result["feature_importance"] = {
                    name: float(abs(shap_array[0][idx]))
                    for idx, name in enumerate(feature_names)
                }

        return result

    def get_global_importance(self, data: np.ndarray | None = None) -> dict:
        """Compute global feature importance across the dataset.

        Returns:
            Dict with mean_abs_shap per feature sorted by importance
        """
        if shap is None:
            return {"error": "SHAP not installed. Install with: uv add shap"}
        if self.explainer is None:
            return {"error": "Explainer not initialized."}

        if data is None:
            return {"error": "Global importance requires data array argument."}

        data = np.asarray(data, dtype=float)
        shap_values = self.explainer.shap_values(data)

        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        if isinstance(shap_values, list):
            merged = np.mean([np.abs(values) for values in shap_values], axis=0)
            values = merged.mean(axis=0)
        else:
            values = np.abs(np.asarray(shap_values)).mean(axis=0)
            if values.ndim > 1:
                values = values.mean(axis=-1)

        ranking = sorted(
            (
                {"feature": feature_names[idx], "mean_abs_shap": float(score)}
                for idx, score in enumerate(values)
            ),
            key=lambda item: item["mean_abs_shap"],
            reverse=True,
        )
        return {"global_importance": ranking}
