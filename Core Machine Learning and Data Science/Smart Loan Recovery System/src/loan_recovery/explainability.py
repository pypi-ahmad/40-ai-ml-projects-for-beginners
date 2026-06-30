"""SHAP explainability for recovery risk models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR


@dataclass(slots=True)
class ExplainabilityOutputs:
    """Paths to generated SHAP assets."""

    shap_summary_path: Path | None
    shap_waterfall_path: Path | None
    shap_force_path: Path | None
    feature_importance_path: Path | None


class ModelExplainer:
    """Generate SHAP explanations for tree-based models inside a preprocessing pipeline."""

    def __init__(self, model, output_dir: Path | None = None) -> None:
        self.model = model
        self.output_dir = output_dir or FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def explain(self, x: pd.DataFrame, max_samples: int = 80) -> ExplainabilityOutputs:
        """Compute SHAP assets with fast tree-based explainers when available."""
        if x.empty:
            return ExplainabilityOutputs(None, None, None, None)

        feature_importance_path = self._plot_feature_importance()

        try:
            import shap
        except Exception:
            return ExplainabilityOutputs(None, None, None, feature_importance_path)

        # Expect sklearn Pipeline: preprocess + model.
        if not hasattr(self.model, "named_steps"):
            return ExplainabilityOutputs(None, None, None, feature_importance_path)

        preprocess = self.model.named_steps.get("preprocess")
        estimator = self.model.named_steps.get("model")
        if preprocess is None or estimator is None:
            return ExplainabilityOutputs(None, None, None, feature_importance_path)

        x_sample = x.sample(n=min(max_samples, len(x)), random_state=42)
        x_t = preprocess.transform(x_sample)
        feature_names = preprocess.get_feature_names_out()

        # Convert sparse matrix to dense for SHAP compatibility if needed.
        if hasattr(x_t, "toarray"):
            x_t = x_t.toarray()

        x_t_df = pd.DataFrame(x_t, columns=feature_names)

        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(x_t_df)
        except Exception:
            # Fallback to model-agnostic explainer on tiny sample to keep runtime bounded.
            try:
                explainer = shap.Explainer(estimator.predict_proba, x_t_df.iloc[:30])
                shap_values = explainer(x_t_df.iloc[:30])
                x_t_df = x_t_df.iloc[:30]
            except Exception:
                return ExplainabilityOutputs(None, None, None, feature_importance_path)

        summary_path = self._plot_summary(shap, shap_values, x_t_df)
        waterfall_path = self._plot_waterfall(shap, shap_values)
        force_path = self._plot_force(shap, shap_values, x_t_df)

        return ExplainabilityOutputs(summary_path, waterfall_path, force_path, feature_importance_path)

    def _plot_feature_importance(self) -> Path | None:
        estimator = None
        preprocess = None
        if hasattr(self.model, "named_steps"):
            estimator = self.model.named_steps.get("model")
            preprocess = self.model.named_steps.get("preprocess")

        if estimator is None or preprocess is None:
            return None

        values = None
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

        if values is None:
            return None

        names = preprocess.get_feature_names_out()
        imp = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.barh(imp["feature"].tail(25), imp["importance"].tail(25), color="#2563EB")
        ax.set_title("Top Feature Importance (Model Native)")
        ax.set_xlabel("Importance")
        path = self.output_dir / "feature_importance.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _to_high_risk_values(self, shap_values):
        if isinstance(shap_values, list):
            # tree explainer multi-class output (list per class)
            idx = 2 if len(shap_values) > 2 else -1
            return shap_values[idx]

        values = getattr(shap_values, "values", shap_values)
        if isinstance(values, np.ndarray) and values.ndim == 3:
            idx = 2 if values.shape[2] > 2 else values.shape[2] - 1
            return values[:, :, idx]
        return values

    def _base_value(self, shap_values):
        base = getattr(shap_values, "base_values", None)
        if base is None:
            return 0.0
        if isinstance(base, np.ndarray):
            if base.ndim == 2:
                return float(base[0, 2 if base.shape[1] > 2 else -1])
            if base.ndim == 1:
                return float(base[2] if len(base) > 2 else base[-1])
        return float(base)

    def _plot_summary(self, shap_module, shap_values, x_df: pd.DataFrame) -> Path | None:
        try:
            values = self._to_high_risk_values(shap_values)
            plt.figure(figsize=(12, 8))
            shap_module.summary_plot(values, x_df, show=False)
            path = self.output_dir / "shap_summary.png"
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            return path
        except Exception:
            return None

    def _plot_waterfall(self, shap_module, shap_values) -> Path | None:
        try:
            values = self._to_high_risk_values(shap_values)
            base = self._base_value(shap_values)
            explanation = shap_module.Explanation(
                values=values[0],
                base_values=base,
                data=None,
                feature_names=None,
            )
            plt.figure(figsize=(12, 7))
            shap_module.plots.waterfall(explanation, show=False)
            path = self.output_dir / "shap_waterfall.png"
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            return path
        except Exception:
            return None

    def _plot_force(self, shap_module, shap_values, x_df: pd.DataFrame) -> Path | None:
        try:
            values = self._to_high_risk_values(shap_values)
            base = self._base_value(shap_values)
            force_html = shap_module.force_plot(base, values[0], x_df.iloc[0], matplotlib=False)
            path = self.output_dir / "shap_force.html"
            shap_module.save_html(str(path), force_html)
            return path
        except Exception:
            return None
