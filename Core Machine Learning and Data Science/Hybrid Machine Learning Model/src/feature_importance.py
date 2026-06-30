from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as sklearn_permutation_importance


logger = logging.getLogger(__name__)



def _feature_names(X: np.ndarray | pd.DataFrame) -> list[str]:
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    arr = np.asarray(X)
    return [f"feature_{idx}" for idx in range(arr.shape[1])]



def permutation_importance(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[str, float]:
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = np.asarray(y)

    result = sklearn_permutation_importance(
        model,
        X_arr,
        y_arr,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )

    names = _feature_names(X)
    out = {names[idx]: float(max(result.importances_mean[idx], 0.0)) for idx in range(len(names))}
    return out



def _builtin_importance(model: Any, names: list[str]) -> dict[str, float] | None:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        return {names[idx]: float(values[idx]) for idx in range(len(names))}
    if hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
        return {names[idx]: float(values[idx]) for idx in range(len(names))}
    return None



def _dict_to_ranked_df(imp: dict[str, float], value_name: str = "importance") -> pd.DataFrame:
    df = pd.DataFrame({"feature": list(imp.keys()), value_name: list(imp.values())})
    return df.sort_values(value_name, ascending=False).reset_index(drop=True)


@dataclass(slots=True)
class FeatureImportanceAnalyzer:
    model: Any
    X_train: np.ndarray | pd.DataFrame
    y_train: np.ndarray | pd.Series

    def compute_all(self) -> dict[str, dict[str, float]]:
        names = _feature_names(self.X_train)
        results: dict[str, dict[str, float]] = {
            "permutation": permutation_importance(self.model, self.X_train, self.y_train, n_repeats=5)
        }

        built_in = _builtin_importance(self.model, names)
        if built_in is not None:
            results["built_in"] = built_in
        return results

    def get_top_features(self, importance: dict[str, float], n: int = 10) -> list[str]:
        sorted_items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in sorted_items[:n]]

    def summary(self) -> str:
        results = self.compute_all()
        lines = ["Feature Importance Summary"]
        for method, importance in results.items():
            top = self.get_top_features(importance, n=5)
            lines.append(f"- {method}: {', '.join(top)}")
        return "\n".join(lines)



def shap_values(
    model: Any,
    X_background: np.ndarray | pd.DataFrame,
    X_explain: np.ndarray | pd.DataFrame,
) -> tuple[Any, np.ndarray]:
    import shap

    bg = X_background.values if isinstance(X_background, pd.DataFrame) else np.asarray(X_background)
    ex = X_explain.values if isinstance(X_explain, pd.DataFrame) else np.asarray(X_explain)

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, bg)
    else:
        explainer = shap.KernelExplainer(model.predict, bg[: min(len(bg), 100)])

    shap_vals = explainer.shap_values(ex)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    return explainer, np.asarray(shap_vals)



def save_shap_summary_plot(
    shap_vals: np.ndarray,
    X_explain: np.ndarray | pd.DataFrame,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt
    import shap

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = X_explain.values if isinstance(X_explain, pd.DataFrame) else np.asarray(X_explain)
    names = _feature_names(X_explain)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals, data, feature_names=names, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path



def save_shap_dependence_plot(
    shap_vals: np.ndarray,
    X_explain: np.ndarray | pd.DataFrame,
    feature: str,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt
    import shap

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = X_explain if isinstance(X_explain, pd.DataFrame) else pd.DataFrame(X_explain, columns=_feature_names(X_explain))

    plt.figure(figsize=(9, 6))
    shap.dependence_plot(feature, shap_vals, data, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path



def importance_to_frame(importance: dict[str, float], value_name: str = "importance") -> pd.DataFrame:
    return _dict_to_ranked_df(importance, value_name=value_name)
