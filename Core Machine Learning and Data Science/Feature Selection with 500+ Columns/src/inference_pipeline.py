"""Reusable end-to-end feature-selection and inference pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from src.benchmark import compute_metrics
from src.feature_selector import FeatureSelector

LOGGER = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas values to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Series):
        return _to_jsonable(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return _to_jsonable(value.to_dict(orient="records"))
    return value


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for feature-selection + model training."""

    var_threshold: float = 0.01
    corr_threshold: float = 0.95
    corr_method: str = "pearson"
    mi_k: int = 120
    rfe_feat: int = 120
    rfe_step: int = 5
    rfe_use_cv: bool = False
    rfe_min_features: int = 10
    l1_C: float = 1.0
    shap_k: int = 120
    random_state: int = 42
    model_name: str = "RandomForestClassifier"
    extra: dict[str, Any] = field(default_factory=dict)


class FeatureSelectionInferencePipeline:
    """Train a selector + model and persist production artifacts."""

    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        model: BaseEstimator | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.selector = FeatureSelector(random_state=self.config.random_state)
        self.model = model or RandomForestClassifier(
            n_estimators=300,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.selected_features_: list[str] = []
        self.fitted_: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "FeatureSelectionInferencePipeline":
        """Fit feature selector and downstream model."""
        self.selected_features_ = self.selector.pipeline(
            X=X_train,
            y=y_train,
            X_val=X_val,
            y_val=y_val,
            var_threshold=self.config.var_threshold,
            corr_threshold=self.config.corr_threshold,
            corr_method=self.config.corr_method,
            mi_k=self.config.mi_k,
            rfe_feat=self.config.rfe_feat,
            rfe_step=self.config.rfe_step,
            rfe_use_cv=self.config.rfe_use_cv,
            rfe_min_features=self.config.rfe_min_features,
            l1_C=self.config.l1_C,
            shap_k=self.config.shap_k,
            verbose=False,
        )
        X_selected = self.transform(X_train)
        self.model.fit(X_selected, y_train)
        self.fitted_ = True
        LOGGER.info(
            "Pipeline fitted: %s -> %s features",
            X_train.shape[1],
            len(self.selected_features_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project input data into the selected feature space."""
        if not self.selected_features_:
            raise ValueError("Selector is not fitted. Call fit() first.")
        missing = [col for col in self.selected_features_ if col not in X.columns]
        if missing:
            raise ValueError(f"Input data missing selected features: {missing[:5]}")
        return X.loc[:, self.selected_features_].copy()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict class labels on raw input dataframe."""
        if not self.fitted_:
            raise ValueError("Pipeline is not fitted. Call fit() first.")
        preds = self.model.predict(self.transform(X))
        return pd.Series(preds, index=X.index, name="prediction")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Evaluate fitted model on a holdout split."""
        if not self.fitted_:
            raise ValueError("Pipeline is not fitted. Call fit() first.")
        X_selected = self.transform(X_test)
        y_pred = self.model.predict(X_selected)
        y_proba = self.model.predict_proba(X_selected) if hasattr(self.model, "predict_proba") else None
        return compute_metrics(y_test.to_numpy(), y_pred, y_proba=y_proba)

    def save_artifacts(self, output_dir: Path | str) -> dict[str, str]:
        """Persist model, selected features, and pipeline configuration."""
        if not self.fitted_:
            raise ValueError("Pipeline is not fitted. Call fit() first.")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / "production_model.joblib"
        features_path = out_dir / "selected_features.csv"
        config_path = out_dir / "pipeline_config.json"
        ranking_path = out_dir / "feature_rankings.json"

        joblib.dump(self.model, model_path)
        pd.DataFrame({"feature": self.selected_features_}).to_csv(features_path, index=False)
        config_payload = asdict(self.config)
        config_payload.update(
            {
                "selected_features_count": len(self.selected_features_),
                "selected_features": self.selected_features_,
            }
        )
        config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        rankings: dict[str, Any] = {}
        for stage_name, stage_value in self.selector.results_.items():
            rankings[stage_name] = _to_jsonable(stage_value)
        ranking_path.write_text(json.dumps(rankings, indent=2), encoding="utf-8")

        LOGGER.info("Saved artifacts to %s", out_dir)
        return {
            "model_path": str(model_path),
            "features_path": str(features_path),
            "config_path": str(config_path),
            "ranking_path": str(ranking_path),
        }
