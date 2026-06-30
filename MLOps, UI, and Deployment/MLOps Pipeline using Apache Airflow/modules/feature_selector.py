"""Feature selection methods: filter, wrapper, embedded."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, VarianceThreshold

from .data_loader import save_csv
from .feature_engineering import temporal_train_test_split
from .settings import load_config, resolve_path


def _prepare_matrix(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = drop_cols or []
    candidates = df.drop(columns=[target_col, *drop_cols], errors="ignore")
    X = candidates.select_dtypes(include=[np.number]).copy()
    y = df[target_col].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return X, y


def variance_filter(X: pd.DataFrame, threshold: float) -> pd.Series:
    """Return binary mask for variance threshold selection."""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    return pd.Series(selector.get_support(), index=X.columns)


def correlation_ranking(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Absolute target correlation ranking."""
    corr = X.assign(_target=y).corr(numeric_only=True)["_target"].drop("_target")
    return corr.abs().sort_values(ascending=False)


def rfe_ranking(X: pd.DataFrame, y: pd.Series, n_features_to_select: int, random_state: int) -> pd.Series:
    """Wrapper-based feature ranking via RFE + random forest."""
    model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    rfe = RFE(model, n_features_to_select=min(n_features_to_select, X.shape[1]), step=1)
    rfe.fit(X, y)
    # RFE rank 1 = selected best. Convert to descending score.
    scores = pd.Series((1.0 / rfe.ranking_), index=X.columns)
    return scores.sort_values(ascending=False)


def random_forest_importance(X: pd.DataFrame, y: pd.Series, random_state: int) -> pd.Series:
    """Embedded importance via random forest."""
    model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)


def flaml_importance(
    X: pd.DataFrame,
    y: pd.Series,
    time_budget_sec: int,
    random_state: int,
) -> pd.Series:
    """Embedded importance via FLAML-best estimator or fallback."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from flaml import AutoML

        automl = AutoML()
        automl.fit(
            X_train=X,
            y_train=y,
            task="regression",
            time_budget=time_budget_sec,
            seed=random_state,
            verbose=0,
        )
        model = automl.model.estimator if hasattr(automl.model, "estimator") else automl.model
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.abs(model.coef_)
        else:
            # Model has no native importance.
            values = np.zeros(X.shape[1], dtype=float)
        return pd.Series(values, index=X.columns).sort_values(ascending=False)
    except Exception:
        return pd.Series(np.zeros(X.shape[1], dtype=float), index=X.columns)


def run_feature_selection(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run filter/wrapper/embedded selection and save combined ranking."""
    config = config or load_config()
    fs_cfg = config["feature_selection"]
    train_cfg = config["training"]
    project_cfg = config["project"]

    target_col = "target_next_day"
    drop_cols = [project_cfg["date_col"], project_cfg["group_col"], project_cfg["target_col"]]

    train_df, _ = temporal_train_test_split(
        df,
        date_col=str(project_cfg["date_col"]),
        test_size=float(train_cfg["test_size"]),
    )
    X, y = _prepare_matrix(train_df, target_col=target_col, drop_cols=[str(c) for c in drop_cols])

    var_mask = variance_filter(X, threshold=float(fs_cfg["variance_threshold"]))
    corr_rank = correlation_ranking(X, y)
    rfe_rank = rfe_ranking(
        X,
        y,
        n_features_to_select=int(fs_cfg["rfe_num_features"]),
        random_state=int(project_cfg["random_state"]),
    )
    rf_rank = random_forest_importance(X, y, random_state=int(project_cfg["random_state"]))
    flaml_rank = flaml_importance(
        X,
        y,
        time_budget_sec=int(train_cfg["flaml_time_budget_sec"]),
        random_state=int(project_cfg["random_state"]),
    )

    ranking = pd.DataFrame({"feature": X.columns})
    ranking["kept_by_variance_filter"] = ranking["feature"].map(var_mask).fillna(False).astype(bool)
    ranking["corr_abs"] = ranking["feature"].map(corr_rank).fillna(0.0)
    ranking["rfe_score"] = ranking["feature"].map(rfe_rank).fillna(0.0)
    ranking["rf_importance"] = ranking["feature"].map(rf_rank).fillna(0.0)
    ranking["flaml_importance"] = ranking["feature"].map(flaml_rank).fillna(0.0)

    ranking["composite_score"] = (
        ranking["corr_abs"].rank(pct=True)
        + ranking["rfe_score"].rank(pct=True)
        + ranking["rf_importance"].rank(pct=True)
        + ranking["flaml_importance"].rank(pct=True)
    )
    ranking = ranking.sort_values("composite_score", ascending=False).reset_index(drop=True)

    output_path = resolve_path(config, "feature_rankings")
    save_csv(ranking, output_path)
    return ranking
