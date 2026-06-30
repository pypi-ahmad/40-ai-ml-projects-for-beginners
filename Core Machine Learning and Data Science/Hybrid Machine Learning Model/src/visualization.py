from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("darkgrid")



def _save_or_return(fig: plt.Figure, path: str | Path | None = None) -> plt.Figure:
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig



def plot_price_history(df: pd.DataFrame, path: str | Path | None = None, title: str = "Historical Close Price") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_ohlc_lines(df: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in ["Open", "High", "Low", "Close"]:
        ax.plot(df.index, df[col], label=col, linewidth=1)
    ax.set_title("OHLC Price Dynamics")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(ncol=4)
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_volume(df: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(df.index, df["Volume"], width=1.0, alpha=0.7)
    ax.set_title("Trading Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_returns_distribution(df: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    returns = df["Close"].pct_change().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(returns, bins=80, alpha=0.8, edgecolor="black")
    axes[0].set_title("Daily Return Distribution")
    axes[0].set_xlabel("Return")

    sns.boxplot(x=returns, orient="h", ax=axes[1])
    axes[1].set_title("Return Boxplot")
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_train_val_test_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index, train["Close"], label="Train")
    ax.plot(val.index, val["Close"], label="Validation")
    ax.plot(test.index, test["Close"], label="Test")
    ax.set_title("Chronological Train/Validation/Test Split")
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction vs Actual",
    path: str | Path | None = None,
) -> plt.Figure:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="Actual", linewidth=1.2)
    ax.plot(y_pred, label="Predicted", linewidth=1.2)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path | None = None,
) -> plt.Figure:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, s=10, alpha=0.6)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Residuals vs Prediction")
    axes[1].hist(residuals, bins=60, alpha=0.8, edgecolor="black")
    axes[1].set_title("Residual Distribution")
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_error_distribution(errors: np.ndarray, path: str | Path | None = None) -> plt.Figure:
    errors = np.asarray(errors)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(errors, bins=70, alpha=0.85, edgecolor="black")
    ax.set_title("Forecast Error Distribution")
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_model_comparison(
    leaderboard: pd.DataFrame,
    metric_col: str = "test_rmse",
    path: str | Path | None = None,
) -> plt.Figure:
    top = leaderboard.sort_values(metric_col).head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top["model"], top[metric_col], color="steelblue")
    ax.invert_yaxis()
    ax.set_title(f"Model Comparison ({metric_col})")
    ax.set_xlabel(metric_col)
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_feature_importance(
    importance: pd.DataFrame | dict[str, float],
    title: str = "Feature Importance",
    n: int = 20,
    path: str | Path | None = None,
) -> plt.Figure:
    if isinstance(importance, dict):
        df = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())})
    else:
        df = importance.copy()

    if "importance" not in df.columns:
        candidate = [c for c in df.columns if c not in {"feature"}]
        if not candidate:
            raise ValueError("importance dataframe must include 'importance' or value column")
        df = df.rename(columns={candidate[0]: "importance"})

    df = df.sort_values("importance", ascending=False).head(n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["feature"], df["importance"], color="teal")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_backtest_equity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path | None = None,
) -> plt.Figure:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_ret = np.diff(y_true) / np.maximum(y_true[:-1], 1e-8)
    signal = np.sign(np.diff(y_pred))
    strat_ret = signal * true_ret[: len(signal)]

    equity = np.cumprod(1 + strat_ret)
    buy_hold = np.cumprod(1 + true_ret[: len(strat_ret)])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(equity, label="Strategy")
    ax.plot(buy_hold, label="Buy & Hold")
    ax.set_title("Backtest Equity Curve")
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_ensemble_contributions(
    weights: dict[str, float],
    path: str | Path | None = None,
) -> plt.Figure:
    series = pd.Series(weights).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    positions = np.arange(len(series))
    ax.bar(positions, series.values, color="slateblue")
    ax.set_xticks(positions)
    ax.set_xticklabels(series.index, rotation=30, ha="right")
    ax.set_title("Ensemble Weight Contributions")
    ax.set_ylabel("Weight")
    fig.tight_layout()
    return _save_or_return(fig, path)



def plot_forecast_with_interval(
    index: Iterable,
    forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str = "Forecast with Confidence Interval",
    path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(index, forecast, label="Forecast")
    ax.fill_between(index, lower, upper, alpha=0.2, label="Confidence Interval")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, path)
