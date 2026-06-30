"""Deterministic analytics and EDA engine."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from ai_spreadsheet_analytics.advanced import AdvancedAnalyticsEngine
from ai_spreadsheet_analytics.kpi import KPIEngine


class AnalyticsEngine:
    """Deterministic analytics operations for trustworthy business answers."""

    def __init__(self) -> None:
        self.kpi_engine = KPIEngine()
        self.advanced = AdvancedAnalyticsEngine()

    def summary_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return summary statistics for numeric and categorical columns."""
        numeric_summary = (
            df.select_dtypes(include=["number"]).describe().T.reset_index().rename(columns={"index": "column"})
        )

        categorical_cols = df.select_dtypes(exclude=["number"]).columns
        categorical_summary: dict[str, dict[str, Any]] = {}
        for col in categorical_cols:
            counts = df[col].value_counts(dropna=False).head(10)
            categorical_summary[col] = {
                "unique": int(df[col].nunique(dropna=True)),
                "top_values": counts.to_dict(),
            }

        return {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "numeric_summary": numeric_summary.to_dict(orient="records"),
            "categorical_summary": categorical_summary,
        }

    def correlations(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return Pearson correlation matrix for numeric columns."""
        numeric = df.select_dtypes(include=["number"])
        if numeric.shape[1] < 2:
            return {"matrix": {}, "note": "Need at least two numeric columns"}
        corr = numeric.corr(numeric_only=True)
        return {"matrix": corr.round(4).to_dict()}

    def time_series_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate automatic time-series summary if date + value columns exist."""
        date_col = self._detect_date_column(df)
        value_col = self._detect_value_column(df)
        if not date_col or not value_col:
            return {"available": False, "reason": "No date/value columns detected"}

        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce", format="mixed")
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[date_col, value_col])
        if work.empty:
            return {"available": False, "reason": "Date/value conversion produced empty frame"}

        work["month"] = work[date_col].dt.to_period("M").astype(str)
        monthly = work.groupby("month", as_index=False)[value_col].sum()
        monthly["pct_change"] = monthly[value_col].pct_change()
        return {
            "available": True,
            "date_column": date_col,
            "value_column": value_col,
            "monthly": monthly.to_dict(orient="records"),
            "peak_month": monthly.loc[monthly[value_col].idxmax(), "month"],
            "lowest_month": monthly.loc[monthly[value_col].idxmin(), "month"],
        }

    def top_bottom(self, df: pd.DataFrame, column: str, n: int = 5) -> dict[str, list[dict[str, Any]]]:
        """Return top/bottom N rows by column."""
        numeric = df.copy()
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        numeric = numeric.dropna(subset=[column])
        top = numeric.nlargest(n, column).to_dict(orient="records")
        bottom = numeric.nsmallest(n, column).to_dict(orient="records")
        return {"top": top, "bottom": bottom}

    def feature_importance_proxy(self, df: pd.DataFrame, target_column: str | None = None) -> dict[str, float]:
        """Proxy feature importance via absolute correlation to target."""
        numeric = df.select_dtypes(include=["number"])
        if numeric.shape[1] < 2:
            return {}

        target = target_column or self._detect_value_column(df)
        if not target or target not in numeric.columns:
            target = numeric.columns[-1]

        corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
        return corr.abs().sort_values(ascending=False).round(4).to_dict()

    def detect_anomalies(self, df: pd.DataFrame, column: str | None = None) -> dict[str, Any]:
        """Detect anomalies using IQR bounds."""
        target_col = column or self._detect_value_column(df)
        if not target_col:
            return {"available": False, "reason": "No numeric value column detected"}

        numeric = pd.to_numeric(df[target_col], errors="coerce")
        numeric = numeric.dropna()
        if len(numeric) < 8:
            return {"available": False, "reason": "Insufficient numeric rows"}

        q1 = float(numeric.quantile(0.25))
        q3 = float(numeric.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (pd.to_numeric(df[target_col], errors="coerce") < lower) | (
            pd.to_numeric(df[target_col], errors="coerce") > upper
        )
        anomalies = df[mask].head(20).to_dict(orient="records")
        return {
            "available": True,
            "column": target_col,
            "lower_bound": lower,
            "upper_bound": upper,
            "count": int(mask.sum()),
            "samples": anomalies,
        }

    def run_full_eda(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run full deterministic EDA pack."""
        value_col = self._detect_value_column(df)
        date_col = self._detect_date_column(df)
        top_bottom_payload = self.top_bottom(df, value_col) if value_col else {"top": [], "bottom": []}

        kpis = self.kpi_engine.compute_kpis(df)
        nl_kpis = self.kpi_engine.generate_nl_kpis(kpis)

        schema = [asdict(field) for field in self.advanced.infer_schema(df)]
        glossary = self.advanced.business_glossary(df)
        trend = (
            self.advanced.detect_trend(df, date_col, value_col)
            if date_col and value_col
            else {"available": False, "reason": "No date/value columns detected"}
        )
        forecast = self.advanced.forecast(df, date_col, value_col, periods=3) if date_col and value_col else []
        clusters = (
            self.advanced.cluster(df, value_col, n_clusters=3).head(50).to_dict(orient="records")
            if value_col
            else []
        )

        return {
            "summary": self.summary_stats(df),
            "correlations": self.correlations(df),
            "time_series": self.time_series_summary(df),
            "top_bottom": top_bottom_payload,
            "feature_importance_proxy": self.feature_importance_proxy(df, value_col),
            "anomalies": self.detect_anomalies(df, value_col),
            "kpis": kpis,
            "kpi_narratives": nl_kpis,
            "schema_inference": schema,
            "business_glossary": glossary,
            "trend_detection": trend,
            "forecast": forecast,
            "clusters": clusters,
        }

    def _detect_date_column(self, df: pd.DataFrame) -> str | None:
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().sum() >= max(5, int(0.5 * len(df[col].dropna()))):
                return col
        return None

    def _detect_value_column(self, df: pd.DataFrame) -> str | None:
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                continue
            parsed_date = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed_date.notna().sum() >= max(5, int(0.8 * len(series.dropna()))):
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() >= max(5, int(0.5 * len(df[col].dropna()))):
                return col
        return None
