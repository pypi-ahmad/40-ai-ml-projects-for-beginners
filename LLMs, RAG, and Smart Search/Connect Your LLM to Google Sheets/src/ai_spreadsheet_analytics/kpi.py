"""Automatic KPI detection and calculation."""

from __future__ import annotations

import pandas as pd


class KPIEngine:
    """Detect and compute business KPIs from tabular data."""

    REVENUE_HINTS = {"revenue", "sales", "amount", "gmv", "income", "price", "total"}
    CUSTOMER_HINTS = {"customer", "user", "client", "buyer"}
    PRODUCT_HINTS = {"product", "sku", "item", "category"}

    def detect_columns(self, df: pd.DataFrame) -> dict[str, str | None]:
        """Infer semantic columns by name heuristics."""
        lower_map = {col: col.lower() for col in df.columns}

        revenue_col = self._first_match(lower_map, self.REVENUE_HINTS)
        customer_col = self._first_match(lower_map, self.CUSTOMER_HINTS)
        product_col = self._first_match(lower_map, self.PRODUCT_HINTS)
        date_col = self._find_date_col(df)

        return {
            "revenue": revenue_col,
            "customer": customer_col,
            "product": product_col,
            "date": date_col,
        }

    def compute_kpis(self, df: pd.DataFrame) -> dict[str, float | int | str | None]:
        """Compute generic KPI set for business analytics."""
        semantic = self.detect_columns(df)
        result: dict[str, float | int | str | None] = {
            "records": len(df),
            "columns": len(df.columns),
        }

        revenue_col = semantic["revenue"]
        if revenue_col:
            revenue_series = pd.to_numeric(df[revenue_col], errors="coerce")
            result["total_revenue"] = float(revenue_series.sum(skipna=True))
            result["avg_revenue"] = float(revenue_series.mean(skipna=True))

            date_col = semantic["date"]
            if date_col:
                dated = df.copy()
                dated[date_col] = pd.to_datetime(dated[date_col], errors="coerce", format="mixed")
                dated["_month"] = dated[date_col].dt.to_period("M").astype(str)
                monthly = (
                    dated.dropna(subset=["_month"])
                    .assign(_revenue=pd.to_numeric(dated[revenue_col], errors="coerce"))
                    .groupby("_month", as_index=False)["_revenue"]
                    .sum()
                )
                if len(monthly) >= 2:
                    start = float(monthly.iloc[0]["_revenue"])
                    end = float(monthly.iloc[-1]["_revenue"])
                    if start:
                        result["revenue_growth_rate"] = (end - start) / start

        customer_col = semantic["customer"]
        if customer_col:
            result["unique_customers"] = int(df[customer_col].nunique(dropna=True))

        product_col = semantic["product"]
        if product_col:
            result["unique_products"] = int(df[product_col].nunique(dropna=True))

        return result

    def generate_nl_kpis(self, kpis: dict[str, float | int | str | None]) -> list[str]:
        """Convert KPIs into natural-language statements."""
        lines: list[str] = []
        if "total_revenue" in kpis:
            lines.append(f"Total revenue is {kpis['total_revenue']:.2f}.")
        if "revenue_growth_rate" in kpis and kpis["revenue_growth_rate"] is not None:
            growth = float(kpis["revenue_growth_rate"]) * 100
            lines.append(f"Revenue growth over observed period is {growth:.2f}%.")
        if "unique_customers" in kpis:
            lines.append(f"Dataset contains {int(kpis['unique_customers'])} unique customers.")
        if "unique_products" in kpis:
            lines.append(f"Dataset contains {int(kpis['unique_products'])} unique products.")
        if not lines:
            lines.append("No domain-specific KPI columns were confidently detected.")
        return lines

    def _first_match(self, lower_map: dict[str, str], hints: set[str]) -> str | None:
        for original, lowered in lower_map.items():
            if any(hint in lowered for hint in hints):
                return original
        return None

    def _find_date_col(self, df: pd.DataFrame) -> str | None:
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().sum() >= max(5, int(0.5 * len(df[col].dropna()))):
                return col
        return None
