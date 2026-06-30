"""Data cleaning engine."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ai_spreadsheet_analytics.schemas import CleaningResult, CleaningStrategy

_CURRENCY_PATTERN = re.compile(r"[^0-9.\-]")


class DataCleaner:
    """Apply configurable cleaning strategies."""

    def clean(self, dataset_key: str, df: pd.DataFrame, strategy: CleaningStrategy) -> CleaningResult:
        """Clean one dataframe.

        Args:
            dataset_key: Dataset identifier.
            df: Input dataframe.
            strategy: Cleaning strategy.

        Returns:
            Cleaning result with action log.
        """
        cleaned = df.copy()
        actions: list[str] = []

        if strategy.drop_duplicate_rows:
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            removed = before - len(cleaned)
            if removed > 0:
                actions.append(f"Removed {removed} duplicate rows")

        if strategy.drop_empty_columns:
            empty_cols = [col for col in cleaned.columns if cleaned[col].isna().all()]
            if empty_cols:
                cleaned = cleaned.drop(columns=empty_cols)
                actions.append(f"Dropped empty columns: {', '.join(empty_cols)}")

        if strategy.drop_constant_columns:
            constant_cols = [col for col in cleaned.columns if cleaned[col].nunique(dropna=False) <= 1]
            if constant_cols:
                cleaned = cleaned.drop(columns=constant_cols)
                actions.append(f"Dropped constant columns: {', '.join(constant_cols)}")

        if strategy.parse_currency:
            currency_columns = self._detect_currency_columns(cleaned)
            for column in currency_columns:
                cleaned[column] = cleaned[column].map(self._parse_currency)
            if currency_columns:
                actions.append(f"Parsed currency columns: {', '.join(currency_columns)}")

        if strategy.parse_percentage:
            percent_columns = self._detect_percent_columns(cleaned)
            for column in percent_columns:
                cleaned[column] = cleaned[column].map(self._parse_percentage)
            if percent_columns:
                actions.append(f"Parsed percentage columns: {', '.join(percent_columns)}")

        if strategy.normalize_dates:
            date_columns = self._detect_date_columns(cleaned)
            for column in date_columns:
                cleaned[column] = pd.to_datetime(
                    cleaned[column], errors="coerce", utc=True, format="mixed"
                ).dt.date
            if date_columns:
                actions.append(f"Normalized date columns: {', '.join(date_columns)}")

        cleaned = self._handle_missing(cleaned, strategy.missing_value_strategy, actions)

        return CleaningResult(dataset_key=dataset_key, cleaned=cleaned, actions=actions)

    def _handle_missing(self, df: pd.DataFrame, strategy: str, actions: list[str]) -> pd.DataFrame:
        updated = df.copy()
        numeric_cols = updated.select_dtypes(include=["number"]).columns

        if strategy == "drop_rows":
            before = len(updated)
            updated = updated.dropna().reset_index(drop=True)
            actions.append(f"Dropped {before - len(updated)} rows with missing values")
        elif strategy == "drop_columns":
            cols_before = len(updated.columns)
            updated = updated.dropna(axis=1)
            actions.append(f"Dropped {cols_before - len(updated.columns)} columns with missing values")
        elif strategy == "mean":
            for col in numeric_cols:
                updated[col] = updated[col].fillna(updated[col].mean())
            updated = updated.fillna("Unknown")
            actions.append("Imputed numeric missing with mean")
        elif strategy == "median":
            for col in numeric_cols:
                updated[col] = updated[col].fillna(updated[col].median())
            updated = updated.fillna("Unknown")
            actions.append("Imputed numeric missing with median")
        elif strategy == "mode":
            for col in updated.columns:
                mode = updated[col].mode(dropna=True)
                if not mode.empty:
                    updated[col] = updated[col].fillna(mode.iloc[0])
            actions.append("Imputed missing values with mode")
        elif strategy == "zero":
            for col in numeric_cols:
                updated[col] = updated[col].fillna(0)
            updated = updated.fillna("Unknown")
            actions.append("Filled numeric missing with zero")
        else:
            actions.append(f"Unknown missing strategy '{strategy}', skipped missing value treatment")
        return updated

    def _detect_currency_columns(self, df: pd.DataFrame) -> list[str]:
        currency_cols: list[str] = []
        for column in df.columns:
            if pd.api.types.is_string_dtype(df[column]) or df[column].dtype == object:
                sample = df[column].dropna().astype(str).head(20)
                if any("$" in value or "€" in value or "£" in value or "₹" in value for value in sample):
                    currency_cols.append(column)
        return currency_cols

    def _detect_percent_columns(self, df: pd.DataFrame) -> list[str]:
        percent_cols: list[str] = []
        for column in df.columns:
            if pd.api.types.is_string_dtype(df[column]) or df[column].dtype == object:
                sample = df[column].dropna().astype(str).head(20)
                if any("%" in value for value in sample):
                    percent_cols.append(column)
        return percent_cols

    def _detect_date_columns(self, df: pd.DataFrame) -> list[str]:
        date_cols: list[str] = []
        for column in df.columns:
            series = df[column]
            if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
                continue
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().sum() >= max(5, int(0.6 * series.notna().sum())):
                date_cols.append(column)
        return date_cols

    def _parse_currency(self, value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        cleaned = _CURRENCY_PATTERN.sub("", str(value))
        if cleaned in {"", "-", "."}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _parse_percentage(self, value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip().replace("%", "")
        try:
            return float(text) / 100
        except ValueError:
            return None
