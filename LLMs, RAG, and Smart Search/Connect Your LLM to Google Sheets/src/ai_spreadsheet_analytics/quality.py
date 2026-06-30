"""Data quality profiler."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from ai_spreadsheet_analytics.schemas import QualityIssue, QualityReport


class DataQualityProfiler:
    """Profile dataframe quality issues."""

    def profile(self, dataset_key: str, df: pd.DataFrame) -> QualityReport:
        """Build quality report for dataframe.

        Args:
            dataset_key: Dataset identifier.
            df: Input dataframe.

        Returns:
            Structured quality report.
        """
        issues: list[QualityIssue] = []

        missing_counts = df.isna().sum().to_dict()
        missing_cells = int(df.isna().sum().sum())
        if missing_cells > 0:
            issues.append(
                QualityIssue(
                    check_name="missing_values",
                    severity="medium",
                    message="Missing values detected",
                    details={"total_missing": missing_cells, "by_column": missing_counts},
                )
            )

        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows > 0:
            issues.append(
                QualityIssue(
                    check_name="duplicate_rows",
                    severity="medium",
                    message="Duplicate rows detected",
                    details={"duplicate_rows": duplicate_rows},
                )
            )

        empty_cols = [col for col in df.columns if df[col].isna().all()]
        if empty_cols:
            issues.append(
                QualityIssue(
                    check_name="empty_columns",
                    severity="high",
                    message="Empty columns detected",
                    details={"columns": empty_cols},
                )
            )

        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        if constant_cols:
            issues.append(
                QualityIssue(
                    check_name="constant_columns",
                    severity="low",
                    message="Constant columns detected",
                    details={"columns": constant_cols},
                )
            )

        mixed_type_columns: dict[str, dict[str, int]] = {}
        for column in df.columns:
            observed_types = Counter(type(value).__name__ for value in df[column].dropna().head(500))
            if len(observed_types) > 1:
                mixed_type_columns[column] = dict(observed_types)
        if mixed_type_columns:
            issues.append(
                QualityIssue(
                    check_name="mixed_types",
                    severity="medium",
                    message="Mixed data types detected",
                    details={"columns": mixed_type_columns},
                )
            )

        invalid_number_cols: dict[str, int] = {}
        invalid_date_cols: dict[str, int] = {}
        outlier_cols: dict[str, int] = {}
        for column in df.columns:
            series = df[column]
            if pd.api.types.is_string_dtype(series) or series.dtype == object:
                number_candidate = pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")
                candidate_non_na = int(number_candidate.notna().sum())
                if candidate_non_na >= max(5, int(0.5 * len(series.dropna()))):
                    invalid_numbers = int(series.notna().sum() - candidate_non_na)
                    if invalid_numbers > 0:
                        invalid_number_cols[column] = invalid_numbers

                date_candidate = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
                date_non_na = int(date_candidate.notna().sum())
                if date_non_na >= max(5, int(0.5 * len(series.dropna()))):
                    invalid_dates = int(series.notna().sum() - date_non_na)
                    if invalid_dates > 0:
                        invalid_date_cols[column] = invalid_dates

            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() >= 8:
                q1 = float(numeric.quantile(0.25))
                q3 = float(numeric.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = int(((numeric < lower) | (numeric > upper)).sum())
                    if outliers > 0:
                        outlier_cols[column] = outliers

        if invalid_number_cols:
            issues.append(
                QualityIssue(
                    check_name="invalid_numbers",
                    severity="medium",
                    message="Invalid numeric values detected",
                    details={"columns": invalid_number_cols},
                )
            )

        if invalid_date_cols:
            issues.append(
                QualityIssue(
                    check_name="invalid_dates",
                    severity="medium",
                    message="Invalid date values detected",
                    details={"columns": invalid_date_cols},
                )
            )

        if outlier_cols:
            issues.append(
                QualityIssue(
                    check_name="outliers",
                    severity="low",
                    message="Potential outliers detected",
                    details={"columns": outlier_cols},
                )
            )

        metrics: dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "empty_columns": len(empty_cols),
            "constant_columns": len(constant_cols),
        }

        return QualityReport(
            dataset_key=dataset_key,
            row_count=len(df),
            column_count=len(df.columns),
            issues=issues,
            metrics=metrics,
        )

    def profile_bundle(self, frames: dict[str, pd.DataFrame]) -> list[QualityReport]:
        """Profile multiple dataframes."""
        return [self.profile(key, frame) for key, frame in frames.items()]
