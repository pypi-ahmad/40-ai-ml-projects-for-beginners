"""Data loading, schema checks, and data-quality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    HIGH_RISK_COLUMN,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


@dataclass(slots=True)
class DataQualityReport:
    """Structured report for data quality checks."""

    row_count: int
    column_count: int
    schema_valid: bool
    missing_by_column: dict[str, int]
    duplicate_rows: int
    invalid_ranges: dict[str, int]
    iqr_outliers: dict[str, int]
    blocking_issues: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for JSON/reporting."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "schema_valid": self.schema_valid,
            "missing_by_column": self.missing_by_column,
            "duplicate_rows": self.duplicate_rows,
            "invalid_ranges": self.invalid_ranges,
            "iqr_outliers": self.iqr_outliers,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
        }


class LoanDataLoader:
    """Load loan recovery dataset and run schema/quality validation."""

    def __init__(self, data_path: str | None = None) -> None:
        self.data_path = str(data_path or DATA_PATH)

    def load(self, add_target_derivatives: bool = False) -> pd.DataFrame:
        """Load CSV as DataFrame and enforce basic schema invariants."""
        df = pd.read_csv(self.data_path)
        self.validate_schema(df)

        # Recovery status is expected to be integer-coded [0,1,2].
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype("Int64")
        if df[TARGET_COLUMN].isna().any():
            raise ValueError("Target column contains non-numeric values that cannot be parsed.")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

        if add_target_derivatives:
            # Optional convenience column for descriptive analysis only.
            df[HIGH_RISK_COLUMN] = (df[TARGET_COLUMN] == 2).astype(int)
        return df

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> None:
        """Validate required column presence and basic target validity."""
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

    @staticmethod
    def class_distribution(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.Series:
        """Return sorted target class distribution."""
        return df[target_col].value_counts(normalize=False).sort_index()

    def quality_report(self, df: pd.DataFrame) -> DataQualityReport:
        """Run practical quality checks with business-relevant invalid-range rules."""
        missing = df.isna().sum().to_dict()
        duplicate_rows = int(df.duplicated().sum())
        schema_valid = all(col in df.columns for col in REQUIRED_COLUMNS)

        invalid_ranges = {
            "Age_outside_18_100": int(((df["Age"] < 18) | (df["Age"] > 100)).sum()),
            "Interest_Rate_outside_0_100": int(((df["Interest_Rate"] < 0) | (df["Interest_Rate"] > 100)).sum()),
            "Credit_Score_outside_300_900": int(((df["Credit_Score"] < 300) | (df["Credit_Score"] > 900)).sum()),
            "Negative_financial_values": int(
                (
                    (df["Monthly_Income"] < 0)
                    | (df["Loan_Amount"] < 0)
                    | (df["Outstanding_Balance"] < 0)
                    | (df["Collateral_Value"] < 0)
                ).sum()
            ),
            "Negative_behavioral_values": int(
                ((df["Missed_Payments"] < 0) | (df["Days_Past_Due"] < 0) | (df["Collection_Attempts"] < 0)).sum()
            ),
            "Debt_to_Income_outside_0_3": int(
                ((df["Debt_to_Income_Ratio"] < 0) | (df["Debt_to_Income_Ratio"] > 3)).sum()
            ),
            "Outstanding_exceeds_loan_amount": int((df["Outstanding_Balance"] > df["Loan_Amount"]).sum()),
            "Recovery_Status_outside_0_2": int((~df[TARGET_COLUMN].isin([0, 1, 2])).sum()),
        }

        iqr_outliers: dict[str, int] = {}
        for col in [col for col in NUMERIC_COLUMNS if col in df.columns]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_outliers[col] = int(((df[col] < lower) | (df[col] > upper)).sum())

        blocking_issues: list[str] = []
        warnings: list[str] = []

        if not schema_valid:
            blocking_issues.append("Schema mismatch: one or more required columns are missing.")
        if any(v > 0 for k, v in invalid_ranges.items() if "outside" in k and "Outstanding_exceeds_loan_amount" not in k):
            blocking_issues.append("Invalid numeric ranges found in critical fields.")
        if duplicate_rows > 0:
            warnings.append(f"{duplicate_rows} duplicate rows detected.")
        if invalid_ranges["Outstanding_exceeds_loan_amount"] > 0:
            warnings.append(
                f"{invalid_ranges['Outstanding_exceeds_loan_amount']} loans have outstanding balance above original loan amount."
            )
        if sum(missing.values()) > 0:
            warnings.append("Dataset contains missing values that require imputation policy.")

        return DataQualityReport(
            row_count=len(df),
            column_count=df.shape[1],
            schema_valid=schema_valid,
            missing_by_column={k: int(v) for k, v in missing.items()},
            duplicate_rows=duplicate_rows,
            invalid_ranges=invalid_ranges,
            iqr_outliers=iqr_outliers,
            blocking_issues=blocking_issues,
            warnings=warnings,
        )

    @staticmethod
    def split_features_target(
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
        drop_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Split model input features and target."""
        drop_cols = drop_cols or []
        x = df.drop(columns=[target_col, *drop_cols], errors="ignore")
        y = df[target_col].copy()
        return x, y

    @staticmethod
    def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Infer numeric/categorical columns from current frame."""
        categorical = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        numeric = [c for c in df.columns if c not in categorical and pd.api.types.is_numeric_dtype(df[c])]
        return numeric, categorical
