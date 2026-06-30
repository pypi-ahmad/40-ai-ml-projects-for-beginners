"""Feature engineering and leakage-safe feature selection for loan recovery analytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_COLUMNS,
    COLLECTION_STAGE_COLUMNS,
    EARLY_WARNING_EXCLUDED_COLUMNS,
    HIGH_RISK_COLUMN,
    RANDOM_STATE,
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
    TARGET_DERIVED_COLUMNS,
)


@dataclass(slots=True)
class DataSplit:
    """Train/test split container."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(slots=True)
class DataSplitWithValidation:
    """Train/validation/test split container."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


class FeatureEngineer:
    """Create finance-aware engineered features and leakage-safe model matrices."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    @staticmethod
    def _safe_divide(num: pd.Series, den: pd.Series, epsilon: float = 1e-6) -> pd.Series:
        """Safely divide two series while handling zero denominators."""
        return num / np.where(np.abs(den) < epsilon, epsilon, den)

    @staticmethod
    def _clip01(values: pd.Series) -> pd.Series:
        return values.clip(lower=0.0, upper=1.0)

    def engineer(self, df: pd.DataFrame, include_target_derivatives: bool = False) -> pd.DataFrame:
        """Add advanced financial and delinquency features.

        Args:
            df: Raw borrower-level frame.
            include_target_derivatives: When True and target exists, adds target-derived helper flags.

        Returns:
            Enriched DataFrame with interpretable, bounded risk features.
        """
        frame = df.copy()

        monthly_rate = frame["Interest_Rate"] / 100.0 / 12.0
        term = frame["Loan_Term_Months"].clip(lower=1)
        loan_amount = frame["Loan_Amount"].clip(lower=0)

        # EMI formula for amortizing loans: P * r * (1+r)^n / ((1+r)^n - 1)
        emi = (
            loan_amount
            * monthly_rate
            * np.power(1 + monthly_rate, term)
            / np.clip(np.power(1 + monthly_rate, term) - 1, self.epsilon, None)
        )
        frame["Estimated_EMI"] = np.where(monthly_rate > 0, emi, loan_amount / term)

        annual_income = (frame["Monthly_Income"] * 12).clip(lower=self.epsilon)
        frame["Loan_to_Income_Ratio"] = self._safe_divide(frame["Loan_Amount"], annual_income, self.epsilon)
        frame["EMI_to_Income_Ratio"] = self._safe_divide(frame["Estimated_EMI"], frame["Monthly_Income"], self.epsilon)

        frame["Debt_Burden_Score"] = (
            0.55 * frame["Debt_to_Income_Ratio"].clip(lower=0)
            + 0.45 * frame["EMI_to_Income_Ratio"].clip(lower=0)
        )

        frame["Collateral_Coverage_Ratio"] = self._safe_divide(
            frame["Collateral_Value"], frame["Outstanding_Balance"] + 1.0, self.epsilon
        )

        # Lower is better; log keeps the feature on a stable scale.
        frame["Recovery_Efficiency_Ratio"] = self._safe_divide(
            np.log1p(frame["Outstanding_Balance"]),
            frame["Collection_Attempts"] + 1.0,
            self.epsilon,
        )

        frame["Risk_Exposure_Score"] = (
            np.log1p(frame["Outstanding_Balance"])
            * (1 + frame["Interest_Rate"] / 100.0)
            * (1 + frame["Previous_Defaults"] * 0.20)
        )

        missed_component = self._clip01(self._safe_divide(frame["Missed_Payments"], pd.Series(12, index=frame.index), self.epsilon))
        dpd_component = self._clip01(self._safe_divide(frame["Days_Past_Due"], pd.Series(180, index=frame.index), self.epsilon))
        credit_component = self._clip01(self._safe_divide(700 - frame["Credit_Score"], pd.Series(400, index=frame.index), self.epsilon))
        default_component = self._clip01(self._safe_divide(frame["Previous_Defaults"], pd.Series(3, index=frame.index), self.epsilon))

        frame["Missed_Payment_Severity"] = self._safe_divide(
            frame["Missed_Payments"] * np.log1p(frame["Days_Past_Due"]),
            frame["Loan_Term_Months"],
            self.epsilon,
        )
        frame["Delinquency_Score"] = 0.50 * dpd_component + 0.30 * missed_component + 0.20 * credit_component

        unsecured_component = self._clip01(
            self._safe_divide(frame["Outstanding_Balance"], frame["Collateral_Value"] + 1.0, self.epsilon) / 2.0
        )
        frame["Recovery_Difficulty_Index"] = (
            0.45 * frame["Delinquency_Score"]
            + 0.35 * unsecured_component
            + 0.20 * default_component
        )

        frame["Collection_Intensity_Score"] = self._safe_divide(
            frame["Collection_Attempts"], frame["Days_Past_Due"] + 30.0, self.epsilon
        )
        frame["Behavioral_Risk_Score"] = 0.50 * missed_component + 0.30 * dpd_component + 0.20 * default_component

        if include_target_derivatives and TARGET_COLUMN in frame.columns:
            frame[HIGH_RISK_COLUMN] = (frame[TARGET_COLUMN] == 2).astype(int)

        for col in CATEGORICAL_COLUMNS:
            if col in frame.columns:
                frame[col] = frame[col].astype(str)

        return frame

    def model_feature_columns(
        self,
        columns: list[str],
        include_sensitive: bool = False,
        early_warning: bool = True,
        drop_cols: list[str] | None = None,
    ) -> list[str]:
        """Return leakage-safe model feature list based on governance policy."""
        drop_cols = drop_cols or []
        excluded = set(drop_cols)
        excluded.update(TARGET_DERIVED_COLUMNS)

        if early_warning:
            excluded.update(EARLY_WARNING_EXCLUDED_COLUMNS)
        elif not include_sensitive:
            excluded.update(SENSITIVE_COLUMNS)

        return [col for col in columns if col not in excluded and col != TARGET_COLUMN]

    def split_features_target(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
        drop_cols: list[str] | None = None,
        include_sensitive: bool = False,
        early_warning: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build leakage-safe feature matrix and target vector."""
        drop_cols = drop_cols or []
        cols = self.model_feature_columns(
            list(df.columns),
            include_sensitive=include_sensitive,
            early_warning=early_warning,
            drop_cols=[target_col, *drop_cols],
        )
        x = df[cols].copy()
        y = df[target_col].copy()
        return x, y

    def train_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
        test_size: float = 0.2,
        random_state: int = RANDOM_STATE,
        drop_cols: list[str] | None = None,
        include_sensitive: bool = False,
        early_warning: bool = True,
    ) -> DataSplit:
        """Create stratified train/test split after leakage-safe feature construction."""
        enriched = self.engineer(df, include_target_derivatives=False)
        x, y = self.split_features_target(
            enriched,
            target_col=target_col,
            drop_cols=drop_cols,
            include_sensitive=include_sensitive,
            early_warning=early_warning,
        )
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        return DataSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    def train_val_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = RANDOM_STATE,
        drop_cols: list[str] | None = None,
        include_sensitive: bool = False,
        early_warning: bool = True,
    ) -> DataSplitWithValidation:
        """Create stratified train/validation/test split for threshold tuning and calibration."""
        enriched = self.engineer(df, include_target_derivatives=False)
        x, y = self.split_features_target(
            enriched,
            target_col=target_col,
            drop_cols=drop_cols,
            include_sensitive=include_sensitive,
            early_warning=early_warning,
        )

        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        val_fraction = val_size / (1.0 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval,
            y_trainval,
            test_size=val_fraction,
            random_state=random_state,
            stratify=y_trainval,
        )

        return DataSplitWithValidation(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

