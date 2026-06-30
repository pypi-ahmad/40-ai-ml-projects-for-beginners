"""Configuration and canonical schema for Smart Loan Recovery System."""

from __future__ import annotations

from pathlib import Path

# Reproducibility seed used across the entire project.
RANDOM_STATE: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_PATH: Path = PROJECT_ROOT / "loan-recovery.csv"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

TARGET_COLUMN: str = "Recovery_Status"
ID_COLUMN: str = "Borrower_ID"
HIGH_RISK_COLUMN: str = "High_Risk_Flag"

# Recovery states in this dataset are encoded as integers.
TARGET_LABELS: dict[int, str] = {
    0: "Fully Recovered",
    1: "Partially Recovered",
    2: "Written Off",
}

NUMERIC_COLUMNS: list[str] = [
    "Age",
    "Monthly_Income",
    "Loan_Amount",
    "Loan_Term_Months",
    "Interest_Rate",
    "Credit_Score",
    "Days_Past_Due",
    "Missed_Payments",
    "Num_Dependents",
    "Years_At_Current_Address",
    "Outstanding_Balance",
    "Collateral_Value",
    "Debt_to_Income_Ratio",
    "Collection_Attempts",
    "Previous_Defaults",
]

CATEGORICAL_COLUMNS: list[str] = [
    "Employment_Status",
    "Loan_Purpose",
    "Education_Level",
    "Marital_Status",
    "Residence_Type",
]

# Features with governance or leakage sensitivity.
SENSITIVE_COLUMNS: list[str] = [
    "Marital_Status",
    "Residence_Type",
]
COLLECTION_STAGE_COLUMNS: list[str] = [
    "Collection_Attempts",
]
TARGET_DERIVED_COLUMNS: list[str] = [
    HIGH_RISK_COLUMN,
]

# Collections/post-action fields that should be excluded for early-warning predictive modeling.
EARLY_WARNING_EXCLUDED_COLUMNS: list[str] = [
    *SENSITIVE_COLUMNS,
    *COLLECTION_STAGE_COLUMNS,
    *TARGET_DERIVED_COLUMNS,
    "Recovery_Efficiency_Ratio",
    "Collection_Intensity_Score",
]

REQUIRED_COLUMNS: list[str] = [ID_COLUMN, *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, TARGET_COLUMN]

# Benchmark model list required by project scope.
BASELINE_MODELS: list[str] = [
    "Logistic Regression",
    "Random Forest",
    "Extra Trees",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "AdaBoost",
    "Gradient Boosting",
    "SVM",
    "KNN",
]

# Default business costs (can be tuned by business team).
DEFAULT_FALSE_POSITIVE_COST: float = 250.0
DEFAULT_FALSE_NEGATIVE_COST: float = 2500.0

# Available strategy actions. The strategy engine picks actions dynamically based on data-driven risk tiers.
STRATEGY_ACTIONS: dict[str, str] = {
    "Low": "Automated reminders via SMS/email and self-service portal nudges",
    "Medium": "Personalized follow-up plus flexible settlement options",
    "High": "Intensive collections with senior agent intervention",
    "Very High": "Legal notice preparation and debt recovery escalation",
}
