"""Generate educational notebook suite for Smart Loan Recovery System."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def edu_block(topic: str, business_example: str) -> str:
    """Return a standard educational block used across notebooks."""
    return f"""
## Definition
{topic}

## Theory
This section explains the statistical or ML theory behind the technique and why it is useful in credit recovery operations.

## Mathematical Intuition
We translate the idea into score/probability/ranking logic so each metric can be interpreted quantitatively.

## Financial Intuition
We connect the method to borrower affordability, delinquency severity, collateral protection, and expected recoverable cashflows.

## Business Impact
We explain what this enables for collection managers, risk teams, and executives.

## Real-World Example
{business_example}

## Visual Explanation
Charts in this notebook show how model/segment behavior changes across borrower groups.

## Code Explanation
Every code cell below is paired with interpretation so beginners can connect implementation details to business outcomes.

## Interpretation of Results
We summarize what changed, why it matters, and how to act on it.
"""


COMMON_IMPORT = """
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loan_recovery import (
    DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    LoanDataLoader,
    FeatureEngineer,
    LoanEDA,
    BorrowerSegmenter,
    ModelTrainer,
    ModelEvaluator,
    RecoveryStrategyEngine,
    ModelExplainer,
    DashboardBuilder,
    LazyPredictBenchmark,
    PyCaretWorkflow,
    FLAMLOptimizer,
    SmartLoanRecoveryPipeline,
    load_model,
)

sns.set_theme(style="whitegrid")
"""

ENSURE_ARTIFACTS = """
def ensure_pipeline_artifacts() -> None:
    required = [
        TABLES_DIR / "manual_model_leaderboard.csv",
        TABLES_DIR / "feature_enriched_data.csv",
        MODELS_DIR / "best_manual_model.joblib",
    ]
    if not all(path.exists() for path in required):
        pipeline = SmartLoanRecoveryPipeline(data_path=DATA_PATH, random_state=42)
        pipeline.run()

ensure_pipeline_artifacts()
"""


def write_notebook(path: Path, cells: list):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def notebook_01():
    cells = [
        md(
            """
# 01 - Loan Recovery Foundations and EDA

## Definition
**Loan recovery** is the process of collecting unpaid loan amounts after delinquency/default.

## Theory
Banks estimate expected recovery to reduce credit losses, optimize collection effort, and improve portfolio quality.

## Mathematical intuition
Expected recovery value per borrower can be approximated as:
\\[
\\mathbb{E}[Recovery] = Outstanding\\_Balance \\times P(Recovery)
\\]

## Financial intuition
A borrower with high outstanding balance and high delinquency can destroy profitability if recovery fails.

## Business impact
Strong recovery analytics improves cashflow, reduces NPA pressure, and guides legal vs non-legal actions.
            """
        ),
        md(edu_block("Loan recovery EDA frames where losses originate and how recoverability differs across borrower profiles.", "Borrowers with high days-past-due and weak collateral often require earlier intervention.")),
        code(COMMON_IMPORT),
        code(ENSURE_ARTIFACTS),
        md(
            """
## Banking Context: Why This Matters
- **Banking**: controls provisioning and capital adequacy impact from bad loans.
- **NBFCs**: reduces collection cost per recovered dollar.
- **FinTech lenders**: enables faster early-warning decisions at scale.
- **Microfinance institutions**: helps prioritize high-touch outreach where social + financial risk is high.
            """
        ),
        code(
            """
loader = LoanDataLoader(DATA_PATH)
df = loader.load()
print(f"Dataset shape: {df.shape}")
display(df.head())
            """
        ),
        md(
            """
## Data Quality Assessment
### Real-world meaning
Missing values, invalid ranges, and outliers are not only technical issues, they are operational risk signals.
            """
        ),
        code(
            """
quality = loader.quality_report(df)
quality_df = pd.DataFrame([
    {"check": k, "count": v} for k, v in quality.invalid_ranges.items()
])
display(pd.DataFrame([quality.to_dict()]).T)
display(quality_df)
            """
        ),
        md(
            """
## Exploratory Data Analysis
We inspect borrower, loan, behavior, and target patterns using histograms, KDEs, boxplots, violin plots, and heatmaps.
            """
        ),
        code(
            """
fe = FeatureEngineer()
enriched = fe.engineer(df)
eda = LoanEDA(enriched)
outputs = eda.generate_all_plots(FIGURES_DIR)
display(eda.summary_table().head(15))
display(eda.relationship_analysis())
outputs
            """
        ),
        code(
            """
from IPython.display import Image, display

for image_path in [
    outputs.target_distribution,
    outputs.correlation_heatmap,
    outputs.numeric_histograms,
    outputs.boxplots,
    outputs.violin_plots,
    outputs.relationship_grid,
]:
    display(Image(filename=str(image_path)))
            """
        ),
        md(
            """
## Interpretation
- Higher `Days_Past_Due` and `Missed_Payments` often align with weaker recovery states.
- Borrowers with stronger collateral coverage generally look easier to recover.
- Income, debt burden, and repayment behavior jointly shape collection strategy decisions.
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "01_loan_recovery_foundations_eda.ipynb", cells)


def notebook_02():
    cells = [
        md(
            """
# 02 - Data Quality and Feature Engineering

## Definition
Feature engineering translates raw loan records into risk-relevant signals.

## Theory
Raw variables are often weak predictors in isolation. Ratios, interaction terms, and domain scores capture borrower stress and recoverability.

## Financial + business intuition
Collections teams act on burden, delinquency severity, and collateral strength, not just raw income or loan amount.
            """
        ),
        md(edu_block("Feature engineering converts raw borrower and loan records into risk-relevant explanatory signals.", "A borrower with low EMI-to-income and high collateral coverage is usually easier to recover.")),
        code(COMMON_IMPORT),
        code(ENSURE_ARTIFACTS),
        code(
            """
df = LoanDataLoader(DATA_PATH).load()
fe = FeatureEngineer()
enriched = fe.engineer(df)
print(enriched.shape)
display(enriched.head())
            """
        ),
        md(
            """
## Engineered Features (with business purpose)
- `Loan_to_Income_Ratio`: affordability pressure
- `Debt_Burden_Score`: repayment stress
- `Collateral_Coverage_Ratio`: lender protection buffer
- `Recovery_Efficiency_Ratio`: outstanding value per collection attempt
- `EMI_to_Income_Ratio`: monthly installment stress
- `Risk_Exposure_Score`: expected risk-weighted exposure
- `Missed_Payment_Severity`: delinquency intensity
- `Delinquency_Score`: normalized delay risk
- `Recovery_Difficulty_Index`: combined recoverability friction
- `Collection_Intensity_Score`: engagement intensity
- `Behavioral_Risk_Score`: repayment behavior health
            """
        ),
        code(
            """
engineered_cols = [
    "Loan_to_Income_Ratio",
    "Debt_Burden_Score",
    "Collateral_Coverage_Ratio",
    "Recovery_Efficiency_Ratio",
    "EMI_to_Income_Ratio",
    "Risk_Exposure_Score",
    "Missed_Payment_Severity",
    "Delinquency_Score",
    "Recovery_Difficulty_Index",
    "Collection_Intensity_Score",
    "Behavioral_Risk_Score",
]
display(enriched[engineered_cols + ["Recovery_Status"]].describe().T)
            """
        ),
        code(
            """
corr = enriched[engineered_cols + ["Recovery_Status"]].corr()[["Recovery_Status"]].sort_values("Recovery_Status")
display(corr)

plt.figure(figsize=(8, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Engineered Feature Correlation with Recovery Status")
plt.show()
            """
        ),
        md(
            """
## Interpretation
The strongest engineered features should be reused in segmentation, risk scoring, and strategy assignment to keep business logic consistent.
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "02_data_quality_feature_engineering.ipynb", cells)


def notebook_03():
    cells = [
        md(
            """
# 03 - Borrower Segmentation

## Theory
Segmentation groups borrowers by behavior and exposure patterns to enable differentiated collection strategy.

## Algorithms compared
1. K-Means
2. MiniBatch K-Means
3. Hierarchical Clustering
4. Gaussian Mixture Models
5. DBSCAN

## Evaluation metrics
- Silhouette Score (higher better)
- Davies-Bouldin Index (lower better)
- Calinski-Harabasz Index (higher better)
            """
        ),
        md(edu_block("Segmentation groups borrowers into actionable cohorts for differentiated recovery treatment.", "A Legal Escalation Candidate segment can be routed to legal preparation while Recovery Friendly gets lighter outreach.")),
        code(COMMON_IMPORT),
        code(ENSURE_ARTIFACTS),
        code(
            """
enriched = pd.read_csv(TABLES_DIR / "feature_enriched_data.csv")
segmenter = BorrowerSegmenter(random_state=42)
seg_out = segmenter.run(enriched)

display(seg_out.metrics_table)
display(seg_out.profile_table)
seg_out.named_segments
            """
        ),
        code(
            """
from IPython.display import Image, display
display(Image(filename=str(FIGURES_DIR / "segmentation_metrics.png")))
display(Image(filename=str(FIGURES_DIR / "segmentation_pca.png")))
            """
        ),
        md(
            """
## Business Interpretation
Segment names are automatically generated from profile patterns (income, risk, burden, recoverability) so collections teams can act immediately.
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "03_borrower_segmentation.ipynb", cells)


def notebook_04():
    cells = [
        md(
            """
# 04 - Risk Prediction Baselines and LazyPredict

## Problem framing
We model `Recovery_Status` as a multi-class prediction task and derive high-risk borrowers from the written-off class.

## Why LazyPredict?
- Fast baseline exploration
- Useful for rapid model sanity checks
- Lower control than hand-crafted pipelines
            """
        ),
        md(edu_block("Risk prediction estimates recovery outcomes and supports high-risk early warning decisions.", "Two borrowers with similar outstanding balances can receive different actions if their written-off probability diverges.")),
        code(COMMON_IMPORT),
        code(ENSURE_ARTIFACTS),
        code(
            """
df = LoanDataLoader(DATA_PATH).load()
fe = FeatureEngineer()
enriched = fe.engineer(df)
split = fe.train_test_split(enriched, target_col="Recovery_Status", drop_cols=["Borrower_ID"])

trainer = ModelTrainer(random_state=42)
manual = trainer.train_baselines(split.x_train, split.y_train, split.x_test, split.y_test)
display(manual.leaderboard)
            """
        ),
        code(
            """
x_train_lazy = pd.get_dummies(split.x_train)
x_test_lazy = pd.get_dummies(split.x_test)
x_train_lazy, x_test_lazy = x_train_lazy.align(x_test_lazy, axis=1, join="left", fill_value=0)

lazy = LazyPredictBenchmark(random_state=42)
lazy_df = lazy.run(x_train_lazy, split.y_train, x_test_lazy, split.y_test)
display(lazy_df)
display(lazy.required_model_snapshot())
            """
        ),
        code(
            """
manual_top = manual.leaderboard[["model", "accuracy", "f1_weighted", "roc_auc_ovr"]].copy()
manual_top["source"] = "Manual"

lazy_top = lazy_df[["model", "Accuracy", "F1 Score"]].copy()
lazy_top.columns = ["model", "accuracy", "f1_weighted"]
lazy_top["roc_auc_ovr"] = np.nan
lazy_top["source"] = "LazyPredict"

comparison = pd.concat([manual_top, lazy_top], ignore_index=True)
display(comparison.sort_values(["source", "f1_weighted"], ascending=[True, False]))
            """
        ),
        md(
            """
## Tradeoff Summary
- **Manual ML**: highest control and deployment clarity.
- **LazyPredict**: strongest for quick benchmark discovery.
- **Weakness**: less configurable preprocessing and less business-specific tuning.
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "04_risk_prediction_baselines_lazypredict.ipynb", cells)


def notebook_05():
    cells = [
        md(
            """
# 05 - PyCaret Workflow vs Manual Workflow

## Why PyCaret exists
PyCaret accelerates experimentation with a compact API and built-in model comparison.

## Tradeoff
- Faster experimentation and less boilerplate
- Lower transparency and lower control than explicit sklearn pipelines
            """
        ),
        md(edu_block("PyCaret accelerates model comparison while manual pipelines retain full control and transparency.", "Teams often prototype in PyCaret, then harden the selected model family in explicit sklearn code.")),
        code(COMMON_IMPORT),
        code(ENSURE_ARTIFACTS),
        code(
            """
enriched = pd.read_csv(TABLES_DIR / "feature_enriched_data.csv")
manual = pd.read_csv(TABLES_DIR / "manual_model_leaderboard.csv")

pycaret = PyCaretWorkflow(random_state=42)
py_out = pycaret.run(enriched.drop(columns=["Borrower_ID"]), target_col="Recovery_Status")

display(py_out.comparison_table)
display(manual)
            """
        ),
        code(
            """
manual_best = manual.iloc[0]
pycaret_best = py_out.comparison_table.iloc[0]
comparison = pd.DataFrame(
    [
        {
            "workflow": "Manual",
            "model": manual_best["model"],
            "accuracy": manual_best["accuracy"],
            "f1": manual_best["f1_weighted"],
            "roc_auc": manual_best["roc_auc_ovr"],
        },
        {
            "workflow": "PyCaret",
            "model": pycaret_best["Model"],
            "accuracy": pycaret_best["Accuracy"],
            "f1": pycaret_best["F1"],
            "roc_auc": pycaret_best["AUC"],
        },
    ]
)
display(comparison)
            """
        ),
        md(
            """
## Interpretation
PyCaret should be treated as a strong accelerator for discovery, then hardened with explicit pipelines for production.
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "05_pycaret_vs_manual_workflow.ipynb", cells)


def notebook_06():
    cells = [
        md(
            """
# 06 - FLAML Optimization and Evaluation

## Why FLAML exists
FLAML performs cost-aware automated model search and hyperparameter optimization.

## Strengths
- Time-budget aware optimization
- Fast candidate prioritization

## Weaknesses
- Lower interpretability of search logic vs fully manual tuning
            """
        ),
        md(edu_block("FLAML performs budget-aware model and hyperparameter search for better cost-performance tradeoffs.", "With fixed model runtime budget, FLAML can discover stronger configurations for written-off detection.")),
        code(COMMON_IMPORT),
        code(ENSURE_ARTIFACTS),
        code(
            """
df = LoanDataLoader(DATA_PATH).load()
fe = FeatureEngineer()
enriched = fe.engineer(df)
split = fe.train_test_split(enriched, target_col="Recovery_Status", drop_cols=["Borrower_ID"])

x_train = pd.get_dummies(split.x_train)
x_test = pd.get_dummies(split.x_test)
x_train, x_test = x_train.align(x_test, axis=1, join="left", fill_value=0)

optimizer = FLAMLOptimizer(time_budget=20, random_state=42)
flaml_out = optimizer.run(x_train, split.y_train, x_test, split.y_test)
flaml_out
            """
        ),
        code(
            """
manual = pd.read_csv(TABLES_DIR / "manual_model_leaderboard.csv")
manual_best = manual.iloc[0]

comparison = pd.DataFrame(
    [
        {
            "workflow": "Manual Best",
            "model": manual_best["model"],
            "accuracy": manual_best["accuracy"],
            "f1_weighted": manual_best["f1_weighted"],
            "roc_auc_ovr": manual_best["roc_auc_ovr"],
        },
        {
            "workflow": "FLAML",
            "model": flaml_out.estimator_name,
            "accuracy": flaml_out.metrics["accuracy"],
            "f1_weighted": flaml_out.metrics["f1_weighted"],
            "roc_auc_ovr": flaml_out.metrics["roc_auc_ovr"],
        },
    ]
)
display(comparison)
            """
        ),
        code(
            """
from IPython.display import Image, display
for image_path in [
    FIGURES_DIR / "confusion_matrix.png",
    FIGURES_DIR / "roc_curves.png",
    FIGURES_DIR / "pr_curve_high_risk.png",
    FIGURES_DIR / "model_comparison.png",
]:
    display(Image(filename=str(image_path)))
            """
        ),
        md(
            """
## Evaluation Notes
Always report both statistical metrics and business metrics (recovery rate, high-risk detection, and cost impact).
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "06_flaml_optimization_evaluation.ipynb", cells)


def notebook_07():
    cells = [
        md(
            """
# 07 - Explainable AI and Recovery Strategy Engine

## Explainability definition
SHAP quantifies how each feature contributes to a prediction for both global and local explanations.

## Business purpose
Collections teams need transparent reasons behind risk scores before acting.
            """
        ),
        md(edu_block("Explainability and strategy logic connect model outputs to actionable, auditable recovery decisions.", "If risk is very high and collateral is weak, the strategy escalates from reminders to legal pathway preparation.")),
        code(COMMON_IMPORT),
        code(ENSURE_ARTIFACTS),
        code(
            """
enriched = pd.read_csv(TABLES_DIR / "feature_enriched_data.csv")
model = load_model(MODELS_DIR / "best_manual_model.joblib")

strategy = RecoveryStrategyEngine(model=model, feature_engineer=FeatureEngineer())
scored = strategy.score_portfolio(enriched)
assigned = strategy.assign_strategies(scored)
scenario_df = strategy.what_if_scenarios(enriched)

display(assigned[["Borrower_ID", "risk_score", "risk_tier", "recommended_strategy", "priority_rank"]].head(20))
display(scenario_df)
            """
        ),
        code(
            """
split = FeatureEngineer().train_test_split(enriched, target_col="Recovery_Status", drop_cols=["Borrower_ID"])
explainer = ModelExplainer(model)
shap_out = explainer.explain(split.x_test)
shap_out
            """
        ),
        code(
            """
from IPython.display import Image, HTML, display

for image_path in [
    FIGURES_DIR / "feature_importance.png",
    FIGURES_DIR / "shap_summary.png",
    FIGURES_DIR / "shap_waterfall.png",
]:
    if image_path.exists():
        display(Image(filename=str(image_path)))

force_html = FIGURES_DIR / "shap_force.html"
if force_html.exists():
    display(HTML(force_html.read_text()))
            """
        ),
        md(
            """
## Strategy logic
Risk tiers are not fixed constants. They are learned from portfolio quantiles, then mapped to operational actions.
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "07_explainable_ai_strategy_engine.ipynb", cells)


def notebook_08():
    cells = [
        md(
            """
# 08 - Business Dashboard and Streamlit Deployment

## Objective
Translate model outputs into executive-friendly portfolio intelligence.

## Dashboard views
- Portfolio health KPIs
- Risk distribution
- Segment mix
- Recovery strategy recommendation distribution
- Scenario analysis
            """
        ),
        md(edu_block("Deployment translates analytics into daily operating tools for business stakeholders.", "Collections managers can filter the top priority queue and immediately see recommended strategy by borrower.")),
        code(COMMON_IMPORT),
        code(ENSURE_ARTIFACTS),
        code(
            """
assigned = pd.read_csv(TABLES_DIR / "portfolio_with_strategies.csv")
scenarios = pd.read_csv(TABLES_DIR / "scenario_analysis.csv")
builder = DashboardBuilder()

risk_fig = builder.risk_distribution(assigned)
recovery_fig = builder.recovery_probability_distribution(assigned)
segment_fig = builder.segment_distribution(assigned)
strategy_fig = builder.strategy_recommendations(assigned)
scenario_fig = builder.scenario_comparison(scenarios)

risk_fig.show()
recovery_fig.show()
segment_fig.show()
strategy_fig.show()
scenario_fig.show()
            """
        ),
        code(
            """
kpis = builder.portfolio_health_kpis(assigned)
pd.DataFrame([kpis])
            """
        ),
        md(
            """
## Deployment Demonstration
Run the production app locally:

```bash
uv run streamlit run app.py
```

The app accepts borrower inputs and returns:
- Risk score
- Recovery probability
- Segment context
- Recommended recovery strategy
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "08_dashboard_and_streamlit_deployment.ipynb", cells)


def main() -> None:
    notebook_01()
    notebook_02()
    notebook_03()
    notebook_04()
    notebook_05()
    notebook_06()
    notebook_07()
    notebook_08()
    print(f"Generated notebooks in {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
