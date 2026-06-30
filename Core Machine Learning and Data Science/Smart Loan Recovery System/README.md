# Smart Loan Recovery System
Production-grade credit risk and collections analytics project for loan recovery optimization using real tabular borrower-loan data.

## Executive Summary
This repository delivers an end-to-end, portfolio-quality system for:
- Recovery likelihood prediction
- High-risk borrower detection
- Multi-algorithm borrower segmentation
- Cost-sensitive collections prioritization
- Explainable AI for model transparency
- Dashboarding and Streamlit deployment

The project is designed for data science, ML engineering, fintech analytics, and credit risk interview demonstrations.

## Banking Context
Loan recovery directly impacts:
- NPA pressure and provisioning (banks)
- Liquidity and collector efficiency (NBFCs)
- Scalable intervention workflows (fintech lenders)
- Targeted outreach capacity (microfinance)

Operational recovery workflows represented in this project:
1. Automated reminders
2. Personalized outreach and settlement
3. Intensive collections
4. Legal escalation

## Credit Risk Fundamentals Covered
- Delinquency, default, and NPA context
- Recovery cost and expected loss framing
- Probability-based recovery forecasting
- High-risk written-off class detection
- Cost-sensitive thresholding for business decisions

## Dataset
- File: `loan-recovery.csv`
- Rows: `500`
- Columns: `22`
- Target: `Recovery_Status` (`0=Fully Recovered`, `1=Partially Recovered`, `2=Written Off`)
- Current class distribution: `{0: 210, 1: 196, 2: 94}`

Feature groups:
- Borrower: age, income, dependents, employment, residence
- Loan: amount, term, interest, outstanding, collateral, debt ratio
- Behavior: missed payments, days past due, prior defaults
- Collections: collection attempts

## Project Structure
```text
src/loan_recovery/
  config.py
  data_loader.py
  features.py
  eda.py
  segmentation.py
  models.py
  lazy_predict.py
  pycaret_workflow.py
  flaml_optimizer.py
  evaluation.py
  strategy.py
  explainability.py
  dashboard.py
  pipeline.py

notebooks/                        # tutorial mini-book (01..08 + executed copies)
scripts/generate_notebooks.py
scripts/run_full_verification.py  # clean run + pipeline + notebooks + tests
app.py                            # Streamlit business app
main.py                           # CLI pipeline entrypoint
outputs/                          # generated artifacts (tables/reports/figures/models)
tests/                            # risk-control unit tests
```

## Modeling Toolchain (Mandatory)
| Tool | Why It Exists | Strengths | Weaknesses / Tradeoff |
|---|---|---|---|
| LazyPredict | Fast baseline sweep | Rapid model-family sanity check | Less control over preprocessing/business logic |
| PyCaret | Accelerated low-code comparison | Fast iteration and leaderboard generation | Less transparent than explicit sklearn pipelines |
| FLAML | Budgeted AutoML optimization | Efficient search under runtime budget | Search internals less interpretable than manual tuning |

Manual benchmark models include: Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, AdaBoost, Gradient Boosting, SVM, KNN.

## Data Quality and Governance Controls
Implemented safeguards include:
- Schema validation and required-column contract
- Missingness, duplicates, invalid-range checks
- Financial plausibility checks (including outstanding vs loan inconsistencies)
- Blocking vs warning-level data quality issues
- Leakage-safe feature policy for early warning models

Leakage controls:
- Target-derived features are excluded from model inputs by default
- Sensitive/proxy features are excluded by default (`Marital_Status`, `Residence_Type`)
- Collections-stage proxy fields are excluded in early-warning modeling mode

## Feature Engineering
Core engineered features:
- Loan-to-Income Ratio
- EMI-to-Income Ratio
- Debt Burden Score
- Collateral Coverage Ratio
- Recovery Efficiency Ratio
- Risk Exposure Score
- Missed Payment Severity
- Delinquency Score
- Recovery Difficulty Index
- Collection Intensity Score
- Behavioral Risk Score

`outputs/tables/feature_audit.csv` documents each feature’s modeling inclusion and governance status.

## Segmentation
Algorithms compared:
1. K-Means (parameter search)
2. MiniBatch K-Means (parameter search)
3. Hierarchical clustering (parameter search)
4. Gaussian Mixture Models (parameter search)
5. DBSCAN (grid search)

Evaluation metrics:
- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score
- Stability score (ARI-based for randomized methods)

Business segment names are auto-generated to be manager-actionable (for example, `Legal Escalation Candidate Segment`, `Recovery Friendly Segment`).

## Evaluation and Business Metrics
Classification metrics:
- Accuracy
- Precision / Recall / F1 (weighted + macro)
- ROC-AUC (OVR)
- PR-AUC (written-off class)
- Brier score (high-risk calibration)

Business metrics:
- Actual vs predicted recovery rate
- Expected recovery amount and gain
- High-risk detection rate and precision
- Collection efficiency
- False positive / false negative cost
- Total misclassification cost
- Collection cost

Cost-sensitive threshold optimization is applied to class-2 (`Written Off`) using validation data, then evaluated on untouched test data.

## Explainability
Artifacts generated under `outputs/figures`:
- Model-native feature importance
- SHAP summary plot
- SHAP waterfall plot
- SHAP force plot (HTML)

## Dashboard and Deployment
### Plotly dashboards
- Portfolio KPIs
- Risk distribution
- Recovery probability distribution
- Segment distribution
- Strategy distribution
- Scenario comparison

### Streamlit app
Inputs:
- Borrower + loan + behavior fields

Outputs:
- Risk score
- Recovery probability
- Written-off probability
- Risk tier
- Recommended strategy
- Priority score

## Reproducible Setup
Python and package manager requirements:
- Python `3.12.10`
- `uv`

```bash
uv venv .venv
source .venv/bin/activate
uv sync --prerelease=allow
```

Run the pipeline:
```bash
uv run python main.py --strict
```

Run the app:
```bash
uv run streamlit run app.py
```

Run full verification (clean artifacts + pipeline + notebooks + unit tests):
```bash
uv run python scripts/run_full_verification.py
```

## Notebook Mini-Book
1. `01_loan_recovery_foundations_eda.ipynb`
2. `02_data_quality_feature_engineering.ipynb`
3. `03_borrower_segmentation.ipynb`
4. `04_risk_prediction_baselines_lazypredict.ipynb`
5. `05_pycaret_vs_manual_workflow.ipynb`
6. `06_flaml_optimization_evaluation.ipynb`
7. `07_explainable_ai_strategy_engine.ipynb`
8. `08_dashboard_and_streamlit_deployment.ipynb`

Executed versions are generated as `*.executed.ipynb`.

## Responsible AI and Model Risk Notes
- Sensitive/proxy fields excluded by default from predictive models.
- Explicit leakage-control policy enforced in feature selection.
- Model explanations are produced for business transparency.
- Reproducibility and quality gates are integrated into strict pipeline mode.

## Final Verification Report
`FINAL_PROJECT_VERIFICATION_REPORT.md` contains:
- full technical/statistical/business audit summary
- data quality and leakage validation
- segmentation/model/strategy/explainability review
- improvements implemented
- residual limitations
- final scoring rubric

