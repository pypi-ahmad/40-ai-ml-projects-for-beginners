# Smart Loan Recovery System — Design Document

## Overview

End-to-end production-quality ML project for financial loan recovery. Predicts recovery likelihood, segments borrowers, recommends optimal recovery strategies using real financial data.

## Dataset

- **File:** `loan-recovery.csv` (500 rows, 21 columns)
- **Target:** `Recovery_Status` — `Fully Recovered` (296), `Partially Recovered` (154), `Written Off` (50)
- **Features:** Borrower (Age, Gender, Employment_Type, Monthly_Income, Dependents), Loan (Amount, Tenure, Interest_Rate, Type, Collateral, Outstanding, EMI), Behavioral (Payment_History, Missed_Payments, Days_Past_Due), Collection (Attempts, Method, Legal_Action)

## Architecture

### Components

1. **Reusable Python Module** (`src/loan_recovery/`)
   - `data_loader.py` — load + validate dataset
   - `eda.py` — statistical summaries, visualizations
   - `features.py` — feature engineering (financial ratios, risk scores)
   - `segmentation.py` — clustering (K-Means, GMM, DBSCAN, Hierarchical, Mini-Batch)
   - `models.py` — LazyPredict, PyCaret, FLAML wrappers
   - `evaluation.py` — metrics, business impact calculation
   - `strategy.py` — recovery strategy engine
   - `explainability.py` — SHAP analysis
   - `dashboard.py` — Plotly dashboard generation
   - `utils.py` — common utilities, config

2. **Main Tutorial Notebook** (`notebooks/smart_loan_recovery_tutorial.ipynb`)
   - Zero-to-hero tutorial format
   - Sections: EDA → Data Quality → Feature Engineering → Segmentation → Risk Prediction → Benchmarking → PyCaret → FLAML → Evaluation → Strategy Engine → SHAP → Dashboard → Scenario Analysis
   - Every concept explained: definition, theory, math intuition, business impact

3. **Streamlit Application** (`app/app.py`)
   - Borrower risk assessment
   - Recovery probability prediction
   - Segment assignment
   - Recovery strategy recommendation
   - Executive dashboard

4. **Documentation** (`README.md`)
   - Mini-book format
   - All sections listed in requirements

## Deliverables

| File | Purpose |
|------|---------|
| `src/loan_recovery/*.py` | Reusable Python modules (10 files) |
| `notebooks/01_smart_loan_recovery_tutorial.ipynb` | Master tutorial covering all topics |
| `app/app.py` | Streamlit interactive application |
| `README.md` | Mini-book documentation |
| `requirements.txt` | Dependencies |

## Approach

1. **EDA → Feature Engineering → Segmentation → Modeling** pipeline
2. All three mandated tools (LazyPredict, FLAML, PyCaret) used with comparison tables
3. Multiple clustering algorithms with quantitative comparison (Silhouette, Davies-Bouldin, Calinski-Harabasz)
4. Multi-class classification (Fully Recovered / Partially Recovered / Written Off)
5. SHAP for explainability
6. Rule-based + data-driven recovery strategy engine
7. Plotly dashboards + Streamlit deployment

## Key Design Decisions

- **Single comprehensive notebook** over many small ones — easier end-to-end execution, better narrative flow for teaching
- **Modular Python library** for reusability across notebook and Streamlit app
- **Multi-class classification** since Recovery_Status has 3 meaningful classes
- **Binary risk flag** derived from status for early warning system
- **Rule-based strategy engine** with data-driven thresholds from segment characteristics
- **Python 3.12.10** with `uv` for environment management

## Verification

- Notebook executes top-to-bottom without errors
- All metrics are computed on held-out test set
- Streamlit app launches successfully
- LazyPredict, FLAML, PyCaret all produce valid outputs
- SHAP visualizations render correctly
