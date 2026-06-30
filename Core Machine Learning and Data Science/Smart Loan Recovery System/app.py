"""Streamlit application for Smart Loan Recovery System."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.loan_recovery import (
    DATA_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    DashboardBuilder,
    FeatureEngineer,
    LoanDataLoader,
    PipelineArtifacts,
    RecoveryStrategyEngine,
    SmartLoanRecoveryPipeline,
    load_model,
)

st.set_page_config(page_title="Smart Loan Recovery System", layout="wide")


@st.cache_resource(show_spinner=True)
def load_pipeline_artifacts():
    """Load pipeline artifacts from disk when available; run pipeline only if needed."""
    required = [
        MODELS_DIR / "best_manual_model.joblib",
        TABLES_DIR / "manual_model_leaderboard.csv",
        TABLES_DIR / "lazypredict_results.csv",
        TABLES_DIR / "portfolio_with_strategies.csv",
        TABLES_DIR / "scenario_analysis.csv",
    ]
    if not all(path.exists() for path in required):
        pipeline = SmartLoanRecoveryPipeline(data_path=DATA_PATH, random_state=42, strict_mode=False)
        artifacts = pipeline.run()
    else:
        artifacts = PipelineArtifacts(
            leaderboard=pd.read_csv(TABLES_DIR / "manual_model_leaderboard.csv"),
            lazypredict_table=pd.read_csv(TABLES_DIR / "lazypredict_results.csv"),
            pycaret_table=pd.read_csv(TABLES_DIR / "pycaret_comparison.csv") if (TABLES_DIR / "pycaret_comparison.csv").exists() else pd.DataFrame(),
            flaml_metrics=_load_json(REPORTS_DIR / "flaml_summary.json"),
            segmentation_metrics=pd.read_csv(TABLES_DIR / "segmentation_metrics.csv"),
            segment_profiles=pd.read_csv(TABLES_DIR / "segment_profiles.csv"),
            assigned_portfolio=pd.read_csv(TABLES_DIR / "portfolio_with_strategies.csv"),
            scenario_table=pd.read_csv(TABLES_DIR / "scenario_analysis.csv"),
            evaluation_summary=_load_json(REPORTS_DIR / "evaluation_summary.json"),
            best_model_name=str(_load_json(REPORTS_DIR / "evaluation_summary.json").get("best_model", "Unknown")),
        )
    model = load_model(MODELS_DIR / "best_manual_model.joblib")
    return artifacts, model


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def borrower_input_form(template_df: pd.DataFrame) -> pd.DataFrame:
    """Collect business-user borrower input and return one-row DataFrame."""
    st.subheader("Borrower Scoring Form")
    st.caption("Enter borrower and loan details to get recovery probability, risk score, segment hint, and strategy recommendation.")

    with st.form("borrower_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=8500)
            loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=200000, value=25000)
            term = st.number_input("Loan Term (Months)", min_value=6, max_value=120, value=36)
            interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=11.0)

        with c2:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=620)
            days_past_due = st.number_input("Days Past Due", min_value=0, max_value=720, value=90)
            missed_payments = st.number_input("Missed Payments", min_value=0, max_value=60, value=6)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
            outstanding = st.number_input("Outstanding Balance", min_value=0, max_value=300000, value=18000)

        with c3:
            collateral = st.number_input("Collateral Value", min_value=0, max_value=500000, value=25000)
            dti = st.number_input("Debt-to-Income Ratio", min_value=0.01, max_value=2.5, value=0.45)
            attempts = st.number_input("Collection Attempts", min_value=0, max_value=50, value=4)
            prev_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=1)
            years_address = st.number_input("Years at Current Address", min_value=0.0, max_value=50.0, value=8.0)

        employment = st.selectbox("Employment Status", sorted(template_df["Employment_Status"].unique()))
        purpose = st.selectbox("Loan Purpose", sorted(template_df["Loan_Purpose"].unique()))
        education = st.selectbox("Education Level", sorted(template_df["Education_Level"].unique()))
        marital = st.selectbox("Marital Status", sorted(template_df["Marital_Status"].unique()))
        residence = st.selectbox("Residence Type", sorted(template_df["Residence_Type"].unique()))

        submitted = st.form_submit_button("Score Borrower", type="primary")

    if not submitted:
        return pd.DataFrame()

    if missed_payments > term:
        st.error("Missed payments cannot exceed loan term months.")
        return pd.DataFrame()
    if dti > 1.2:
        st.warning("Debt-to-income ratio above 1.2 indicates severe affordability stress.")
    if outstanding > loan_amount:
        st.warning("Outstanding balance exceeds original loan amount. Proceeding, but review penalties/fee assumptions.")

    row = pd.DataFrame(
        [
            {
                "Borrower_ID": "NEW_BORROWER",
                "Age": age,
                "Monthly_Income": income,
                "Loan_Amount": loan_amount,
                "Loan_Term_Months": term,
                "Interest_Rate": interest_rate,
                "Credit_Score": credit_score,
                "Days_Past_Due": days_past_due,
                "Missed_Payments": missed_payments,
                "Employment_Status": employment,
                "Loan_Purpose": purpose,
                "Education_Level": education,
                "Marital_Status": marital,
                "Num_Dependents": dependents,
                "Residence_Type": residence,
                "Years_At_Current_Address": years_address,
                "Outstanding_Balance": outstanding,
                "Collateral_Value": collateral,
                "Debt_to_Income_Ratio": dti,
                "Collection_Attempts": attempts,
                "Previous_Defaults": prev_defaults,
            }
        ]
    )
    return row


def main() -> None:
    """Render application."""
    st.title("Smart Loan Recovery System")
    st.markdown(
        "Production-style credit risk analytics system for borrower segmentation, recovery forecasting, and strategy optimization."
    )

    artifacts, model = load_pipeline_artifacts()
    assigned = artifacts.assigned_portfolio
    scenario_table = artifacts.scenario_table

    dashboard = DashboardBuilder()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Executive Dashboard",
            "Portfolio Prioritization",
            "Model Intelligence",
            "Borrower Scoring",
        ]
    )

    with tab1:
        kpis = dashboard.portfolio_health_kpis(assigned)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Borrowers", f"{int(kpis['borrowers']):,}")
        c2.metric("Outstanding", f"${kpis['total_outstanding']:,.0f}")
        c3.metric("Avg Recovery Prob", f"{kpis['avg_recovery_probability']:.2%}")
        c4.metric("Avg Risk Score", f"{kpis['avg_risk_score']:.2f}")
        c5.metric("High-Risk Share", f"{kpis['high_risk_share']:.2%}")

        st.plotly_chart(dashboard.risk_distribution(assigned), use_container_width=True)
        st.plotly_chart(dashboard.recovery_probability_distribution(assigned), use_container_width=True)
        st.plotly_chart(dashboard.scenario_comparison(scenario_table), use_container_width=True)

    with tab2:
        st.subheader("Recovery Prioritization Queue")
        top_n = st.slider("Top prioritized borrowers", min_value=10, max_value=200, value=50, step=10)
        display_cols = [
            "Borrower_ID",
            "risk_tier",
            "risk_score",
            "recovery_probability",
            "Outstanding_Balance",
            "recommended_strategy",
            "priority_rank",
            "segment_name",
        ]
        st.dataframe(assigned[display_cols].sort_values("priority_rank").head(top_n), use_container_width=True)
        st.plotly_chart(dashboard.strategy_recommendations(assigned), use_container_width=True)
        st.plotly_chart(dashboard.segment_distribution(assigned), use_container_width=True)

    with tab3:
        st.subheader("Model Benchmarking Snapshot")
        st.markdown("Manual benchmark leaderboard")
        st.dataframe(artifacts.leaderboard, use_container_width=True)

        st.markdown("LazyPredict benchmark snapshot")
        st.dataframe(artifacts.lazypredict_table, use_container_width=True)

        st.markdown("PyCaret comparison table")
        if artifacts.pycaret_table.empty:
            st.info("PyCaret output unavailable in this run. Check `outputs/reports/pycaret_error.json`.")
        else:
            st.dataframe(artifacts.pycaret_table, use_container_width=True)

        st.markdown("FLAML summary")
        st.json(artifacts.flaml_metrics)

    with tab4:
        loader = LoanDataLoader(DATA_PATH)
        template_df = loader.load()
        borrower_df = borrower_input_form(template_df)

        if not borrower_df.empty:
            try:
                strategy_engine = RecoveryStrategyEngine(model=model, feature_engineer=FeatureEngineer())
                scored = strategy_engine.score_portfolio(borrower_df)
                assigned_single = strategy_engine.assign_strategies(scored)
            except Exception as exc:
                st.error(f"Unable to score borrower: {exc}")
                return

            output = assigned_single[
                [
                    "risk_score",
                    "recovery_probability",
                    "prob_written_off",
                    "risk_tier",
                    "recommended_strategy",
                    "priority_score",
                ]
            ].copy()
            st.success("Borrower scored successfully.")
            st.dataframe(output, use_container_width=True)

            st.markdown("### Business Guidance")
            tier = assigned_single.iloc[0]["risk_tier"]
            strategy = assigned_single.iloc[0]["recommended_strategy"]
            st.write(f"Risk Tier: **{tier}**")
            st.write(f"Recommended Strategy: **{strategy}**")


if __name__ == "__main__":
    main()
