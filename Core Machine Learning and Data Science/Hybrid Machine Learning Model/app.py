from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import load_stock_data
from src.forecast_pipeline import ForecastingFramework


st.set_page_config(page_title="Hybrid ML Forecasting System", layout="wide")
st.title("Hybrid Machine Learning Forecasting System")
st.caption("Portfolio-grade time-series forecasting with ML, DL, and hybrid ensembles")


@st.cache_resource
def load_framework(config_path: str) -> ForecastingFramework:
    return ForecastingFramework(config_path=config_path)


config_path = st.sidebar.text_input("Config Path", value="config/config.yaml")
framework = load_framework(config_path)

uploaded = st.file_uploader(
    "Upload stock CSV (Date, Open, High, Low, Close/Last, Volume)",
    type=["csv"],
)
if uploaded is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.read())
            temp_path = Path(tmp.name)

        validated = load_stock_data(temp_path)
        if len(validated) < 300:
            st.error("Uploaded dataset is too short. Provide at least 300 rows for stable training.")
        else:
            framework.config["data"]["path"] = str(temp_path)
            st.success(f"Validated upload with {len(validated):,} rows.")
    except Exception as exc:
        st.error(f"Uploaded CSV failed validation: {exc}")

horizons = framework.config.get("features", {}).get("horizons", [1, 5, 10, 30])
selected_horizon = st.selectbox("Forecast Horizon (days)", options=horizons, index=0)

col1, col2 = st.columns(2)
run_baseline = col1.checkbox("Train Baseline + AutoML", value=True)
run_deep = col2.checkbox("Train Deep + Hybrid", value=True)

if st.button("Run Training"):
    try:
        with st.spinner("Loading data..."):
            df = framework.load_data()
            st.success(f"Loaded {len(df):,} rows")
            st.dataframe(df.tail(10))

        ds = framework.prepare_horizon_dataset(selected_horizon)
        seq_len = int(framework.config.get("models", {}).get("deep_learning", {}).get("sequence_length", 30))
        if len(ds.y_val) <= seq_len or len(ds.y_test) <= seq_len:
            st.error(
                "Selected horizon + sequence length leaves no validation/test sequences. "
                "Use shorter sequence length or longer history."
            )
            st.stop()

        baseline_bundle = None
        deep_bundle = None
        hybrid_bundle = None

        if run_baseline:
            with st.spinner("Training baseline models and running AutoML tools..."):
                baseline_bundle = framework.train_baselines(selected_horizon)
            st.subheader("Baseline Leaderboard")
            st.dataframe(baseline_bundle["leaderboard"], use_container_width=True)

            if baseline_bundle.get("automl"):
                st.subheader("AutoML Results")
                for tool, table in baseline_bundle["automl"].items():
                    st.markdown(f"**{tool}**")
                    st.dataframe(table, use_container_width=True)

        if run_deep:
            with st.spinner("Training deep learning models..."):
                deep_bundle = framework.train_deep_models(selected_horizon)
            st.subheader("Deep Learning Leaderboard")
            st.dataframe(deep_bundle["leaderboard"], use_container_width=True)

            with st.spinner("Building hybrid ensembles..."):
                hybrid_bundle = framework.train_hybrids(selected_horizon)
            st.subheader("Hybrid Leaderboard (Test)")
            st.dataframe(hybrid_bundle["leaderboard"], use_container_width=True)
            st.subheader("Hybrid Leaderboard (Validation)")
            st.dataframe(hybrid_bundle["val_leaderboard"], use_container_width=True)

            best_name = hybrid_bundle["leaderboard"].iloc[0]["model"]
            y_true = hybrid_bundle["y_test_true"]
            y_pred = hybrid_bundle["test_predictions"][best_name]

            chart_df = pd.DataFrame({"Actual": y_true, "Forecast": y_pred})
            st.line_chart(chart_df)

            st.metric("Best Hybrid", best_name)
            st.metric("Best RMSE (Test)", f"{hybrid_bundle['leaderboard'].iloc[0]['rmse']:.4f}")

            try:
                weights = framework.optimize_weights(
                    selected_horizon,
                    hybrid_bundle["val_predictions"],
                    hybrid_bundle["y_val_true"],
                    method="grid",
                    evaluation_predictions=hybrid_bundle["test_predictions"],
                    evaluation_y_true=hybrid_bundle["y_test_true"],
                )
                st.subheader("Optimized Ensemble Weights")
                st.json(weights["weights"])
                st.caption(
                    "Weights are learned on validation predictions and evaluated on holdout test predictions."
                )
            except Exception as exc:
                st.warning(f"Weight optimization failed: {exc}")
    except Exception as exc:
        st.exception(exc)

st.divider()
st.markdown("Outputs saved under `outputs/` directories for plots, tables, predictions, and artifacts.")
