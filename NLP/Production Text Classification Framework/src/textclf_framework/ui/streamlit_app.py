"""Professional Streamlit dashboard for prediction and benchmarking."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import streamlit as st

from textclf_framework.serving.inference import InferenceEngine


@st.cache_resource

def _load_engine(model_path: str, label_names: list[str]) -> InferenceEngine:
    return InferenceEngine(model_path=model_path, label_names=label_names)


def _single_prediction_page(engine: InferenceEngine) -> None:
    st.header("Single Prediction")
    text = st.text_area("Input text", height=180)
    top_k = st.slider("Top-K", min_value=1, max_value=5, value=3)

    if st.button("Predict") and text.strip():
        preds, latency_ms = asyncio.run(engine.predict(text, top_k=top_k))
        st.metric("Latency (ms)", f"{latency_ms:.2f}")
        for item in preds:
            st.write(f"{item.label_name} ({item.label_id}) - {item.confidence:.4f}")


def _batch_prediction_page(engine: InferenceEngine) -> None:
    st.header("Batch Prediction")
    uploaded = st.file_uploader("Upload CSV with 'text' column", type=["csv"])
    if uploaded is None:
        return

    frame = pd.read_csv(uploaded)
    if "text" not in frame.columns:
        st.error("CSV must include a 'text' column.")
        return

    if st.button("Run Batch Prediction"):
        preds, latency_ms = asyncio.run(engine.predict_batch(frame["text"].tolist()))
        frame["pred_label"] = [group[0].label_name if group else "unknown" for group in preds]
        frame["pred_confidence"] = [group[0].confidence if group else 0.0 for group in preds]
        st.metric("Batch Latency (ms)", f"{latency_ms:.2f}")
        st.dataframe(frame.head(100))
        st.download_button(
            "Download predictions",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )


def _benchmark_page(report_path: Path) -> None:
    st.header("Benchmark Results")
    if not report_path.exists():
        st.info("No benchmark matrix found yet. Run benchmark pipeline first.")
        return

    frame = pd.read_csv(report_path)
    st.dataframe(frame)


def _evaluation_page() -> None:
    st.header("Evaluation Dashboard")
    st.caption("Load generated confusion matrices and calibration plots from reports/ directory.")


def _explainability_page() -> None:
    st.header("Explainability")
    st.caption("LIME, SHAP and attention analysis artifacts are generated in reports/explainability/.")


def main() -> None:
    """Run Streamlit dashboard."""
    st.set_page_config(page_title="Project21 Text Classification", layout="wide")

    st.title("Production Text Classification Framework")
    page = st.sidebar.selectbox(
        "Page",
        [
            "Single Prediction",
            "Batch Prediction",
            "Explainability",
            "Evaluation Dashboard",
            "Benchmark Results",
        ],
    )

    model_path = st.sidebar.text_input("Model path", value="artifacts/models/champion")
    labels = st.sidebar.text_input("Label names (comma separated)", value="class_0,class_1")
    label_names = [label.strip() for label in labels.split(",") if label.strip()]

    engine = _load_engine(model_path=model_path, label_names=label_names)

    if page == "Single Prediction":
        _single_prediction_page(engine)
    elif page == "Batch Prediction":
        _batch_prediction_page(engine)
    elif page == "Explainability":
        _explainability_page()
    elif page == "Evaluation Dashboard":
        _evaluation_page()
    elif page == "Benchmark Results":
        _benchmark_page(Path("reports/benchmark_matrix.csv"))


if __name__ == "__main__":
    main()
