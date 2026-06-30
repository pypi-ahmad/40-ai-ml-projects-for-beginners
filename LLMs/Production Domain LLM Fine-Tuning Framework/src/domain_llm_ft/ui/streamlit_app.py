"""Streamlit dashboard for domain fine-tuning framework."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from domain_llm_ft.inference.engine import InferenceEngine


@st.cache_resource
def get_engine(model_name: str) -> InferenceEngine:
    return InferenceEngine(model_name)


def page_single_prediction() -> None:
    st.header("Single Prediction")
    model = st.text_input("Model", value="distilbert-base-uncased")
    text = st.text_area("Input text")
    if st.button("Predict") and text:
        pred = get_engine(model).predict(text)
        st.write({"label": pred.label, "score": pred.score, "probabilities": pred.probabilities})


def page_batch_prediction() -> None:
    st.header("Batch Prediction")
    model = st.text_input("Batch model", value="distilbert-base-uncased")
    uploaded = st.file_uploader("Upload CSV/JSON", type=["csv", "json"], key="batch_upload")
    if uploaded is None:
        return

    if uploaded.name.endswith(".csv"):
        frame = pd.read_csv(uploaded)
    else:
        frame = pd.read_json(uploaded, lines=True)

    text_column = st.selectbox("Text column", frame.columns)
    if st.button("Run batch"):
        preds = get_engine(model).predict_batch(frame[text_column].astype(str).tolist())
        frame["prediction"] = [p.label for p in preds]
        frame["confidence"] = [p.score for p in preds]
        st.dataframe(frame.head(100))
        st.download_button(
            "Download predictions",
            frame.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )


def page_dataset_explorer() -> None:
    st.header("Dataset Explorer")
    path = st.text_input("Dataset path", value="artifacts/reports/dataset_stats.json")
    if st.button("Load stats"):
        p = Path(path)
        if p.exists():
            st.json(p.read_text(encoding="utf-8"))
        else:
            st.warning("Stats file not found")


def page_training_dashboard() -> None:
    st.header("Training Dashboard")
    st.info("Use MLflow UI for full run drill-down. This page surfaces key artifact files.")
    artifacts = list(Path("artifacts/figures").glob("*.png"))
    if not artifacts:
        st.warning("No training figures found yet.")
    for artifact in artifacts:
        st.image(str(artifact), caption=artifact.name)


def page_benchmark_dashboard() -> None:
    st.header("Benchmark Dashboard")
    table_path = Path("artifacts/reports/benchmark.csv")
    if table_path.exists():
        frame = pd.read_csv(table_path)
        st.dataframe(frame)
        st.bar_chart(frame.set_index("model")["latency_ms"])
        st.bar_chart(frame.set_index("model")["throughput"])
    else:
        st.warning("Run benchmark first.")


def page_model_comparison() -> None:
    st.header("Model Comparison")
    table_path = Path("artifacts/reports/benchmark.csv")
    if table_path.exists():
        frame = pd.read_csv(table_path)
        chosen = st.multiselect("Models", frame["model"].tolist(), default=frame["model"].tolist()[:3])
        st.dataframe(frame[frame["model"].isin(chosen)])
    else:
        st.warning("No benchmark results yet.")


def page_confusion_matrix() -> None:
    st.header("Confusion Matrix")
    image_path = Path("artifacts/figures/confusion_matrix.png")
    if image_path.exists():
        st.image(str(image_path))
    else:
        st.warning("Confusion matrix not found.")


def page_error_analysis() -> None:
    st.header("Error Analysis")
    path = Path("artifacts/reports/error_analysis/misclassified.csv")
    if path.exists():
        st.dataframe(pd.read_csv(path).head(200))
    else:
        st.warning("Error analysis artifacts missing.")


def page_model_manager() -> None:
    st.header("Model Manager")
    checkpoints = sorted(Path("artifacts/checkpoints").glob("**/*"))
    st.write(f"Found {len(checkpoints)} checkpoint artifacts")
    for checkpoint in checkpoints[:200]:
        st.code(str(checkpoint))


def run_app() -> None:
    """Run Streamlit multipage dashboard."""
    st.set_page_config(page_title="Domain LLM FT Framework", layout="wide")
    st.title("Production Domain LLM Fine-Tuning Framework")

    pages = {
        "Single Prediction": page_single_prediction,
        "Batch Prediction": page_batch_prediction,
        "Dataset Explorer": page_dataset_explorer,
        "Training Dashboard": page_training_dashboard,
        "Benchmark Dashboard": page_benchmark_dashboard,
        "Model Comparison": page_model_comparison,
        "Confusion Matrix": page_confusion_matrix,
        "Error Analysis": page_error_analysis,
        "Model Manager": page_model_manager,
    }

    selected = st.sidebar.radio("Pages", list(pages.keys()))
    pages[selected]()


if __name__ == "__main__":
    run_app()
