"""Streamlit dashboard for model operations."""

from __future__ import annotations


def run_streamlit_app() -> None:
    """Launch Streamlit dashboard pages.

    Raises:
        RuntimeError: Streamlit is unavailable.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Streamlit not installed. Install extra: llmft-framework[ui]") from exc

    st.set_page_config(page_title="LLM Fine-Tuning Framework", layout="wide")
    st.title("Production LLM Fine-Tuning Framework")

    pages = [
        "Chat",
        "Model Selector",
        "Dataset Explorer",
        "Training Dashboard",
        "Evaluation",
        "Benchmarks",
        "Adapter Manager",
        "Inference Settings",
    ]
    selected = st.sidebar.radio("Page", pages)

    st.sidebar.subheader("Generation Controls")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.05)
    top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.sidebar.slider("Top-k", 1, 100, 40, 1)
    max_tokens = st.sidebar.slider("Max tokens", 16, 2048, 256, 16)

    st.write(f"Current page: {selected}")
    st.caption(
        f"Controls -> temperature={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_tokens}"
    )

    if selected == "Chat":
        prompt = st.text_area("Prompt", height=180)
        if st.button("Generate"):
            st.info("Connect this page to backend endpoints for live inference.")
            st.code(prompt)

    if selected == "Dataset Explorer":
        dataset_file = st.file_uploader("Upload dataset", type=["json", "jsonl", "csv"])
        if dataset_file is not None:
            st.success(f"Uploaded: {dataset_file.name}")

    if selected == "Adapter Manager":
        st.button("Download Adapters")
        st.button("Compare Adapters")
