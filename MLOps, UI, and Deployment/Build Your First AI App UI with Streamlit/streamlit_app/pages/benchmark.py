"""Benchmark dashboard page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_app.components.ui_components import (
    render_latency_distribution,
    render_model_comparison,
    render_quality_radar,
    render_throughput_chart,
    render_usage_stats,
)
from streamlit_app.config import APP_CONFIG
from streamlit_app.utils.helpers import METRICS_DIR, load_sample_text, now_utc_iso, save_json_artifact
from streamlit_app.utils.models import is_ollama_available, run_benchmark_matrix


SAMPLE_PROMPTS = {
    "Short": load_sample_text("sentiment_neutral"),
    "Medium": load_sample_text("summary"),
    "Long": "\n\n".join([load_sample_text("summary")] * 3),
}

SYSTEM_PROMPTS = {
    "General": None,
    "Summarization": "Summarize key points concisely and accurately.",
    "Classification": "Classify topic and explain brief rationale.",
    "Creative": "Respond creatively while staying coherent.",
}


def _persist_benchmark_artifacts(summary_rows: list[dict], run_rows: list[dict]) -> tuple[Path, Path]:
    timestamp = now_utc_iso().replace(":", "-")
    summary_path = METRICS_DIR / f"benchmark_summary_{timestamp}.csv"
    run_path = METRICS_DIR / f"benchmark_runs_{timestamp}.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(run_rows).to_csv(run_path, index=False)
    save_json_artifact(summary_rows, f"benchmark_summary_{timestamp}.json")

    return summary_path, run_path


def render() -> None:
    st.title("Benchmark Dashboard")

    with st.expander("Learning Module: Benchmarking Local Models", expanded=False):
        st.markdown(
            """
**Definition**: Benchmarking measures performance and quality under repeatable conditions.

**Theory**: Single-run latency is noisy. Use repeated runs and aggregate statistics.

**Motivation**: Model choice is tradeoff between response speed, memory, and output quality.

**Real-world example**: Teams benchmark candidate models before production rollout.

**Visual explanation**: fixed prompt set -> repeated inference -> metrics table + charts.

**Code explanation**: This page records run-level rows and model-level aggregates.

**Best practices**: Fix prompt, temperature, token budget, and run count for fair comparison.

**Common mistakes**: Comparing models with different prompts and calling it objective.
            """
        )

    if not is_ollama_available():
        st.error("Ollama daemon not reachable. Start local daemon to run real benchmarks.")
        st.caption("Command: `ollama serve` on your machine.")
        return

    prompt_mode = st.selectbox("Prompt template", options=list(SAMPLE_PROMPTS.keys()), index=1)
    custom_prompt = st.text_area("Custom prompt (optional)", height=120)
    prompt = custom_prompt.strip() if custom_prompt.strip() else SAMPLE_PROMPTS[prompt_mode]

    selected_models = st.multiselect(
        "Models to benchmark",
        options=APP_CONFIG.models.benchmark_models,
        default=APP_CONFIG.models.benchmark_models,
    )

    runs = st.slider("Runs per model", min_value=1, max_value=5, value=APP_CONFIG.benchmark_runs)
    system_mode = st.selectbox("System prompt mode", options=list(SYSTEM_PROMPTS.keys()), index=0)
    temperature = st.slider("Benchmark temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    if st.button("Run benchmark", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Select at least one model.")
            return

        with st.spinner("Running benchmark matrix..."):
            summary_rows, run_rows = run_benchmark_matrix(
                models=selected_models,
                prompt=prompt,
                runs=runs,
                system_prompt=SYSTEM_PROMPTS[system_mode],
                temperature=temperature,
            )

        st.session_state["last_benchmark_summary"] = summary_rows
        st.session_state["last_benchmark_runs"] = run_rows

        summary_csv, runs_csv = _persist_benchmark_artifacts(summary_rows, run_rows)
        st.success("Benchmark run complete. Artifacts saved to outputs/metrics.")
        st.caption(f"Summary CSV: {summary_csv.name}")
        st.caption(f"Run-level CSV: {runs_csv.name}")

    summary_rows = st.session_state.get("last_benchmark_summary", [])
    run_rows = st.session_state.get("last_benchmark_runs", [])
    if not summary_rows:
        st.info("No benchmark results yet. Run benchmark to generate live metrics and charts.")
        return

    st.subheader("Model Comparison")
    render_model_comparison(summary_rows)

    st.subheader("Latency Distribution")
    render_latency_distribution(run_rows)

    st.subheader("Throughput")
    render_throughput_chart(summary_rows)

    st.subheader("Tradeoff Radar")
    render_quality_radar(summary_rows)

    total_requests = len(run_rows)
    avg_latency = float(pd.DataFrame(summary_rows)["mean_latency"].mean())
    total_tokens = int(pd.DataFrame(run_rows)["output_word_count"].sum())
    active_models = ", ".join(row["model"] for row in summary_rows)

    render_usage_stats(
        total_requests=total_requests,
        avg_latency=avg_latency,
        total_tokens=total_tokens,
        active_model=active_models,
    )
