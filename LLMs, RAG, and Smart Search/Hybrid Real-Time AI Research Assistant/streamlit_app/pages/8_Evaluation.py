from __future__ import annotations

import streamlit as st

from hybrid_research_assistant.evaluation import evaluate_benchmark, load_eval_samples
from streamlit_app.utils import get_runtime, init_state

init_state()
runtime = get_runtime()

st.title("Evaluation")
st.caption("Run benchmark evaluation on local dataset")

if st.button("Run Evaluation", type="primary"):
    with st.spinner("Evaluating benchmark dataset..."):
        samples = load_eval_samples(runtime.settings.evaluation.benchmark_path)
        rows, report = evaluate_benchmark(runtime.workflow, samples)
    st.success(f"Evaluation complete on {len(rows)} samples")
    st.json(report.model_dump())
