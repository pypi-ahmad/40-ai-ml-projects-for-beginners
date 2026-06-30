"""Streamlit dashboard for PEFT platform."""

from __future__ import annotations

import streamlit as st

from peft_platform.inference.engine import InferenceEngine
from peft_platform.model_registry import list_models
from peft_platform.monitoring.system import collect_metrics
from peft_platform.peft.registry import list_methods


st.set_page_config(page_title="Production PEFT Platform", layout="wide")
st.title("Production-Grade PEFT Fine-Tuning Platform")

pages = [
    "Chat",
    "Instruction Playground",
    "Adapter Manager",
    "Training Dashboard",
    "Benchmark Dashboard",
    "Dataset Explorer",
    "Evaluation Dashboard",
    "Model Comparison",
    "Generation Playground",
]
selected = st.sidebar.selectbox("Page", pages)

if selected == "Chat":
    st.subheader("Chat")
    prompt = st.text_area("Prompt", "Explain LoRA in simple terms.")
    temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    if st.button("Generate"):
        engine = InferenceEngine()
        response = engine.generate(prompt=prompt, temperature=temp)
        st.write(response.text)

elif selected == "Instruction Playground":
    st.subheader("Instruction Playground")
    st.write("Test instruction prompts against selected model profiles.")
    st.json([model.__dict__ for model in list_models()])

elif selected == "Adapter Manager":
    st.subheader("Adapter Manager")
    st.write("Supported PEFT methods")
    st.write(list_methods())

elif selected == "Training Dashboard":
    st.subheader("Training Dashboard")
    st.info("Connect this page to MLflow run history for live training monitoring.")

elif selected == "Benchmark Dashboard":
    st.subheader("Benchmark Dashboard")
    st.info("Benchmark artifacts will appear here after running CLI benchmark command.")

elif selected == "Dataset Explorer":
    st.subheader("Dataset Explorer")
    st.info("Dataset schema, stats, and sample preview live here.")

elif selected == "Evaluation Dashboard":
    st.subheader("Evaluation Dashboard")
    st.info("Show BLEU/ROUGE/BERTScore/EM/F1/accuracy + latency summaries.")

elif selected == "Model Comparison":
    st.subheader("Model Comparison")
    st.table([
        {"Model": m.id, "Family": m.family, "Tier": m.tier.value}
        for m in list_models()
    ])

else:
    st.subheader("Generation Playground")
    prompt = st.text_input("Input", "Write 3 bullet points about QLoRA.")
    if st.button("Run"):
        engine = InferenceEngine()
        st.write(engine.generate(prompt).text)

st.sidebar.markdown("---")
st.sidebar.subheader("System Metrics")
metrics = collect_metrics()
st.sidebar.write({
    "cpu_load": round(metrics.cpu_percent, 2),
    "memory_mb": round(metrics.memory_mb, 2),
    "gpu": metrics.gpu_name,
    "gpu_mem_mb": metrics.gpu_memory_mb,
})
