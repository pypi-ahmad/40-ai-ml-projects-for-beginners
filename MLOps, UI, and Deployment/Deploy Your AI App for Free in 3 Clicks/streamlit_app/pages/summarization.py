"""
Text Summarization page.

Condenses long text into short summary.
Teaches: length control, text comparison, API parameters.
"""

import streamlit as st
from streamlit_app.utils.models import summarize_text
from streamlit_app.components.ui_components import (
    render_text_comparison,
    render_example_buttons,
    render_feedback_buttons,
    render_metrics_row,
)
from streamlit_app.utils.helpers import validate_text_input
from streamlit_app.utils.caching import timed_cache, inference_timer


EXAMPLES = [
    (
        "Artificial intelligence is transforming industries across the globe. "
        "From healthcare diagnostics to autonomous vehicles, AI systems are "
        "increasingly capable of performing tasks that once required human "
        "intelligence. Machine learning, a subset of AI, enables computers to "
        "learn from data without explicit programming. Deep learning, which uses "
        "neural networks with multiple layers, has achieved remarkable results "
        "in image recognition, natural language processing, and game playing. "
        "However, challenges remain including bias in training data, interpretability "
        "of models, and the environmental cost of training large models."
    ),
    (
        "Python is one of the most popular programming languages in the world. "
        "It was created by Guido van Rossum and first released in 1991. Python's "
        "design philosophy emphasizes code readability through significant "
        "indentation. It supports multiple programming paradigms including "
        "procedural, object-oriented, and functional programming. Python has a "
        "large standard library and a thriving ecosystem of third-party packages. "
        "It is widely used in web development, data science, artificial intelligence, "
        "scientific computing, and automation. Major companies like Google, "
        "Netflix, and NASA use Python extensively."
    ),
    (
        "Climate change poses significant challenges to global ecosystems and "
        "human societies. Rising temperatures are causing polar ice caps to melt, "
        "sea levels to rise, and weather patterns to become more extreme. "
        "Scientists warn that without substantial reductions in greenhouse gas "
        "emissions, the consequences could be catastrophic. However, renewable "
        "energy technologies like solar and wind power are becoming increasingly "
        "cost-effective. Electric vehicles are reducing transportation emissions. "
        "International agreements like the Paris Accord aim to coordinate global "
        "action. Individual actions, such as reducing waste and choosing sustainable "
        "products, also contribute to the solution."
    ),
]


@timed_cache(max_age_seconds=600)
@inference_timer
def _run_summarization_inference(text: str, max_length: int) -> dict:
    """Run summarization inference through the Streamlit cache layer."""
    return summarize_text(text, max_length=max_length, use_cache=False)


def render():
    st.markdown("# 📝 Text Summarization")
    st.markdown(
        "Condense long text into concise, meaningful summaries. "
        "Choose summary length and compare source vs. result side-by-side."
    )

    with st.expander("📖 What is Text Summarization?", expanded=False):
        st.markdown(
            """
        **Text Summarization** creates a shorter version preserving key information.

        **Two main approaches:**
        - **Extractive:** picks important sentences verbatim from source
        - **Abstractive:** generates new sentences capturing meaning
          (like a human would write)

        **Real-world uses:**
        - 📰 News aggregation: summarize articles for quick scanning
        - 📄 Document review: condense legal/medical reports
        - 📧 Email triage: generate message previews
        - 🔬 Research: quickly evaluate paper relevance

        **Model:** `facebook/bart-large-cnn` — abstractive summarization
        """
        )

    st.markdown("---")

    example_col, input_col = st.columns([1, 2])
    with example_col:
        render_example_buttons(
            [e[:50] + "..." for e in EXAMPLES],
            key_prefix="summarization",
            columns=1,
        )

    with input_col:
        if "input_text" in st.session_state and st.session_state["input_text"]:
            default_text = st.session_state["input_text"]
        else:
            default_text = ""

        text = st.text_area(
            "Enter text to summarize:",
            value=default_text,
            placeholder="Paste long text here...",
            height=150,
            key="summarization_input",
        )

        max_length = st.slider(
            "Maximum summary length (words):",
            min_value=30,
            max_value=200,
            value=80,
            help="Controls how detailed the summary should be.",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            summarize_clicked = st.button(
                "📝 Summarize", type="primary", use_container_width=True
            )
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state["summarization_input"] = ""
                st.session_state.pop("summarization_result", None)
                st.rerun()

    st.markdown("---")

    if summarize_clicked and text:
        error_msg = validate_text_input(text, min_length=50)
        if error_msg:
            st.error(error_msg)
            st.stop()

        with st.spinner("Generating summary..."):
            result = _run_summarization_inference(text, max_length=max_length)

        if "error" in result:
            st.error(f"Summarization failed: {result['error']}")
            st.stop()

        st.session_state["summarization_result"] = result
        st.session_state["last_method"] = result.get("method", "unknown")
        st.session_state["last_inference_time"] = result.get("inference_time")

    if st.session_state.get("summarization_result") is not None:
        result = st.session_state["summarization_result"]

        summary = result.get("summary", "")
        inference_time = result.get("inference_time", 0.0)
        method = result.get("method", "unknown")

        metrics = {
            "Original Words": len(text.split()),
            "Summary Words": len(summary.split()),
            "Compression": f"{len(summary.split()) / max(len(text.split()), 1):.0%}",
            "Method": method,
        }
        render_metrics_row(metrics, columns=4)

        render_text_comparison(
            source=text,
            result=summary,
            source_label="📄 Original Text",
            result_label="📋 Summary",
            max_length=1000,
        )

        st.caption(f"⚡ Response time: {inference_time:.2f}s")
        render_feedback_buttons("summarization")

    elif summarize_clicked and not text:
        st.warning("Please enter some text to summarize.")
    elif summarize_clicked and text and len(text.split()) < 50:
        st.warning("For best results, use text with at least 50 words.")


def show():
    """Backward-compatible alias used by older tests/material."""
    render()
