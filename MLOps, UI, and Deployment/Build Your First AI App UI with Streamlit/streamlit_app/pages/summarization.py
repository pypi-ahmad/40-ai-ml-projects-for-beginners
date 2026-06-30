"""Text summarization page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.ui_components import render_result_box
from streamlit_app.utils.caching import cached_summarize
from streamlit_app.utils.helpers import load_sample_text, save_prompt, validate_text_input


def render() -> None:
    st.title("Text Summarization")

    with st.expander("Learning Module: Abstractive Summarization", expanded=False):
        st.markdown(
            """
**Definition**: Summarization compresses long text while preserving key meaning.

**Theory**: Abstractive summarization generates new sentences from semantic representation.

**Motivation**: Teams need short decision briefs from long documents.

**Real-world example**: Compliance analysts summarize policy changes into action bullets.

**Visual explanation**: Long input -> token budget + instructions -> concise response.

**Code explanation**: Slider controls summary budget; output stats show compression ratio.

**Best practices**: Preserve named entities, numbers, and deadlines.

**Common mistakes**: Setting too-small token budget and losing critical details.
            """
        )

    input_text = st.text_area(
        "Paste long text",
        height=220,
        placeholder="Paste report, article, transcript, or meeting notes...",
    )

    max_words = st.slider("Maximum summary length (words)", min_value=40, max_value=350, value=140, step=10)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load sample", use_container_width=True):
            st.session_state["summary_input"] = load_sample_text("summary")
    with col2:
        summarize = st.button("Generate summary", type="primary", use_container_width=True)

    if st.session_state.get("summary_input"):
        input_text = st.session_state["summary_input"]

    if summarize:
        error = validate_text_input(input_text, min_length=20)
        if error:
            st.error(error)
            return

        save_prompt(input_text)
        temperature = float(st.session_state["model_settings"].get("temperature", 0.2))
        model_name = st.session_state["selected_models"].get("summarization", "qwen3.5:4b")

        with st.spinner(f"Summarizing with {model_name}..."):
            summary = cached_summarize(input_text, max_words, model_name, temperature)

        render_result_box(summary, title="Summary")

        original_words = len(input_text.split())
        summary_words = len(summary.split())
        compression = (summary_words / original_words) * 100 if original_words else 0.0

        st.caption(
            f"Original words: {original_words} | Summary words: {summary_words} | "
            f"Compression ratio: {compression:.1f}%"
        )
