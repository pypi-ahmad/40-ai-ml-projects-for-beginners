"""Sentiment analysis page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.ui_components import render_confidence_gauge, render_result_box
from streamlit_app.utils.caching import cached_sentiment, compare_cached_vs_uncached_sentiment
from streamlit_app.utils.helpers import load_sample_text, save_prompt, validate_text_input


def render() -> None:
    st.title("Sentiment Analysis")

    with st.expander("Learning Module: Sentiment Analysis Fundamentals", expanded=False):
        st.markdown(
            """
**Definition**: Sentiment analysis detects emotional polarity in text.

**Theory**: Modern LLM sentiment prompting frames output as structured JSON.

**Motivation**: Product teams monitor user reviews and support tickets at scale.

**Real-world example**: E-commerce uses sentiment trends to prioritize bug fixes.

**Visual explanation**: Input text -> prompt template -> model -> JSON parser -> UI gauge.

**Code explanation**: This page validates input, calls cached inference, then renders confidence.

**Best practices**: Use strict schema prompts and clamp confidence to [0,1].

**Common mistakes**: Trusting raw free-form model output without parsing safeguards.
            """
        )

    input_text = st.text_area(
        "Enter review or comment",
        height=160,
        placeholder="Example: The checkout flow is fast but payment retry keeps failing.",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Load sample", use_container_width=True):
            st.session_state["sentiment_input"] = load_sample_text("sentiment_positive")
    with col2:
        run_benchmark = st.checkbox("Show cache impact")
    with col3:
        analyze = st.button("Analyze sentiment", type="primary", use_container_width=True)

    if st.session_state.get("sentiment_input"):
        input_text = st.session_state["sentiment_input"]

    if analyze:
        error = validate_text_input(input_text)
        if error:
            st.error(error)
            return

        save_prompt(input_text)
        model_name = st.session_state["selected_models"].get("sentiment", "granite4.1:3b")

        with st.spinner(f"Running local inference with {model_name}..."):
            result = cached_sentiment(input_text, model_name)

        render_result_box(
            (
                f"Sentiment: **{str(result.get('sentiment', 'neutral')).title()}**\n\n"
                f"Confidence: **{float(result.get('confidence', 0.0)):.2%}**\n\n"
                f"Explanation: {result.get('explanation', 'No explanation returned.')}."
            ),
            title="Inference Result",
        )
        render_confidence_gauge(float(result.get("confidence", 0.0)))

        if run_benchmark:
            with st.spinner("Measuring cache behavior..."):
                metrics = compare_cached_vs_uncached_sentiment(input_text, model_name)
            st.caption(
                f"Uncached: {metrics['uncached_seconds']:.3f}s | "
                f"Cached (first): {metrics['cached_first_seconds']:.3f}s | "
                f"Cached (second): {metrics['cached_second_seconds']:.3f}s"
            )
