"""
Reusable UI components for the AI Application.

Provides consistent visual elements across all pages:
result cards, metric displays, confidence visualizations,
loading states, and feedback collection.
"""

import streamlit as st
import pandas as pd
from streamlit_app.utils.helpers import (
    format_confidence,
    estimate_reading_time,
    truncate_text,
)


def render_result_card(
    title: str,
    content: str,
    method: str = "",
    inference_time: float | None = None,
    confidence: float | None = None,
    expandable: bool = False,
):
    """
    Render a consistent result display card.

    Teaches:
    - Reusable UI components pattern
    - Conditional rendering based on data presence
    - Consistent visual design across pages
    """
    st.markdown(f"### {title}")

    with st.container(border=True):
        if expandable:
            with st.expander("View Result", expanded=True):
                st.markdown(content)
        else:
            st.markdown(content)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if method:
                st.caption(f"🔧 Method: {method}")
        with col2:
            if inference_time is not None:
                st.caption(f"⚡ {inference_time:.2f}s")
        with col3:
            if confidence is not None:
                st.caption(f"🎯 Confidence: {format_confidence(confidence)}")


def render_confidence_bars(
    scores: dict[str, float],
    title: str = "Confidence Scores",
    top_k: int = 5,
):
    """
    Render horizontal bar chart for confidence/scores.

    Teaches:
    - Visualizing model confidence
    - DataFrame construction from dictionaries
    - Streamlit native chart components
    """
    if not scores:
        st.info("No scores available.")
        return

    sorted_scores = dict(
        sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    )

    st.markdown(f"### {title}")
    df = pd.DataFrame(
        {
            "Category": list(sorted_scores.keys()),
            "Confidence": list(sorted_scores.values()),
        }
    )
    df["Confidence"] = df["Confidence"].round(3)

    st.bar_chart(df, x="Category", y="Confidence", horizontal=True)


def render_text_comparison(
    source: str,
    result: str,
    source_label: str = "Source Text",
    result_label: str = "Result",
    source_lang: str = "",
    result_lang: str = "",
    max_length: int = 500,
):
    """
    Side-by-side or stacked comparison of source and result.

    Teaches:
    - Before/after display pattern
    - Column layout for comparison
    - Length-aware display with truncation
    """
    source_display = truncate_text(source, max_length)
    result_display = truncate_text(result, max_length)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{source_label}**")
        if source_lang:
            st.caption(f"Language: {source_lang}")
        with st.container(border=True):
            st.markdown(source_display)
        st.caption(f"Words: {len(source.split())}")

    with col2:
        st.markdown(f"**{result_label}**")
        if result_lang:
            st.caption(f"Language: {result_lang}")
        with st.container(border=True):
            st.markdown(result_display)
        st.caption(f"Words: {len(result.split())}")


def render_example_buttons(
    examples: list[str],
    key_prefix: str = "example",
    columns: int = 2,
):
    """
    Render clickable example text buttons.

    Teaches:
    - Quick-start examples pattern
    - Dynamic column layout
    - Button-based input selection
    """
    st.markdown("**Try an example:**")
    cols = st.columns(columns)
    for i, example in enumerate(examples):
        with cols[i % columns]:
            if st.button(
                truncate_text(example, 60),
                key=f"{key_prefix}_{i}",
                use_container_width=True,
            ):
                st.session_state["input_text"] = example
                st.rerun()


def render_loading_state(message: str = "Processing..."):
    """
    Render a loading placeholder.

    Teaches:
    - Explicit loading states
    - User experience patterns
    - Status communication
    """
    return st.status(message, state="running")


def render_feedback_buttons(result_key: str = "feedback"):
    """
    Render thumbs up/down feedback buttons.

    Teaches:
    - User feedback collection pattern
    - Session state for feedback persistence
    - Simple evaluation mechanism
    """
    st.markdown("**Was this helpful?**")
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("👍 Yes", key=f"{result_key}_thumbs_up"):
            st.session_state[f"{result_key}_rating"] = "positive"
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 No", key=f"{result_key}_thumbs_down"):
            st.session_state[f"{result_key}_rating"] = "negative"
            st.info("Thanks! We'll work on improving.")


def render_metrics_row(metrics: dict[str, str | float], columns: int = 4):
    """
    Render a row of metric cards.

    Teaches:
    - Key metric display pattern
    - Dynamic column allocation
    - Streamlit metric component usage
    """
    cols = st.columns(columns)
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            st.metric(label=label, value=value)


def render_code_block(code: str, language: str = "python"):
    """
    Render a formatted code block.

    Teaches:
    - Code display for educational content
    - Consistent code formatting
    """
    st.code(code, language=language)
