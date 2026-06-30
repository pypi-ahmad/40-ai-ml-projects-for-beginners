"""
Text Classification page.

Categorizes text into predefined classes (news categories).
Teaches: zero-shot classification, multi-label scores, confidence visualization.
"""

import streamlit as st
from streamlit_app.utils.models import classify_text
from streamlit_app.components.ui_components import (
    render_confidence_bars,
    render_example_buttons,
    render_feedback_buttons,
    render_metrics_row,
)
from streamlit_app.utils.helpers import validate_text_input, validate_categories
from streamlit_app.utils.caching import timed_cache, inference_timer


DEFAULT_CATEGORIES = [
    "technology",
    "sports",
    "business",
    "health",
    "science",
    "entertainment",
    "politics",
    "education",
]

EXAMPLES = [
    (
        "Apple released a new iPhone with advanced AI capabilities "
        "and a revolutionary camera system that changes photography.",
        ["technology", "business"],
    ),
    (
        "The local team won the championship after a thrilling overtime "
        "victory that had fans on their feet until the final buzzer.",
        ["sports", "entertainment"],
    ),
    (
        "Scientists discovered a new species of deep-sea creatures "
        "in the Mariana Trench, expanding our understanding of life.",
        ["science", "education"],
    ),
]


@timed_cache(max_age_seconds=600)
@inference_timer
def _run_classification_inference(text: str, labels: tuple[str, ...]) -> dict:
    """Run classification inference through the Streamlit cache layer."""
    return classify_text(text, list(labels), use_cache=False)


def render():
    st.markdown("# 🏷️ Text Classification")
    st.markdown(
        "Categorize text into one or more categories using zero-shot "
        "classification. No training needed — just define your categories."
    )

    with st.expander("📖 What is Text Classification?", expanded=False):
        st.markdown(
            """
        **Text Classification** assigns predefined categories to text.

        **Zero-shot classification** means the model can classify text into categories
        it was never explicitly trained on — by understanding semantic relationships.

        **Real-world uses:**
        - 📧 Email routing: auto-sort into folders (inbox, spam, promotions)
        - 🎫 Ticket triage: categorize support requests by department
        - 📰 Content moderation: flag inappropriate content automatically
        - 🏢 Document organization: auto-tag documents by department/project

        **Model:** `facebook/bart-large-mnli` — zero-shot classification
        """
        )

    st.markdown("---")

    example_col, input_col = st.columns([1, 2])
    with example_col:
        st.markdown("**Try an example:**")
        for i, (example_text, _) in enumerate(EXAMPLES):
            if st.button(
                example_text[:60] + "...",
                key=f"class_example_{i}",
                use_container_width=True,
            ):
                st.session_state["classification_input"] = example_text
                st.rerun()

    with input_col:
        if "input_text" in st.session_state and st.session_state["input_text"]:
            default_text = st.session_state["input_text"]
        else:
            default_text = ""

        text = st.text_area(
            "Enter text to classify:",
            value=default_text,
            placeholder="Type or paste text here...",
            height=120,
            key="classification_input",
        )

        categories_input = st.text_input(
            "Categories (comma-separated):",
            value=", ".join(DEFAULT_CATEGORIES),
            help="Enter custom categories or use defaults.",
        )
        categories = [
            c.strip() for c in categories_input.split(",") if c.strip()
        ]

        col1, col2 = st.columns([3, 1])
        with col1:
            classify_clicked = st.button(
                "🏷️ Classify", type="primary", use_container_width=True
            )
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state["classification_input"] = ""
                st.session_state.pop("classification_result", None)
                st.rerun()

    st.markdown("---")

    if classify_clicked and text:
        error_msg = validate_text_input(text)
        if error_msg:
            st.error(error_msg)
            st.stop()

        category_error = validate_categories(categories)
        if category_error:
            st.error(category_error)
            st.stop()

        with st.spinner("Classifying text..."):
            result = _run_classification_inference(text, tuple(categories))

        if "error" in result:
            st.error(f"Classification failed: {result['error']}")
            st.stop()

        st.session_state["classification_result"] = result
        st.session_state["last_method"] = result.get("method", "unknown")
        st.session_state["last_inference_time"] = result.get("inference_time")

    if st.session_state.get("classification_result") is not None:
        result = st.session_state["classification_result"]

        label = result.get("label", "")
        confidence = result.get("confidence", 0.0)
        scores = result.get("scores", {})
        inference_time = result.get("inference_time", 0.0)
        method = result.get("method", "unknown")

        st.markdown(f"## Top Category: **{label}**")

        metrics = {
            "Top Category": label,
            "Confidence": f"{confidence:.1%}",
            "Categories": len(scores),
            "Method": method,
        }
        render_metrics_row(metrics, columns=4)

        if scores:
            render_confidence_bars(scores, "All Categories", top_k=8)

        st.caption(f"⚡ Response time: {inference_time:.2f}s")
        render_feedback_buttons("classification")

    elif classify_clicked and not text:
        st.warning("Please enter some text to classify.")


def show():
    """Backward-compatible alias used by older tests/material."""
    render()
