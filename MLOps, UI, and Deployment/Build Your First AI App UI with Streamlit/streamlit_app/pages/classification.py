"""Zero-shot text classification page."""

from __future__ import annotations

import streamlit as st

from streamlit_app.components.ui_components import render_confidence_gauge, render_result_box
from streamlit_app.utils.caching import cached_classify
from streamlit_app.utils.helpers import load_sample_text, save_prompt, validate_categories, validate_text_input


DEFAULT_CATEGORIES = [
    "Technology",
    "Business",
    "Finance",
    "Healthcare",
    "Education",
    "Politics",
    "Sports",
    "Other",
]


def render() -> None:
    st.title("Text Classification")

    with st.expander("Learning Module: Classification Patterns", expanded=False):
        st.markdown(
            """
**Definition**: Classification maps input text to a label from a fixed taxonomy.

**Theory**: Zero-shot prompting lets LLMs classify categories they were not specifically fine-tuned on.

**Motivation**: Unified ticket triage and inbox routing reduce manual workload.

**Real-world example**: Support center classifies incidents as billing/bug/feature request.

**Visual explanation**: Input text + category list -> model -> validated JSON result.

**Code explanation**: Category validation prevents duplicate/underspecified taxonomies.

**Best practices**: Keep mutually exclusive category definitions.

**Common mistakes**: Using overlapping classes like "Tech" and "Software" without clear boundaries.
            """
        )

    text = st.text_area("Input text", height=160, placeholder="Paste content to classify...")

    categories_text = st.text_input(
        "Categories (comma-separated)",
        value=", ".join(DEFAULT_CATEGORIES),
        help="Tip: keep category labels distinct and concise.",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load sample", use_container_width=True):
            st.session_state["classify_input"] = load_sample_text("classify")
    with col2:
        classify = st.button("Run classification", type="primary", use_container_width=True)

    if st.session_state.get("classify_input"):
        text = st.session_state["classify_input"]

    if classify:
        text_error = validate_text_input(text, min_length=10)
        if text_error:
            st.error(text_error)
            return

        categories = [item.strip() for item in categories_text.split(",") if item.strip()]
        category_error = validate_categories(categories)
        if category_error:
            st.error(category_error)
            return

        save_prompt(text)
        model_name = st.session_state["selected_models"].get("classification", "granite4.1:3b")

        with st.spinner(f"Classifying with {model_name}..."):
            result = cached_classify(text=text, categories=tuple(categories), model=model_name)

        render_result_box(
            (
                f"Category: **{result.get('category', 'Other')}**\n\n"
                f"Confidence: **{float(result.get('confidence', 0.0)):.2%}**\n\n"
                f"Reason: {result.get('reason', 'No reason returned.')}"
            ),
            title="Classification Result",
        )
        render_confidence_gauge(float(result.get("confidence", 0.0)))
