"""
Sentiment Analysis page.

Classifies text as POSITIVE, NEGATIVE, or NEUTRAL.
Teaches: API inference, confidence visualization, error handling.
"""

import streamlit as st
from streamlit_app.utils.models import analyze_sentiment
from streamlit_app.components.ui_components import (
    render_result_card,
    render_example_buttons,
    render_feedback_buttons,
    render_metrics_row,
)
from streamlit_app.utils.helpers import validate_text_input
from streamlit_app.utils.caching import timed_cache, inference_timer


EXAMPLES = [
    "I absolutely love this product! It has changed my life completely.",
    "This is the worst experience I have ever had. Terrible service.",
    "The weather today is partly cloudy with a chance of rain.",
    "The movie was okay, not great but not terrible either.",
    "I am so excited about the new features you guys released!",
]


@timed_cache(max_age_seconds=600)
@inference_timer
def _run_sentiment_inference(text: str) -> dict:
    """Run sentiment inference through the Streamlit cache layer."""
    return analyze_sentiment(text, use_cache=False)


def render():
    st.markdown("# 💬 Sentiment Analysis")
    st.markdown(
        "Detect emotional tone in text — positive, negative, or neutral. "
        "Powered by Hugging Face Inference API with automatic fallback."
    )

    with st.expander("📖 What is Sentiment Analysis?", expanded=False):
        st.markdown(
            """
        **Sentiment Analysis** (opinion mining) uses NLP to determine emotional tone.

        **Real-world uses:**
        - 📊 Brand monitoring: track public perception on social media
        - 🎯 Customer feedback: auto-tag support tickets as positive/negative
        - 📈 Market research: analyze product reviews at scale
        - 🔍 Fraud detection: identify unusual sentiment patterns

        **How it works:**
        1. Text gets tokenized into word/subword pieces
        2. Model computes probability for each sentiment class
        3. Highest probability class wins (with confidence score)
        4. App displays result with visual indicators

        **Models used:** `distilbert-base-uncased-finetuned-sst-2-english` (primary)
        """
        )

    st.markdown("---")

    example_col, input_col = st.columns([1, 2])
    with example_col:
        render_example_buttons(EXAMPLES, key_prefix="sentiment", columns=1)

    with input_col:
        if "input_text" in st.session_state and st.session_state["input_text"]:
            default_text = st.session_state["input_text"]
        else:
            default_text = ""

        text = st.text_area(
            "Enter text to analyze:",
            value=default_text,
            placeholder="Type or paste text here...",
            height=120,
            key="sentiment_input",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_clicked = st.button(
                "🔍 Analyze Sentiment", type="primary", use_container_width=True
            )
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state["sentiment_input"] = ""
                st.session_state.pop("sentiment_result", None)
                st.rerun()

    st.markdown("---")

    if analyze_clicked and text:
        error_msg = validate_text_input(text)
        if error_msg:
            st.error(error_msg)
            st.stop()

        with st.spinner("Analyzing sentiment..."):
            result = _run_sentiment_inference(text)

        if "error" in result:
            st.error(f"Analysis failed: {result['error']}")
            st.stop()

        st.session_state["sentiment_result"] = result
        st.session_state["last_method"] = result.get("method", "unknown")
        st.session_state["last_inference_time"] = result.get("inference_time")

    if st.session_state.get("sentiment_result") is not None:
        result = st.session_state["sentiment_result"]

        label = result.get("label", "").upper()
        confidence = result.get("confidence", 0.0)
        inference_time = result.get("inference_time", 0.0)
        method = result.get("method", "unknown")
        scores = result.get("scores", {})

        emoji_map = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}
        emoji = emoji_map.get(label, "⚪")

        st.markdown(f"## Result: {emoji} {label}")

        metrics = {
            "Sentiment": label,
            "Confidence": f"{confidence:.1%}",
            "Method": method,
            "Response": f"{inference_time:.2f}s",
        }
        render_metrics_row(metrics, columns=4)

        if scores:
            with st.expander("📊 Score Breakdown", expanded=True):
                score_col1, score_col2 = st.columns([1, 1])
                with score_col1:
                    for cls, score in sorted(
                        scores.items(), key=lambda x: x[1], reverse=True
                    ):
                        st.markdown(f"**{cls}:** {score:.3f}")
                with score_col2:
                    import pandas as pd

                    df = pd.DataFrame(
                        {
                            "Class": list(scores.keys()),
                            "Score": list(scores.values()),
                        }
                    )
                    st.bar_chart(df, x="Class", y="Score", horizontal=True)

        render_feedback_buttons("sentiment")

    elif analyze_clicked and not text:
        st.warning("Please enter some text to analyze.")


def show():
    """Backward-compatible alias used by older tests/material."""
    render()
