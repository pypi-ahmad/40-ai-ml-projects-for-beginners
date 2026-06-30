"""Streamlit demo app for predictive keyboard suggestions."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import streamlit as st
import torch

from utils.keyboard_engine import PredictiveKeyboardEngine
from utils.models import (
    CNN_LSTM_LM,
    GRU_LM,
    LSTM_LM,
    BiLSTM_LM,
    StackedLSTM_LM,
    TransformerLM,
)
from utils.vocabulary import Vocabulary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
RESULTS = OUTPUTS / "results"

MODEL_BUILDERS = {
    "LSTM": lambda cfg: LSTM_LM(
        vocab_size=int(cfg["vocab_size"]),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=1,
    ),
    "StackedLSTM": lambda cfg: StackedLSTM_LM(
        vocab_size=int(cfg["vocab_size"]),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=2,
    ),
    "BiLSTM": lambda cfg: BiLSTM_LM(
        vocab_size=int(cfg["vocab_size"]),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=max(int(cfg["hidden_dim"]) // 2, 64),
        num_layers=2,
    ),
    "GRU": lambda cfg: GRU_LM(
        vocab_size=int(cfg["vocab_size"]),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=2,
    ),
    "CNN_LSTM": lambda cfg: CNN_LSTM_LM(
        vocab_size=int(cfg["vocab_size"]),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_filters=max(int(cfg["embedding_dim"]) // 2, 64),
    ),
    "Transformer": lambda cfg: TransformerLM(
        vocab_size=int(cfg["vocab_size"]),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        nhead=int(cfg["transformer_heads"]),
        num_layers=int(cfg["transformer_layers"]),
    ),
}


def _load_model_registry() -> dict[str, dict[str, object]]:
    for candidate in sorted(RESULTS.glob("model_registry_*.json"), reverse=True):
        return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _load_best_model_name() -> str | None:
    candidates = sorted(RESULTS.glob("leaderboard_*.csv"), reverse=True)
    if not candidates:
        return None
    import pandas as pd

    df = pd.read_csv(candidates[0])
    if df.empty:
        return None
    for model_name in df["model"].astype(str).tolist():
        if _load_engine(model_name) is not None:
            return model_name
    return str(df.iloc[0]["model"])


def _load_engine(model_name: str) -> PredictiveKeyboardEngine | None:
    registry = _load_model_registry()
    cfg = registry.get(model_name)
    if cfg is None:
        return None

    vocab_path_str = cfg.get("vocab_path")
    vocab_path = Path(str(vocab_path_str)) if vocab_path_str else OUTPUTS / "vocab.json"
    if not vocab_path.exists():
        return None
    vocab = Vocabulary.load(vocab_path)

    builder = MODEL_BUILDERS.get(model_name)
    if builder is None:
        return None

    model = builder(cfg)
    checkpoint_path = Path(str(cfg["checkpoint_path"]))
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        return None

    return PredictiveKeyboardEngine(
        model=model,
        vocabulary=vocab,
        context_length=int(cfg["context_len"]),
        device="cpu",
    )


def render_app() -> None:
    st.set_page_config(page_title="Predictive Keyboard", page_icon="⌨️", layout="wide")

    st.title("Predictive Keyboard Demo")
    st.caption(
        "Top-3 / Top-5 next-word predictions with probabilities, autocomplete, and confidence scores."
    )

    default_model = _load_best_model_name() or "LSTM"
    model_name = st.selectbox("Model", options=list(MODEL_BUILDERS.keys()), index=list(MODEL_BUILDERS.keys()).index(default_model) if default_model in MODEL_BUILDERS else 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.selectbox("Suggestions", [3, 5], index=0)
    with col2:
        strategy = st.selectbox("Strategy", ["topk", "beam", "temperature", "top_p"], index=0)
    with col3:
        temperature = st.slider("Temperature", 0.5, 1.5, 1.0, 0.1)
    top_p = st.slider("Top-p (nucleus)", 0.5, 1.0, 0.9, 0.05)

    text = st.text_input("Type text", value="I would like to")

    engine = _load_engine(model_name)
    if engine is None:
        st.error(
            "No trained model artifacts found. Run `uv run python scripts/train_and_benchmark.py --profile quick --include-wikitext --prefer-gpu`."
        )
        return

    cleaned = text.strip()
    if not cleaned:
        st.warning("Enter text to get predictions.")
        return
    if len(cleaned) > 500:
        st.warning("Input is long. Using first 500 characters for responsive inference.")
        cleaned = cleaned[:500]

    try:
        start = perf_counter()
        suggestions = engine.predict(
            cleaned,
            top_k=top_k,
            strategy=strategy,
            temperature=temperature,
            top_p=top_p,
        )
        latency_ms = (perf_counter() - start) * 1000
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return
    if not suggestions:
        st.warning("No valid suggestions generated for this input.")
        return

    st.subheader("Suggestion Bar")
    bar = st.columns(top_k)
    for idx, suggestion in enumerate(suggestions):
        with bar[idx]:
            st.metric(
                label=f"#{idx + 1}",
                value=str(suggestion["token"]),
                delta=f"{float(suggestion['probability']) * 100:.2f}%",
            )

    st.subheader("Prediction Details")
    st.dataframe(suggestions, use_container_width=True)
    st.caption(f"Inference latency: {latency_ms:.2f} ms")

    completion = engine.autocomplete(cleaned)
    st.subheader("Autocomplete")
    st.write(completion["completed_text"])
    st.caption(f"Confidence: {float(completion['confidence']) * 100:.2f}%")


if __name__ == "__main__":
    render_app()
