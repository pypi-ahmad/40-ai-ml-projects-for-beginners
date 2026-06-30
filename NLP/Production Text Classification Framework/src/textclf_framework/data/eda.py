"""Exploratory data analysis utilities for text classification datasets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud


def class_distribution_frame(texts: list[str], labels: list[int], label_names: list[str]) -> pd.DataFrame:
    """Create class distribution dataframe."""
    counts = Counter(labels)
    rows = []
    total = max(len(labels), 1)
    for label_id, count in sorted(counts.items()):
        label_name = label_names[label_id] if label_id < len(label_names) else str(label_id)
        rows.append(
            {
                "label_id": int(label_id),
                "label_name": label_name,
                "count": int(count),
                "proportion": float(count / total),
            }
        )
    return pd.DataFrame(rows)


def top_words(texts: list[str], top_n: int = 30, ngram_range: tuple[int, int] = (1, 1)) -> pd.DataFrame:
    """Return top words or n-grams by corpus frequency."""
    vectorizer = CountVectorizer(stop_words="english", ngram_range=ngram_range, min_df=2)
    matrix = vectorizer.fit_transform(texts)
    freqs = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    frame = pd.DataFrame({"term": terms, "frequency": freqs})
    return frame.sort_values("frequency", ascending=False).head(top_n)


def tfidf_top_terms(texts: list[str], top_n: int = 30) -> pd.DataFrame:
    """Return top terms by mean TF-IDF score."""
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(texts)
    mean_scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    frame = pd.DataFrame({"term": terms, "mean_tfidf": mean_scores})
    return frame.sort_values("mean_tfidf", ascending=False).head(top_n)


def vocabulary_growth(texts: list[str], checkpoints: int = 20) -> pd.DataFrame:
    """Track unique vocabulary growth over corpus progression."""
    step = max(len(texts) // checkpoints, 1)
    seen: set[str] = set()
    rows: list[dict[str, int]] = []
    for idx, text in enumerate(texts, start=1):
        seen.update(text.split())
        if idx % step == 0 or idx == len(texts):
            rows.append({"documents": idx, "vocabulary_size": len(seen)})
    return pd.DataFrame(rows)


def create_wordcloud(texts: list[str], output_path: str | Path) -> Path:
    """Create and save corpus word cloud image."""
    joined = " ".join(texts)
    cloud = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(joined)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cloud.to_file(output.as_posix())
    return output


def plot_class_distribution(frame: pd.DataFrame) -> go.Figure:
    """Build interactive class distribution chart."""
    return px.bar(
        frame,
        x="label_name",
        y="count",
        title="Class Distribution",
        color="label_name",
    )


def plot_length_histogram(texts: list[str]) -> go.Figure:
    """Build interactive document-length histogram."""
    lengths = [len(text.split()) for text in texts]
    return px.histogram(
        x=lengths,
        nbins=40,
        title="Token Length Distribution",
        labels={"x": "Token Count"},
    )


def plot_vocabulary_growth(frame: pd.DataFrame) -> go.Figure:
    """Build interactive vocabulary growth curve."""
    return px.line(frame, x="documents", y="vocabulary_size", title="Vocabulary Growth")
