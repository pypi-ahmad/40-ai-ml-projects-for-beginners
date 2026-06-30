"""Embedding training and visualization helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def train_word2vec(
    sentences: list[list[str]],
    *,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 5,
    sg: int = 1,
    seed: int = 42,
) -> Word2Vec:
    """Train Word2Vec model using gensim.

    API usage based on official gensim examples.
    """

    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=sg,
        seed=seed,
    )


def train_fasttext(
    sentences: list[list[str]],
    *,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 5,
    sg: int = 1,
    seed: int = 42,
) -> FastText:
    """Train FastText model using gensim implementation."""

    return FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=sg,
        seed=seed,
    )


def nearest_neighbors(
    keyed_vectors,
    word: str,
    topn: int = 10,
) -> list[tuple[str, float]]:
    """Return nearest neighbors from gensim keyed vectors."""

    if word not in keyed_vectors:
        return []
    return [(neighbor, float(score)) for neighbor, score in keyed_vectors.most_similar(word, topn=topn)]


def reduce_embeddings(
    vectors: np.ndarray,
    *,
    method: Literal["pca", "tsne", "umap"] = "pca",
    random_state: int = 42,
) -> np.ndarray:
    """Project embedding matrix to 2D for visualization."""

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state, init="pca")
    elif method == "umap":
        reducer = UMAP(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"Unsupported reduction method: {method}")

    return reducer.fit_transform(vectors)


def embedding_dataframe(
    keyed_vectors,
    *,
    max_words: int = 300,
    method: Literal["pca", "tsne", "umap"] = "pca",
) -> pd.DataFrame:
    """Create DataFrame with 2D coordinates for embedding plotting."""

    words = keyed_vectors.index_to_key[:max_words]
    vectors = np.array([keyed_vectors[word] for word in words])
    reduced = reduce_embeddings(vectors, method=method)

    return pd.DataFrame(
        {
            "word": words,
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "method": method,
        }
    )
