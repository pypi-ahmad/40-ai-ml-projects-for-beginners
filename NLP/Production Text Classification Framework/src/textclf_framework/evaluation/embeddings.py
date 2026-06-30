"""Embedding extraction and projection analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

try:
    import umap
except Exception:  # pragma: no cover
    umap = None


@dataclass(slots=True)
class EmbeddingProjection:
    method: str
    coordinates: np.ndarray


def extract_cls_embeddings(model: PreTrainedModel, dataloader: DataLoader, device: str = "cuda") -> np.ndarray:
    """Extract CLS embeddings from hidden states."""
    model.eval()
    model.to(device)
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
            result = model(**batch, output_hidden_states=True, return_dict=True)
            cls_embed = result.hidden_states[-1][:, 0, :].detach().cpu().numpy()
            outputs.append(cls_embed)

    if not outputs:
        return np.empty((0, 0))
    return np.concatenate(outputs, axis=0)


def project_embeddings(
    embeddings: np.ndarray,
    method: Literal["pca", "tsne", "umap"] = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> EmbeddingProjection:
    """Project embeddings to 2D/3D for visualization."""
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state, init="pca")
    elif method == "umap":
        if umap is None:
            raise RuntimeError("UMAP is unavailable; install umap-learn.")
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unsupported projection method: {method}")

    coords = reducer.fit_transform(embeddings)
    return EmbeddingProjection(method=method, coordinates=coords)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Cluster embeddings using KMeans."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return model.fit_predict(embeddings)


def nearest_neighbors(embeddings: np.ndarray, query_index: int, k: int = 5) -> np.ndarray:
    """Return nearest neighbor indices for one embedding row."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings[query_index : query_index + 1])
    return indices[0][1:]
