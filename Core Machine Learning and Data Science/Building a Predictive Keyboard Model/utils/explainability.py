"""Explainability helpers for transformer attention and embedding neighborhoods."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def transformer_attention_for_context(
    model: torch.nn.Module,
    context_ids: torch.Tensor,
    *,
    device: str = "cpu",
) -> list[np.ndarray]:
    """Extract attention maps from transformer model for one context.

    Model is expected to support `return_attention=True` in forward.
    """

    model.eval()
    context_ids = context_ids.to(device)

    outputs = model(context_ids, return_attention=True)
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise ValueError(
            "Model did not return attention weights. Expected (logits, attentions)."
        )

    _, attentions = outputs
    maps: list[np.ndarray] = []
    for layer_attention in attentions:
        # layer_attention: [batch, heads, seq, seq]
        avg_head = layer_attention.mean(dim=1)[0].detach().cpu().numpy()
        maps.append(avg_head)
    return maps


@torch.no_grad()
def prediction_probability_table(
    logits: torch.Tensor,
    *,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k token IDs and probabilities from logits."""

    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
    return top_ids[0].detach().cpu().numpy(), top_probs[0].detach().cpu().numpy()


def embedding_neighbors(
    keyed_vectors,
    word: str,
    *,
    topn: int = 10,
) -> list[tuple[str, float]]:
    """Get nearest neighbors for interpretability of embedding space."""

    if word not in keyed_vectors:
        return []
    return [(w, float(score)) for w, score in keyed_vectors.most_similar(word, topn=topn)]
