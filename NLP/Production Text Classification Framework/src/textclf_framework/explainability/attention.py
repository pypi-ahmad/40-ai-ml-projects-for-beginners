"""Attention visualization helpers."""

from __future__ import annotations

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def attention_scores_for_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    layer: int = -1,
    head: int | None = None,
) -> dict[str, list[float] | list[str]]:
    """Extract token-level attention importances for one input."""
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    attentions = outputs.attentions[layer][0].cpu().numpy()  # [heads, seq, seq]
    if head is None:
        matrix = attentions.mean(axis=0)
    else:
        matrix = attentions[head]

    cls_attention = matrix[0]
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    norm = cls_attention / (cls_attention.sum() + 1e-12)
    return {"tokens": tokens, "scores": norm.astype(float).tolist()}


def top_attention_tokens(tokens: list[str], scores: list[float], top_k: int = 10) -> list[tuple[str, float]]:
    """Return top-k attention tokens sorted descending."""
    idxs = np.argsort(np.asarray(scores))[::-1][:top_k]
    return [(tokens[i], float(scores[i])) for i in idxs]
