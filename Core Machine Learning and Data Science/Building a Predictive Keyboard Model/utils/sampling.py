"""Backward-compatible re-export of decoding helpers."""

from .decoding import (
    beam_search_next,
    generate,
    greedy_sample,
    temperature_scale,
    top_k_sample,
    top_p_sample,
)

__all__ = [
    "temperature_scale",
    "greedy_sample",
    "top_k_sample",
    "top_p_sample",
    "beam_search_next",
    "generate",
]
