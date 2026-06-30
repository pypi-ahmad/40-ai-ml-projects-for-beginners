"""Decoding strategies used by predictive keyboard inference engine."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _safe_temperature(temperature: float) -> float:
    return max(temperature, 1e-6)


def temperature_scale(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Apply temperature scaling to logits."""

    return logits / _safe_temperature(temperature)


def greedy_sample(logits: torch.Tensor) -> int:
    """Select highest-probability token index."""

    return int(logits.argmax(dim=-1).item())


def top_k_sample(logits: torch.Tensor, k: int = 10, temperature: float = 1.0) -> int:
    """Sample one token from top-k candidate distribution."""

    scaled = temperature_scale(logits, temperature)
    k = min(k, scaled.shape[-1])
    top_logits, top_indices = torch.topk(scaled, k=k, dim=-1)
    probs = F.softmax(top_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    return int(top_indices[0, sampled_idx.item()].item())


def top_p_sample(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> int:
    """Sample one token from nucleus (top-p) candidate distribution."""

    scaled = temperature_scale(logits, temperature)
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    mask = cumulative_probs > p
    # Keep at least one token in candidate set.
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))

    sampled_local = torch.multinomial(F.softmax(sorted_logits, dim=-1), num_samples=1)
    return int(sorted_indices[0, sampled_local.item()].item())


def beam_search_next(
    logits: torch.Tensor,
    beam_width: int = 3,
) -> list[tuple[int, float]]:
    """Return top beam candidates for next-token expansion.

    Returns tuples of (token_id, log_probability).
    """

    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(beam_width, probs.shape[-1]), dim=-1)
    return [
        (int(token_id.item()), float(math.log(max(prob.item(), 1e-12))))
        for token_id, prob in zip(top_ids[0], top_probs[0], strict=False)
    ]


def generate(
    model: torch.nn.Module,
    context: torch.Tensor,
    max_len: int = 20,
    strategy: str = "greedy",
    temperature: float = 1.0,
    k: int = 10,
    p: float = 0.9,
    device: str = "cpu",
    eos_id: int = 3,
) -> list[int]:
    """Generate token IDs autoregressively."""

    model.eval()
    generated: list[int] = []
    input_ids = context.to(device)

    for _ in range(max_len):
        logits = model(input_ids)
        step_logits = logits[0, -1, :] if logits.dim() == 3 else logits[0, :]

        strategy_l = strategy.lower()
        if strategy_l == "greedy":
            next_id = greedy_sample(step_logits)
        elif strategy_l == "temperature":
            probs = F.softmax(temperature_scale(step_logits, temperature), dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        elif strategy_l == "top_k":
            next_id = top_k_sample(step_logits.unsqueeze(0), k=k, temperature=temperature)
        elif strategy_l == "top_p":
            next_id = top_p_sample(step_logits.unsqueeze(0), p=p, temperature=temperature)
        else:
            next_id = greedy_sample(step_logits)

        generated.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device, dtype=torch.long)],
            dim=-1,
        )

        if next_id == eos_id:
            break

    return generated
