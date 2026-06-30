"""Inference engine for predictive keyboard suggestions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .decoding import beam_search_next
from .tokenization import RegexTokenizerBackend, normalize_text
from .vocabulary import Vocabulary


@dataclass(slots=True)
class PredictiveKeyboardEngine:
    """Generate top-k next-word suggestions from trained language model."""

    model: torch.nn.Module
    vocabulary: Vocabulary
    context_length: int
    device: str = "cpu"
    _tokenizer: RegexTokenizerBackend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.model.eval()
        self._tokenizer = RegexTokenizerBackend()

    @property
    def _blocked_ids(self) -> set[int]:
        return {
            self.vocabulary.pad_idx,
            self.vocabulary.unk_idx,
            self.vocabulary.bos_idx,
            self.vocabulary.eos_idx,
        }

    def _context_tensor_from_text(self, text: str) -> torch.Tensor:
        normalized = normalize_text(text)
        tokens = self._tokenizer.tokenize(normalized)
        ids = self.vocabulary.encode(tokens)

        if len(ids) < self.context_length:
            padding = [self.vocabulary.pad_idx] * (self.context_length - len(ids))
            ids = padding + ids
        else:
            ids = ids[-self.context_length :]

        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _format_candidates(
        self,
        token_probs: list[tuple[int, float]],
        *,
        top_k: int,
    ) -> list[dict[str, float | str | int]]:
        out: list[dict[str, float | str | int]] = []
        for token_id, prob in token_probs:
            if token_id in self._blocked_ids:
                continue
            out.append(
                {
                    "token_id": int(token_id),
                    "token": self.vocabulary.idx2word.get(int(token_id), "<unk>"),
                    "probability": float(prob),
                }
            )
            if len(out) >= top_k:
                break
        return out

    @torch.no_grad()
    def predict(
        self,
        text: str,
        *,
        top_k: int = 5,
        strategy: str = "topk",
        temperature: float = 1.0,
        beam_width: int = 3,
        top_p: float = 0.9,
    ) -> list[dict[str, float | str | int]]:
        """Predict next tokens with probabilities.

        Args:
            text: User input text.
            top_k: Number of candidates to return.
            strategy: "topk", "beam", "temperature", or "top_p".
            temperature: Temperature for probability scaling.
            beam_width: Width for beam strategy.
            top_p: Nucleus threshold for top-p strategy.
        """

        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0 < top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")

        context = self._context_tensor_from_text(text)
        logits = self.model(context)
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        strategy_l = strategy.lower()
        if strategy_l not in {"topk", "beam", "temperature", "top_p"}:
            raise ValueError("strategy must be one of: topk, beam, temperature, top_p")

        if strategy_l == "beam":
            beam = beam_search_next(scaled_logits, beam_width=max(beam_width, top_k * 2))
            beam_probs = [(token_id, math.exp(log_prob)) for token_id, log_prob in beam]
            return self._format_candidates(beam_probs, top_k=top_k)

        if strategy_l == "top_p":
            vocab_size = probs.shape[-1]
            all_ids = torch.arange(vocab_size, device=probs.device, dtype=torch.long)
            allowed_mask = torch.ones(vocab_size, device=probs.device, dtype=torch.bool)
            for blocked_id in self._blocked_ids:
                if 0 <= blocked_id < vocab_size:
                    allowed_mask[blocked_id] = False

            allowed_probs = probs[0][allowed_mask]
            allowed_ids = all_ids[allowed_mask]
            if allowed_probs.numel() == 0:
                return []

            sorted_probs, sort_idx = torch.sort(allowed_probs, descending=True, dim=-1)
            sorted_ids = allowed_ids[sort_idx]
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cumulative <= top_p
            keep_mask[..., 0] = True
            filtered_probs = sorted_probs * keep_mask
            norm = filtered_probs.sum().clamp_min(1e-12)
            filtered_probs = filtered_probs / norm
            token_probs = [
                (int(token_id.item()), float(prob.item()))
                for token_id, prob in zip(sorted_ids, filtered_probs, strict=False)
                if prob.item() > 0
            ]
            return self._format_candidates(token_probs, top_k=top_k)

        k = min(max(top_k * 4, top_k), probs.shape[-1])
        top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
        token_probs = [
            (int(token_id.item()), float(prob.item()))
            for token_id, prob in zip(top_ids[0], top_probs[0], strict=False)
        ]
        return self._format_candidates(token_probs, top_k=top_k)

    def autocomplete(self, text: str) -> dict[str, str | float | int]:
        """Autocomplete text using highest-probability next word."""

        predictions = self.predict(text, top_k=1)
        if not predictions:
            return {"input": text, "completed_text": text, "confidence": 0.0}

        top = predictions[0]
        token = str(top["token"])
        completed = f"{text.rstrip()} {token}".strip()
        return {
            "input": text,
            "completed_text": completed,
            "confidence": float(top["probability"]),
        }

    def simulate_keyboard_step(self, text: str, suggestion_count: int = 3) -> dict[str, object]:
        """Return smartphone-style suggestion bar payload."""

        suggestions = self.predict(text, top_k=suggestion_count)
        return {
            "typed_text": text,
            "suggestions": suggestions,
            "best_completion": self.autocomplete(text),
        }
