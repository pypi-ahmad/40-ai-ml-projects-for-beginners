"""Core metric implementations (lightweight fallback)."""

from __future__ import annotations

from collections import Counter


def exact_match(pred: str, ref: str) -> float:
    """Return exact-match score."""
    return 1.0 if pred.strip() == ref.strip() else 0.0


def bleu_unigram(pred: str, ref: str) -> float:
    """Compute unigram BLEU-like precision with brevity penalty."""
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum(min(count, ref_counter[token]) for token, count in pred_counter.items())
    precision = overlap / len(pred_tokens)
    bp = min(1.0, len(pred_tokens) / max(1, len(ref_tokens)))
    return round(precision * bp, 4)


def rouge_l(pred: str, ref: str) -> float:
    """Compute simplified Rouge-L F1."""
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _lcs(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, token_a in enumerate(a, start=1):
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]
