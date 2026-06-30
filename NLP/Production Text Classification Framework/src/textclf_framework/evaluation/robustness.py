"""Robustness testing against input perturbations."""

from __future__ import annotations

import random
import re
from collections.abc import Callable

import numpy as np
import pandas as pd


def perturb_typo(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    idx = rng.randint(1, len(text) - 2)
    chars = list(text)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def perturb_noise(text: str) -> str:
    return re.sub(r"\s+", " !! ", text, count=1)


def perturb_caps(text: str) -> str:
    return text.upper()


def perturb_emoji(text: str) -> str:
    return f"{text} 😊"


def perturb_short(text: str) -> str:
    words = text.split()
    return " ".join(words[: max(1, len(words) // 4)])


def perturb_long(text: str) -> str:
    return " ".join([text] * 4)


def evaluate_robustness(
    texts: list[str],
    predict_fn: Callable[[list[str]], np.ndarray],
    labels: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run perturbation tests and return confidence/accuracy deltas."""
    rng = random.Random(seed)
    scenarios = {
        "clean": lambda x: x,
        "typo": lambda x: perturb_typo(x, rng),
        "noise": perturb_noise,
        "caps": perturb_caps,
        "emoji": perturb_emoji,
        "short": perturb_short,
        "long": perturb_long,
    }

    baseline_probs = predict_fn(texts)
    baseline_preds = baseline_probs.argmax(axis=1)
    base_conf = baseline_probs.max(axis=1).mean()

    rows: list[dict[str, float | str]] = []
    for scenario, transform in scenarios.items():
        transformed = [transform(text) for text in texts]
        probs = predict_fn(transformed)
        preds = probs.argmax(axis=1)
        conf = float(probs.max(axis=1).mean())
        row: dict[str, float | str] = {
            "scenario": scenario,
            "mean_confidence": conf,
            "confidence_delta": conf - float(base_conf),
            "prediction_shift_rate": float(np.mean(preds != baseline_preds)),
        }
        if labels is not None:
            row["accuracy"] = float(np.mean(np.asarray(preds) == np.asarray(labels)))
        rows.append(row)

    return pd.DataFrame(rows)
