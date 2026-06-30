"""Utility helpers for reproducibility and persistence."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import ollama

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Persist dictionary as UTF-8 JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def configure_logging(level: int = logging.INFO) -> None:
    """Standard logging config shared by scripts/apps."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def assert_ollama_available(host: str) -> None:
    """Fail early with clear message if Ollama server is unavailable."""
    try:
        client = ollama.Client(host=host, timeout=10)
        client.ps()
    except Exception as exc:
        raise RuntimeError(
            "Ollama server is not reachable. Start Ollama and ensure the configured host is correct."
        ) from exc
