"""Runtime verification for local Python + Ollama readiness."""

from __future__ import annotations

import logging
import platform
import sys

from src.benchmarking import BENCHMARK_MODELS
from src.config import get_config
from src.ollama_client import OllamaClient
from src.ui_handlers import CHAT_MODELS, OCR_MODELS, SENTIMENT_MODELS, SUMMARY_MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Verify runtime prerequisites and fail fast with actionable errors."""

    cfg = get_config()

    current_version = platform.python_version()
    expected_prefix = "3.12"
    if not current_version.startswith(expected_prefix):
        raise RuntimeError(
            f"Python {expected_prefix}.x required. Current interpreter: {current_version}. "
            "Activate project venv and rerun."
        )

    required_models = sorted(
        set(
            SENTIMENT_MODELS
            + SUMMARY_MODELS
            + CHAT_MODELS
            + OCR_MODELS
            + BENCHMARK_MODELS
            + [cfg.translation_model, cfg.embedding_model]
        )
    )

    client = OllamaClient()
    try:
        installed_models = set(client.list_models())
    finally:
        client.close()

    if not installed_models:
        raise RuntimeError(
            "Could not query Ollama models. Ensure Ollama is running at "
            f"{cfg.ollama_base_url} and accessible from this shell."
        )

    missing_models = sorted(set(required_models) - installed_models)
    if missing_models:
        commands = "\n".join(f"  ollama pull {model}" for model in missing_models)
        raise RuntimeError(
            "Missing required models for full app workflow:\n"
            f"{', '.join(missing_models)}\n"
            "Pull commands:\n"
            f"{commands}"
        )

    logger.info(
        "Runtime verification passed. Python=%s, Ollama models=%d",
        current_version,
        len(installed_models),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI wrapper
        logger.error("Runtime verification failed: %s", exc)
        sys.exit(1)
