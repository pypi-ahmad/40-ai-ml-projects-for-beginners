"""Download and preprocess canonical HF corpus for semantic search project."""

from __future__ import annotations

from semantic_search.config import load_config
from semantic_search.logging_utils import configure_logging
from semantic_search.logging_utils import get_logger
from semantic_search.service import SemanticSearchService


def main() -> None:
    config = load_config()
    configure_logging(config)
    logger = get_logger()
    service = SemanticSearchService(config)
    service.ensure_ollama_models(include_optional_qwen=False)
    docs = service.ingest_huggingface()
    logger.info("dataset_download_complete", count=len(docs))


if __name__ == "__main__":
    main()
