"""Build chunked index for default model configuration."""

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
    service.load_documents()
    service.chunk_documents()
    service.build_indexes(config.embedding.primary)
    logger.info("index_build_complete")


if __name__ == "__main__":
    main()
