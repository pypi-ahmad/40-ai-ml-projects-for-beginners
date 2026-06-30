"""Application runtime composition."""

from __future__ import annotations

from hybrid_research_assistant.cache import SemanticResponseCache
from hybrid_research_assistant.embeddings import build_embedding_provider
from hybrid_research_assistant.graph import GraphComponents
from hybrid_research_assistant.index_manifest import IndexManifestManager
from hybrid_research_assistant.indexing import IndexingService
from hybrid_research_assistant.llm import LLMJudge, OllamaLLM
from hybrid_research_assistant.loaders import DocumentLoader
from hybrid_research_assistant.logging_utils import configure_logging
from hybrid_research_assistant.memory import ConversationMemory
from hybrid_research_assistant.ocr import OllamaOCRBackend
from hybrid_research_assistant.rerank import Reranker
from hybrid_research_assistant.retrieval import IntentRouter, RetrievalService
from hybrid_research_assistant.settings import AppSettings, load_settings
from hybrid_research_assistant.vectordb import ChromaVectorStore
from hybrid_research_assistant.web_search import (
    BraveProvider,
    CacheStore,
    DuckDuckGoProvider,
    TavilyProvider,
    WebSearchService,
)
from hybrid_research_assistant.workflow import WorkflowRuntime


class AppRuntime:
    """Container for all reusable runtime services."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        settings.ensure_directories()
        configure_logging(settings.paths.logs_dir)

        ocr_backend = OllamaOCRBackend(
            host=settings.ollama_host,
            model=settings.models.ocr_model,
        )
        self.ocr_backend = ocr_backend
        self.loader = DocumentLoader(
            base_dir=settings.paths.documents_dir,
            namespace=settings.indexing.namespace,
            duplicate_policy="skip_exact" if settings.indexing.deduplicate_chunks else "keep_all",
            ocr_backend=ocr_backend,
        )

        self.embedder = build_embedding_provider(
            model_name=settings.models.embedding_default,
            ollama_host=settings.ollama_host,
            normalize=settings.indexing.normalize_embeddings,
        )

        self.vector_store = ChromaVectorStore(
            db_path=settings.paths.vector_db_dir,
            collection_name=settings.active_collection_name,
        )

        self.manifest = IndexManifestManager(settings.active_manifest_path)

        self.indexer = IndexingService(
            loader=self.loader,
            embedder=self.embedder,
            vector_store=self.vector_store,
            manifest=self.manifest,
            collection_name=settings.active_collection_name,
            namespace=settings.indexing.namespace,
        )

        providers = {"duckduckgo": DuckDuckGoProvider(timeout_seconds=settings.web_search.timeout_seconds)}
        providers["tavily"] = TavilyProvider(
            api_key="" if "TAVILY_API_KEY" not in __import__("os").environ else __import__("os").environ["TAVILY_API_KEY"],
            timeout_seconds=settings.web_search.timeout_seconds,
        )
        providers["brave"] = BraveProvider(
            api_key="" if "BRAVE_API_KEY" not in __import__("os").environ else __import__("os").environ["BRAVE_API_KEY"],
            timeout_seconds=settings.web_search.timeout_seconds,
        )
        self.web_search = WebSearchService(
            default_provider=settings.web_search.provider_default,
            cache_store=CacheStore(settings.paths.web_cache_dir, settings.web_search.cache_ttl_seconds),
            providers=providers,
        )

        self.retrieval = RetrievalService(
            vector_store=self.vector_store,
            embedder=self.embedder,
            web_search=self.web_search,
        )

        self.reranker = Reranker(settings.models.reranker_model)
        self.llm = OllamaLLM(
            host=settings.ollama_host,
            model=settings.models.primary_llm,
            temperature=settings.generation.temperature,
            max_tokens=settings.generation.max_tokens,
        )
        self.judge = LLMJudge(host=settings.ollama_host, model=settings.models.judge_llm)

        self.memory = ConversationMemory(
            max_turns=settings.memory.max_turns,
            summary_trigger_turns=settings.memory.summary_trigger_turns,
        )
        self.cache = SemanticResponseCache(
            embedder=self.embedder,
            similarity_threshold=settings.cache.semantic_similarity_threshold,
            ttl_local=settings.cache.response_ttl_local_seconds,
            ttl_web=settings.cache.response_ttl_web_seconds,
        )

        self.workflow = WorkflowRuntime(
            GraphComponents(
                intent_router=IntentRouter(),
                retrieval=self.retrieval,
                reranker=self.reranker,
                llm=self.llm,
                judge=self.judge,
                fallback_text=settings.generation.grounded_fallback,
                retrieval_top_k=settings.retrieval.top_k_default,
                candidate_k=settings.retrieval.candidate_k_default,
            )
        )

    def close(self) -> None:
        """Release runtime resources such as HTTP clients."""

        for resource in (self.llm, self.judge, self.ocr_backend):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass


def build_runtime(config_path: str = "configs/app.yaml") -> AppRuntime:
    """Build runtime from resolved settings."""

    settings = load_settings(config_path=config_path)
    return AppRuntime(settings)


__all__ = ["AppRuntime", "build_runtime", "load_settings"]
