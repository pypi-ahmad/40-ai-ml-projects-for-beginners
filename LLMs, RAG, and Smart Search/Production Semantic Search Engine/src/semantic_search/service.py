"""End-to-end semantic search service orchestration."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from semantic_search.analytics import compute_collection_analytics, summarize_search_logs, write_search_log
from semantic_search.benchmark import estimate_dir_size_bytes
from semantic_search.cache import EmbeddingCache, QueryCache, SemanticQueryCache
from semantic_search.chunking import Chunker, ChunkingParams
from semantic_search.config import AppConfig, EmbeddingModelConfig, load_config
from semantic_search.embeddings import EmbeddingBackend, build_embedding_backend, embed_text_batches
from semantic_search.evaluation import EvaluationOutput, evaluate_retrieval, load_evaluation_cases, save_evaluation_output
from semantic_search.loaders import (
    RecursiveDocumentLoader,
    enrich_documents_with_urls,
    export_multiformat_subset,
    load_huggingface_huffpost,
    read_documents_jsonl,
    write_documents_jsonl,
)
from semantic_search.logging_utils import get_logger
from semantic_search.query_processing import QueryProcessor
from semantic_search.rerank import CrossEncoderReranker
from semantic_search.retrieval import HybridRetriever
from semantic_search.schemas import DocumentChunk, DocumentRecord, SearchLogEvent, SearchRequest, SearchResponse
from semantic_search.utils import ensure_dir
from semantic_search.vector_store import ChromaVectorStore, FaissVectorStore

logger = get_logger()


class SemanticSearchService:
    """Main orchestrator for ingestion, indexing, search, and evaluation."""

    def __init__(self, config: AppConfig):
        self.config = config
        ensure_dir(config.paths["processed_data_dir"])
        ensure_dir(config.paths["cache_dir"])
        ensure_dir(config.paths["reports_dir"])
        ensure_dir(config.paths["logs_dir"])
        ensure_dir(config.paths["chroma_dir"])
        ensure_dir(config.paths["faiss_dir"])

        self.documents: list[DocumentRecord] = []
        self.chunks: list[DocumentChunk] = []
        self.retriever: HybridRetriever | None = None

        self.embedding_backend: EmbeddingBackend | None = None
        self.embedding_model_cfg: EmbeddingModelConfig = config.embedding.primary
        self.chroma_store: ChromaVectorStore | None = None
        self.faiss_store: FaissVectorStore | None = None
        self.reranker: CrossEncoderReranker | None = None

        self.embedding_cache = EmbeddingCache(Path(config.paths["cache_dir"]) / "embeddings.sqlite")
        self.query_cache = QueryCache(Path(config.paths["cache_dir"]) / "queries.sqlite")
        self.semantic_query_cache = SemanticQueryCache(Path(config.paths["cache_dir"]) / "semantic_queries.sqlite")

        qcfg = config.query_processing
        self.query_processor = QueryProcessor(
            lowercase=bool(qcfg.get("lowercase", True)),
            remove_stopwords=bool(qcfg.get("remove_stopwords", True)),
            spell_correction=bool(qcfg.get("spell_correction", True)),
            synonym_expansion=bool(qcfg.get("synonym_expansion", True)),
            abbreviation_expansion=bool(qcfg.get("abbreviation_expansion", True)),
        )
        self._collection_version = 0

    def _collection_name(self) -> str:
        return f"{self.config.vector_db.collection_name}__{self.config.vector_db.namespace}"

    def _model_slug(self) -> str:
        return self.embedding_model_cfg.model_name.replace("/", "__")

    def _chroma_model_dir(self) -> Path:
        return Path(self.config.paths["chroma_dir"]) / self._model_slug()

    def _faiss_model_dir(self) -> Path:
        return Path(self.config.paths["faiss_dir"]) / self._model_slug()

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "SemanticSearchService":
        """Construct service from YAML config."""
        return cls(load_config(config_path))

    def ensure_ollama_models(self, include_optional_qwen: bool = True) -> None:
        """Hard requirement: ensure Ollama is reachable and required models exist."""
        required = ["nomic-embed-text", self.config.evaluation.llm_judge_model]
        if include_optional_qwen:
            required.append("qwen3.5:4b")

        probe = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        if probe.returncode != 0:
            raise RuntimeError(f"Ollama not ready: {probe.stderr.strip()}")
        installed_output = probe.stdout.lower()

        for model in required:
            if model.lower() in installed_output:
                continue
            logger.info("pulling_ollama_model", model=model)
            pull = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, check=False)
            if pull.returncode != 0:
                raise RuntimeError(f"Failed to pull model {model}: {pull.stderr.strip()}")

    def ingest_huggingface(self) -> list[DocumentRecord]:
        """Load, enrich, deduplicate, and persist HF dataset documents."""
        docs = load_huggingface_huffpost(self.config)
        docs = asyncio.run(enrich_documents_with_urls(docs, self.config))

        unique: dict[str, DocumentRecord] = {}
        for doc in docs:
            if doc.document_hash not in unique:
                unique[doc.document_hash] = doc
        self.documents = list(unique.values())

        output_path = Path(self.config.paths["processed_data_dir"]) / "documents.jsonl"
        write_documents_jsonl(self.documents, output_path)

        export_multiformat_subset(
            self.documents,
            Path(self.config.paths["processed_data_dir"]) / "multiformat_subset",
            limit=200,
        )
        return self.documents

    def ingest_folder(self, folder_path: str | Path) -> list[DocumentRecord]:
        """Ingest local files recursively from supported document types."""
        loader = RecursiveDocumentLoader(self.config)
        self.documents = loader.load(folder_path)
        output_path = Path(self.config.paths["processed_data_dir"]) / "documents.jsonl"
        write_documents_jsonl(self.documents, output_path)
        return self.documents

    def load_documents(self, path: str | Path | None = None) -> list[DocumentRecord]:
        """Load already-processed documents from JSONL."""
        target = Path(path) if path else Path(self.config.paths["processed_data_dir"]) / "documents.jsonl"
        self.documents = read_documents_jsonl(target)
        return self.documents

    def chunk_documents(self, strategy: str | None = None, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[DocumentChunk]:
        """Chunk loaded documents and persist chunk records."""
        if not self.documents:
            self.load_documents()

        params = ChunkingParams(
            strategy=strategy or self.config.chunking.strategy,
            chunk_size=chunk_size or self.config.chunking.chunk_size,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else self.config.chunking.chunk_overlap,
            sentence_similarity_threshold=self.config.chunking.sentence_similarity_threshold,
        )
        chunker = Chunker(params)
        self.chunks = chunker.chunk_documents(self.documents)

        chunks_path = Path(self.config.paths["processed_data_dir"]) / "chunks.jsonl"
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(chunk.model_dump_json() + "\n")

        self.query_processor.build_vocabulary([chunk.text for chunk in self.chunks])
        return self.chunks

    def load_chunks(self, path: str | Path | None = None) -> list[DocumentChunk]:
        """Load chunk records from disk."""
        target = Path(path) if path else Path(self.config.paths["processed_data_dir"]) / "chunks.jsonl"
        chunks: list[DocumentChunk] = []
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                chunks.append(DocumentChunk.model_validate_json(line))
        self.chunks = chunks
        self.query_processor.build_vocabulary([chunk.text for chunk in chunks])
        return chunks

    def _embed_chunks(self, backend: EmbeddingBackend, batch_size: int) -> np.ndarray:
        texts = [chunk.text for chunk in self.chunks]
        vectors: list[np.ndarray] = []
        misses: list[str] = []
        miss_positions: list[int] = []

        for idx, text in enumerate(texts):
            cached = self.embedding_cache.get(backend.model_name, text)
            if cached is None:
                misses.append(text)
                miss_positions.append(idx)
                vectors.append(np.zeros(1, dtype=np.float32))
            else:
                vectors.append(cached)

        if misses:
            encoded = embed_text_batches(misses, backend=backend, batch_size=batch_size)
            cache_rows: list[tuple[str, np.ndarray]] = []
            for pos, vector in zip(miss_positions, encoded, strict=False):
                vectors[pos] = vector
                cache_rows.append((texts[pos], vector))
            self.embedding_cache.set_many(backend.model_name, cache_rows)

        matrix = np.vstack(vectors).astype(np.float32)
        return matrix

    def build_indexes(self, embedding_model: EmbeddingModelConfig | None = None) -> None:
        """Build Chroma primary and FAISS comparison indexes."""
        if not self.chunks:
            self.load_chunks()

        model_cfg = embedding_model or self.embedding_model_cfg
        self.embedding_model_cfg = model_cfg
        backend = build_embedding_backend(model_cfg, self.config)
        self.embedding_backend = backend

        embeddings = self._embed_chunks(backend=backend, batch_size=model_cfg.batch_size)
        self.chroma_store = ChromaVectorStore(self._chroma_model_dir(), self._collection_name())
        self.chroma_store.reset_collection()
        self.chroma_store.upsert(self.chunks, embeddings)

        self.faiss_store = FaissVectorStore(self._faiss_model_dir(), metric=self.config.retrieval.vector_metric)
        self.faiss_store.build(self.chunks, embeddings)
        self.faiss_store.save()

        self.retriever = HybridRetriever(
            chunks=self.chunks,
            vector_metric=self.config.retrieval.vector_metric,
            mmr_lambda=self.config.retrieval.mmr_lambda,
            rrf_k=self.config.retrieval.rrf_k,
        )

        if self.config.reranker.enabled:
            self.reranker = CrossEncoderReranker(self.config.reranker.model_name)
        self._write_collection_manifest()

    def incremental_index_documents(self, new_documents: list[DocumentRecord]) -> int:
        """Add new documents with dedup and incremental Chroma upsert."""
        existing_hashes = {doc.document_hash for doc in self.documents}
        added = [doc for doc in new_documents if doc.document_hash not in existing_hashes]
        if not added:
            return 0

        self.documents.extend(added)
        write_documents_jsonl(
            self.documents,
            Path(self.config.paths["processed_data_dir"]) / "documents.jsonl",
        )

        params = ChunkingParams(
            strategy=self.config.chunking.strategy,
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            sentence_similarity_threshold=self.config.chunking.sentence_similarity_threshold,
        )
        chunker = Chunker(params)
        new_chunks = chunker.chunk_documents(added)
        self.chunks.extend(new_chunks)
        with (Path(self.config.paths["processed_data_dir"]) / "chunks.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            for chunk in new_chunks:
                handle.write(chunk.model_dump_json() + "\n")

        if self.embedding_backend is None or self.chroma_store is None:
            self.build_indexes(self.embedding_model_cfg)
            return len(added)

        embeddings = self._embed_chunks_for_subset(new_chunks)
        self.chroma_store.upsert(new_chunks, embeddings)

        if self.faiss_store is not None:
            all_embeddings = self._embed_chunks(self.embedding_backend, self.embedding_model_cfg.batch_size)
            self.faiss_store.build(self.chunks, all_embeddings)
            self.faiss_store.save()

        self.retriever = HybridRetriever(
            chunks=self.chunks,
            vector_metric=self.config.retrieval.vector_metric,
            mmr_lambda=self.config.retrieval.mmr_lambda,
            rrf_k=self.config.retrieval.rrf_k,
        )
        self._write_collection_manifest()
        return len(added)

    def delete_documents(self, document_ids: list[str]) -> int:
        """Delete documents and associated chunks from indexes."""
        if not document_ids:
            return 0
        initial_docs = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.doc_id not in set(document_ids)]
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id not in set(document_ids)]

        write_documents_jsonl(
            self.documents,
            Path(self.config.paths["processed_data_dir"]) / "documents.jsonl",
        )
        with (Path(self.config.paths["processed_data_dir"]) / "chunks.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for chunk in self.chunks:
                handle.write(chunk.model_dump_json() + "\n")

        if self.chroma_store is not None:
            self.chroma_store.delete_by_document_ids(document_ids)
        if self.embedding_backend is not None and self.faiss_store is not None:
            all_embeddings = self._embed_chunks(self.embedding_backend, self.embedding_model_cfg.batch_size)
            self.faiss_store.build(self.chunks, all_embeddings)
            self.faiss_store.save()

        self.retriever = HybridRetriever(
            chunks=self.chunks,
            vector_metric=self.config.retrieval.vector_metric,
            mmr_lambda=self.config.retrieval.mmr_lambda,
            rrf_k=self.config.retrieval.rrf_k,
        )
        self._write_collection_manifest()
        return initial_docs - len(self.documents)

    def list_collection_versions(self) -> dict[str, Any]:
        """Return latest collection manifest for explorer UI."""
        manifest_path = Path(self.config.paths["reports_dir"]) / "collection_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {}

    def _ensure_runtime(self) -> None:
        if not self.chunks:
            self.load_chunks()

        if self.embedding_backend is None:
            self.embedding_backend = build_embedding_backend(self.embedding_model_cfg, self.config)

        if self.chroma_store is None:
            self.chroma_store = ChromaVectorStore(self._chroma_model_dir(), self._collection_name())

        chroma_count = self.chroma_store.count()
        if chroma_count == 0:
            self.build_indexes(self.embedding_model_cfg)
            return

        if self.faiss_store is None:
            self.faiss_store = FaissVectorStore(self._faiss_model_dir(), metric=self.config.retrieval.vector_metric)
            index_path = self._faiss_model_dir() / "index.faiss"
            if index_path.exists():
                self.faiss_store.load()

        if self.retriever is None:
            self.retriever = HybridRetriever(
                chunks=self.chunks,
                vector_metric=self.config.retrieval.vector_metric,
                mmr_lambda=self.config.retrieval.mmr_lambda,
                rrf_k=self.config.retrieval.rrf_k,
            )

        if self.config.reranker.enabled and self.reranker is None:
            self.reranker = CrossEncoderReranker(self.config.reranker.model_name)

    def _embed_chunks_for_subset(self, chunks: list[DocumentChunk]) -> np.ndarray:
        assert self.embedding_backend is not None
        vectors: list[np.ndarray] = []
        misses: list[str] = []
        miss_idx: list[int] = []
        for idx, chunk in enumerate(chunks):
            cached = self.embedding_cache.get(self.embedding_backend.model_name, chunk.text)
            if cached is None:
                vectors.append(np.zeros(1, dtype=np.float32))
                misses.append(chunk.text)
                miss_idx.append(idx)
            else:
                vectors.append(cached)

        if misses:
            encoded = embed_text_batches(
                misses,
                backend=self.embedding_backend,
                batch_size=self.embedding_model_cfg.batch_size,
            )
            cache_rows: list[tuple[str, np.ndarray]] = []
            for idx, vector in zip(miss_idx, encoded, strict=False):
                vectors[idx] = vector
                cache_rows.append((chunks[idx].text, vector))
            self.embedding_cache.set_many(self.embedding_backend.model_name, cache_rows)
        return np.vstack(vectors).astype(np.float32)

    def _write_collection_manifest(self) -> None:
        self._collection_version += 1
        documents_count = len(self.documents)
        if documents_count == 0:
            documents_path = Path(self.config.paths["processed_data_dir"]) / "documents.jsonl"
            if documents_path.exists():
                with documents_path.open("r", encoding="utf-8") as handle:
                    documents_count = sum(1 for line in handle if line.strip())
            elif self.chunks:
                documents_count = len({chunk.document_id for chunk in self.chunks})

        manifest = {
            "version": self._collection_version,
            "documents": documents_count,
            "chunks": len(self.chunks),
            "embedding_model": self.embedding_model_cfg.model_name,
            "chunking": {
                "strategy": self.config.chunking.strategy,
                "chunk_size": self.config.chunking.chunk_size,
                "chunk_overlap": self.config.chunking.chunk_overlap,
            },
        }
        path = Path(self.config.paths["reports_dir"]) / "collection_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def search(self, request: SearchRequest) -> SearchResponse:
        """Run semantic/lexical/hybrid search with optional reranking."""
        self._ensure_runtime()

        assert self.embedding_backend is not None
        assert self.retriever is not None
        assert self.chroma_store is not None

        processed_query = self.query_processor.process(request.query)
        cache_key = self.query_cache.make_key(
            query=processed_query,
            mode=request.mode,
            model_name=self.embedding_backend.model_name,
            top_k=request.top_k,
            filters=request.filters,
        )
        if request.use_cache:
            cached = self.query_cache.get(cache_key)
            if cached:
                return SearchResponse.model_validate(cached)

        overall_start = perf_counter()

        query_vector = self.embedding_backend.encode([processed_query])
        if request.use_cache:
            semantic_cached = self.semantic_query_cache.get_similar(
                query_vector=query_vector[0],
                mode=request.mode,
                model_name=self.embedding_backend.model_name,
                top_k=request.top_k,
                filters=request.filters,
                similarity_threshold=0.92,
            )
            if semantic_cached:
                return SearchResponse.model_validate(semantic_cached)

        vector_filters = {
            key: value
            for key, value in request.filters.items()
            if key not in {"date_from", "date_to"}
        }
        vector_response = self.chroma_store.query(
            query_embeddings=query_vector,
            top_k=max(request.top_k, self.config.retrieval.vector_candidates),
            where=vector_filters if vector_filters else None,
        )
        semantic_hits, vector_latency_ms = self.retriever.semantic_search(
            vector_response=vector_response,
            top_k=max(request.top_k, self.config.retrieval.vector_candidates),
            filters=request.filters,
        )

        lexical_hits: list[Any] = []
        lexical_latency_ms = 0.0
        if request.mode in {"lexical", "hybrid"}:
            lexical_hits, lexical_latency_ms = self.retriever.lexical_search(
                query=processed_query,
                top_k=max(request.top_k, self.config.retrieval.vector_candidates),
                filters=request.filters,
            )

        if request.mode == "semantic":
            merged_hits = semantic_hits[: request.top_k]
        elif request.mode == "lexical":
            merged_hits = lexical_hits[: request.top_k]
        else:
            merged_hits = self.retriever.hybrid_search(
                semantic_hits=semantic_hits,
                lexical_hits=lexical_hits,
                top_k=max(request.top_k, self.config.retrieval.vector_candidates),
            )

        if self.config.retrieval.mmr_enabled:
            merged_hits = self.retriever.apply_mmr(query=processed_query, hits=merged_hits, top_k=request.top_k)
        else:
            merged_hits = merged_hits[: request.top_k]

        rerank_latency_ms = None
        if request.rerank and self.reranker is not None:
            merged_hits, rerank_latency_ms = self.reranker.rerank(
                query=processed_query,
                hits=merged_hits,
                top_n=request.top_k,
            )

        threshold = request.similarity_threshold if request.similarity_threshold is not None else self.config.retrieval.similarity_threshold
        final_hits = [hit for hit in merged_hits if hit.score >= threshold]
        final_hits = [hit.model_copy(update={"rank": idx + 1}) for idx, hit in enumerate(final_hits[: request.top_k])]

        latency_ms = (perf_counter() - overall_start) * 1000
        response = SearchResponse(
            request=request,
            hits=final_hits,
            latency_ms=latency_ms,
            vector_latency_ms=vector_latency_ms,
            lexical_latency_ms=lexical_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
        )

        search_log_path = Path(self.config.paths["logs_dir"]) / "search_events.jsonl"
        write_search_log(
            search_log_path,
            SearchLogEvent(
                query=request.query,
                mode=request.mode,
                top_k=request.top_k,
                latency_ms=latency_ms,
                hit_count=len(final_hits),
                success=len(final_hits) > 0,
            ),
        )

        if request.use_cache:
            self.query_cache.set(cache_key, response.model_dump(mode="json"))
            self.semantic_query_cache.set(
                query_vector=query_vector[0],
                mode=request.mode,
                model_name=self.embedding_backend.model_name,
                top_k=request.top_k,
                filters=request.filters,
                response_payload=response.model_dump(mode="json"),
            )
        return response

    def evaluate(self, cases_path: str | Path, output_path: str | Path, mode: str = "hybrid") -> EvaluationOutput:
        """Evaluate retrieval quality against labeled dataset."""
        cases = load_evaluation_cases(cases_path)

        def _search_fn(query: str) -> SearchResponse:
            req = SearchRequest(query=query, mode=mode, top_k=10, rerank=True, use_cache=False)
            return self.search(req)

        output = evaluate_retrieval(system_name=f"{mode}+rerank", cases=cases, search_fn=_search_fn, k=10)
        save_evaluation_output(output, output_path)
        return output

    def analytics(self) -> dict[str, Any]:
        """Return dashboard-ready analytics summary."""
        vector_size = estimate_dir_size_bytes(self.config.paths["chroma_dir"])
        collection_stats = compute_collection_analytics(
            documents=self.documents,
            chunks=self.chunks,
            embedding_model=self.embedding_model_cfg.model_name,
            vector_db_size_bytes=vector_size,
        )
        search_stats = summarize_search_logs(Path(self.config.paths["logs_dir"]) / "search_events.jsonl")
        return {
            **collection_stats,
            **search_stats,
        }

    def save_response_json(self, response: SearchResponse, output_path: str | Path) -> None:
        """Persist search response for audit and downloads."""
        Path(output_path).write_text(
            json.dumps(response.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
