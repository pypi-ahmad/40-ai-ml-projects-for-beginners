"""RAG ingestion and retrieval pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph_platform.memory.vector_store import ChromaMemoryStore
from langgraph_platform.rag.loaders import DocumentLoader
from langgraph_platform.state.models import RetrievedDocument


@dataclass(slots=True)
class IngestionReport:
    """Summary of ingestion run."""

    documents_loaded: int
    chunks_created: int


class RAGPipeline:
    """Load -> chunk -> embed -> store -> retrieve pipeline."""

    def __init__(
        self, vector_store: ChromaMemoryStore, chunk_size: int = 1000, chunk_overlap: int = 150
    ) -> None:
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def ingest_paths(self, paths: list[str | Path], source_type: str = "file") -> IngestionReport:
        """Ingest local files into vector store."""

        all_ids: list[str] = []
        all_docs: list[str] = []
        all_meta: list[dict[str, str]] = []

        for path in paths:
            content = DocumentLoader.load_path(path)
            chunks = self.splitter.split_text(content)
            for index, chunk in enumerate(chunks):
                chunk_id = hashlib.sha1(
                    f"{path}-{index}".encode(), usedforsecurity=False
                ).hexdigest()
                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_meta.append(
                    {"source": str(path), "source_type": source_type, "chunk": str(index)}
                )

        if all_docs:
            self.vector_store.add_documents(ids=all_ids, documents=all_docs, metadatas=all_meta)

        return IngestionReport(documents_loaded=len(paths), chunks_created=len(all_docs))

    def ingest_urls(self, urls: list[str]) -> IngestionReport:
        """Ingest URLs into vector store."""

        all_ids: list[str] = []
        all_docs: list[str] = []
        all_meta: list[dict[str, str]] = []

        for url in urls:
            content = DocumentLoader.load_url(url)
            chunks = self.splitter.split_text(content)
            for index, chunk in enumerate(chunks):
                chunk_id = hashlib.sha1(
                    f"{url}-{index}".encode(), usedforsecurity=False
                ).hexdigest()
                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_meta.append({"source": url, "source_type": "url", "chunk": str(index)})

        if all_docs:
            self.vector_store.add_documents(ids=all_ids, documents=all_docs, metadatas=all_meta)

        return IngestionReport(documents_loaded=len(urls), chunks_created=len(all_docs))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Retrieve semantically similar chunks for query."""

        result = self.vector_store.search(query=query, top_k=top_k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        outputs: list[RetrievedDocument] = []
        for doc_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            score = 1.0 - float(distance) if isinstance(distance, (float, int)) else 0.5
            outputs.append(
                RetrievedDocument(
                    doc_id=str(doc_id),
                    source=str(metadata.get("source", "knowledge")),
                    content=content,
                    metadata=dict(metadata or {}),
                    score=max(min(score, 1.0), 0.0),
                )
            )
        return outputs
