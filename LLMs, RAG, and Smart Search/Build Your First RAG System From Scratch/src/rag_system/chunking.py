"""Chunking strategies for educational and production-minded RAG pipelines."""

from __future__ import annotations

import enum
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Iterable

from rag_system.types import ChunkRecord, DocumentRecord

logger = logging.getLogger(__name__)


class ChunkingStrategy(enum.StrEnum):
    """Supported chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"


@dataclass(slots=True)
class ChunkingResult:
    """Container for chunking outputs and basic stats."""

    chunks: list[ChunkRecord]
    strategy: ChunkingStrategy

    @property
    def avg_chunk_length(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(len(chunk.text) for chunk in self.chunks) / len(self.chunks)


class TextChunker:
    """Text chunker with fixed, recursive, semantic, and parent-child modes."""

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 900,
        chunk_overlap: int = 180,
        parent_chunk_size: int = 1600,
        parent_chunk_overlap: int = 200,
        separators: list[str] | None = None,
        semantic_similarity_threshold: float = 0.78,
        embedding_engine: object | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if parent_chunk_overlap >= parent_chunk_size:
            raise ValueError("parent_chunk_overlap must be smaller than parent_chunk_size")

        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.embedding_engine = embedding_engine

    def chunk_documents(self, documents: Iterable[DocumentRecord]) -> ChunkingResult:
        """Chunk multiple documents and keep rich metadata for teaching/debugging."""
        documents_list = list(documents)
        chunks: list[ChunkRecord] = []
        for doc in documents_list:
            doc_chunks = self.chunk_document(doc)
            chunks.extend(doc_chunks)

        logger.info(
            "Chunked %d documents into %d chunks using %s",
            len(documents_list),
            len(chunks),
            self.strategy,
        )
        return ChunkingResult(chunks=chunks, strategy=self.strategy)

    def chunk_document(self, document: DocumentRecord) -> list[ChunkRecord]:
        """Chunk one document using configured strategy."""
        text = document.text.strip()
        if not text:
            return []

        if self.strategy == ChunkingStrategy.FIXED:
            return self._to_chunk_records(document, self._fixed_chunks(text), self.strategy)
        if self.strategy == ChunkingStrategy.RECURSIVE:
            return self._to_chunk_records(document, self._recursive_chunks(text), self.strategy)
        if self.strategy == ChunkingStrategy.SEMANTIC:
            return self._to_chunk_records(document, self._semantic_chunks(text), self.strategy)
        if self.strategy == ChunkingStrategy.PARENT_CHILD:
            return self._parent_child_chunks(document)
        raise ValueError(f"Unsupported chunking strategy: {self.strategy}")

    def _fixed_chunks(self, text: str) -> list[str]:
        """Fixed-size sliding window chunking in character space."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _recursive_chunks(self, text: str) -> list[str]:
        """Recursive splitting on natural boundaries, then greedy merge."""
        segments = [text]
        for sep in self.separators:
            next_segments: list[str] = []
            for segment in segments:
                if len(segment) <= self.chunk_size:
                    next_segments.append(segment)
                else:
                    split_parts = segment.split(sep)
                    # Re-add separator to keep natural punctuation context.
                    for idx, part in enumerate(split_parts):
                        if not part:
                            continue
                        if idx < len(split_parts) - 1 and sep.strip():
                            next_segments.append(part + sep.strip())
                        else:
                            next_segments.append(part)
            segments = next_segments
            if all(len(segment) <= self.chunk_size for segment in segments):
                break

        return self._merge_segments(segments, chunk_size=self.chunk_size, overlap=self.chunk_overlap)

    def _semantic_chunks(self, text: str) -> list[str]:
        """Semantic chunking by sentence similarity boundaries.

        If no embedding engine is provided, method falls back to recursive
        chunking while preserving same API.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]

        if self.embedding_engine is None:
            return self._recursive_chunks(text)

        embeddings = self.embedding_engine.embed_batch(sentences, batch_size=16)
        breaks: list[int] = [0]

        # A low adjacent similarity usually signals topic transition.
        for i in range(1, len(sentences)):
            sim = self.embedding_engine.cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.semantic_similarity_threshold:
                breaks.append(i)
        breaks.append(len(sentences))

        segments: list[str] = []
        for i in range(len(breaks) - 1):
            part = " ".join(sentences[breaks[i] : breaks[i + 1]]).strip()
            if part:
                segments.append(part)

        if not segments:
            return self._recursive_chunks(text)

        return self._merge_segments(segments, chunk_size=self.chunk_size, overlap=self.chunk_overlap)

    def _parent_child_chunks(self, document: DocumentRecord) -> list[ChunkRecord]:
        """Parent-child chunking.

        Parent chunks store broader context, child chunks store fine-grained
        retrieval units. Child chunks retain parent_id for context expansion.
        """
        parent_texts = self._merge_segments(
            self._recursive_chunks(document.text),
            chunk_size=self.parent_chunk_size,
            overlap=self.parent_chunk_overlap,
        )
        all_chunks: list[ChunkRecord] = []

        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = self._chunk_id(document.doc_id, parent_index, "parent")
            parent_meta = {
                **document.metadata,
                "strategy": ChunkingStrategy.PARENT_CHILD.value,
                "level": "parent",
                "parent_index": parent_index,
                "char_length": len(parent_text),
            }
            all_chunks.append(
                ChunkRecord(
                    chunk_id=parent_id,
                    doc_id=document.doc_id,
                    text=parent_text,
                    metadata=parent_meta,
                    parent_id=None,
                )
            )

            # Create child chunks from each parent chunk.
            child_texts = self._merge_segments(
                self._split_sentences(parent_text),
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            for child_index, child_text in enumerate(child_texts):
                child_id = self._chunk_id(document.doc_id, parent_index * 10_000 + child_index, "child")
                child_meta = {
                    **document.metadata,
                    "strategy": ChunkingStrategy.PARENT_CHILD.value,
                    "level": "child",
                    "parent_index": parent_index,
                    "child_index": child_index,
                    "char_length": len(child_text),
                }
                all_chunks.append(
                    ChunkRecord(
                        chunk_id=child_id,
                        doc_id=document.doc_id,
                        text=child_text,
                        metadata=child_meta,
                        parent_id=parent_id,
                    )
                )

        return all_chunks

    def _to_chunk_records(
        self,
        document: DocumentRecord,
        chunk_texts: list[str],
        strategy: ChunkingStrategy,
    ) -> list[ChunkRecord]:
        """Convert raw chunk texts into typed chunk records."""
        records: list[ChunkRecord] = []
        for index, text in enumerate(chunk_texts):
            chunk_id = self._chunk_id(document.doc_id, index, strategy.value)
            metadata = {
                **document.metadata,
                "strategy": strategy.value,
                "chunk_index": index,
                "total_chunks": len(chunk_texts),
                "char_length": len(text),
            }
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    text=text,
                    metadata=metadata,
                    parent_id=None,
                )
            )
        return records

    @staticmethod
    def _merge_segments(segments: list[str], chunk_size: int, overlap: int) -> list[str]:
        """Greedy segment merge with optional overlap carry-over."""
        clean_segments = [segment.strip() for segment in segments if segment.strip()]
        if not clean_segments:
            return []

        chunks: list[str] = []
        current = clean_segments[0]

        for segment in clean_segments[1:]:
            candidate = f"{current} {segment}".strip()
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            chunks.append(current)
            if overlap > 0 and len(current) > overlap:
                overlap_text = current[-overlap:]
                current = f"{overlap_text} {segment}".strip()
            else:
                current = segment

        if current:
            chunks.append(current)

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Simple sentence splitter robust for tutorial usage."""
        # Regex keeps punctuation in sentence and handles common line breaks.
        raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [part.strip() for part in raw_parts if part.strip()]

    @staticmethod
    def _chunk_id(doc_id: str, index: int, strategy: str) -> str:
        """Generate stable chunk id."""
        digest = hashlib.sha1(f"{doc_id}:{strategy}:{index}".encode("utf-8")).hexdigest()[:12]
        return f"chunk_{digest}"
