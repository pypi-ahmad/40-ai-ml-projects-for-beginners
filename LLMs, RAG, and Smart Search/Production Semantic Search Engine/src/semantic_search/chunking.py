"""Chunking strategies for semantic retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from semantic_search.schemas import DocumentChunk, DocumentRecord
from semantic_search.utils import hash_text


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class ChunkingParams:
    strategy: str
    chunk_size: int
    chunk_overlap: int
    sentence_similarity_threshold: float = 0.72


class Chunker:
    """Build chunks using recursive, token, or semantic strategy."""

    def __init__(self, params: ChunkingParams):
        self.params = params

    def chunk_documents(self, documents: list[DocumentRecord]) -> list[DocumentChunk]:
        """Chunk all documents and return flattened chunks."""
        chunks: list[DocumentChunk] = []
        for doc in documents:
            if self.params.strategy == "semantic":
                texts = self._semantic_chunks(doc.text)
            elif self.params.strategy == "token":
                texts = self._token_chunks(doc.text)
            else:
                texts = self._recursive_chunks(doc.text)

            for idx, chunk_text in enumerate(texts):
                chunk_hash = hash_text(f"{doc.doc_id}:{idx}:{chunk_text}")[:12]
                metadata = {
                    "document_id": doc.doc_id,
                    "source": doc.source,
                    "filename": doc.filename,
                    "category": doc.category,
                    "tags": ",".join(doc.tags),
                    "author": doc.author,
                    "date": doc.published_date,
                    "language": doc.language,
                    "document_hash": doc.document_hash,
                    "title": doc.title,
                    "url": doc.url,
                }
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.doc_id}-chunk-{idx}-{chunk_hash}",
                        document_id=doc.doc_id,
                        chunk_index=idx,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )
        return chunks

    def _recursive_chunks(self, text: str) -> list[str]:
        if len(text) <= self.params.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.params.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - self.params.chunk_overlap)
        return chunks

    def _token_chunks(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        if len(tokens) <= self.params.chunk_size:
            return [" ".join(tokens)]

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.params.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            if chunk_tokens:
                chunks.append(" ".join(chunk_tokens))
            if end == len(tokens):
                break
            start = max(0, end - self.params.chunk_overlap)
        return chunks

    def _semantic_chunks(self, text: str) -> list[str]:
        sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        vectors = TfidfVectorizer(min_df=1).fit_transform(sentences)
        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_len = len(sentences[0].split())

        for idx in range(1, len(sentences)):
            prev_vec = vectors[idx - 1]
            curr_vec = vectors[idx]
            denom = np.linalg.norm(prev_vec.toarray()) * np.linalg.norm(curr_vec.toarray())
            similarity = float((prev_vec @ curr_vec.T).toarray()[0, 0] / denom) if denom else 0.0
            next_len = len(sentences[idx].split())

            force_new = (
                similarity < self.params.sentence_similarity_threshold
                or current_len + next_len > self.params.chunk_size
            )
            if force_new:
                chunks.append(" ".join(current).strip())
                current = [sentences[idx]]
                current_len = next_len
            else:
                current.append(sentences[idx])
                current_len += next_len

        if current:
            chunks.append(" ".join(current).strip())

        if self.params.chunk_overlap > 0 and len(chunks) > 1:
            overlapped: list[str] = []
            overlap_words = self.params.chunk_overlap
            for idx, chunk in enumerate(chunks):
                if idx == 0:
                    overlapped.append(chunk)
                    continue
                prev_words = chunks[idx - 1].split()
                overlap_prefix = " ".join(prev_words[-overlap_words:]) if prev_words else ""
                overlapped.append(f"{overlap_prefix} {chunk}".strip())
            return overlapped
        return chunks

    def _tokenize(self, text: str) -> list[str]:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            token_ids = enc.encode(text)
            return [enc.decode([tid]) for tid in token_ids]
        except Exception:  # noqa: BLE001
            return text.split()
