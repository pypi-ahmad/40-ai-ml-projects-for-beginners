"""Keyword retrieval with BM25 and persistent lexical index."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_rag.types import ChunkRecord, RetrievalResult
from local_rag.utils import json_dump, json_load

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Tokenize string for lexical retrieval."""

    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass(slots=True)
class LexicalIndexStats:
    """Stored lexical index stats."""

    chunk_count: int
    avg_tokens_per_chunk: float


class BM25LexicalIndex:
    """Persistent BM25 index used for keyword retrieval."""

    def __init__(self, index_path: Path, k1: float = 1.5, b: float = 0.75) -> None:
        self.index_path = index_path
        self.k1 = k1
        self.b = b

        self.chunk_ids: list[str] = []
        self.chunks_by_id: dict[str, ChunkRecord] = {}
        self.doc_tokens: list[list[str]] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    def build(self, chunks: list[ChunkRecord]) -> LexicalIndexStats:
        """Build in-memory BM25 index from chunk records."""

        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self._fit_stats()
        self.save()
        token_lengths = [len(tokens) for tokens in self.doc_tokens]
        avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
        return LexicalIndexStats(chunk_count=len(chunks), avg_tokens_per_chunk=avg_tokens)

    def _fit_stats(self) -> None:
        total_len = sum(len(tokens) for tokens in self.doc_tokens)
        self.avgdl = total_len / len(self.doc_tokens) if self.doc_tokens else 0.0
        self.df.clear()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1

        doc_count = max(len(self.doc_tokens), 1)
        self.idf = {
            token: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in self.df.items()
        }

    def query(
        self,
        text: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Run BM25 keyword retrieval and return top-k hits."""

        query_tokens = tokenize(text)
        if not query_tokens or not self.doc_tokens:
            return []

        scores: list[tuple[str, float]] = []
        for chunk_id, tokens in zip(self.chunk_ids, self.doc_tokens, strict=False):
            chunk = self.chunks_by_id[chunk_id]
            if filters and not _metadata_matches(chunk.metadata, filters):
                continue
            score = self._score_doc(query_tokens, tokens)
            if score <= 0:
                continue
            scores.append((chunk_id, score))

        scores.sort(key=lambda row: row[1], reverse=True)
        selected = scores[:top_k]

        max_score = selected[0][1] if selected else 1.0
        if max_score <= 0:
            max_score = 1.0

        hits: list[RetrievalResult] = []
        for chunk_id, score in selected:
            chunk = self.chunks_by_id[chunk_id]
            normalized = score / max_score
            hits.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=normalized,
                    metadata=dict(chunk.metadata),
                    strategy="keyword",
                    keyword_score=normalized,
                )
            )
        return hits

    def _score_doc(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0

        score = 0.0
        tf: dict[str, int] = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1

        doc_len = len(doc_tokens)
        avgdl = self.avgdl if self.avgdl > 0 else 1.0
        for token in query_tokens:
            token_tf = tf.get(token, 0)
            if token_tf == 0:
                continue
            idf = self.idf.get(token, 0.0)
            numerator = token_tf * (self.k1 + 1)
            denominator = token_tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            score += idf * (numerator / denominator)
        return score

    def save(self) -> None:
        """Persist lexical index to disk."""

        payload = {
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "chunk_ids": self.chunk_ids,
            "doc_tokens": self.doc_tokens,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                for chunk in self.chunks_by_id.values()
            ],
        }
        json_dump(self.index_path, payload)

    def load(self) -> bool:
        """Load lexical index from disk. Returns False when missing."""

        if not self.index_path.exists():
            return False

        payload = json_load(self.index_path)
        self.k1 = float(payload.get("k1", self.k1))
        self.b = float(payload.get("b", self.b))
        self.avgdl = float(payload.get("avgdl", 0.0))
        self.chunk_ids = [str(row) for row in payload.get("chunk_ids", [])]
        self.doc_tokens = [list(row) for row in payload.get("doc_tokens", [])]

        chunks: dict[str, ChunkRecord] = {}
        for row in payload.get("chunks", []):
            chunk = ChunkRecord(
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                text=str(row["text"]),
                metadata=dict(row.get("metadata", {})),
            )
            chunks[chunk.chunk_id] = chunk
        self.chunks_by_id = chunks

        self._fit_stats()
        return True


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        value = metadata.get(key)
        if value != expected:
            return False
    return True
