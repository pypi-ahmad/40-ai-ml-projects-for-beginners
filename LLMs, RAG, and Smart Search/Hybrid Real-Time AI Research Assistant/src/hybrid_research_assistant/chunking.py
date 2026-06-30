"""Chunking strategies and benchmark helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_research_assistant.schemas import ChunkingStrategy, ChunkRecord, DocumentRecord
from hybrid_research_assistant.utils import sha256_text


class Chunker:
    """Chunk documents with recursive, token, or semantic strategies."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        strategy: ChunkingStrategy,
        semantic_window: int = 3,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.semantic_window = semantic_window

        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        self._encoder = None
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            # Offline-safe fallback for environments where tokenizer assets are not downloadable.
            self._encoder = None

    def split_documents(self, docs: list[DocumentRecord]) -> list[ChunkRecord]:
        """Split loaded documents into normalized chunks."""

        if self.strategy == ChunkingStrategy.RECURSIVE:
            return self._split_recursive(docs)
        if self.strategy == ChunkingStrategy.TOKEN:
            return self._split_token(docs)
        return self._split_semantic(docs)

    def _split_recursive(self, docs: list[DocumentRecord]) -> list[ChunkRecord]:
        texts = [doc.text for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        split_docs = self._recursive_splitter.create_documents(texts=texts, metadatas=metadatas)

        chunks: list[ChunkRecord] = []
        for split_doc in split_docs:
            metadata = dict(split_doc.metadata)
            doc_id = str(metadata["doc_id"])
            start_index = int(metadata.get("start_index", 0))
            chunk_id = sha256_text(f"{doc_id}::{start_index}::{split_doc.page_content[:64]}")
            metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "start_index": start_index,
                    "chunking_strategy": self.strategy.value,
                }
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=split_doc.page_content,
                    metadata=metadata,
                )
            )
        return chunks

    def _split_token(self, docs: list[DocumentRecord]) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for doc in docs:
            step = max(1, self.chunk_size - self.chunk_overlap)

            if self._encoder is not None:
                tokens = list(self._encoder.encode(doc.text))
                for start in range(0, len(tokens), step):
                    token_slice = tokens[start : start + self.chunk_size]
                    if not token_slice:
                        continue
                    text = self._encoder.decode(token_slice).strip()
                    if not text:
                        continue
                    chunk_id = sha256_text(f"{doc.doc_id}::{start}::{text[:64]}")
                    metadata = dict(doc.metadata)
                    metadata.update(
                        {
                            "chunk_id": chunk_id,
                            "start_index": start,
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                            "chunking_strategy": self.strategy.value,
                        }
                    )
                    chunks.append(
                        ChunkRecord(chunk_id=chunk_id, doc_id=doc.doc_id, text=text, metadata=metadata)
                    )
            else:
                words = doc.text.split()
                for start in range(0, len(words), step):
                    word_slice = words[start : start + self.chunk_size]
                    if not word_slice:
                        continue
                    text = " ".join(word_slice).strip()
                    if not text:
                        continue
                    chunk_id = sha256_text(f"{doc.doc_id}::{start}::{text[:64]}")
                    metadata = dict(doc.metadata)
                    metadata.update(
                        {
                            "chunk_id": chunk_id,
                            "start_index": start,
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                            "chunking_strategy": self.strategy.value,
                        }
                    )
                    chunks.append(
                        ChunkRecord(chunk_id=chunk_id, doc_id=doc.doc_id, text=text, metadata=metadata)
                    )
        return chunks

    def _split_semantic(self, docs: list[DocumentRecord]) -> list[ChunkRecord]:
        """Approximate semantic chunking using sentence boundaries and window merge."""

        chunks: list[ChunkRecord] = []
        for doc in docs:
            sentences = self._sentence_split(doc.text)
            if not sentences:
                continue
            window = max(1, self.semantic_window)
            i = 0
            chunk_index = 0
            while i < len(sentences):
                selected: list[str] = []
                token_count = 0
                while i < len(sentences) and len(selected) < window:
                    sent = sentences[i]
                    if self._encoder is not None:
                        sent_tokens = len(self._encoder.encode(sent))
                    else:
                        sent_tokens = len(sent.split())
                    if selected and token_count + sent_tokens > self.chunk_size:
                        break
                    selected.append(sent)
                    token_count += sent_tokens
                    i += 1
                if not selected:
                    i += 1
                    continue
                text = " ".join(selected).strip()
                chunk_id = sha256_text(f"{doc.doc_id}::{chunk_index}::{text[:64]}")
                metadata = dict(doc.metadata)
                metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "start_index": chunk_index,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": 0,
                        "chunking_strategy": self.strategy.value,
                    }
                )
                chunks.append(ChunkRecord(chunk_id=chunk_id, doc_id=doc.doc_id, text=text, metadata=metadata))
                chunk_index += 1
        return chunks

    @staticmethod
    def _sentence_split(text: str) -> Sequence[str]:
        sentence_break = re.compile(r"(?<=[.!?])\s+")
        return [segment.strip() for segment in sentence_break.split(text) if segment.strip()]
