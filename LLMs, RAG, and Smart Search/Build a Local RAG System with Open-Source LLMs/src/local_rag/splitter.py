"""Text chunking utilities."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag.types import ChunkRecord, LoadedDocument
from local_rag.utils import sha256_text


class TextChunker:
    """Chunk documents using RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def split_documents(self, docs: list[LoadedDocument]) -> list[ChunkRecord]:
        """Split loaded documents into normalized chunks."""

        texts = [doc.text for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        split_docs = self._splitter.create_documents(texts=texts, metadatas=metadatas)

        chunks: list[ChunkRecord] = []
        for split_doc in split_docs:
            metadata = dict(split_doc.metadata)
            doc_id = metadata["doc_id"]
            start_index = int(metadata.get("start_index", 0))
            chunk_id = sha256_text(f"{doc_id}::{start_index}::{split_doc.page_content[:64]}")
            metadata["chunk_size"] = self.chunk_size
            metadata["chunk_overlap"] = self.chunk_overlap
            metadata["chunk_id"] = chunk_id
            metadata["start_index"] = start_index
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=split_doc.page_content,
                    metadata=metadata,
                )
            )
        return chunks
