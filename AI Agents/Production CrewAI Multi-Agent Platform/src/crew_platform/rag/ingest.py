"""RAG ingestion pipeline for local files and web documents."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import trafilatura
from pypdf import PdfReader

from crew_platform.config import Settings
from crew_platform.memory.runtime import RuntimeMemory


class RAGIngestionService:
    """Loads documents, chunks them, and stores in semantic memory."""

    def __init__(self, settings: Settings, memory: RuntimeMemory) -> None:
        self.settings = settings
        self.memory = memory

    def ingest_path(self, path: str, source_type: str | None = None) -> dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(path)

        suffix = (source_type or file_path.suffix.lower().lstrip(".")).lower()
        text = self._read_file(file_path, suffix)
        chunks = self._chunk(text)
        for idx, chunk in enumerate(chunks):
            self.memory.remember_run(
                run_id=f"doc-{file_path.stem}-{idx}",
                objective=f"ingest:{file_path.name}",
                summary=chunk,
            )

        return {
            "source": str(file_path),
            "chunks_indexed": len(chunks),
            "type": suffix,
        }

    async def ingest_url(self, url: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
        extracted = trafilatura.extract(response.text, include_links=True, include_comments=False)
        text = extracted or response.text
        chunks = self._chunk(text)
        for idx, chunk in enumerate(chunks):
            self.memory.remember_run(
                run_id=f"web-{hashlib.sha1(url.encode()).hexdigest()[:10]}-{idx}",
                objective=f"ingest_url:{url}",
                summary=chunk,
            )
        return {"source": url, "chunks_indexed": len(chunks), "type": "web"}

    def _read_file(self, file_path: Path, suffix: str) -> str:
        if suffix in {"md", "markdown", "txt", "rst"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        if suffix == "pdf":
            reader = PdfReader(str(file_path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if suffix == "csv":
            df = pl.read_csv(file_path)
            return df.head(200).write_csv()
        if suffix == "json":
            return file_path.read_text(encoding="utf-8", errors="ignore")
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _chunk(self, text: str) -> list[str]:
        chunk_size = self.settings.rag.chunk_size
        overlap = self.settings.rag.chunk_overlap
        if chunk_size <= 0:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text) and len(chunks) < self.settings.rag.max_chunks_per_doc:
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(end - overlap, 0)
        return chunks
