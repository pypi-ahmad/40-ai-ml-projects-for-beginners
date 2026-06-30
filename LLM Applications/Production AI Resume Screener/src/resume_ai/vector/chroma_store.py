"""ChromaDB storage for resume and job embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from resume_ai.config.loader import AppConfig

try:
    import chromadb
    from chromadb.config import Settings
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]


class ChromaStore:
    """Wrapper around Chroma collections used by platform."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._client = None
        self._collections: dict[str, Any] = {}
        self._bootstrap()

    def _bootstrap(self) -> None:
        if chromadb is None:
            return
        settings = None
        if Settings is not None:
            settings = Settings(anonymized_telemetry=False)
        self._client = chromadb.PersistentClient(path=self.config.runtime.chroma_path, settings=settings)
        for name in [
            "resume_chunks",
            "job_descriptions",
            "candidate_summaries",
            "projects",
            "experience",
            "interview_feedback",
        ]:
            self._collections[name] = self._client.get_or_create_collection(name=name)

    def upsert_resume(
        self,
        resume_id: int,
        candidate_id: int,
        text: str,
        metadata: dict[str, Any],
        embedding: np.ndarray,
    ) -> None:
        collection = self._collections.get("resume_chunks")
        if collection is None:
            return
        collection.upsert(
            ids=[f"resume:{resume_id}"],
            documents=[text],
            metadatas=[{**metadata, "candidate_id": candidate_id, "resume_id": resume_id}],
            embeddings=[embedding.tolist()],
        )

    def upsert_job(self, job_id: int, text: str, metadata: dict[str, Any], embedding: np.ndarray) -> None:
        collection = self._collections.get("job_descriptions")
        if collection is None:
            return
        collection.upsert(
            ids=[f"job:{job_id}"],
            documents=[text],
            metadatas=[{**metadata, "job_id": job_id}],
            embeddings=[embedding.tolist()],
        )

    def query_resumes(
        self,
        query_embedding: np.ndarray,
        n_results: int = 20,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        collection = self._collections.get("resume_chunks")
        if collection is None:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
        )
