"""RAG assistant for recruiter queries over candidate embeddings."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from resume_ai.config.loader import AppConfig
from resume_ai.db import models
from resume_ai.embeddings.service import EmbeddingService
from resume_ai.vector.chroma_store import ChromaStore


class RecruiterAssistant:
    """Semantic search and grounded recruiter answers."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.embedding = EmbeddingService(config)
        self.vector_store = ChromaStore(config)

    def search_candidates(self, query: str, top_k: int = 10) -> list[dict]:
        query_emb = self.embedding.embed_text(query)
        payload = self.vector_store.query_resumes(query_embedding=query_emb, n_results=top_k)

        ids = payload.get("ids", [[]])[0]
        documents = payload.get("documents", [[]])[0]
        metadatas = payload.get("metadatas", [[]])[0]
        distances = payload.get("distances", [[]])[0]

        rows: list[dict] = []
        for idx, item_id in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            rows.append(
                {
                    "chunk_id": item_id,
                    "candidate_id": meta.get("candidate_id"),
                    "resume_id": meta.get("resume_id"),
                    "snippet": documents[idx][:300] if idx < len(documents) else "",
                    "distance": distances[idx] if idx < len(distances) else None,
                }
            )
        return rows

    def answer(self, session: Session, query: str, top_k: int = 5) -> dict:
        hits = self.search_candidates(query=query, top_k=top_k)
        candidate_ids = [hit["candidate_id"] for hit in hits if hit.get("candidate_id") is not None]

        if not candidate_ids:
            return {"answer": "No matching candidates found.", "citations": []}

        rows = session.execute(
            select(models.Candidate.id, models.Candidate.name).where(models.Candidate.id.in_(candidate_ids))
        ).all()
        name_map = {row.id: row.name for row in rows}

        cited = []
        for hit in hits:
            cid = hit.get("candidate_id")
            if cid is None:
                continue
            cited.append(
                {
                    "candidate_id": cid,
                    "candidate_name": name_map.get(cid),
                    "snippet": hit.get("snippet", ""),
                }
            )

        answer = "Top matching candidates: " + ", ".join(
            f"{item['candidate_name'] or f'Candidate {item['candidate_id']}'}"
            for item in cited[:3]
        )
        return {"answer": answer, "citations": cited}
