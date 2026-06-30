"""Duplicate and near-duplicate detection."""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from resume_ai.db import models
from resume_ai.embeddings.service import cosine_similarity


def find_exact_duplicate(session: Session, file_hash: str) -> models.Resume | None:
    return session.scalar(select(models.Resume).where(models.Resume.file_hash == file_hash))


def find_near_duplicate(
    current_embedding: np.ndarray,
    existing_embeddings: list[tuple[int, np.ndarray]],
    threshold: float = 0.96,
) -> int | None:
    best_id = None
    best_score = threshold
    for candidate_id, embedding in existing_embeddings:
        score = cosine_similarity(current_embedding, embedding)
        if score >= best_score:
            best_score = score
            best_id = candidate_id
    return best_id
