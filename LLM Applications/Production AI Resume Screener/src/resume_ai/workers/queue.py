"""SQLite-backed queue for resumable batch processing."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from resume_ai.db import models


class ProcessingQueue:
    """Persistent queue over processing_jobs table."""

    def enqueue_paths(self, session: Session, paths: list[Path]) -> int:
        count = 0
        for path in paths:
            row = models.ProcessingJob(source_path=str(path), status="queued", attempts=0)
            session.add(row)
            count += 1
        session.flush()
        return count

    def pop_next(self, session: Session) -> models.ProcessingJob | None:
        row = session.scalar(
            select(models.ProcessingJob)
            .where(models.ProcessingJob.status == "queued")
            .order_by(models.ProcessingJob.id.asc())
            .limit(1)
        )
        if row is None:
            return None
        row.status = "running"
        row.attempts += 1
        session.flush()
        return row

    def mark_done(self, session: Session, row_id: int) -> None:
        row = session.get(models.ProcessingJob, row_id)
        if row is None:
            return
        row.status = "done"
        session.flush()

    def mark_failed(self, session: Session, row_id: int, error_message: str) -> None:
        row = session.get(models.ProcessingJob, row_id)
        if row is None:
            return
        row.status = "failed"
        row.error_message = error_message[:4000]
        session.flush()
