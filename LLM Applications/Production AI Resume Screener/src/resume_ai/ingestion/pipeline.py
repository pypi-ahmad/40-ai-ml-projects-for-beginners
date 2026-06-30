"""End-to-end ingestion pipeline for single and batch resumes."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from resume_ai.config.loader import AppConfig
from resume_ai.db import models
from resume_ai.db.repository import insert_resume, sync_candidate_details, upsert_candidate
from resume_ai.embeddings.service import EmbeddingService
from resume_ai.ingestion.dedupe import find_exact_duplicate, find_near_duplicate
from resume_ai.ingestion.readers import ResumeReader
from resume_ai.models import ResumeParseResult
from resume_ai.parsing.resume_parser import ResumeParser
from resume_ai.vector.chroma_store import ChromaStore


class IngestionPipeline:
    """Parse and persist resumes, then index embeddings."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.reader = ResumeReader(config)
        self.parser = ResumeParser(config)
        self.embedding = EmbeddingService(config)
        self.vector_store = ChromaStore(config)

    def ingest_file(self, session: Session, path: Path, blind_mode: bool = True) -> ResumeParseResult:
        text, ocr_mode = self.reader.read(path)
        file_hash = self.reader.compute_file_hash(path)
        duplicate = find_exact_duplicate(session, file_hash)
        if duplicate:
            parsed_duplicate = ResumeParseResult.model_validate(duplicate.parsed_json)
            sync_candidate_details(
                session,
                candidate_id=duplicate.candidate_id,
                parsed=parsed_duplicate,
            )
            return parsed_duplicate

        parsed = self.parser.parse(text=text, ocr_mode=ocr_mode, blind_mode=blind_mode)
        candidate = upsert_candidate(session, parsed)
        resume = insert_resume(
            session=session,
            candidate_id=candidate.id,
            file_name=path.name,
            file_hash=file_hash,
            raw_text=text,
            parsed=parsed,
        )
        sync_candidate_details(session, candidate_id=candidate.id, parsed=parsed)

        candidate_text = parsed.redacted_text or text
        emb = self.embedding.embed_text(candidate_text)

        existing_embeddings = self._load_existing_candidate_embeddings(session)
        near_id = find_near_duplicate(emb, existing_embeddings)
        if near_id is not None and near_id != candidate.id:
            resume.ingestion_status = "near_duplicate"

        self.vector_store.upsert_resume(
            resume_id=resume.id,
            candidate_id=candidate.id,
            text=candidate_text,
            metadata={
                "candidate_id": candidate.id,
                "resume_id": resume.id,
                "ocr_mode": parsed.ocr_mode.value,
                "blinded": bool(parsed.redacted_text),
            },
            embedding=emb,
        )
        return parsed

    def ingest_folder(self, session: Session, folder: Path, blind_mode: bool = True) -> list[ResumeParseResult]:
        results: list[ResumeParseResult] = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"}:
                continue
            results.append(self.ingest_file(session, path, blind_mode=blind_mode))
        return results

    def _load_existing_candidate_embeddings(self, session: Session) -> list[tuple[int, object]]:
        rows = session.execute(select(models.Resume.id, models.Resume.raw_text)).all()
        output: list[tuple[int, object]] = []
        for resume_id, raw_text in rows:
            embedding = self.embedding.embed_text(raw_text[:3000])
            output.append((resume_id, embedding))
        return output
