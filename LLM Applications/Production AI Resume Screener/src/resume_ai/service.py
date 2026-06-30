"""Application service orchestrating ingestion, parsing, scoring, RAG, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from resume_ai.analytics.metrics import build_snapshot
from resume_ai.config.loader import AppConfig, load_config
from resume_ai.db import Base, models
from resume_ai.db.repository import save_score, upsert_job
from resume_ai.db.session import create_engine_from_config, get_session_factory, session_scope
from resume_ai.embeddings.service import EmbeddingService
from resume_ai.ingestion.pipeline import IngestionPipeline
from resume_ai.interview.generator import InterviewGenerator
from resume_ai.matching.engine import MatchingEngine
from resume_ai.models import JobRequirementProfile, ResumeParseResult, ScoreBreakdown
from resume_ai.parsing.jd_parser import JobDescriptionParser
from resume_ai.rag.assistant import RecruiterAssistant
from resume_ai.ranking.engine import compare_candidates as compare_fn
from resume_ai.ranking.engine import rank_scores
from resume_ai.reasoning.recommender import build_recommendation
from resume_ai.reports.generator import export_html, export_json, export_markdown, export_pdf
from resume_ai.vector.chroma_store import ChromaStore
from resume_ai.workers.queue import ProcessingQueue


class ResumeAIService:
    """Facade for all platform features."""

    def __init__(self, config_path: str | Path = "config"):
        self.config = load_config(config_path)
        self.engine = create_engine_from_config(self.config)
        self.session_factory = get_session_factory(self.engine)
        Base.metadata.create_all(self.engine)

        self.ingestion = IngestionPipeline(self.config)
        self.jd_parser = JobDescriptionParser(self.config)
        self.matching = MatchingEngine(self.config)
        self.interview = InterviewGenerator(self.config)
        self.assistant = RecruiterAssistant(self.config)
        self.embedding = EmbeddingService(self.config)
        self.vector_store = ChromaStore(self.config)
        self.queue = ProcessingQueue()

    def health(self) -> dict[str, Any]:
        with session_scope(self.session_factory) as session:
            total_candidates = session.scalar(select(func.count(models.Candidate.id))) or 0
            total_resumes = session.scalar(select(func.count(models.Resume.id))) or 0
            total_jobs = session.scalar(select(func.count(models.JobDescription.id))) or 0
        return {
            "status": "ok",
            "database": "sqlite",
            "vector_db": "chromadb",
            "models": self.config.models.model_dump(),
            "counts": {
                "candidates": total_candidates,
                "resumes": total_resumes,
                "jobs": total_jobs,
            },
        }

    def upload_resume(self, file_path: str, blind_mode: bool = True) -> dict[str, Any]:
        path = Path(file_path)
        with session_scope(self.session_factory) as session:
            parsed = self.ingestion.ingest_file(session, path=path, blind_mode=blind_mode)
            resume = session.scalar(
                select(models.Resume).order_by(models.Resume.id.desc()).limit(1)
            )
            candidate = session.get(models.Candidate, resume.candidate_id) if resume else None
            return {
                "candidate_id": candidate.id if candidate else None,
                "resume_id": resume.id if resume else None,
                "parsed": parsed.model_dump(mode="json"),
            }

    def enqueue_folder(self, folder_path: str) -> dict[str, Any]:
        folder = Path(folder_path)
        paths = [p for p in folder.rglob("*") if p.is_file()]
        with session_scope(self.session_factory) as session:
            count = self.queue.enqueue_paths(session, paths)
        return {"queued": count}

    def run_queue(self, blind_mode: bool = True, max_items: int = 1000) -> dict[str, Any]:
        processed = 0
        failed = 0
        with session_scope(self.session_factory) as session:
            while processed + failed < max_items:
                row = self.queue.pop_next(session)
                if row is None:
                    break
                try:
                    self.ingestion.ingest_file(session, Path(row.source_path), blind_mode=blind_mode)
                    self.queue.mark_done(session, row.id)
                    processed += 1
                except Exception as exc:  # pragma: no cover - runtime path
                    self.queue.mark_failed(session, row.id, str(exc))
                    failed += 1
        return {"processed": processed, "failed": failed}

    def get_resume(self, resume_id: int) -> dict[str, Any]:
        with session_scope(self.session_factory) as session:
            resume = session.get(models.Resume, resume_id)
            if resume is None:
                raise ValueError(f"Resume {resume_id} not found")
            return {
                "resume_id": resume.id,
                "candidate_id": resume.candidate_id,
                "parsed": resume.parsed_json,
                "ocr_mode": resume.ocr_mode,
            }

    def get_candidate(self, candidate_id: int) -> dict[str, Any]:
        with session_scope(self.session_factory) as session:
            candidate = session.get(models.Candidate, candidate_id)
            if candidate is None:
                raise ValueError(f"Candidate {candidate_id} not found")
            resumes = session.execute(
                select(models.Resume.id, models.Resume.file_name, models.Resume.created_at).where(
                    models.Resume.candidate_id == candidate_id
                )
            ).all()
            notes = session.execute(
                select(models.RecruiterNote.note, models.RecruiterNote.status).where(
                    models.RecruiterNote.candidate_id == candidate_id
                )
            ).all()
            return {
                "candidate": {
                    "id": candidate.id,
                    "name": candidate.name,
                    "email": candidate.email,
                    "phone": candidate.phone,
                    "location": candidate.location,
                    "linkedin": candidate.linkedin,
                    "github": candidate.github,
                    "portfolio": candidate.portfolio,
                },
                "resumes": [dict(row._mapping) for row in resumes],
                "notes": [dict(row._mapping) for row in notes],
            }

    def create_job(self, jd_text: str) -> dict[str, Any]:
        parsed_jd = self.jd_parser.parse(jd_text)
        with session_scope(self.session_factory) as session:
            job = upsert_job(session, jd_text=jd_text, parsed=parsed_jd)
            embedding = self.embedding.embed_text(jd_text)
            self.vector_store.upsert_job(
                job_id=job.id,
                text=jd_text,
                metadata={"job_id": job.id, "title": parsed_jd.title or "untitled"},
                embedding=embedding,
            )
        return {"job_id": job.id, "parsed": parsed_jd.model_dump(mode="json")}

    def score(self, candidate_id: int, job_id: int, weight_override: dict[str, float] | None = None) -> ScoreBreakdown:
        with session_scope(self.session_factory) as session:
            resume = session.scalar(
                select(models.Resume)
                .where(models.Resume.candidate_id == candidate_id)
                .order_by(models.Resume.id.desc())
                .limit(1)
            )
            job = session.get(models.JobDescription, job_id)
            if resume is None:
                raise ValueError(f"No resume found for candidate {candidate_id}")
            if job is None:
                raise ValueError(f"Job {job_id} not found")

            parsed_resume = ResumeParseResult.model_validate(resume.parsed_json)
            parsed_jd = JobRequirementProfile.model_validate(job.parsed_json)
            breakdown = self.matching.score_candidate(
                parsed_resume=parsed_resume,
                parsed_job=parsed_jd,
                candidate_id=candidate_id,
                job_id=job_id,
                override_weights=weight_override or job.weight_override,
            )
            save_score(session, breakdown)
            return breakdown

    def rank_for_job(self, job_id: int) -> list[dict[str, Any]]:
        with session_scope(self.session_factory) as session:
            rows = session.execute(
                select(models.CandidateJobScore.breakdown_json).where(models.CandidateJobScore.job_id == job_id)
            ).scalars()
            scores = [ScoreBreakdown.model_validate(row) for row in rows]
        ranked = rank_scores(scores)
        return [item.model_dump(mode="json") for item in ranked]

    def compare(self, job_id: int, candidate_ids: list[int]) -> list[dict[str, Any]]:
        with session_scope(self.session_factory) as session:
            rows = session.execute(
                select(models.CandidateJobScore.breakdown_json).where(models.CandidateJobScore.job_id == job_id)
            ).scalars()
            scores = [ScoreBreakdown.model_validate(row) for row in rows]
        compared = compare_fn(scores, candidate_ids)
        return [item.model_dump(mode="json") for item in compared]

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        with session_scope(self.session_factory) as session:
            answer = self.assistant.answer(session=session, query=query, top_k=top_k)
        return answer

    def generate_interview(self, candidate_id: int, job_id: int) -> dict[str, Any]:
        score = self.score(candidate_id=candidate_id, job_id=job_id)
        questions = self.interview.generate(score)
        with session_scope(self.session_factory) as session:
            session.add(
                models.InterviewPack(
                    candidate_id=candidate_id,
                    job_id=job_id,
                    questions_json=questions.model_dump(mode="json"),
                )
            )
        return questions.model_dump(mode="json")

    def generate_report(self, candidate_id: int, job_id: int, output_dir: str = "outputs/reports") -> dict[str, str]:
        score = self.score(candidate_id, job_id)
        with session_scope(self.session_factory) as session:
            resume = session.scalar(
                select(models.Resume)
                .where(models.Resume.candidate_id == candidate_id)
                .order_by(models.Resume.id.desc())
            )
            if resume is None:
                raise ValueError("Candidate resume not found")
            parsed = ResumeParseResult.model_validate(resume.parsed_json)

        recommendation = build_recommendation(parsed, score)
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        md_path = export_markdown(recommendation, score, root / f"candidate_{candidate_id}_job_{job_id}.md")
        html_path = export_html(recommendation, score, root / f"candidate_{candidate_id}_job_{job_id}.html")
        json_path = export_json(recommendation, score, root / f"candidate_{candidate_id}_job_{job_id}.json")
        pdf_path = export_pdf(recommendation, score, root / f"candidate_{candidate_id}_job_{job_id}.pdf")

        with session_scope(self.session_factory) as session:
            for path, report_type in [
                (md_path, "markdown"),
                (html_path, "html"),
                (json_path, "json"),
                (pdf_path, "pdf"),
            ]:
                session.add(
                    models.Report(
                        candidate_id=candidate_id,
                        job_id=job_id,
                        report_type=report_type,
                        artifact_path=str(path),
                    )
                )

        return {
            "markdown": str(md_path),
            "html": str(html_path),
            "json": str(json_path),
            "pdf": str(pdf_path),
        }

    def analytics(self) -> dict[str, Any]:
        with session_scope(self.session_factory) as session:
            snapshot = build_snapshot(session)
            scores = session.execute(select(models.CandidateJobScore.total_score)).scalars().all()
        return {
            "snapshot": snapshot.model_dump(mode="json"),
            "scores": list(scores),
        }

    def list_reports(self) -> list[dict[str, Any]]:
        with session_scope(self.session_factory) as session:
            rows = session.execute(select(models.Report)).scalars().all()
            return [
                {
                    "id": row.id,
                    "candidate_id": row.candidate_id,
                    "job_id": row.job_id,
                    "report_type": row.report_type,
                    "artifact_path": row.artifact_path,
                }
                for row in rows
            ]

    def add_note(self, candidate_id: int, note: str, tags: list[str] | None = None, status: str = "new") -> int:
        with session_scope(self.session_factory) as session:
            row = models.RecruiterNote(
                candidate_id=candidate_id,
                note=note,
                tags_json=tags or [],
                status=status,
            )
            session.add(row)
            session.flush()
            return row.id

    def export_multi_jd_comparison(self, candidate_id: int, job_ids: list[int]) -> dict[str, Any]:
        comparisons: dict[str, Any] = {"candidate_id": candidate_id, "jobs": []}
        for job_id in job_ids:
            score = self.score(candidate_id=candidate_id, job_id=job_id)
            comparisons["jobs"].append({"job_id": job_id, "score": score.model_dump(mode="json")})
        return comparisons

    def dump_state(self) -> str:
        with session_scope(self.session_factory) as session:
            rows = session.execute(select(models.Candidate.id, models.Candidate.name)).all()
        return json.dumps([dict(row._mapping) for row in rows], indent=2)
