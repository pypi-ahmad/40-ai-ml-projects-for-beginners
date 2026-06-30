"""FastAPI request and response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadRequest(BaseModel):
    file_path: str
    blind_mode: bool = True


class JobRequest(BaseModel):
    jd_text: str


class ScoreRequest(BaseModel):
    candidate_id: int
    job_id: int
    weight_override: dict[str, float] | None = None


class CompareRequest(BaseModel):
    job_id: int
    candidate_ids: list[int] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class InterviewRequest(BaseModel):
    candidate_id: int
    job_id: int


class ReportRequest(BaseModel):
    candidate_id: int
    job_id: int
    output_dir: str = "outputs/reports"


class NoteRequest(BaseModel):
    candidate_id: int
    note: str
    tags: list[str] = Field(default_factory=list)
    status: str = "new"
