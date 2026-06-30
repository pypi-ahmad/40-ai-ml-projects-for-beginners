"""FastAPI app exposing resume intelligence endpoints."""

from __future__ import annotations

import uuid
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, HTTPException

from resume_ai.api.schemas import (
    CompareRequest,
    InterviewRequest,
    JobRequest,
    NoteRequest,
    ReportRequest,
    ScoreRequest,
    SearchRequest,
    UploadRequest,
)
from resume_ai.models import APIEnvelope
from resume_ai.service import ResumeAIService

app = FastAPI(title="Production AI Resume Screener", version="0.1.0")
service: ResumeAIService | None = None


@lru_cache(maxsize=1)
def _build_service() -> ResumeAIService:
    return ResumeAIService()


def get_service() -> ResumeAIService:
    global service
    if service is None:
        service = _build_service()
    return service


def envelope(data: dict, errors: list[str] | None = None) -> APIEnvelope:
    return APIEnvelope(status="ok" if not errors else "error", data=data, errors=errors or [], trace_id=str(uuid.uuid4()))


@app.get("/health")
def health() -> APIEnvelope:
    return envelope(get_service().health())


@app.post("/upload")
def upload(payload: UploadRequest) -> APIEnvelope:
    try:
        return envelope(get_service().upload_resume(payload.file_path, blind_mode=payload.blind_mode))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/resume/{resume_id}")
def get_resume(resume_id: int) -> APIEnvelope:
    try:
        return envelope(get_service().get_resume(resume_id))
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/candidate/{candidate_id}")
def get_candidate(candidate_id: int) -> APIEnvelope:
    try:
        return envelope(get_service().get_candidate(candidate_id))
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/job")
def create_job(payload: JobRequest) -> APIEnvelope:
    try:
        return envelope(get_service().create_job(payload.jd_text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/score")
def score(payload: ScoreRequest) -> APIEnvelope:
    try:
        out = get_service().score(payload.candidate_id, payload.job_id, payload.weight_override)
        return envelope(out.model_dump(mode="json"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/compare")
def compare(payload: CompareRequest) -> APIEnvelope:
    try:
        out = get_service().compare(payload.job_id, payload.candidate_ids)
        return envelope({"results": out})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search")
def search(payload: SearchRequest) -> APIEnvelope:
    try:
        out = get_service().search(payload.query, top_k=payload.top_k)
        return envelope(out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/interview")
def interview(payload: InterviewRequest) -> APIEnvelope:
    try:
        out = get_service().generate_interview(payload.candidate_id, payload.job_id)
        return envelope({"questions": out})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reports")
def reports(payload: ReportRequest) -> APIEnvelope:
    try:
        out = get_service().generate_report(payload.candidate_id, payload.job_id, output_dir=payload.output_dir)
        return envelope(out)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/reports")
def list_reports() -> APIEnvelope:
    return envelope({"reports": get_service().list_reports()})


@app.get("/analytics")
def analytics() -> APIEnvelope:
    return envelope(get_service().analytics())


@app.post("/notes")
def notes(payload: NoteRequest) -> APIEnvelope:
    note_id = get_service().add_note(
        candidate_id=payload.candidate_id,
        note=payload.note,
        tags=payload.tags,
        status=payload.status,
    )
    return envelope({"note_id": note_id})


@app.post("/mcp/tool")
def mcp_tool(payload: dict) -> APIEnvelope:
    """HTTP bridge for MCP-like tool calls.

    payload shape:
    {"tool": "resume_search", "args": {...}}
    """
    tool = payload.get("tool")
    args = payload.get("args", {})

    if tool == "resume_search":
        return envelope(get_service().search(args.get("query", ""), top_k=args.get("top_k", 10)))
    if tool == "candidate_lookup":
        return envelope(get_service().get_candidate(args["candidate_id"]))
    if tool == "generate_interview":
        return envelope(
            {
                "questions": get_service().generate_interview(
                    candidate_id=args["candidate_id"],
                    job_id=args["job_id"],
                )
            }
        )
    if tool == "score_candidate":
        result = get_service().score(candidate_id=args["candidate_id"], job_id=args["job_id"])
        return envelope(result.model_dump(mode="json"))
    if tool == "generate_report":
        result = get_service().generate_report(candidate_id=args["candidate_id"], job_id=args["job_id"])
        return envelope(result)

    raise HTTPException(status_code=404, detail=f"Unknown tool: {tool}")


def run() -> None:
    uvicorn.run(
        "resume_ai.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
