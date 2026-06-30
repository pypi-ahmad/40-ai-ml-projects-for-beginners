"""MCP server exposing resume intelligence tools."""

from __future__ import annotations

from resume_ai.service import ResumeAIService

service = ResumeAIService()


def build_server():
    """Build MCP server instance."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("resume-ai")

    @mcp.tool(name="resume_search")
    def resume_search(query: str, top_k: int = 10) -> dict:
        return service.search(query=query, top_k=top_k)

    @mcp.tool(name="candidate_lookup")
    def candidate_lookup(candidate_id: int) -> dict:
        return service.get_candidate(candidate_id)

    @mcp.tool(name="generate_interview")
    def generate_interview(candidate_id: int, job_id: int) -> dict:
        return service.generate_interview(candidate_id=candidate_id, job_id=job_id)

    @mcp.tool(name="score_candidate")
    def score_candidate(candidate_id: int, job_id: int) -> dict:
        return service.score(candidate_id=candidate_id, job_id=job_id).model_dump(mode="json")

    @mcp.tool(name="generate_report")
    def generate_report(candidate_id: int, job_id: int) -> dict:
        return service.generate_report(candidate_id=candidate_id, job_id=job_id)

    return mcp


def run() -> None:
    mcp = build_server()
    mcp.run()


if __name__ == "__main__":
    run()
