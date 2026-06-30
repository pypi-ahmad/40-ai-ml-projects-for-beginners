"""Enterprise agent registry with 20 required agents."""

from __future__ import annotations

from langgraph_platform.agents.prompts import (
    PLANNER_PROMPT,
    RESEARCH_PROMPT,
    SUPERVISOR_PROMPT,
    VERIFICATION_PROMPT,
    WRITER_PROMPT,
)
from langgraph_platform.agents.spec import AgentSpec


def _default_schema() -> dict[str, str]:
    return {"content": "str", "confidence": "float", "sources": "list[dict]"}


AGENT_REGISTRY: dict[str, AgentSpec] = {
    "planner": AgentSpec(
        name="planner",
        role="Planner Agent",
        objective="Create plan and routing decisions",
        system_prompt=PLANNER_PROMPT,
        tools=["memory_search", "calculator"],
        constraints=["No fabricated facts", "Use structured output"],
        output_schema={"plan": "str", "subtasks": "list[str]", "routing": "dict"},
    ),
    "research": AgentSpec(
        name="research",
        role="Research Agent",
        objective="General knowledge retrieval",
        system_prompt=RESEARCH_PROMPT,
        tools=["duckduckgo_search", "wikipedia", "url_fetcher"],
        constraints=["Cite every factual claim"],
        output_schema=_default_schema(),
    ),
    "technical_research": AgentSpec(
        name="technical_research",
        role="Technical Research Agent",
        objective="Technical deep dive and architecture evidence",
        system_prompt=RESEARCH_PROMPT,
        tools=["github_search", "documentation_search", "markdown_reader"],
        constraints=["Prefer official docs"],
        output_schema=_default_schema(),
    ),
    "web_search": AgentSpec(
        name="web_search",
        role="Web Search Agent",
        objective="Find latest web signals",
        system_prompt=RESEARCH_PROMPT,
        tools=["duckduckgo_search", "news_search", "url_fetcher"],
        output_schema=_default_schema(),
    ),
    "documentation": AgentSpec(
        name="documentation",
        role="Documentation Agent",
        objective="Retrieve docs and summarize authoritative guidance",
        system_prompt=RESEARCH_PROMPT,
        tools=["documentation_search", "markdown_reader", "pdf_reader"],
        output_schema=_default_schema(),
    ),
    "github": AgentSpec(
        name="github",
        role="GitHub Agent",
        objective="Query repositories/issues/README data",
        system_prompt=RESEARCH_PROMPT,
        tools=["github_search", "url_fetcher"],
        output_schema=_default_schema(),
    ),
    "rag": AgentSpec(
        name="rag",
        role="RAG Agent",
        objective="Retrieve semantic chunks from knowledge base",
        system_prompt=RESEARCH_PROMPT,
        tools=["chroma_search", "memory_search"],
        output_schema=_default_schema(),
    ),
    "memory": AgentSpec(
        name="memory",
        role="Memory Agent",
        objective="Use long-term memory and session context",
        system_prompt=RESEARCH_PROMPT,
        tools=["memory_search", "sql_tool"],
        output_schema=_default_schema(),
    ),
    "knowledge_manager": AgentSpec(
        name="knowledge_manager",
        role="Knowledge Manager",
        objective="Normalize, deduplicate, and rank evidence",
        system_prompt=RESEARCH_PROMPT,
        tools=["chroma_search", "json_reader"],
        output_schema=_default_schema(),
    ),
    "business_analyst": AgentSpec(
        name="business_analyst",
        role="Business Analyst",
        objective="Business impact assessment",
        system_prompt=WRITER_PROMPT,
        tools=["calculator", "currency_converter"],
        output_schema=_default_schema(),
    ),
    "data_analyst": AgentSpec(
        name="data_analyst",
        role="Data Analyst",
        objective="Data/statistical interpretation",
        system_prompt=WRITER_PROMPT,
        tools=["csv_reader", "python_repl", "calculator"],
        output_schema=_default_schema(),
    ),
    "financial_analyst": AgentSpec(
        name="financial_analyst",
        role="Financial Analyst",
        objective="Financial trend and valuation analysis",
        system_prompt=WRITER_PROMPT,
        tools=["currency_converter", "calculator", "news_search"],
        output_schema=_default_schema(),
    ),
    "technical_writer": AgentSpec(
        name="technical_writer",
        role="Technical Writer",
        objective="Produce technical summary",
        system_prompt=WRITER_PROMPT,
        tools=["markdown_reader"],
        output_schema=_default_schema(),
    ),
    "report_writer": AgentSpec(
        name="report_writer",
        role="Report Writer",
        objective="Create final report",
        system_prompt=WRITER_PROMPT,
        tools=["markdown_reader"],
        output_schema={
            "report_markdown": "str",
            "citation_ids": "list[str]",
            "confidence": "float",
        },
    ),
    "fact_checker": AgentSpec(
        name="fact_checker",
        role="Fact Checker",
        objective="Verify factual consistency",
        system_prompt=VERIFICATION_PROMPT,
        tools=["url_fetcher", "chroma_search"],
        output_schema={"status": "str", "issues": "list[str]", "confidence": "float"},
    ),
    "reflection": AgentSpec(
        name="reflection",
        role="Reflection Agent",
        objective="Critique and revise quality",
        system_prompt=VERIFICATION_PROMPT,
        tools=["markdown_reader"],
        output_schema={"improvements": "list[str]", "confidence": "float"},
    ),
    "critic": AgentSpec(
        name="critic",
        role="Critic Agent",
        objective="Red-team the output",
        system_prompt=VERIFICATION_PROMPT,
        tools=["markdown_reader"],
        output_schema={"risks": "list[str]", "confidence": "float"},
    ),
    "qa": AgentSpec(
        name="qa",
        role="QA Agent",
        objective="Perform quality gates",
        system_prompt=VERIFICATION_PROMPT,
        tools=["json_reader"],
        output_schema={"qa_status": "str", "issues": "list[str]", "confidence": "float"},
    ),
    "citation": AgentSpec(
        name="citation",
        role="Citation Agent",
        objective="Attach and normalize citations",
        system_prompt=VERIFICATION_PROMPT,
        tools=["url_fetcher"],
        output_schema={"citations": "list[dict]", "confidence": "float"},
    ),
    "supervisor": AgentSpec(
        name="supervisor",
        role="Supervisor Agent",
        objective="Final approval and governance",
        system_prompt=SUPERVISOR_PROMPT,
        tools=["memory_search"],
        constraints=["Can reject low-confidence output"],
        output_schema={"approve": "bool", "reason": "str", "confidence": "float"},
    ),
}


def get_agent(name: str) -> AgentSpec:
    """Retrieve one agent spec by name."""

    if name not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent: {name}")
    return AGENT_REGISTRY[name]


def list_agents() -> list[AgentSpec]:
    """List all configured agent specs."""

    return list(AGENT_REGISTRY.values())
