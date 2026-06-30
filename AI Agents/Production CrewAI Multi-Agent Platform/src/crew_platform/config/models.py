"""Configuration models and loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "Production CrewAI Multi-Agent Platform"
    environment: str = "dev"


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    default_model: str = "qwen3:8b"
    planner_model: str = "qwen3:8b"
    verifier_model: str = "llama3.1:8b"
    reflection_model: str = "gemma3:4b"
    consensus_models: list[str] = Field(default_factory=lambda: ["llama3.1:8b", "qwen3:8b", "gemma3:4b"])
    available_models: list[str] = Field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 1200
    request_timeout_seconds: int = 180
    auto_pull_missing_models: bool = False


class OrchestrationConfig(BaseModel):
    mode: str = "hybrid"
    max_parallel_tasks: int = 2
    max_iterations: int = 20
    default_retry_limit: int = 2
    retry_backoff_seconds: float = 2.0
    use_crewai_execution: bool = False
    consensus_enabled: bool = True
    consensus_trigger_confidence: float = 0.65
    plan_approval_required: bool = True


class MemoryConfig(BaseModel):
    sqlite_path: str = "artifacts/platform.db"
    chroma_enabled: bool = True
    chroma_path: str = "artifacts/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    retrieval_top_k: int = 5
    conversation_window: int = 30


class RAGConfig(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 200
    max_chunks_per_doc: int = 200


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class ToolsConfig(BaseModel):
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])
    optional_tools: list[str] = Field(default_factory=list)
    enable_python_tool: bool = False
    python_timeout_seconds: int = 5
    python_memory_limit_mb: int = 128
    graceful_network_fallback: bool = True
    workspace_root: str = "."


class MonitoringConfig(BaseModel):
    collect_interval_seconds: int = 5
    enable_gpu_metrics: bool = True


class ReportsConfig(BaseModel):
    output_dir: str = "data/reports"
    always_formats: list[str] = Field(default_factory=lambda: ["markdown", "json"])
    on_demand_formats: list[str] = Field(default_factory=lambda: ["html", "pdf"])


class MCPConfig(BaseModel):
    enabled: bool = True
    server_name: str = "crew-platform-mcp"
    external_servers: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = True
    run_log_path: str = "logs/agent_runs.jsonl"


class AgentProfile(BaseModel):
    id: str
    role: str
    goal: str
    backstory: str
    tools: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    output_schema: str


class AgentCatalogConfig(BaseModel):
    agents: list[AgentProfile] = Field(default_factory=list)


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping in {path}")
    return data


def load_settings(path: str | Path = "configs/settings.yaml") -> Settings:
    """Load and validate platform settings from YAML."""

    raw = _load_yaml(Path(path))
    return Settings.model_validate(raw)


def load_agent_catalog(path: str | Path = "configs/agents.yaml") -> AgentCatalogConfig:
    """Load agent catalog from YAML."""

    raw = _load_yaml(Path(path))
    return AgentCatalogConfig.model_validate(raw)
