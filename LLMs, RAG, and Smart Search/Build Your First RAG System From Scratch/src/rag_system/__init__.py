"""Public API for Build Your First RAG System project."""

from .advanced_retrieval import AdvancedRetriever, AdvancedRetrievalOutput
from .chunking import ChunkingResult, ChunkingStrategy, TextChunker
from .config import RAGConfig, load_config
from .data import prepare_dataset_artifacts
from .diagnostics import embedding_integrity_report, index_integrity_report, retrieval_diagnostics
from .embeddings import EmbeddingEngine
from .evaluation import EvaluationBundle, JudgeMetricSummary, LLMJudge, RAGEvaluator
from .faiss_benchmark import FaissBenchmark, FaissUnavailableError
from .generation import GenerationEngine, RAGPipeline
from .metrics import (
    GenerationMetricSummary,
    RetrievalMetricRow,
    RetrievalMetricSummary,
    compute_generation_metrics,
    compute_retrieval_metrics,
)
from .pipeline import ProjectRunner
from .prompts import PromptLibrary
from .retrieval import RetrievalEngine
from .types import ChunkRecord, DocumentRecord, JudgeResult, QueryRecord, RAGResponse, RetrievedChunk

__all__ = [
    "AdvancedRetriever",
    "AdvancedRetrievalOutput",
    "ChunkingResult",
    "ChunkingStrategy",
    "TextChunker",
    "RAGConfig",
    "load_config",
    "prepare_dataset_artifacts",
    "embedding_integrity_report",
    "index_integrity_report",
    "retrieval_diagnostics",
    "EmbeddingEngine",
    "EvaluationBundle",
    "JudgeMetricSummary",
    "LLMJudge",
    "RAGEvaluator",
    "FaissBenchmark",
    "FaissUnavailableError",
    "GenerationEngine",
    "RAGPipeline",
    "GenerationMetricSummary",
    "RetrievalMetricRow",
    "RetrievalMetricSummary",
    "compute_generation_metrics",
    "compute_retrieval_metrics",
    "ProjectRunner",
    "PromptLibrary",
    "RetrievalEngine",
    "ChunkRecord",
    "DocumentRecord",
    "JudgeResult",
    "QueryRecord",
    "RAGResponse",
    "RetrievedChunk",
]
