"""LangGraph workflow builder and runtime executor."""

from __future__ import annotations

import os
from typing import Any

from langgraph.graph import END, START, StateGraph

from langgraph_platform.config.settings import AppConfig
from langgraph_platform.engine.nodes import (
    NodeRuntime,
    citation_node,
    consensus_reasoning_node,
    critic_node,
    fact_checker_node,
    finalize_node,
    knowledge_merge_node,
    parallel_research_node,
    parse_node_result,
    planner_node,
    qa_node,
    reflection_node,
    supervisor_node,
    writer_node,
)
from langgraph_platform.engine.router import should_retry_reflection
from langgraph_platform.memory.sqlite_store import SQLiteStore
from langgraph_platform.memory.vector_store import ChromaMemoryStore
from langgraph_platform.rag.pipeline import RAGPipeline
from langgraph_platform.state.models import ExecutionMetadata, WorkflowResult, WorkflowState
from langgraph_platform.utils.time import new_session_id, new_workflow_id


class LangGraphWorkflowEngine:
    """Production-oriented workflow engine backed by LangGraph."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.sqlite_store = SQLiteStore(config.sqlite_url)
        self.sqlite_store.init()
        self.vector_store = ChromaMemoryStore(
            chroma_path=config.memory.chroma_path,
            embed_model=config.model.embedder,
        )
        self.rag_pipeline = RAGPipeline(vector_store=self.vector_store)
        self.runtime = NodeRuntime(
            config=config,
            sqlite_store=self.sqlite_store,
            rag_pipeline=self.rag_pipeline,
        )
        self.workflow_graph = self._build_graph()
        self.graph = self.workflow_graph.compile()

    def close(self) -> None:
        """Cleanup runtime resources."""

        self.runtime.close()
        self.vector_store.close()
        self.sqlite_store.close()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(dict)

        graph.add_node("planner", lambda state: planner_node(state, self.runtime))
        graph.add_node(
            "parallel_research", lambda state: parallel_research_node(state, self.runtime)
        )
        graph.add_node("knowledge_merge", lambda state: knowledge_merge_node(state, self.runtime))
        graph.add_node(
            "consensus_reasoning", lambda state: consensus_reasoning_node(state, self.runtime)
        )
        graph.add_node("writer", lambda state: writer_node(state, self.runtime))
        graph.add_node("fact_checker", lambda state: fact_checker_node(state, self.runtime))
        graph.add_node("reflection", lambda state: reflection_node(state, self.runtime))
        graph.add_node("critic", lambda state: critic_node(state, self.runtime))
        graph.add_node("qa", lambda state: qa_node(state, self.runtime))
        graph.add_node("citation", lambda state: citation_node(state, self.runtime))
        graph.add_node("supervisor", lambda state: supervisor_node(state, self.runtime))
        graph.add_node("finalize", lambda state: finalize_node(state, self.runtime))

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "parallel_research")
        graph.add_edge("parallel_research", "knowledge_merge")
        graph.add_edge("knowledge_merge", "consensus_reasoning")
        graph.add_edge("consensus_reasoning", "writer")
        graph.add_edge("writer", "fact_checker")
        graph.add_edge("fact_checker", "reflection")
        graph.add_edge("reflection", "critic")
        graph.add_edge("critic", "qa")
        graph.add_edge("qa", "citation")
        graph.add_edge("citation", "supervisor")

        graph.add_conditional_edges(
            "supervisor",
            self._supervisor_route,
            {
                "retry": "writer",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)

        return graph

    def _supervisor_route(self, state_dict: dict[str, Any]) -> str:
        state = parse_node_result(state_dict)
        if should_retry_reflection(
            state=state,
            confidence_threshold=self.config.retry.confidence_threshold,
            max_retries=self.config.retry.max_retries,
        ):
            return "retry"
        return "finalize"

    def create_initial_state(
        self, user_request: str, session_id: str | None = None
    ) -> WorkflowState:
        """Create new typed workflow state."""

        workflow_id = new_workflow_id()
        return WorkflowState(
            user_request=user_request,
            execution_metadata=ExecutionMetadata(
                workflow_id=workflow_id,
                session_id=session_id or new_session_id(),
            ),
        )

    def run(self, user_request: str, session_id: str | None = None) -> WorkflowResult:
        """Run full workflow end to end."""

        initial = self.create_initial_state(user_request=user_request, session_id=session_id)
        raw = self.graph.invoke(initial.model_dump(mode="json"))
        final_state = parse_node_result(raw)
        if (
            os.environ.get("ENABLE_MLFLOW", "0") == "1"
            and self.config.monitoring.mlflow_tracking_uri
        ):
            self._log_run_to_mlflow(final_state)

        return WorkflowResult(
            workflow_id=final_state.execution_metadata.workflow_id,
            final_report=final_state.reports[-1] if final_state.reports else "No report generated.",
            confidence=final_state.confidence_score,
            verification_status=final_state.verification_status,
            citations=final_state.citations,
            metadata=final_state.execution_metadata,
        )

    def _log_run_to_mlflow(self, state: WorkflowState) -> None:
        """Log key run metrics to MLflow when available."""

        try:
            import mlflow

            mlflow.set_tracking_uri(self.config.monitoring.mlflow_tracking_uri)
            mlflow.set_experiment("langgraph_platform")
            with mlflow.start_run(run_name=state.execution_metadata.workflow_id):
                mlflow.log_metric("confidence", state.confidence_score)
                mlflow.log_metric("citations_count", len(state.citations))
                mlflow.log_metric("total_tokens", state.token_usage.total_tokens)
                mlflow.log_param("verification_status", state.verification_status.value)
        except Exception:
            return

    def inspect_graph(self) -> dict[str, Any]:
        """Return graph topology and metadata."""

        node_names = sorted(self.workflow_graph.nodes.keys())
        edge_pairs: list[dict[str, str]] = []
        for source, target in self.workflow_graph.edges:
            edge_pairs.append({"source": source, "target": target})
        for source, branch_map in self.workflow_graph.branches.items():
            for _, branch_spec in branch_map.items():
                for _, target in branch_spec.ends.items():
                    edge_pairs.append({"source": source, "target": target})

        return {
            "nodes": node_names,
            "edges": edge_pairs,
            "retry_max": self.config.retry.max_retries,
            "confidence_threshold": self.config.retry.confidence_threshold,
        }
