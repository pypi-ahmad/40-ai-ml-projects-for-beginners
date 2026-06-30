"""Evaluation harness for retrieval, generation, and LLM-as-a-judge."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rag_system.generation import GenerationEngine, RAGPipeline
from rag_system.metrics import (
    GenerationMetricSummary,
    RetrievalMetricSummary,
    RetrievalMetricRow,
    compute_generation_metrics,
    compute_retrieval_metrics,
)
from rag_system.prompts import PromptLibrary
from rag_system.types import JudgeResult, QueryRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JudgeMetricSummary:
    """Aggregate summary for judge scores."""

    relevance: float
    correctness: float
    groundedness: float
    completeness: float
    faithfulness: float
    num_examples: int


@dataclass(slots=True)
class EvaluationBundle:
    """Complete evaluation output object."""

    retrieval: RetrievalMetricSummary
    generation: GenerationMetricSummary
    judge: JudgeMetricSummary


class LLMJudge:
    """Local LLM-as-a-judge using granite4.1:3b."""

    def __init__(self, generation_engine: GenerationEngine) -> None:
        self.engine = generation_engine

    def score(self, query: str, context: str, answer: str) -> JudgeResult:
        """Evaluate answer using strict JSON schema output."""
        messages = PromptLibrary.judge_prompt(query=query, context=context, answer=answer)
        response = self.engine.generate(
            messages=messages,
            temperature=0.0,
            max_tokens=220,
            think=False,
            response_format={
                "type": "object",
                "properties": {
                    "relevance": {"type": "number"},
                    "correctness": {"type": "number"},
                    "groundedness": {"type": "number"},
                    "completeness": {"type": "number"},
                    "faithfulness": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "relevance",
                    "correctness",
                    "groundedness",
                    "completeness",
                    "faithfulness",
                    "rationale",
                ],
            },
        )

        raw = response.get("text", "")
        data = self._safe_json(raw)

        return JudgeResult(
            relevance=self._clip_score(data.get("relevance", 1.0)),
            correctness=self._clip_score(data.get("correctness", 1.0)),
            groundedness=self._clip_score(data.get("groundedness", 1.0)),
            completeness=self._clip_score(data.get("completeness", 1.0)),
            faithfulness=self._clip_score(data.get("faithfulness", 1.0)),
            rationale=str(data.get("rationale", "")),
            raw_output=raw,
        )

    @staticmethod
    def _clip_score(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 1.0
        return float(np.clip(numeric, 1.0, 5.0))

    @staticmethod
    def _safe_json(text: str) -> dict[str, Any]:
        if not text.strip():
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return {}
            return {}


class RAGEvaluator:
    """End-to-end evaluator for retrieval/generation/judge metrics."""

    def __init__(self, pipeline: RAGPipeline, judge_engine: GenerationEngine) -> None:
        self.pipeline = pipeline
        self.judge = LLMJudge(generation_engine=judge_engine)

    def evaluate_retrieval(
        self,
        queries: list[QueryRecord],
        top_k: int = 6,
        max_queries: int | None = None,
    ) -> tuple[RetrievalMetricSummary, list[RetrievalMetricRow], pd.DataFrame]:
        """Compute retrieval metrics from gold doc ids."""
        subset = queries[:max_queries] if max_queries else queries

        query_ids: list[str] = []
        retrieved_doc_ids: list[list[str]] = []
        gold_doc_ids: list[list[str]] = []
        diagnostic_rows: list[dict[str, Any]] = []

        for query_record in subset:
            chunks = self.pipeline.retrieval_engine.query(
                query=query_record.query,
                top_k=top_k,
                dedupe_by_doc=True,
            )
            docs = [chunk.doc_id for chunk in chunks]
            failure_bucket = self.pipeline.retrieval_engine.classify_retrieval(
                chunks=chunks,
                gold_doc_ids=query_record.gold_doc_ids,
                min_relevance_score=self.pipeline.min_relevance_score,
            )

            query_ids.append(query_record.query_id)
            retrieved_doc_ids.append(docs)
            gold_doc_ids.append(query_record.gold_doc_ids)

            diagnostic_rows.append(
                {
                    "query_id": query_record.query_id,
                    "query": query_record.query,
                    "retrieved_doc_ids": docs,
                    "gold_doc_ids": query_record.gold_doc_ids,
                    "top_score": chunks[0].score if chunks else 0.0,
                    "hit_at_k": int(any(doc_id in query_record.gold_doc_ids for doc_id in docs)),
                    "failure_bucket": failure_bucket,
                }
            )

        summary, rows = compute_retrieval_metrics(
            query_ids=query_ids,
            retrieved_doc_ids=retrieved_doc_ids,
            gold_doc_ids=gold_doc_ids,
            top_k=top_k,
        )
        return summary, rows, pd.DataFrame(diagnostic_rows)

    def evaluate_generation(
        self,
        queries: list[QueryRecord],
        top_k: int = 6,
        max_queries: int | None = None,
    ) -> tuple[GenerationMetricSummary, pd.DataFrame]:
        """Compute automated generation metrics (EM/BLEU/ROUGE/METEOR/BERTScore)."""
        subset = queries[:max_queries] if max_queries else queries

        predictions: list[str] = []
        references: list[str] = []
        rows: list[dict[str, Any]] = []

        for query_record in subset:
            result = self.pipeline.answer(query=query_record.query, top_k=top_k)
            predictions.append(result.answer)
            references.append(query_record.gold_answer or "")
            rows.append(
                {
                    "query_id": query_record.query_id,
                    "query": query_record.query,
                    "prediction": result.answer,
                    "reference": query_record.gold_answer or "",
                    "citations": result.citations,
                    "abstained": result.abstained,
                    "abstain_reason": result.abstain_reason,
                    "retrieval_latency_s": result.retrieval_latency_s,
                    "generation_latency_s": result.generation_latency_s,
                }
            )

        summary = compute_generation_metrics(predictions=predictions, references=references)
        return summary, pd.DataFrame(rows)

    def evaluate_judge(
        self,
        queries: list[QueryRecord],
        top_k: int = 6,
        max_queries: int | None = None,
    ) -> tuple[JudgeMetricSummary, pd.DataFrame]:
        """Compute LLM-as-a-judge quality scores."""
        subset = queries[:max_queries] if max_queries else queries

        rows: list[dict[str, Any]] = []
        for query_record in subset:
            result = self.pipeline.answer(query=query_record.query, top_k=top_k)
            judge = self.judge.score(
                query=query_record.query,
                context=result.context,
                answer=result.answer,
            )
            rows.append(
                {
                    "query_id": query_record.query_id,
                    "query": query_record.query,
                    "relevance": judge.relevance,
                    "correctness": judge.correctness,
                    "groundedness": judge.groundedness,
                    "completeness": judge.completeness,
                    "faithfulness": judge.faithfulness,
                    "rationale": judge.rationale,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return JudgeMetricSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0), df

        summary = JudgeMetricSummary(
            relevance=float(df["relevance"].mean()),
            correctness=float(df["correctness"].mean()),
            groundedness=float(df["groundedness"].mean()),
            completeness=float(df["completeness"].mean()),
            faithfulness=float(df["faithfulness"].mean()),
            num_examples=len(df),
        )
        return summary, df

    def evaluate_hallucination_reduction(
        self,
        queries: list[QueryRecord],
        max_queries: int = 200,
    ) -> pd.DataFrame:
        """Compare no-RAG vs RAG answers with judge-based groundedness proxy."""
        subset = queries[:max_queries]
        rows: list[dict[str, Any]] = []

        for query_record in subset:
            rag_result = self.pipeline.answer(query=query_record.query)
            no_rag = self.pipeline.generation_engine.generate_baseline_answer(query_record.query)

            rag_judge = self.judge.score(query_record.query, rag_result.context, rag_result.answer)
            no_rag_judge = self.judge.score(query_record.query, "No retrieval context provided.", no_rag["text"])

            rows.append(
                {
                    "query_id": query_record.query_id,
                    "query": query_record.query,
                    "rag_answer": rag_result.answer,
                    "no_rag_answer": no_rag["text"],
                    "rag_groundedness": rag_judge.groundedness,
                    "no_rag_groundedness": no_rag_judge.groundedness,
                    "rag_faithfulness": rag_judge.faithfulness,
                    "no_rag_faithfulness": no_rag_judge.faithfulness,
                    "groundedness_delta": rag_judge.groundedness - no_rag_judge.groundedness,
                }
            )

        return pd.DataFrame(rows)

    def run_full_evaluation(
        self,
        queries: list[QueryRecord],
        top_k: int,
        retrieval_limit: int,
        generation_limit: int,
        judge_limit: int,
        run_judge: bool = True,
    ) -> tuple[EvaluationBundle, dict[str, pd.DataFrame]]:
        """Run complete evaluation suite and return summaries + raw frames."""
        retrieval_summary, retrieval_rows, retrieval_df = self.evaluate_retrieval(
            queries=queries,
            top_k=top_k,
            max_queries=retrieval_limit,
        )
        generation_summary, generation_df = self.evaluate_generation(
            queries=queries,
            top_k=top_k,
            max_queries=generation_limit,
        )
        if run_judge:
            judge_summary, judge_df = self.evaluate_judge(
                queries=queries,
                top_k=top_k,
                max_queries=judge_limit,
            )
        else:
            judge_summary = JudgeMetricSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)
            judge_df = pd.DataFrame(
                columns=[
                    "query_id",
                    "query",
                    "relevance",
                    "correctness",
                    "groundedness",
                    "completeness",
                    "faithfulness",
                    "rationale",
                ]
            )

        bundle = EvaluationBundle(
            retrieval=retrieval_summary,
            generation=generation_summary,
            judge=judge_summary,
        )
        if "failure_bucket" in retrieval_df.columns:
            failure_summary = (
                retrieval_df["failure_bucket"].value_counts(dropna=False).rename_axis("failure_bucket").reset_index(name="count")
            )
        else:
            failure_summary = pd.DataFrame(columns=["failure_bucket", "count"])

        frames = {
            "retrieval": retrieval_df,
            "retrieval_per_query": pd.DataFrame([asdict(row) for row in retrieval_rows]),
            "retrieval_failure_summary": failure_summary,
            "generation": generation_df,
            "judge": judge_df,
        }
        return bundle, frames
