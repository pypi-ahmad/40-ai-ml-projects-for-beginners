"""Retrieval and generation evaluation utilities."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from local_rag.rag import RAGPipeline
from local_rag.retriever import RetrievalStrategy, Retriever
from local_rag.types import EvalExample, GenerationMetrics, ResponseMetrics, RetrievalMetrics
from local_rag.utils import write_jsonl


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@K."""

    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@K."""

    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for item in top if item in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute reciprocal rank for first relevant hit."""

    for idx, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute NDCG@K with binary relevance."""

    dcg = 0.0
    for idx, item in enumerate(retrieved_ids[:k], start=1):
        if item in relevant_ids:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token.strip()]


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _safe_bleu(reference: str, prediction: str) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        ref_tokens = _tokenize(reference)
        pred_tokens = _tokenize(prediction)
        if not ref_tokens or not pred_tokens:
            return 0.0
        return float(
            sentence_bleu(
                [ref_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method1,
            )
        )
    except Exception:
        return 0.0


def _safe_meteor(reference: str, prediction: str) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score

        ref_tokens = _tokenize(reference)
        pred_tokens = _tokenize(prediction)
        if not ref_tokens or not pred_tokens:
            return 0.0
        return float(meteor_score([ref_tokens], pred_tokens))
    except Exception:
        return 0.0


def _safe_rouge(reference: str, prediction: str) -> tuple[float, float]:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        result = scorer.score(reference, prediction)
        rouge1 = float(result["rouge1"].fmeasure)
        rouge_l = float(result["rougeL"].fmeasure)
        return rouge1, rouge_l
    except Exception:
        return 0.0, 0.0


def _safe_bertscore(rows: list[tuple[str, str]]) -> float:
    try:
        from bert_score import score as bert_score

        candidates = [candidate for candidate, _ in rows]
        references = [reference for _, reference in rows]
        if not candidates:
            return 0.0
        # Force CPU scoring to avoid GPU OOM when Ollama models occupy VRAM.
        _, _, f1 = bert_score(
            candidates,
            references,
            lang="en",
            verbose=False,
            device="cpu",
        )
        return float(f1.mean().item())
    except Exception:
        return 0.0


class RetrievalEvaluator:
    """Evaluate retriever quality against labeled examples."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def evaluate(
        self,
        examples: Iterable[EvalExample],
        ks: tuple[int, ...] = (1, 3, 5, 10),
        strategies: tuple[RetrievalStrategy, ...] = ("vector", "keyword", "hybrid"),
    ) -> list[RetrievalMetrics]:
        """Compute aggregate metrics for candidate K values and strategies."""

        rows = list(examples)
        if not rows:
            return []

        metrics: list[RetrievalMetrics] = []
        for strategy in strategies:
            for k in ks:
                p_vals: list[float] = []
                r_vals: list[float] = []
                rr_vals: list[float] = []
                ndcg_vals: list[float] = []
                latencies: list[float] = []

                for row in rows:
                    started = time.perf_counter()
                    retrieved, _ = self.retriever.retrieve(
                        row.query,
                        top_k=k,
                        strategy=strategy,
                    )
                    latencies.append((time.perf_counter() - started) * 1000)

                    ids = [hit.doc_id for hit in retrieved]
                    relevant = set(row.relevant_doc_ids)

                    p_vals.append(precision_at_k(ids, relevant, k))
                    r_vals.append(recall_at_k(ids, relevant, k))
                    rr_vals.append(reciprocal_rank(ids, relevant))
                    ndcg_vals.append(ndcg_at_k(ids, relevant, k))

                metrics.append(
                    RetrievalMetrics(
                        k=k,
                        strategy=strategy,
                        precision_at_k=statistics.mean(p_vals),
                        recall_at_k=statistics.mean(r_vals),
                        mrr=statistics.mean(rr_vals),
                        ndcg=statistics.mean(ndcg_vals),
                        avg_retrieval_latency_ms=statistics.mean(latencies),
                    )
                )

        return metrics


class ResponseEvaluator:
    """Evaluate generation latency/length/citations over query set."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def evaluate(
        self,
        queries: Iterable[str],
        *,
        top_k: int,
        strategy: RetrievalStrategy = "hybrid",
    ) -> ResponseMetrics:
        """Compute response-level summary metrics."""

        rows = [query.strip() for query in queries if query.strip()]
        if not rows:
            return ResponseMetrics(
                avg_generation_latency_ms=0.0,
                avg_answer_length=0.0,
                avg_citation_count=0.0,
            )

        generation_latencies: list[float] = []
        answer_lengths: list[float] = []
        citation_counts: list[float] = []

        for query in rows:
            response = self.pipeline.ask(query, top_k=top_k, strategy=strategy)
            generation_latencies.append(response.timings.generation_ms)
            answer_lengths.append(float(len(response.answer.split())))
            citation_counts.append(float(len(response.citations)))

        return ResponseMetrics(
            avg_generation_latency_ms=statistics.mean(generation_latencies),
            avg_answer_length=statistics.mean(answer_lengths),
            avg_citation_count=statistics.mean(citation_counts),
        )


class GenerationEvaluator:
    """Evaluate generation quality metrics including groundedness proxies."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def evaluate(
        self,
        examples: Iterable[EvalExample],
        *,
        top_k: int,
        strategy: RetrievalStrategy = "hybrid",
    ) -> GenerationMetrics:
        """Compute generation metrics for labeled examples."""

        rows = [row for row in examples if row.query.strip() and row.answer]
        if not rows:
            return GenerationMetrics(
                strategy=strategy,
                avg_generation_latency_ms=0.0,
                avg_answer_length=0.0,
                avg_citation_count=0.0,
                bleu=0.0,
                rouge_l=0.0,
                rouge_1=0.0,
                meteor=0.0,
                bertscore_f1=0.0,
                groundedness=0.0,
                faithfulness=0.0,
                context_precision=0.0,
                context_recall=0.0,
                answer_relevancy=0.0,
                hallucination_rate=0.0,
            )

        generation_latencies: list[float] = []
        answer_lengths: list[float] = []
        citation_counts: list[float] = []
        bleu_scores: list[float] = []
        rouge1_scores: list[float] = []
        rougel_scores: list[float] = []
        meteor_scores: list[float] = []
        groundedness_scores: list[float] = []
        faithfulness_scores: list[float] = []
        context_precision_scores: list[float] = []
        context_recall_scores: list[float] = []
        answer_relevancy_scores: list[float] = []
        hallucination_flags: list[float] = []
        bert_rows: list[tuple[str, str]] = []

        for row in rows:
            response = self.pipeline.ask(row.query, top_k=top_k, strategy=strategy)
            reference = row.answer or ""
            answer = response.answer
            context = "\n".join(hit.text for hit in response.retrieved)

            generation_latencies.append(response.timings.generation_ms)
            answer_lengths.append(float(len(answer.split())))
            citation_counts.append(float(len(response.citations)))

            bleu_scores.append(_safe_bleu(reference, answer))
            rouge1, rouge_l = _safe_rouge(reference, answer)
            rouge1_scores.append(rouge1)
            rougel_scores.append(rouge_l)
            meteor_scores.append(_safe_meteor(reference, answer))
            bert_rows.append((answer, reference))

            groundedness = _overlap_ratio(answer, context)
            context_precision = _overlap_ratio(answer, context)
            context_recall = _overlap_ratio(reference, context)
            answer_relevancy = _overlap_ratio(answer, f"{row.query} {reference}")

            faithful = 1.0 if groundedness >= 0.5 else 0.0
            hallucinated = 1.0 if groundedness < 0.25 else 0.0

            groundedness_scores.append(groundedness)
            faithfulness_scores.append(faithful)
            context_precision_scores.append(context_precision)
            context_recall_scores.append(context_recall)
            answer_relevancy_scores.append(answer_relevancy)
            hallucination_flags.append(hallucinated)

        return GenerationMetrics(
            strategy=strategy,
            avg_generation_latency_ms=statistics.mean(generation_latencies),
            avg_answer_length=statistics.mean(answer_lengths),
            avg_citation_count=statistics.mean(citation_counts),
            bleu=statistics.mean(bleu_scores),
            rouge_l=statistics.mean(rougel_scores),
            rouge_1=statistics.mean(rouge1_scores),
            meteor=statistics.mean(meteor_scores),
            bertscore_f1=_safe_bertscore(bert_rows),
            groundedness=statistics.mean(groundedness_scores),
            faithfulness=statistics.mean(faithfulness_scores),
            context_precision=statistics.mean(context_precision_scores),
            context_recall=statistics.mean(context_recall_scores),
            answer_relevancy=statistics.mean(answer_relevancy_scores),
            hallucination_rate=statistics.mean(hallucination_flags),
        )


def dump_retrieval_metrics(path: Path, metrics: list[RetrievalMetrics]) -> None:
    """Persist retrieval metrics to JSONL."""

    write_jsonl(path, [asdict(metric) for metric in metrics])


def dump_generation_metrics(path: Path, metric: GenerationMetrics) -> None:
    """Persist generation metrics to JSON."""

    write_jsonl(path, [asdict(metric)])
