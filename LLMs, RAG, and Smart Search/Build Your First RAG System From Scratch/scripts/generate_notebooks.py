"""Generate tutorial notebooks for Project #13 (RAG Zero-to-Hero)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


COMMON_SETUP = """from pathlib import Path
import json
import pandas as pd

from rag_system import (
    RAGConfig,
    ProjectRunner,
    ChunkingStrategy,
)

cfg = RAGConfig(project_root=Path(".."), profile="balanced")
runner = ProjectRunner(cfg)
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def write_notebook(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    path = NB_DIR / name
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def teaching_block(title: str, definition: str, theory: str, motivation: str, architecture: str) -> str:
    return f"""## Definition
{definition}

## Theory
{theory}

## Motivation
{motivation}

## Architecture
{architecture}

## Real-world Examples
- Enterprise support assistant over product docs
- Compliance assistant over policy text
- Research assistant over paper abstracts

## Best Practices
- Measure retrieval and generation separately
- Track latency and grounding together
- Keep prompts strict about citations and uncertainty

## Common Mistakes
- Skipping retrieval diagnostics
- Using mismatched embedding models for index/query
- Reporting only one metric and claiming broad quality
"""


def build_notebooks() -> None:
    write_notebook(
        "01_rag_foundations.ipynb",
        [
            md("""# 01 - RAG Foundations (Zero to Hero)

This notebook introduces why Retrieval-Augmented Generation (RAG) exists and what problem it solves."""),
            md(
                teaching_block(
                    title="RAG Foundations",
                    definition="Retrieval-Augmented Generation combines retrieval from an external corpus with LLM generation.",
                    theory="A base LLM relies on parametric memory, while RAG supplements memory with non-parametric evidence at inference time.",
                    motivation="Hallucinations, stale knowledge, and context-window limits make pure generation unreliable for factual QA.",
                    architecture="Traditional LLM: query -> model -> answer. RAG: query -> retrieval -> grounded prompt -> answer with citations.",
                )
            ),
            code(COMMON_SETUP),
            code("""from rag_system.visualization import llm_vs_rag_diagram, architecture_diagram
from pathlib import Path

fig_dir = Path("../data/artifacts/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

llm_vs_rag_diagram(fig_dir / "01_llm_vs_rag.png")
architecture_diagram(fig_dir / "01_rag_flow.png")
fig_dir
"""),
            md("""## Visual Explanation
Open generated figures:
- `data/artifacts/figures/01_llm_vs_rag.png`
- `data/artifacts/figures/01_rag_flow.png`

These visuals explain why retrieval reduces hallucination risk."""),
            md("""## Code Explanation
The visualization module renders architecture diagrams that are reused in README and audit report artifacts."""),
        ],
    )

    write_notebook(
        "02_modern_rag_flow.ipynb",
        [
            md("""# 02 - How Modern RAG Works"""),
            md(
                teaching_block(
                    title="Modern RAG Flow",
                    definition="Modern RAG is a staged pipeline that separates retrieval quality from generation quality.",
                    theory="Dense embeddings map semantics to vectors; vector search returns nearest contexts; prompt assembly grounds response synthesis.",
                    motivation="Debugging is easier when each stage has independent metrics and diagnostics.",
                    architecture="User Query -> Embedding -> Vector Search -> Retrieved Context -> Prompt Construction -> Generation -> Answer.",
                )
            ),
            code(COMMON_SETUP),
            code("""bundle = runner.ingest_dataset()
len(bundle['documents']), len(bundle['queries']), bundle['summary']
"""),
            code("""chunking = runner.run_chunking(bundle['documents'][:260], strategy=ChunkingStrategy.RECURSIVE)
runner.index_chunks(chunking.chunks, reset=True)

sample_query = bundle['queries'][0].query
result = runner.pipeline.answer(sample_query, top_k=6)

{
    'query': sample_query,
    'answer_preview': result.answer[:280],
    'abstained': result.abstained,
    'num_retrieved': len(result.retrieved_chunks),
    'citations': result.citations,
}
"""),
            md("""## Code Explanation
This walkthrough runs a minimal retrieve-augment-generate loop and inspects retrieved chunks before trusting output text."""),
        ],
    )

    write_notebook(
        "03_dataset_eda.ipynb",
        [
            md("""# 03 - Dataset Exploration and Quality Audit (SQuAD v2)"""),
            md(
                teaching_block(
                    title="Dataset EDA",
                    definition="EDA (Exploratory Data Analysis) validates whether a dataset is suitable for retrieval and generation tasks.",
                    theory="RAG quality depends on corpus diversity, metadata quality, and clean train/eval split design.",
                    motivation="Weak datasets create misleading retrieval metrics and unstable generation quality.",
                    architecture="Data acquisition -> deduplication -> split policy -> quality checks -> artifact persistence.",
                )
            ),
            code(COMMON_SETUP),
            code("""bundle = runner.ingest_dataset()
summary = bundle['summary']
summary
"""),
            code("""from rag_system.data import documents_to_frame, queries_to_frame
from rag_system.visualization import plot_document_lengths
from pathlib import Path


doc_df = documents_to_frame(bundle['documents'])
query_df = queries_to_frame(bundle['queries'])

fig_dir = Path('../data/artifacts/figures')
fig_dir.mkdir(parents=True, exist_ok=True)
plot_document_lengths(doc_df, fig_dir / '03_doc_lengths.png')

{
    'documents_shape': doc_df.shape,
    'queries_shape': query_df.shape,
    'answerable_ratio': summary['answerable_ratio'],
    'query_split_counts': summary['query_split_counts'],
    'leakage_audit': bundle.get('leakage_audit', {}),
    'figure': str(fig_dir / '03_doc_lengths.png'),
}
"""),
            md("""## Findings and Interpretation
Use these outputs to justify dataset quality and confirm split-aware evaluation integrity."""),
        ],
    )

    write_notebook(
        "04_chunking_deep_dive.ipynb",
        [
            md("""# 04 - Chunking Deep Dive"""),
            md(
                teaching_block(
                    title="Chunking",
                    definition="Chunking splits long documents into retrieval units suitable for embedding and vector search.",
                    theory="Chunk size and overlap trade off between context completeness and retrieval precision.",
                    motivation="Poor chunking causes context fragmentation and low recall for answer-bearing passages.",
                    architecture="Document -> strategy-specific segmentation -> chunk metadata -> vector index.",
                )
            ),
            code(COMMON_SETUP),
            code("""bundle = runner.ingest_dataset()
docs = bundle['documents'][:80]
queries = bundle['queries'][:80]

strategies = [
    ChunkingStrategy.FIXED,
    ChunkingStrategy.RECURSIVE,
    ChunkingStrategy.SEMANTIC,
    ChunkingStrategy.PARENT_CHILD,
]

rows = []
for strategy in strategies:
    out = runner.run_chunking(docs, strategy=strategy)
    runner.index_chunks(out.chunks, reset=True)
    summary, _, _ = runner.evaluator.evaluate_retrieval(queries=queries, top_k=6, max_queries=len(queries))
    rows.append({
        'strategy': strategy.value,
        'num_chunks': len(out.chunks),
        'avg_chunk_chars': out.avg_chunk_length,
        'mrr': summary.mrr,
        'ndcg': summary.ndcg,
    })

pd.DataFrame(rows).sort_values('mrr', ascending=False)
"""),
            md("""## Tradeoffs
- Fixed: simple and fast, weakest boundary quality
- Recursive: strong practical default
- Semantic: better topical coherence, extra embedding compute
- Parent-child: stronger context recovery, higher storage/index cost"""),
        ],
    )

    write_notebook(
        "05_embeddings_deep_dive.ipynb",
        [
            md("""# 05 - Embeddings Deep Dive"""),
            md(
                teaching_block(
                    title="Embeddings",
                    definition="Embeddings are dense vectors representing semantic meaning of text.",
                    theory="Semantic similarity is measured by vector geometry, commonly cosine similarity for dense retrieval.",
                    motivation="Embedding quality directly controls retrieval recall and downstream answer grounding.",
                    architecture="Text -> embedding model -> vector -> nearest-neighbor search.",
                )
            ),
            code(COMMON_SETUP),
            code("""engine = runner.embedding_engine
texts = [
    'RAG combines retrieval and generation.',
    'Vector databases support semantic search.',
    'Bananas are yellow fruit.'
]
vectors = engine.embed_batch(texts)

sim_01 = engine.cosine_similarity(vectors[0], vectors[1])
sim_02 = engine.cosine_similarity(vectors[0], vectors[2])
(sim_01, sim_02, len(vectors[0]))
"""),
            code("""from rag_system.diagnostics import embedding_integrity_report
embedding_integrity_report(engine, texts=texts, batch_size=2)
"""),
            md("""## Visual Explanation
A similarity matrix can be generated to show that semantically related pairs cluster more closely than unrelated text."""),
        ],
    )

    write_notebook(
        "06_chromadb_retrieval.ipynb",
        [
            md("""# 06 - ChromaDB Retrieval"""),
            md(
                teaching_block(
                    title="Vector Database",
                    definition="A vector database stores embeddings and supports efficient nearest-neighbor retrieval with metadata filtering.",
                    theory="Approximate nearest-neighbor indexing enables scalable semantic retrieval.",
                    motivation="Without a vector DB, large-scale dense retrieval becomes too slow and operationally fragile.",
                    architecture="Chunks + embeddings -> Chroma collection -> query embedding -> top-k results.",
                )
            ),
            code(COMMON_SETUP),
            code("""bundle = runner.ingest_dataset()
chunking = runner.run_chunking(bundle['documents'][:650], strategy=ChunkingStrategy.RECURSIVE)
runner.index_chunks(chunking.chunks, reset=True)

query = 'What is the main argument in this passage?'
hits = runner.retrieval_engine.query(query=query, top_k=6, dedupe_by_doc=True)

pd.DataFrame([
    {
        'rank': i + 1,
        'doc_id': h.doc_id,
        'score': h.score,
        'title': h.metadata.get('title', 'unknown'),
    }
    for i, h in enumerate(hits)
])
"""),
            md("""## Comparison: ChromaDB vs FAISS vs Pinecone vs Weaviate vs Qdrant
- ChromaDB: local-first simplicity and beginner-friendly persistence
- FAISS: library-level ANN speed, fewer database features
- Pinecone: managed vector DB, stronger ops but paid
- Weaviate/Qdrant: production vector DB features with richer deployment patterns"""),
        ],
    )

    write_notebook(
        "07_prompt_and_generation.ipynb",
        [
            md("""# 07 - Prompt Engineering and Generation"""),
            md(
                teaching_block(
                    title="Prompt Engineering",
                    definition="Prompt engineering defines behavior contracts for grounded, citation-aware generation.",
                    theory="Instruction clarity and constraints reduce unsupported claims and improve consistency.",
                    motivation="A high-recall retriever still fails if prompts allow context-ignoring behavior.",
                    architecture="Retrieved context + policy instructions + question -> constrained generation.",
                )
            ),
            code(COMMON_SETUP),
            code("""from rag_system.prompts import PromptLibrary

query = 'Explain retrieval-augmented generation in simple terms.'
context = '[1] RAG retrieves relevant documents before generation.'

bad = PromptLibrary.bad_prompt_example(query, context)
good = PromptLibrary.good_prompt_example(query, context)
bad[:300], good[:300]
"""),
            code("""bundle = runner.ingest_dataset()
chunking = runner.run_chunking(bundle['documents'][:650], strategy=ChunkingStrategy.RECURSIVE)
runner.index_chunks(chunking.chunks, reset=True)

result = runner.pipeline.answer(bundle['queries'][10].query, top_k=6)
{
    'answer': result.answer[:420],
    'citations': result.citations,
    'abstained': result.abstained,
    'abstain_reason': result.abstain_reason,
}
"""),
            md("""## Code Explanation
The prompt library enforces context-only answering and explicit abstention when evidence is insufficient."""),
        ],
    )

    write_notebook(
        "08_evaluation.ipynb",
        [
            md("""# 08 - Evaluation: Retrieval, Generation, and Judge"""),
            md(
                teaching_block(
                    title="Evaluation",
                    definition="RAG evaluation measures retrieval quality and generation quality as separate but linked objectives.",
                    theory="Poor retrieval creates an upper bound on generation quality regardless of model fluency.",
                    motivation="Without robust metrics, improvements may be cosmetic rather than factual.",
                    architecture="Query set -> retrieval metrics -> generation metrics -> LLM judge -> hallucination analysis.",
                )
            ),
            code(COMMON_SETUP),
            code("""bundle = runner.ingest_dataset()
chunking = runner.run_chunking(bundle['documents'][:320], strategy=ChunkingStrategy.RECURSIVE)
runner.index_chunks(chunking.chunks, reset=True)

queries = bundle['queries'][:24]
summary, frames = runner.evaluator.run_full_evaluation(
    queries=queries,
    top_k=6,
    retrieval_limit=18,
    generation_limit=6,
    judge_limit=6,
)

{
    'retrieval': summary.retrieval,
    'generation': summary.generation,
    'judge': summary.judge,
}
"""),
            code("""hall = runner.evaluator.evaluate_hallucination_reduction(queries=queries[:8], max_queries=8)
hall[['groundedness_delta', 'rag_faithfulness', 'no_rag_faithfulness']].describe()
"""),
        ],
    )

    write_notebook(
        "09_advanced_retrieval_and_faiss_appendix.ipynb",
        [
            md("""# 09 - Advanced Retrieval and FAISS Appendix"""),
            md(
                teaching_block(
                    title="Advanced Retrieval",
                    definition="Advanced retrieval augments baseline similarity search with query expansion, multi-query fusion, and reranking.",
                    theory="Diverse query rewrites can improve recall; reranking can improve top-rank precision.",
                    motivation="Baseline retrievers often miss relevant context for ambiguous or under-specified questions.",
                    architecture="Original query -> rewrites -> multi-query retrieval -> hybrid rerank -> final top-k.",
                )
            ),
            code(COMMON_SETUP),
            code("""bundle = runner.ingest_dataset()
chunking = runner.run_chunking(bundle['documents'][:700], strategy=ChunkingStrategy.RECURSIVE)
runner.index_chunks(chunking.chunks, reset=True)

query = bundle['queries'][5].query
compare = runner.advanced_retriever.compare_base_vs_advanced(query=query, top_k=6)
compare
"""),
            md("""## FAISS Appendix
FAISS is included only for comparative study. ChromaDB remains the primary retrieval engine for this project."""),
        ],
    )

    write_notebook(
        "10_gradio_demo_and_production_notes.ipynb",
        [
            md("""# 10 - Gradio Demo and Production Considerations"""),
            md(
                teaching_block(
                    title="Production Considerations",
                    definition="Production RAG needs reliable latency, reproducibility, observability, and clear failure behavior.",
                    theory="System quality depends on end-to-end contracts, not only model quality.",
                    motivation="Portfolio-ready projects should show engineering discipline beyond a toy chatbot demo.",
                    architecture="Data + index lifecycle, runtime serving, monitoring loops, regression evaluation pipeline.",
                )
            ),
            code("""# Run from project root:
# uv run python app/gradio_app.py
"""),
            code("""from rag_system.config import RAGConfig
cfg = RAGConfig(profile='max_depth')
{
    'corpus_splits': cfg.corpus_splits,
    'eval_splits': cfg.eval_splits,
    'retrieval_eval_queries': cfg.retrieval_eval_queries,
    'generation_eval_queries': cfg.generation_eval_queries,
    'judge_eval_queries': cfg.judge_eval_queries,
}
"""),
            md("""## Failure Mode Checklist
- vague query handling
- out-of-domain handling
- low-relevance abstention
- retrieval transparency and diagnostics"""),
        ],
    )


if __name__ == "__main__":
    build_notebooks()
    print(f"Generated notebooks in {NB_DIR}")
