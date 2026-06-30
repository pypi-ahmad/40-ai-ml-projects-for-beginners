"""Generate tutorial notebook series for predictive keyboard project."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def notebook_01() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 01. Foundations: What Is a Predictive Keyboard?

A **predictive keyboard** estimates next likely word(s) from typed context.

## Definition
Predictive keyboards are probabilistic language systems that map context tokens to candidate next tokens.

## Why this matters
- Faster typing
- Lower effort on mobile devices
- Fewer spelling and grammar interruptions
- Better assistive communication tools

## Real-world systems
- Android keyboard suggestions
- iPhone QuickType
- Gmail Smart Compose
- Search autocomplete
- AI assistant response prefill

## Theory intuition
Given context tokens \\(x_{1:t}\\), model estimates \\(P(x_{t+1} | x_{1:t})\\). We return top-k highest probabilities.

## NLP intuition
Language has patterns (syntax, collocations, style, domain terms). Models learn these patterns from corpus statistics and gradients.

## Mathematical intuition
Prediction is argmax over vocabulary:
\\[
\\hat{w} = \arg\\max_{w \\in V} P(w\\mid \text{context})
\\]
Top-k mode returns ranked candidates instead of single argmax.

## What this mini-book builds
1. Data ingestion from Sherlock + WikiText subset
2. Preprocessing and tokenization comparisons
3. Vocabulary and sequence engineering
4. N-gram baselines
5. Neural LMs (LSTM/GRU/CNN-LSTM/Transformer)
6. Embeddings + explainability
7. Benchmarking + keyboard engine + Streamlit deployment
"""
        ),
        md(
            """
## Sequence Modeling Concepts

### Context windows
- 3 words: short memory, fast inference
- 5 words: practical baseline
- 10 words: stronger semantics
- 20 words: richer context, higher compute

### Language model families
- **Count-based**: unigram, bigram, trigram
- **Neural recurrent**: LSTM, GRU, BiLSTM, stacked LSTM
- **Hybrid**: CNN + LSTM
- **Attention-based**: Transformer

### Why LSTMs became popular
RNN vanishing gradients made long dependencies hard. LSTM gating stabilized memory flow.

### Why Transformers replaced them
Self-attention gives global token interactions, parallel training, and better scaling.
"""
        ),
        code(
            """
from pathlib import Path
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
print("Project root:", PROJECT_ROOT)
"""
        ),
        code(
            """
from utils.tokenization import RegexTokenizerBackend

text = "I would like to learn machine learning with practical projects"
tokens = RegexTokenizerBackend().tokenize(text.lower())
print("Tokens:", tokens)

for context_len in [3, 5]:
    context = tokens[-context_len:]
    print(f"Context-{context_len}:", context)
"""
        ),
        md(
            """
## Interpretation
This notebook establishes full conceptual map. Next notebook starts real data profiling and corpus statistics.
"""
        ),
    ]

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def notebook_02() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 02. Dataset Profiling and Corpus Diagnostics

## Goal
Understand corpus structure before training: documents, sentences, words, vocabulary, and distribution tails.

## Theory
Poor data understanding causes poor model behavior. Profiling catches imbalance, noise, sparsity, and long-tail effects.
"""
        ),
        code(
            """
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from utils.data import corpus_statistics, prepare_combined_corpus
from utils.reproducibility import set_global_seed

set_global_seed(42)
PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()

bundle = prepare_combined_corpus(
    project_root=PROJECT_ROOT,
    include_wikitext=True,
    wikitext_train_tokens=30_000,
    wikitext_val_tokens=5_000,
    wikitext_test_tokens=5_000,
)

stats = {
    "train": corpus_statistics(bundle.train_text),
    "val": corpus_statistics(bundle.val_text),
    "test": corpus_statistics(bundle.test_text),
}

pd.DataFrame(stats).T
"""
        ),
        code(
            """
# Save profile for downstream notebooks.
out_dir = PROJECT_ROOT / "outputs" / "results"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "dataset_profile_notebook.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print("Saved profile:", out_dir / "dataset_profile_notebook.json")
"""
        ),
        code(
            """
# Document length inspection.
def split_paragraphs(text: str):
    return [p.strip() for p in text.split("\\n\\n") if p.strip()]

train_docs = split_paragraphs(bundle.train_text)
lengths = [len(doc.split()) for doc in train_docs]

plt.figure(figsize=(10, 4))
plt.hist(lengths, bins=40, color="#3b82f6", alpha=0.8)
plt.title("Train document length distribution")
plt.xlabel("Tokens per document")
plt.ylabel("Count")
plt.show()
"""
        ),
        md(
            """
## Interpretation
- We now have real multi-source corpus.
- Vocabulary will include literary + encyclopedic styles.
- Next notebook compares tokenizer behavior before vocabulary engineering.
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def notebook_03() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 03. NLP Preprocessing and Tokenization Comparison

## Definitions
- **Tokenization**: split text into model units
- **Normalization**: canonicalize case/spacing/accent
- **Stop words**: high-frequency function words (optional filtering)

## Why compare tokenizers?
Different tokenizers change vocabulary size, OOV rate, sequence length, and model cost.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd

from utils.data import prepare_combined_corpus
from utils.tokenization import (
    HuggingFaceBPETokenizerBackend,
    NLTKTokenizerBackend,
    RegexTokenizerBackend,
    SpacyTokenizerBackend,
    compare_tokenizer_outputs,
    normalize_text,
)

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
bundle = prepare_combined_corpus(
    project_root=PROJECT_ROOT,
    include_wikitext=True,
    wikitext_train_tokens=80_000,
    wikitext_val_tokens=10_000,
    wikitext_test_tokens=10_000,
)

sample_text = normalize_text(bundle.train_text[:200_000])

hf_backend = HuggingFaceBPETokenizerBackend(vocab_size=3000, min_frequency=2)
hf_backend.fit([sample_text])

summary = compare_tokenizer_outputs(
    text=sample_text,
    backends=[
        RegexTokenizerBackend(),
        NLTKTokenizerBackend(),
        SpacyTokenizerBackend(),
        hf_backend,
    ],
)

summary_df = pd.DataFrame(summary)
summary_df
"""
        ),
        code(
            """
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.bar(summary_df["backend"], summary_df["token_count"], color="#0ea5e9")
plt.title("Token count by tokenizer backend")
plt.ylabel("Token count")
plt.show()
"""
        ),
        code(
            """
# Word frequency analysis with NLTK tokenizer (baseline for later notebooks).
from collections import Counter

tokens = NLTKTokenizerBackend().tokenize(sample_text)
freq = Counter(tokens)

most_common_df = pd.DataFrame(freq.most_common(25), columns=["token", "count"])
most_common_df.head(10)
"""
        ),
        md(
            """
## Interpretation
- Tokenizer choice changes token granularity and throughput.
- We use NLTK word-level path for baseline model comparability.
- Subword BPE backend remains available for extension.
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def notebook_04() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 04. Vocabulary Engineering and Sequence Construction

## Why words must become numbers
Neural models operate on tensors, not strings. Vocabulary maps text tokens to integer IDs.

## Definitions
- `word2idx`: token -> integer ID
- `idx2word`: integer ID -> token
- `<unk>`: out-of-vocabulary fallback
- `<pad>`: sequence padding token
- `<bos>/<eos>`: sequence boundaries
"""
        ),
        code(
            """
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data import prepare_combined_corpus
from utils.tokenization import NLTKTokenizerBackend, normalize_text
from utils.vocabulary import Vocabulary
from utils.sequence_builder import build_context_target_pairs

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
bundle = prepare_combined_corpus(
    project_root=PROJECT_ROOT,
    include_wikitext=True,
    wikitext_train_tokens=120_000,
    wikitext_val_tokens=20_000,
    wikitext_test_tokens=20_000,
)

tokens = NLTKTokenizerBackend().tokenize(normalize_text(bundle.train_text))
"""
        ),
        code(
            """
vocab_sizes = {}
for min_freq in [1, 2, 5, 10]:
    vocab = Vocabulary(min_freq=min_freq, max_size=20_000)
    vocab.build([tokens])
    vocab_sizes[min_freq] = len(vocab)

pd.DataFrame(
    [{"min_freq": k, "vocab_size": v} for k, v in vocab_sizes.items()]
).sort_values("min_freq")
"""
        ),
        code(
            """
vocab = Vocabulary(min_freq=2, max_size=20_000)
vocab.build([tokens])
ids = vocab.encode_with_special(tokens)

print("Vocab size:", len(vocab))
print("Sample word2idx:", list(vocab.word2idx.items())[:10])
print("Sample idx2word:", [(i, vocab.idx2word[i]) for i in range(10)])
"""
        ),
        code(
            """
context_options = [3, 5, 10, 20]
counts = []
for context_len in context_options:
    pairs = build_context_target_pairs(ids, context_len=context_len)
    counts.append({"context_len": context_len, "num_pairs": len(pairs)})

counts_df = pd.DataFrame(counts)
counts_df
"""
        ),
        code(
            """
plt.figure(figsize=(8, 4))
plt.plot(counts_df["context_len"], counts_df["num_pairs"], marker="o")
plt.title("Sequence count vs context window")
plt.xlabel("Context length")
plt.ylabel("Number of (context,target) pairs")
plt.show()
"""
        ),
        code(
            """
# Save vocabulary for training + app.
out_path = PROJECT_ROOT / "outputs" / "vocab_notebook.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
vocab.save(out_path)
print("Saved:", out_path)
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def notebook_05() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 05. Baseline Models: Most-Frequent, Unigram, Bigram, Trigram

## Why baselines matter
A neural model is only useful if it beats strong classical baselines.

## Theory
- **Most Frequent**: always predicts single most common token.
- **Unigram**: context-free token probabilities.
- **Bigram**: uses one previous token.
- **Trigram**: uses two previous tokens.
"""
        ),
        code(
            """
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

from utils.data import prepare_combined_corpus
from utils.evaluation import dataloader_metrics
from utils.models import MostFrequentWordModel, UnigramModel, BigramModel, TrigramModel
from utils.sequence_builder import build_dataloaders_from_ids
from utils.tokenization import NLTKTokenizerBackend, normalize_text
from utils.vocabulary import Vocabulary

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()

bundle = prepare_combined_corpus(
    project_root=PROJECT_ROOT,
    include_wikitext=True,
    wikitext_train_tokens=120_000,
    wikitext_val_tokens=20_000,
    wikitext_test_tokens=20_000,
)

tokenizer = NLTKTokenizerBackend()
train_tokens = tokenizer.tokenize(normalize_text(bundle.train_text))
test_tokens = tokenizer.tokenize(normalize_text(bundle.test_text))

vocab = Vocabulary(min_freq=2, max_size=20_000)
vocab.build([train_tokens])

train_ids = vocab.encode_with_special(train_tokens)
test_ids = vocab.encode_with_special(test_tokens)

train_loader = build_dataloaders_from_ids(train_ids, context_len=5, batch_size=128, val_ratio=0, test_ratio=0)["train"]
test_loader = build_dataloaders_from_ids(test_ids, context_len=5, batch_size=128, val_ratio=0, test_ratio=0)["train"]
"""
        ),
        code(
            """
from itertools import islice

criterion = nn.NLLLoss()
models = {
    "MostFrequent": MostFrequentWordModel(vocab_size=len(vocab)),
    "Unigram": UnigramModel(vocab_size=len(vocab), smoothing=1.0),
    "Bigram": BigramModel(vocab_size=len(vocab), smoothing=1.0),
    "Trigram": TrigramModel(vocab_size=len(vocab), smoothing=1.0),
}

rows = []
for name, model in models.items():
    model.fit(train_ids)
    small_eval_loader = islice(test_loader, 20)
    metrics = dataloader_metrics(model, small_eval_loader, criterion, device="cpu")
    rows.append({"model": name, **metrics})

baseline_df = pd.DataFrame(rows).sort_values("top5_accuracy", ascending=False)
baseline_df
"""
        ),
        code(
            """
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.bar(baseline_df["model"], baseline_df["perplexity"], color="#f97316")
plt.title("Baseline perplexity comparison")
plt.ylabel("Perplexity")
plt.show()
"""
        ),
        md(
            """
## Interpretation
Higher-order n-grams usually improve next-word quality but suffer sparsity on rare contexts.
Neural models next: better generalization under sparse patterns.
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def notebook_06() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 06. Neural Language Models and Training Pipeline

## Models covered
1. Vanilla LSTM
2. Stacked LSTM
3. BiLSTM
4. GRU
5. CNN + LSTM
6. Transformer LM

## Training workflow
- Train/val/test separation
- Early stopping
- Learning-rate scheduling
- Gradient clipping
- Checkpointing
- Cross-entropy and perplexity monitoring
"""
        ),
        code(
            """
from pathlib import Path
import subprocess
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

leaderboard_path = RESULTS_DIR / "leaderboard_quick_cpu.csv"
if not leaderboard_path.exists():
    cmd = [
        "uv", "run", "python", "scripts/train_and_benchmark.py",
        "--profile", "quick",
        "--include-wikitext",
        "--prefer-gpu",
        "--wikitext-train-tokens", "60000",
        "--wikitext-val-tokens", "10000",
        "--wikitext-test-tokens", "10000",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

leaderboard = pd.read_csv(leaderboard_path)
leaderboard
"""
        ),
        code(
            """
import matplotlib.pyplot as plt

neural = leaderboard[leaderboard["family"] == "neural"].copy()
neural = neural.sort_values("top5_accuracy", ascending=False)

plt.figure(figsize=(10, 4))
plt.bar(neural["model"], neural["top5_accuracy"], color="#22c55e")
plt.title("Neural model Top-5 accuracy")
plt.ylabel("Top-5 accuracy")
plt.xticks(rotation=30)
plt.show()
"""
        ),
        md(
            """
## Interpretation
Evaluate trade-off triangle:
- Accuracy/Perplexity
- Training time
- Inference latency and memory

Production model choice depends on quality target and latency budget.
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def notebook_07() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 07. Embeddings and Explainable NLP

## Embedding theory
Word embeddings map tokens into dense vectors where semantic similarity becomes geometric proximity.

## Methods implemented
- Trainable neural embeddings (inside language models)
- Word2Vec (gensim)
- FastText (gensim, subword-aware)

## Explainability in this notebook
- Embedding neighborhood analysis
- PCA / t-SNE / UMAP projections
- Transformer attention heatmaps
"""
        ),
        code(
            """
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data import prepare_combined_corpus
from utils.embeddings import embedding_dataframe, nearest_neighbors, train_fasttext, train_word2vec
from utils.tokenization import NLTKTokenizerBackend, normalize_text

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()

bundle = prepare_combined_corpus(
    project_root=PROJECT_ROOT,
    include_wikitext=True,
    wikitext_train_tokens=120_000,
    wikitext_val_tokens=20_000,
    wikitext_test_tokens=20_000,
)

sentences = normalize_text(bundle.train_text).split(".")
tokenized = [NLTKTokenizerBackend().tokenize(sent) for sent in sentences]
tokenized = [s for s in tokenized if len(s) > 3][:8000]

w2v = train_word2vec(tokenized, vector_size=100, epochs=5, min_count=2)
ft = train_fasttext(tokenized, vector_size=100, epochs=5, min_count=2)

print("Word2Vec neighbors for 'time':", nearest_neighbors(w2v.wv, "time", topn=5))
print("FastText neighbors for 'time':", nearest_neighbors(ft.wv, "time", topn=5))
"""
        ),
        code(
            """
pca_df = embedding_dataframe(w2v.wv, method="pca", max_words=250)

plt.figure(figsize=(8, 6))
plt.scatter(pca_df["x"], pca_df["y"], s=8, alpha=0.7)
for _, row in pca_df.head(40).iterrows():
    plt.text(row["x"], row["y"], row["word"], fontsize=7)
plt.title("Word2Vec PCA projection")
plt.show()
"""
        ),
        code(
            """
tsne_df = embedding_dataframe(w2v.wv, method="tsne", max_words=250)
plt.figure(figsize=(8, 6))
plt.scatter(tsne_df["x"], tsne_df["y"], s=8, alpha=0.7, color="#a855f7")
plt.title("Word2Vec t-SNE projection")
plt.show()
"""
        ),
        code(
            """
umap_df = embedding_dataframe(w2v.wv, method="umap", max_words=250)
plt.figure(figsize=(8, 6))
plt.scatter(umap_df["x"], umap_df["y"], s=8, alpha=0.7, color="#0ea5e9")
plt.title("Word2Vec UMAP projection")
plt.show()
"""
        ),
        md(
            """
## Interpretation
FastText tends to perform better on rare or morphologically related words due to subword modeling.
Embedding projections are approximate but useful for cluster-level interpretation.
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def notebook_08() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 08. Benchmarking, Keyboard Engine, Advanced Decoding, and Deployment

## Objectives
- Build production-style inference API
- Compare decoding strategies (beam, temperature, top-k, top-p)
- Simulate smartphone keyboard suggestion bar
- Prepare Streamlit deployment workflow
"""
        ),
        code(
            """
from pathlib import Path
import json
import pandas as pd
import torch

from utils.keyboard_engine import PredictiveKeyboardEngine
from utils.models import LSTM_LM, StackedLSTM_LM, BiLSTM_LM, GRU_LM, CNN_LSTM_LM, TransformerLM
from utils.vocabulary import Vocabulary

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
RESULTS = PROJECT_ROOT / "outputs" / "results"

leaderboard_path = sorted(RESULTS.glob("leaderboard_*.csv"))[-1]
registry_path = sorted(RESULTS.glob("model_registry_*.json"))[-1]

leaderboard = pd.read_csv(leaderboard_path)
registry = json.loads(registry_path.read_text(encoding="utf-8"))

leaderboard.head(10)
"""
        ),
        code(
            """
MODEL_BUILDERS = {
    "LSTM": lambda c: LSTM_LM(vocab_size=int(c["vocab_size"]), embedding_dim=int(c["embedding_dim"]), hidden_dim=int(c["hidden_dim"]), num_layers=1),
    "StackedLSTM": lambda c: StackedLSTM_LM(vocab_size=int(c["vocab_size"]), embedding_dim=int(c["embedding_dim"]), hidden_dim=int(c["hidden_dim"]), num_layers=2),
    "BiLSTM": lambda c: BiLSTM_LM(vocab_size=int(c["vocab_size"]), embedding_dim=int(c["embedding_dim"]), hidden_dim=max(int(c["hidden_dim"])//2, 64), num_layers=2),
    "GRU": lambda c: GRU_LM(vocab_size=int(c["vocab_size"]), embedding_dim=int(c["embedding_dim"]), hidden_dim=int(c["hidden_dim"]), num_layers=2),
    "CNN_LSTM": lambda c: CNN_LSTM_LM(vocab_size=int(c["vocab_size"]), embedding_dim=int(c["embedding_dim"]), hidden_dim=int(c["hidden_dim"]), num_filters=max(int(c["embedding_dim"])//2, 64)),
    "Transformer": lambda c: TransformerLM(vocab_size=int(c["vocab_size"]), embedding_dim=int(c["embedding_dim"]), hidden_dim=int(c["hidden_dim"]), nhead=int(c["transformer_heads"]), num_layers=int(c["transformer_layers"])),
}

best_model_name = str(leaderboard.iloc[0]["model"])
cfg = registry[best_model_name]
vocab_path = Path(str(cfg.get("vocab_path", PROJECT_ROOT / "outputs" / "vocab.json")))
vocab = Vocabulary.load(vocab_path)

model = MODEL_BUILDERS[best_model_name](cfg)
checkpoint = torch.load(Path(cfg["checkpoint_path"]), map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])

engine = PredictiveKeyboardEngine(model=model, vocabulary=vocab, context_length=int(cfg["context_len"]), device="cpu")
print("Loaded model:", best_model_name)
"""
        ),
        code(
            """
prompt = "I would like to"

print("Top-3 suggestions:")
for row in engine.predict(prompt, top_k=3):
    print(row)

print("\\nTop-5 suggestions:")
for row in engine.predict(prompt, top_k=5):
    print(row)

print("\\nBeam suggestions:")
for row in engine.predict(prompt, top_k=5, strategy="beam"):
    print(row)
"""
        ),
        code(
            """
simulation = engine.simulate_keyboard_step("the meaning of", suggestion_count=3)
simulation
"""
        ),
        md(
            """
## Deployment
Run local app:

```bash
uv run streamlit run app/streamlit_app.py
```

App exposes top-3/top-5 suggestions, probabilities, autocomplete, and strategy controls.
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_dir = project_root / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "01-foundations-predictive-keyboard.ipynb": notebook_01(),
        "02-dataset-profiling.ipynb": notebook_02(),
        "03-nlp-preprocessing-tokenization.ipynb": notebook_03(),
        "04-vocabulary-sequence-engineering.ipynb": notebook_04(),
        "05-baseline-language-models.ipynb": notebook_05(),
        "06-neural-language-modeling.ipynb": notebook_06(),
        "07-embeddings-and-explainability.ipynb": notebook_07(),
        "08-benchmarking-engine-deployment.ipynb": notebook_08(),
    }

    for name, notebook in notebooks.items():
        path = notebook_dir / name
        nbf.write(notebook, path)
        print("Wrote", path)


if __name__ == "__main__":
    main()
