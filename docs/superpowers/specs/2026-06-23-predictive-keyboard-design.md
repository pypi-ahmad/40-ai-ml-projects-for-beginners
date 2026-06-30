# Predictive Keyboard Model — Design Document

## Overview

Build a production-quality Predictive Keyboard System — learns language patterns from Sherlock Holmes text corpus and predicts next most likely words. Real NLP pipeline, real deep learning, real evaluation, real inference engine. Portfolio-grade project.

## Dataset

- Source: `3dd01-book.zip` → `sherlock-holm.es_stories_plain-text_advs.txt`
- "The Adventures of Sherlock Holmes" by Arthur Conan Doyle
- ~104K words, 12K lines, 610KB
- Single literary text corpus

## Stack

| Component | Choice |
|-----------|--------|
| Runtime | uv, Python 3.12.10, local venv |
| Deep learning | PyTorch |
| NLP preprocessing | NLTK, spaCy, HuggingFace Tokenizers |
| Classical baselines | scikit-learn, NLTK (n-gram models) |
| Visualization | matplotlib, seaborn, plotly, wordcloud |
| Embedding viz | PCA, t-SNE, UMAP (from sklearn/umap-learn) |
| Deployment | Streamlit |
| Project mgmt | uv-managed venv inside project folder |

## Notebook Structure (6 notebooks)

| # | Notebook | Content |
|---|----------|---------|
| 01 | `01-data-exploration.ipynb` | Dataset extraction, profiling, EDA, Zipf's law, word clouds, n-gram analysis |
| 02 | `02-text-preprocessing.ipynb` | Tokenization (NLTK/spaCy/HF comparison), vocabulary engineering, word2idx/idx2word, sequence construction (3/5/10/20-word contexts) |
| 03 | `03-baseline-models.ipynb` | Most-frequent-word, unigram/bigram/trigram baselines, accuracy benchmarks |
| 04 | `04-neural-language-models.ipynb` | Embedding layer, Vanilla LSTM, Stacked LSTM, BiLSTM, GRU, CNN+LSTM, Transformer LM. Full training pipeline (early stopping, LR scheduling, checkpointing, gradient clipping). Embedding visualization (PCA/t-SNE/UMAP) |
| 05 | `05-evaluation-engine.ipynb` | All metrics (accuracy@1/3/5, perplexity, cross-entropy, MRR, recall@3/5). Beam search, temperature/top-k/top-p sampling. Model comparison tables. Training/validation curves |
| 06 | `06-predictive-keyboard-demo.ipynb` | Keyboard inference engine, smartphone simulation, attention visualization, prediction confidence plots, final end-to-end demo |

## Shared Python Modules

| Module | Purpose |
|--------|---------|
| `utils/preprocessing.py` | Text cleaning, normalization, tokenization |
| `utils/vocabulary.py` | Vocabulary construction, word2idx/idx2word, embedding layer builder |
| `utils/sequence_builder.py` | Context window → target pair generation for varying context lengths |
| `utils/trainer.py` | Training loop, early stopping, LR scheduler, checkpointing, gradient clipping |
| `utils/evaluation.py` | accuracy@k, perplexity, MRR, recall@k, cross-entropy |
| `utils/models/` | Each model as separate class (`lstm.py`, `stacked_lstm.py`, `bilstm.py`, `gru.py`, `cnn_lstm.py`, `transformer_lm.py`) |
| `utils/sampling.py` | Greedy, beam search, temperature, top-k, top-p decoding |

## Streamlit Application

- `app/app.py`
- Text input field
- Real-time top-5 predictions with confidence bars
- Clean, minimal UI

## Model Architecture Summary

| Model | Parameters | Notes |
|-------|-----------|-------|
| Most Frequent Word | 0 | Baseline |
| Unigram | vocab_size | Baseline |
| Bigram | vocab_size² | Sparse |
| Trigram | vocab_size³ | Very sparse |
| Vanilla LSTM | ~500K–1M | Single layer |
| Stacked LSTM | ~1M–3M | 2–3 layers |
| BiLSTM | ~1M–2M | Bidirectional context |
| GRU | ~500K–1M | Gated alternative |
| CNN+LSTM | ~800K–2M | N-gram feature extractor + LSTM |
| Transformer | ~1M–5M | Self-attention, positional encoding |

## Evaluation Metrics

- Accuracy@1, Accuracy@3, Accuracy@5
- Cross-Entropy Loss
- Perplexity (exp(loss))
- Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)

## Decoding Strategies

- Greedy (argmax)
- Beam Search (width 3/5)
- Temperature Sampling
- Top-k Sampling
- Top-p (nucleus) Sampling

## Project Files

```
Building a Predictive Keyboard Model/
├── 3dd01-book.zip
├── README.md
├── pyproject.toml
├── 01-data-exploration.ipynb
├── 02-text-preprocessing.ipynb
├── 03-baseline-models.ipynb
├── 04-neural-language-models.ipynb
├── 05-evaluation-engine.ipynb
├── 06-predictive-keyboard-demo.ipynb
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── vocabulary.py
│   ├── sequence_builder.py
│   ├── trainer.py
│   ├── evaluation.py
│   ├── sampling.py
│   └── models/
│       ├── __init__.py
│       ├── lstm.py
│       ├── stacked_lstm.py
│       ├── bilstm.py
│       ├── gru.py
│       ├── cnn_lstm.py
│       └── transformer_lm.py
├── app/
│   └── app.py
├── outputs/
│   └── (figures, checkpoints, results)
└── images/
    └── (README figures)
```

## Educational Approach

Every notebook section follows: Definition → Theory → Math Intuition → NLP Intuition → Real-world examples → Visual explanation → Code → Interpretation. Written as zero-to-hero tutorial.
