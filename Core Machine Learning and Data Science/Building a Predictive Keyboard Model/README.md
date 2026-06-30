# Building a Predictive Keyboard Model

End-to-end NLP and language-modeling project that learns next-word probabilities from real text and serves smartphone-style suggestions.

## Executive Summary
- Problem: build a realistic predictive keyboard system (not a toy notebook).
- Data: Sherlock corpus from `3dd01-book.zip` with optional WikiText-103 augmentation.
- Models: Most-Frequent, Unigram, Bigram, Trigram, LSTM family, GRU, CNN-LSTM, Transformer LM.
- Outputs: top-3/top-5 next-word suggestions with calibrated probabilities, benchmark tables, explainability views, and Streamlit demo.

## Predictive Keyboard Theory
Given context tokens `x_1 ... x_t`, the model estimates:

`P(x_{t+1} | x_1 ... x_t)`

The keyboard returns the top-k candidates ranked by probability. This is the same core objective used by mobile keyboards, smart compose systems, and autocomplete engines.

## NLP Pipeline
1. Extract and validate corpus from `3dd01-book.zip`.
2. Build train/val/test text splits with punctuation preserved.
3. Normalize and tokenize text (NLTK, spaCy, Hugging Face tokenizers supported).
4. Build vocabulary from **train split only**.
5. Convert tokens to IDs and build context-target sequences.
6. Train baselines and neural LMs.
7. Evaluate with accuracy, recall@k, MRR, cross-entropy, perplexity.
8. Benchmark latency, throughput, and memory.

## Leakage and Evaluation Rigor
- Split-safe sequence construction avoids cross-split context leakage.
- Vocabulary and token-frequency fitting is train-only.
- Weighted metric aggregation is used for cross-entropy/perplexity and top-k metrics.
- Transformer uses causal masking for autoregressive next-token prediction.
- Best checkpoint is restored before final test evaluation.

## Vocabulary Engineering
- `word2idx` / `idx2word`
- special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- configurable `min_freq` and `max_size`
- OOV statistics and long-tail behavior support

## Models Implemented
### Classical Language Models
- Most Frequent Word
- Unigram (smoothed)
- Bigram (smoothed)
- Trigram (smoothed)

### Neural Language Models
- Vanilla LSTM
- Stacked LSTM
- BiLSTM
- GRU
- CNN + LSTM
- Transformer LM (causal attention)

## Embeddings and Explainability
- Trainable embeddings inside neural models
- Word2Vec and FastText (gensim)
- Projection tools: PCA, t-SNE, UMAP
- Explainability helpers: attention map extraction and probability tables

## Repository Layout
- `utils/`: reusable NLP, modeling, training, evaluation, inference modules
- `scripts/`: reproducible data prep, training, benchmarking, notebook generation/execution
- `notebooks/`: mini-book tutorial sequence (01-08)
- `app/streamlit_app.py`: local deployment demo
- `outputs/`: checkpoints, leaderboards, executed notebooks, reports

## Quickstart (Fresh Clone)
```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

### 1) Prepare and validate data
```bash
uv run python scripts/prepare_data.py --include-wikitext
```

### 2) Train and benchmark
```bash
uv run python scripts/train_and_benchmark.py \
  --profile quick \
  --include-wikitext \
  --prefer-gpu \
  --wikitext-train-tokens 60000 \
  --wikitext-val-tokens 10000 \
  --wikitext-test-tokens 10000
```

### 3) Execute tutorial notebooks
```bash
uv run python scripts/execute_notebooks.py
```

### 4) Launch predictive keyboard app
```bash
uv run streamlit run app/streamlit_app.py
```

## Main Artifacts
- `outputs/results/leaderboard_*.csv`: model ranking
- `outputs/results/model_registry_*.json`: checkpoint + model metadata
- `outputs/results/dataset_profile.json`: corpus profiling
- `outputs/results/dataset_validation.json`: corpus integrity checks
- `outputs/checkpoints/*_quick_cpu_*.pt`: trained checkpoints (run-suffixed for safety)
- `outputs/executed_notebooks/*-executed.ipynb`: executed tutorial notebooks
- `FINAL_PROJECT_VERIFICATION_REPORT.md`: final audit report

## Results and Lessons Learned
- Neural models outperform count-based baselines on top-k suggestion quality in this setup.
- Strong baselines remain competitive in latency-constrained settings.
- Correct evaluation math and leakage-safe splitting materially affect reported scores.
- FastText is useful for morphology-heavy and rare-word neighborhoods.

## Future Improvements
- Distilled transformer for lower-latency edge inference
- Quantization/ONNX export for mobile deployment
- Personalized user-adaptive language models
- Safety filtering for toxic/unsafe completions
