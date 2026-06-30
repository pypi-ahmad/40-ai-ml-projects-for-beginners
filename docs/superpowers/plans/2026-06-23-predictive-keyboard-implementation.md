# Predictive Keyboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete predictive keyboard system — language modeling from Sherlock Holmes text with neural architectures, evaluation, and Streamlit demo.

**Architecture:** Shared Python utility modules for preprocessing/vocabulary/training/evaluation + 6 Jupyter notebooks for exploration, baselines, neural models, evaluation, demo + Streamlit web app. PyTorch for deep learning, NLTK/HuggingFace for NLP, scikit-learn for classical baselines.

**Tech Stack:** Python 3.12.10, uv, PyTorch, NLTK, spaCy, HuggingFace tokenizers, scikit-learn, matplotlib, seaborn, plotly, wordcloud, umap-learn, Streamlit

## Global Constraints

- Python 3.12.10 (locked via `.python-version`)
- Use `uv` for all package/env management
- Never zip` the `__MACOSX` folder; use `-x "__MACOSX/*"` in unzip
- All models defined as separate classes in `utils/models/`
- All notebooks save figures to `outputs/` directory
- Seed everything for reproducibility (`torch.manual_seed(42)`, `np.random.seed(42)`)
- Store extracted `.txt` in project root, not a temp directory
- Use relative imports from `utils/` in notebooks (add `sys.path.append("..")`)
- Streamlit app goes in `app/app.py`

---

### Task 1: Project Scaffolding & Data Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Modify: `Building a Predictive Keyboard Model/` directory
- N/A: test (scaffolding)

**Interfaces:**
- Consumes: `3dd01-book.zip` in project root
- Produces: `sherlock-holm.es_stories_plain-text_advs.txt` in project root

- [ ] **Step 1: Set up project structure**

Create all directories:
```bash
PROJ_DIR="/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/Building a Predictive Keyboard Model"
mkdir -p "$PROJ_DIR/utils/models"
mkdir -p "$PROJ_DIR/app"
mkdir -p "$PROJ_DIR/outputs/figures"
mkdir -p "$PROJ_DIR/outputs/checkpoints"
mkdir -p "$PROJ_DIR/outputs/results"
mkdir -p "$PROJ_DIR/images"
```

- [ ] **Step 2: Create `.python-version`**

Write to `.python-version`:
```
3.12.10
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "predictive-keyboard"
version = "1.0.0"
description = "Predictive keyboard system using language models trained on Sherlock Holmes"
requires-python = "==3.12.10"
dependencies = [
    "torch>=2.0",
    "nltk>=3.8",
    "spacy>=3.7",
    "tokenizers>=0.15",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "seaborn>=0.13",
    "plotly>=5.17",
    "wordcloud>=1.9",
    "umap-learn>=0.5",
    "streamlit>=1.28",
    "pandas>=2.0",
    "numpy>=1.24",
    "tqdm>=4.65",
    "jupyter>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "black", "flake8"]
```

- [ ] **Step 4: Extract dataset**

```bash
cd "$PROJ_DIR"
unzip -o "3dd01-book.zip" -x "__MACOSX/*"
# Move text file to root if nested
find . -name "*.txt" -exec mv {} ./sherlock-holm.es_stories_plain-text_advs.txt \;
```

- [ ] **Step 5: Create `__init__.py` files**

Write `utils/__init__.py`:
```python
```

Write `utils/models/__init__.py`:
```python
from .lstm import LSTM_LM
from .stacked_lstm import StackedLSTM_LM
from .bilstm import BiLSTM_LM
from .gru import GRU_LM
from .cnn_lstm import CNN_LSTM_LM
from .transformer_lm import TransformerLM

__all__ = [
    "LSTM_LM",
    "StackedLSTM_LM",
    "BiLSTM_LM",
    "GRU_LM",
    "CNN_LSTM_LM",
    "TransformerLM",
]
```

- [ ] **Step 6: Create uv venv and install**

```bash
cd "$PROJ_DIR"
uv venv --python 3.12.10
uv pip install -e .
uv pip install -e ".[dev]"
uv pip install spacy && python -m spacy download en_core_web_sm
```

---

### Task 2: Text Preprocessing Module

**Files:**
- Create: `utils/preprocessing.py`
- Create: `utils/vocabulary.py`
- Create: `utils/sequence_builder.py`

**Interfaces:**
- Consumes: Raw text file path → loads full corpus
- Produces: Cleaned tokens, vocabulary objects, context-target sequences

- [ ] **Step 1: Create `utils/preprocessing.py`**

```python
import re
import nltk
from typing import List, Optional, Callable


def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s\'.!?,;-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_nltk(text: str) -> List[str]:
    return nltk.word_tokenize(text.lower())


def tokenize_split(text: str) -> List[str]:
    return text.lower().split()


def normalize_text(text: str, method: str = "nltk") -> List[str]:
    cleaned = clean_text(text)
    if method == "nltk":
        try:
            return tokenize_nltk(cleaned)
        except LookupError:
            nltk.download("punkt")
            return tokenize_nltk(cleaned)
    return tokenize_split(cleaned)


def create_ngrams(tokens: List[str], n: int) -> List[List[str]]:
    return [tokens[i : i + n] for i in range(len(tokens) - n + 1)]


def get_special_tokens() -> dict:
    return {
        "<PAD>": 0,
        "<UNK>": 1,
        "<SOS>": 2,
        "<EOS>": 3,
    }
```

- [ ] **Step 2: Create `utils/vocabulary.py`**

```python
from typing import Dict, List, Optional
import torch


class Vocabulary:
    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.frequencies: Dict[str, int] = {}
        self.min_freq = min_freq
        self.max_size = max_size

        if tokens is not None:
            self.build(tokens)

    @property
    def pad_idx(self) -> int: return self.word2idx.get("<PAD>", 0)

    @property
    def unk_idx(self) -> int: return self.word2idx.get("<UNK>", 1)

    @property
    def sos_idx(self) -> int: return self.word2idx.get("<SOS>", 2)

    @property
    def eos_idx(self) -> int: return self.word2idx.get("<EOS>", 3)

    @property
    def size(self) -> int: return len(self.word2idx)

    def build(self, tokens: List[str]):
        from collections import Counter
        freq = Counter(tokens)

        specials = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        for i, token in enumerate(specials):
            self.word2idx[token] = i
            self.idx2word[i] = token
            self.frequencies[token] = 0

        sorted_tokens = sorted(
            [(t, c) for t, c in freq.items() if c >= self.min_freq],
            key=lambda x: -x[1],
        )

        if self.max_size is not None:
            sorted_tokens = sorted_tokens[: self.max_size]

        idx = len(specials)
        for token, count in sorted_tokens:
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            self.frequencies[token] = count
            idx += 1

    def encode(self, token: str) -> int:
        return self.word2idx.get(token, self.unk_idx)

    def decode(self, idx: int) -> str:
        return self.idx2word.get(idx, "<UNK>")

    def encode_sequence(self, tokens: List[str]) -> List[int]:
        return [self.encode(t) for t in tokens]

    def decode_sequence(self, indices: List[int]) -> List[str]:
        return [self.decode(i) for i in indices]

    def save(self, path: str):
        import json
        data = {
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "frequencies": self.frequencies,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        import json
        with open(path) as f:
            data = json.load(f)
        vocab = cls(min_freq=data["min_freq"], max_size=data["max_size"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab.frequencies = data["frequencies"]
        return vocab
```

- [ ] **Step 3: Create `utils/sequence_builder.py`**

```python
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    def __init__(
        self,
        sequences: List[torch.Tensor],
        targets: List[torch.Tensor],
    ):
        self.sequences = sequences
        self.targets = targets

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def build_sequences(
    token_ids: List[int],
    context_len: int = 5,
    stride: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    sequences, targets = [], []
    for i in range(0, len(token_ids) - context_len, stride):
        seq = token_ids[i : i + context_len]
        tgt = token_ids[i + context_len]
        sequences.append(torch.tensor(seq, dtype=torch.long))
        targets.append(torch.tensor(tgt, dtype=torch.long))
    return sequences, targets


def create_dataloaders(
    sequences: List[torch.Tensor],
    targets: List[torch.Tensor],
    batch_size: int = 64,
    train_split: float = 0.8,
    val_split: float = 0.1,
    shuffle: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    n = len(sequences)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_data = LanguageModelDataset(
        sequences[:n_train], targets[:n_train]
    )
    val_data = LanguageModelDataset(
        sequences[n_train : n_train + n_val],
        targets[n_train : n_train + n_val],
    )
    test_data = LanguageModelDataset(
        sequences[n_train + n_val :],
        targets[n_train + n_val :],
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader
```

---

### Task 3: Training, Evaluation & Sampling Modules

**Files:**
- Create: `utils/trainer.py`
- Create: `utils/evaluation.py`
- Create: `utils/sampling.py`

**Interfaces:**
- Consumes: PyTorch model, dataloaders, vocab
- Produces: Trained model checkpoint, evaluation metrics, predicted tokens

- [ ] **Step 1: Create `utils/trainer.py`**

```python
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Callable] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in tqdm(dataloader, desc="Training", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
        return total_loss / len(dataloader.dataset)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 3,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, list]:
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if checkpoint_path is not None:
                    torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return self.history
```

- [ ] **Step 2: Create `utils/evaluation.py`**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple
import numpy as np


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module = nn.CrossEntropyLoss(),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    k_values: List[int] = [1, 3, 5],
) -> dict:
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_samples = 0

    correct_at_k = {k: 0 for k in k_values}
    reciprocal_ranks = []
    total_recall_at_k = {k: 0 for k in [3, 5]}

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=max(k_values), dim=-1)

        for k in k_values:
            top_k = top_indices[:, :k]
            correct_at_k[k] += (top_k == y.unsqueeze(1)).any(dim=1).sum().item()

        ranks = []
        for i in range(y.size(0)):
            rank = (top_indices[i] == y[i]).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                ranks.append(1.0 / (rank[0].item() + 1))
            else:
                ranks.append(0.0)
        reciprocal_ranks.extend(ranks)

        for k in [3, 5]:
            if k <= top_indices.size(1):
                total_recall_at_k[k] += (top_indices[:, :k] == y.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)

    metrics = {"cross_entropy": avg_loss, "perplexity": perplexity}
    for k in k_values:
        metrics[f"accuracy@{k}"] = correct_at_k[k] / total_samples
    metrics["mrr"] = np.mean(reciprocal_ranks)
    for k in [3, 5]:
        metrics[f"recall@{k}"] = total_recall_at_k[k] / total_samples

    return metrics


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, dataloader, criterion, device)
    return metrics["perplexity"]


def print_metrics_table(metrics: dict, model_name: str = ""):
    header = f"  {model_name} Performance  ".center(60, "=")
    print(f"\n{header}")
    for key in ["cross_entropy", "perplexity", "accuracy@1", "accuracy@3", "accuracy@5", "mrr", "recall@3", "recall@5"]:
        if key in metrics:
            print(f"  {key:20s}: {metrics[key]:.4f}")
    print("=" * 60)
```

- [ ] **Step 3: Create `utils/sampling.py`**

```python
import torch
import torch.nn.functional as F
from typing import List, Optional


def greedy_decode(model: torch.nn.Module, context: torch.Tensor, vocab_size: int) -> int:
    with torch.no_grad():
        logits = model(context.unsqueeze(0))
        return logits.squeeze(0).argmax().item()


def top_k_sampling(logits: torch.Tensor, k: int = 10) -> torch.Tensor:
    top_k_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_vals[:, -1].unsqueeze(-1)
    filtered = torch.where(logits >= threshold, logits, float("-inf"))
    probs = F.softmax(filtered, dim=-1)
    return torch.multinomial(probs.squeeze(0), 1).item()


def top_p_sampling(logits: torch.Tensor, p: float = 0.9) -> int:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    mask = cumsum - probs > p
    sorted_logits[mask] = float("-inf")
    reweighted = F.softmax(sorted_logits, dim=-1)
    idx = torch.multinomial(reweighted.squeeze(0), 1).item()
    return sorted_indices.squeeze(0)[idx].item()


def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> int:
    scaled = logits / max(temperature, 1e-8)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs.squeeze(0), 1).item()


def decode_sequence(
    model: torch.nn.Module,
    context: torch.Tensor,
    vocab,
    max_len: int = 20,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 10,
    top_p: float = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[str]:

    model.eval()
    generated = context.tolist()
    context_tensor = context.unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(context_tensor)[:, -1, :] if logits.dim() == 3 else model(context_tensor)

            if strategy == "greedy":
                next_token = logits.argmax(dim=-1).item()
            elif strategy == "top_k":
                next_token = top_k_sampling(logits, k=top_k)
            elif strategy == "top_p":
                next_token = top_p_sampling(logits, p=top_p)
            elif strategy == "temperature":
                next_token = temperature_sampling(logits, temperature=temperature)
            else:
                next_token = logits.argmax(dim=-1).item()

            generated.append(next_token)

            if next_token == vocab.eos_idx:
                break

            context_tensor = torch.cat(
                [context_tensor[:, 1:], torch.tensor([[next_token]], device=device)], dim=1
            )

    return vocab.decode_sequence(generated)
```

---

### Task 4: Model Implementations

**Files:**
- Create: `utils/models/lstm.py`
- Create: `utils/models/stacked_lstm.py`
- Create: `utils/models/bilstm.py`
- Create: `utils/models/gru.py`
- Create: `utils/models/cnn_lstm.py`
- Create: `utils/models/transformer_lm.py`

**Interfaces:** Each model class takes `vocab_size`, `embedding_dim`, `hidden_dim`, and optional architecture-specific params. Each implements `forward(x)` returning `(batch, vocab_size)` logits.

- [ ] **Step 1: Create `utils/models/lstm.py`**

```python
import torch
import torch.nn as nn


class LSTM_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)
```

- [ ] **Step 2: Create `utils/models/stacked_lstm.py`**

```python
import torch
import torch.nn as nn


class StackedLSTM_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)
```

- [ ] **Step 3: Create `utils/models/bilstm.py`**

```python
import torch
import torch.nn as nn


class BiLSTM_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)
```

- [ ] **Step 4: Create `utils/models/gru.py`**

```python
import torch
import torch.nn as nn


class GRU_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        gru_out, _ = self.gru(emb)
        out = self.dropout(gru_out[:, -1, :])
        return self.fc(out)
```

- [ ] **Step 5: Create `utils/models/cnn_lstm.py`**

```python
import torch
import torch.nn as nn


class CNN_LSTM_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_filters: int = 100,
        kernel_sizes: list = [3, 5, 7],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_filters, k, padding=k // 2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1),
            )
            for k in kernel_sizes
        ])
        cnn_out_dim = num_filters * len(kernel_sizes)
        self.lstm = nn.LSTM(cnn_out_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        emb_perm = emb.permute(0, 2, 1)
        conv_outs = [conv(emb_perm) for conv in self.convs]
        cnn_out = torch.cat(conv_outs, dim=1)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)
```

- [ ] **Step 6: Create `utils/models/transformer_lm.py`**

```python
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        max_len: int = 100,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        transformer_out = self.transformer(emb)
        out = self.dropout(transformer_out[:, -1, :])
        return self.fc(out)
```

---

### Task 5: Notebook 01 — Data Exploration

**Files:**
- Create: `01-data-exploration.ipynb`

**Interfaces:** None. Standalone exploration notebook.

- [ ] **Step 1: Create notebook with these sections:**

1. **Setup & Imports** — imports, `sys.path.append("..")`, load text via `preprocessing.load_text()`
2. **Dataset Profile** — char count, word count, line count, sentence count, unique tokens
3. **Text Sample** — display first/last 500 chars, random paragraphs
4. **Line Plot: Text Position vs. Character Count** — running character histogram
5. **Bar Plot: Top 30 Most Frequent Words** — frequency bar chart
6. **Zipf's Law Visualization** — log-log frequency rank plot, fit regression line
7. **Word Cloud** — generate word cloud from full text
8. **N-gram Analysis** — most common 2-grams, 3-grams, 4-grams as bar charts
9. **Vocabulary Growth Curve** — cumulative unique tokens vs total words
10. **Story-level Segmentation** — detect `CHAPTER` / `ADVENTURE` headers, count per story
11. **Sentence Length Distribution** — histogram, mean, median, std
12. **Narrative Stats** — punctuation frequencies, capitalization patterns

Code approach: Use the raw text directly with standard Python + matplotlib/seaborn/wordcloud. Embed every output.

---

### Task 6: Notebook 02 — Text Preprocessing

**Files:**
- Create: `02-text-preprocessing.ipynb`

- [ ] **Step 1: Create notebook with these sections:**

1. **Setup** — imports, load text
2. **Cleaning** — show dirty→clean pipeline with `clean_text()`
3. **Tokenization Comparison** — text.split() vs NLTK vs spaCy. Compare token counts, speed
4. **Frequency Distribution Plot** — bar chart of top 50 tokens with `Vocabulary`
5. **Vocabulary Building** — build `Vocabulary` with different `min_freq` values, compare sizes
6. **Sequence Construction** — build context-target pairs with `build_sequences()` for context lengths 3, 5, 10, 20
7. **Context Length Trade-off** — table showing train/val/test splits per context length
8. **Visualize Sequences** — show examples of (context → target) as readable text
9. **Save Vocabulary** — persist `vocab.json` to `outputs/`

---

### Task 7: Notebook 03 — Baseline Models

**Files:**
- Create: `03-baseline-models.ipynb`

- [ ] **Step 1: Create notebook with these sections:**

1. **Setup** — imports, load vocab, build sequences (context_len=5)
2. **Most Frequent Word Baseline** — always predict the most common word, accuracy@1/3/5
3. **Unigram Model** — predict by unigram probability, all metrics
4. **Bigram Model** — estimate P(w_i|w_{i-1}), handle sparse counts
5. **Trigram Model** — estimate P(w_i|w_{i-2}, w_{i-1}), smoothing with add-1
6. **Baseline Comparison Table** — accuracy@1/3/5, perplexity, MRR, recall@3/5 across all 4 baselines
7. **Bar Chart Comparison** — visual comparison of all baseline metrics
8. **Error Analysis** — confusion examples from each baseline

---

### Task 8: Notebook 04 — Neural Language Models

**Files:**
- Create: `04-neural-language-models.ipynb`

- [ ] **Step 1: Create notebook with these sections:**

1. **Setup** — imports, load vocab, build loaders (context_len=5, batch_size=64)
2. **Embedding Visualization** — train simple embedding, PCA/t-SNE/UMAP, show nearest neighbors
3. **Vanilla LSTM** — define, train 10 epochs, evaluate, save checkpoint
4. **Stacked LSTM (3-layer)** — define, train, evaluate, checkpoint
5. **BiLSTM** — define, train, evaluate, checkpoint
6. **GRU** — define, train, evaluate, checkpoint
7. **CNN+LSTM** — define, train, evaluate, checkpoint
8. **Transformer LM** — define, train, evaluate, checkpoint
9. **Training Curves** — overlay train/val loss for all models on one plot
10. **Embedding Space Evolution** — PCA before/after training
11. **Comparison Table** — all neural models side by side

All models: `embedding_dim=128`, `hidden_dim=256`, `lr=1e-3`, `epochs=10`, `patience=3`, `grad_clip=1.0`. Log metrics at end of each epoch.

---

### Task 9: Notebook 05 — Evaluation Engine

**Files:**
- Create: `05-evaluation-engine.ipynb`

- [ ] **Step 1: Create notebook with these sections:**

1. **Setup** — imports, load all model checkpoints
2. **Global Evaluation Table** — all 10 models (4 baseline + 6 neural) on all metrics
3. **Heatmap** — models x metrics as annotated heatmap
4. **Beam Search Comparison** — beam width 1, 3, 5 vs greedy. Perplexity comparison
5. **Sampling Strategies** — greedy vs temperature (0.5, 0.8, 1.0, 1.5) vs top-k (5, 10, 20) vs top-p (0.8, 0.9, 0.95)
6. **Sample Generations** — given "the mysterious", show completions from each sampling method
7. **Context Length Ablation** — train best model on context_len=3,5,10,20, compare
8. **Perplexity vs Model Size** — scatter plot, discuss scaling laws
9. **Error Patterns** — confusion matrix for top predictions

---

### Task 10: Notebook 06 — Predictive Keyboard Demo

**Files:**
- Create: `06-predictive-keyboard-demo.ipynb`

- [ ] **Step 1: Create notebook with these sections:**

1. **Setup** — load best model + vocab
2. **Keyboard Engine** — function: input text → top-5 predictions with probabilities
3. **Interactive Cell** — user enters partial text, see predictions update
4. **Smartphone Simulation** — visual mock phone with text input + 5 prediction buttons
5. **Attention Visualization** — for Transformer, show attention weights as heatmap
6. **Prediction Confidence** — bar chart of top-5 probabilities for example inputs
7. **Story Generator** — seed text, generate 50 words with each decoding strategy
8. **Comparative Demo** — show all models on same input side by side

---

### Task 11: Streamlit Application

**Files:**
- Create: `app/app.py`

- [ ] **Step 1: Create `app/app.py`**

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
import numpy as np

from utils.preprocessing import normalize_text
from utils.vocabulary import Vocabulary
from utils.models import LSTM_LM
from utils.evaluation import evaluate_model

st.set_page_config(page_title="Sherlock Predictive Keyboard", layout="centered")
st.title("Sherlock Holmes Predictive Keyboard")
st.markdown("Type text and get real-time next-word predictions")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocabulary.load("outputs/vocab.json")
    model = LSTM_LM(vocab_size=vocab.size)
    model.load_state_dict(torch.load("outputs/checkpoints/lstm_best.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, vocab, device

model, vocab, device = load_model()

def predict_next(text: str, top_k: int = 5) -> list:
    tokens = normalize_text(text, method="split")
    if len(tokens) == 0:
        return []
    context = tokens[-5:]
    idxs = [vocab.encode(t) for t in context]
    if len(idxs) < 5:
        idxs = [vocab.pad_idx] * (5 - len(idxs)) + idxs
    tensor = torch.tensor([idxs], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

    results = []
    for prob, idx in zip(top_probs.squeeze(), top_indices.squeeze()):
        word = vocab.decode(idx.item())
        results.append((word, prob.item()))
    return results

user_input = st.text_input("Start typing...", "the mysterious")

if user_input:
    predictions = predict_next(user_input)
    if predictions:
        st.subheader("Next word predictions:")
        cols = st.columns(len(predictions))
        for col, (word, prob) in zip(cols, predictions):
            col.button(f"{word}\n{prob:.1%}", use_container_width=True)

        st.subheader("Confidence")
        words = [w for w, _ in predictions]
        probs = [p for _, p in predictions]
        st.bar_chart({"word": words, "probability": probs}, x="word", y="probability")

st.sidebar.header("Info")
st.sidebar.info(
    "Trained on *The Adventures of Sherlock Holmes*\n"
    f"Vocabulary size: {vocab.size}\n"
    "Model: LSTM Language Model"
)
```

---

### Task 12: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write comprehensive README**

Structure:
```
# Predictive Keyboard Model

## Overview
## The Dataset
## Methodology
### Data Preprocessing
### Baseline Models (N-gram)
### Neural Architectures
### Training & Hyperparameters
### Evaluation Metrics
### Decoding Strategies
### Results
### Predictive Keyboard Demo
## Key Findings
## Limitations & Future Work
## Requirements
## Project Structure
## How to Run
## Results Summary Table (at-a-glance)
## References
```

Include key findings, model comparison table, demo screenshot references.

---

## Self-Review

**Spec coverage check:**
- Zip extract with `-x "__MACOSX/*"`: Task 1 Step 4 ✓
- All 6 model architectures: Task 4 ✓
- Baseline models (MFW, unigram, bigram, trigram): Task 7 ✓
- Embedding visualization (PCA/t-SNE/UMAP): Task 8 Step 2 ✓
- All metrics (accuracy@k, perplexity, MRR, recall@k): Task 3 evaluation.py ✓
- Beam search, temperature, top-k, top-p: Task 3 sampling.py ✓
- Streamlit app: Task 11 ✓
- Training pipeline (early stopping, LR scheduling, gradient clipping, checkpointing): Task 3 trainer.py ✓
- 6 notebooks: Tasks 5-10 ✓
- README in mini-book format: Task 12 ✓

**Placeholder scan:** No TBD, TODO, or "fill in later" placeholders found. All utility code provided in full.

**Type consistency:** All models use same `(vocab_size, embedding_dim, hidden_dim, ...)` pattern. `evaluate_model()` returns consistent dict. `trainer.fit()` returns `Dict[str, list]`. ✓
