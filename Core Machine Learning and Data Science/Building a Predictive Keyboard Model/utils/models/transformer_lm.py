"""Transformer language model with optional attention export."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Minimal transformer encoder block with exposed attention weights."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_output, attn_weights = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_output))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights if return_attention else None


class TransformerLM(nn.Module):
    """Decoder-style transformer for next-token prediction."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_scale = math.sqrt(float(embedding_dim))
        self.position = PositionalEncoding(embedding_dim, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embedding_dim,
                    num_heads=nhead,
                    ff_hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, vocab_size)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # True values are masked positions for nn.MultiheadAttention.
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = self.embedding(x) * self.embedding_scale
        hidden = self.position(hidden)

        mask = self._causal_mask(x.size(1), x.device)
        attentions: list[torch.Tensor] = []

        for layer in self.layers:
            hidden, attn = layer(hidden, mask, return_attention=return_attention)
            if return_attention and attn is not None:
                attentions.append(attn)

        hidden = self.dropout(hidden[:, -1, :])
        logits = self.classifier(hidden)

        if return_attention:
            return logits, attentions
        return logits
