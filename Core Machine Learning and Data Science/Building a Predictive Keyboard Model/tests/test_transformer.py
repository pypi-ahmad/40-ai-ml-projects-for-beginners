import torch

from utils.models import TransformerLM


def test_transformer_causal_mask_blocks_future_attention() -> None:
    model = TransformerLM(
        vocab_size=32,
        embedding_dim=16,
        hidden_dim=32,
        nhead=4,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()

    x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    logits, attentions = model(x, return_attention=True)

    assert logits.shape == (1, 32)
    assert len(attentions) == 1
    # [batch, heads, seq, seq]
    attention = attentions[0][0]
    seq_len = attention.shape[-1]

    for i in range(seq_len):
        future_weights = attention[:, i, i + 1 :]
        if future_weights.numel() > 0:
            assert torch.allclose(
                future_weights,
                torch.zeros_like(future_weights),
                atol=1e-6,
            )
