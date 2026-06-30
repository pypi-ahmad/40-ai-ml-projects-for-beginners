import torch

from utils.keyboard_engine import PredictiveKeyboardEngine
from utils.vocabulary import Vocabulary


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, self.vocab_size)
        # Make regular tokens highest, but also inject high score on special IDs.
        logits[:, 4] = 8.0
        logits[:, 5] = 7.0
        logits[:, 6] = 6.0
        logits[:, 0] = 9.0  # <pad> should be filtered
        logits[:, 2] = 8.5  # <bos> should be filtered
        return logits


def _build_engine() -> PredictiveKeyboardEngine:
    vocab = Vocabulary(min_freq=1)
    vocab.build([["i", "would", "like", "to", "know", "see", "make"]])
    model = DummyModel(vocab_size=len(vocab))
    return PredictiveKeyboardEngine(
        model=model,
        vocabulary=vocab,
        context_length=4,
        device="cpu",
    )


def test_keyboard_engine_filters_special_tokens() -> None:
    engine = _build_engine()
    preds = engine.predict("i would like to", top_k=3)
    assert len(preds) == 3
    blocked = {"<pad>", "<unk>", "<bos>", "<eos>"}
    assert all(str(item["token"]) not in blocked for item in preds)


def test_keyboard_engine_supports_top_p_strategy() -> None:
    engine = _build_engine()
    preds = engine.predict("i would like to", top_k=3, strategy="top_p", top_p=0.9)
    assert preds
    assert all(0.0 <= float(item["probability"]) <= 1.0 for item in preds)


def test_keyboard_engine_rejects_invalid_strategy() -> None:
    engine = _build_engine()
    try:
        engine.predict("hello world", top_k=3, strategy="invalid")
    except ValueError as exc:
        assert "strategy" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for invalid strategy")
