from pathlib import Path

from utils.vocabulary import Vocabulary


def test_vocabulary_build_encode_decode_and_persistence(tmp_path: Path) -> None:
    corpus = [
        ["hello", "world", "hello"],
        ["predictive", "keyboard", "world"],
    ]
    vocab = Vocabulary(min_freq=1, max_size=16)
    vocab.build(corpus)

    encoded = vocab.encode(["hello", "unknown"])
    assert encoded[0] == vocab.word2idx["hello"]
    assert encoded[1] == vocab.unk_idx

    decoded = vocab.decode(encoded, remove_special=False)
    assert decoded[0] == "hello"

    save_path = tmp_path / "vocab.json"
    vocab.save(save_path)

    loaded = Vocabulary.load(save_path)
    assert loaded.word2idx == vocab.word2idx
    assert loaded.token_frequencies["hello"] == 2


def test_vocabulary_stats_include_oov_rate() -> None:
    corpus = [["a", "b", "a"], ["c"]]
    vocab = Vocabulary(min_freq=2)
    vocab.build(corpus)

    stats = vocab.statistics(["a", "b", "x"])
    assert stats["vocab_size"] >= 4
    assert 0.0 <= stats["oov_rate"] <= 1.0
