from utils.tokenization import (
    NLTKTokenizerBackend,
    RegexTokenizerBackend,
    compare_tokenizer_outputs,
    normalize_text,
)


def test_normalize_and_regex_tokenization() -> None:
    text = "Hello, WORLD!! Predictive Keyboard."
    clean = normalize_text(text)
    tokens = RegexTokenizerBackend().tokenize(clean)
    assert tokens[:2] == ["hello", "world"]


def test_compare_tokenizers_includes_expected_backends() -> None:
    text = "This is a tiny corpus for keyboard predictions."
    summary = compare_tokenizer_outputs(
        text=text,
        backends=[RegexTokenizerBackend(), NLTKTokenizerBackend()],
    )

    names = {item["backend"] for item in summary}
    assert {"regex", "nltk"}.issubset(names)
    assert all(item["token_count"] > 0 for item in summary)
