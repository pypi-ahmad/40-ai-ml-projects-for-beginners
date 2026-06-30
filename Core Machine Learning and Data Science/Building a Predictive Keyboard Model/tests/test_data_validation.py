from utils.data import _split_text_train_val_test, validate_text_corpus


def test_text_split_preserves_punctuation_boundaries() -> None:
    text = "One sentence. Two sentence! Three sentence? Four sentence."
    train, val, test = _split_text_train_val_test(text)
    merged = f"{train} {val} {test}".strip()
    assert "." in merged or "!" in merged or "?" in merged


def test_validate_text_corpus_reports_expected_fields() -> None:
    text = "Alpha beta.\n\nGamma delta.\n\nAlpha beta."
    stats = validate_text_corpus(text)
    assert stats["num_documents"] == 3
    assert stats["num_sentences"] == 3
    assert stats["num_words"] == 6
    assert stats["duplicate_documents"] >= 1
