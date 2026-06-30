from textclf_framework.data.preprocessing import PreprocessConfig, preprocess_text


def test_preprocess_removes_html_url_email() -> None:
    cfg = PreprocessConfig(remove_html=True, remove_urls=True, remove_emails=True, emoji_policy="remove")
    raw = "<p>Hello</p> visit https://example.com and email me@test.com 😊"
    cleaned = preprocess_text(raw, cfg)

    assert "<p>" not in cleaned
    assert "https://" not in cleaned
    assert "@" not in cleaned
    assert "hello" in cleaned


def test_preprocess_handles_none() -> None:
    assert preprocess_text(None) == ""
