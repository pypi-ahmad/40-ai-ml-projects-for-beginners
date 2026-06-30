from unittest.mock import MagicMock, patch

import pytest

from src.document_analyzer import DocumentAnalyzer


@pytest.fixture
def analyzer() -> DocumentAnalyzer:
    a = DocumentAnalyzer()
    yield a
    a.close()


def test_extract_text_file_not_found(analyzer: DocumentAnalyzer) -> None:
    result = analyzer.extract_text("/nonexistent/path.png")
    assert "File not found" in result


@patch("src.document_analyzer.Path.exists", return_value=True)
@patch("src.document_analyzer.Path.read_bytes", return_value=b"fake_image_bytes")
@patch("src.document_analyzer.OllamaClient")
def test_extract_text_success(
    mock_ollama: MagicMock,
    mock_read_bytes: MagicMock,
    mock_exists: MagicMock,
) -> None:
    mock_ollama.return_value.generate.return_value = {"response": "Extracted text content"}
    a = DocumentAnalyzer()
    result = a.extract_text("/path/to/image.png")
    assert result == "Extracted text content"
    mock_ollama.return_value.generate.assert_called_once()
    a.close()


@patch("src.document_analyzer.Path.exists", return_value=True)
@patch("src.document_analyzer.Path.read_bytes", return_value=b"data")
@patch("src.document_analyzer.OllamaClient")
def test_extract_text_empty_response(
    mock_ollama: MagicMock,
    mock_read_bytes: MagicMock,
    mock_exists: MagicMock,
) -> None:
    mock_ollama.return_value.generate.return_value = {"response": "   "}
    a = DocumentAnalyzer()
    result = a.extract_text("/path/to/img.png")
    assert result == "No text extracted."
    a.close()


@patch("src.document_analyzer.OllamaClient")
def test_answer_question(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.generate.return_value = {"response": "42"}
    a = DocumentAnalyzer()
    result = a.answer_question("Context about life", "What is the answer?")
    assert result == "42"
    mock_ollama.return_value.generate.assert_called_once()
    a.close()


@patch("src.document_analyzer.OllamaClient")
def test_answer_question_empty_context(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.generate.return_value = {"response": ""}
    a = DocumentAnalyzer()
    result = a.answer_question("", "Any ideas?")
    assert result == ""
    a.close()
