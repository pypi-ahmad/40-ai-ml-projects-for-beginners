from resume_ai.config.loader import AppConfig
from resume_ai.models import OCRMode
from resume_ai.parsing.resume_parser import ResumeParser


class DummyLLM:
    def generate_json(self, *args, **kwargs):
        return {}


def test_resume_parser_extracts_contact_and_skills() -> None:
    parser = ResumeParser(AppConfig())
    parser.llm = DummyLLM()  # type: ignore[assignment]

    text = """
    Jane Doe
    jane@example.com
    +1 555 222 3333
    Skills
    Python FastAPI AWS Kubernetes LangChain
    Experience
    Built production ML platform
    """

    parsed = parser.parse(text=text, ocr_mode=OCRMode.DIGITAL, blind_mode=True)

    assert parsed.candidate.email == "jane@example.com"
    assert "python" in parsed.skills.programming_languages
    assert "fastapi" in parsed.skills.frameworks
    assert parsed.redacted_text is not None
    assert "[REDACTED_EMAIL]" in parsed.redacted_text
