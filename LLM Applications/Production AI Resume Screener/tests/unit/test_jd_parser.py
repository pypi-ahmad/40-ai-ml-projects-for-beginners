from resume_ai.config.loader import AppConfig
from resume_ai.parsing.jd_parser import JobDescriptionParser


class DummyLLM:
    def generate_json(self, *args, **kwargs):
        return {}


def test_jd_parser_normalizes_aliases() -> None:
    parser = JobDescriptionParser(AppConfig())
    parser.llm = DummyLLM()  # type: ignore[assignment]

    jd = """
    Senior AI Engineer
    Must have ReactJS, NodeJS, K8s, Python
    5+ years experience
    Bachelor's degree
    """

    parsed = parser.parse(jd)

    assert "react" in parsed.required_skills
    assert "node.js" in parsed.keywords or "node.js" in parsed.technologies
    assert "kubernetes" in parsed.technologies
