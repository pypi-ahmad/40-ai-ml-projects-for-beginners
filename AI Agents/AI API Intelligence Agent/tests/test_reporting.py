from api_intel_agent.core.schemas import AnalyzeResponse, OutputFormat, RunStatus
from api_intel_agent.reporting import ReportGenerator


def test_markdown_report_generation(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENT__REPORTS__OUTPUT_DIR', str(tmp_path))
    generator = ReportGenerator()
    response = AnalyzeResponse(
        run_id='r1',
        status=RunStatus.SUCCESS,
        summary='Summary',
    )
    out = generator.generate(response, output_format=OutputFormat.MARKDOWN)
    assert out.endswith('.md')
