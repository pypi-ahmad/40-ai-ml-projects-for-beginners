from __future__ import annotations

from pathlib import Path

from reasoning_agent.tooling.base import ToolContext
from reasoning_agent.tooling.tools.csv_analyzer import CSVAnalyzerInput, analyze_csv
from reasoning_agent.tooling.tools.datetime_tool import DatetimeInput, current_datetime
from reasoning_agent.tooling.tools.document_search import DocumentSearchInput, search_documents
from reasoning_agent.tooling.tools.file_reader import FileReaderInput, read_file
from reasoning_agent.tooling.tools.json_explorer import JSONExplorerInput, explore_json
from reasoning_agent.tooling.tools.markdown_reader import MarkdownReaderInput, read_markdown
from reasoning_agent.tooling.tools.unit_converter import UnitConverterInput, convert_units


def test_datetime_tool_utc() -> None:
    out = current_datetime(DatetimeInput(timezone="UTC"), ToolContext("s", "r", Path(".")))
    assert out.timezone == "UTC"
    assert "T" in out.iso


def test_unit_converter_linear_and_temp() -> None:
    linear = convert_units(
        UnitConverterInput(value=1000, from_unit="m", to_unit="km"),
        ToolContext("s", "r", Path(".")),
    )
    assert round(linear.converted_value, 5) == 1.0

    temp = convert_units(
        UnitConverterInput(value=0, from_unit="c", to_unit="f"),
        ToolContext("s", "r", Path(".")),
    )
    assert round(temp.converted_value, 2) == 32.0


def test_file_json_markdown_and_document_tools(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# Title\n\nhello world\n", encoding="utf-8")
    (tmp_path / "data.json").write_text('{"a":{"b":3}}', encoding="utf-8")
    (tmp_path / "data.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")

    ctx = ToolContext("s", "r", tmp_path)

    f_out = read_file(FileReaderInput(path="note.md"), ctx)
    assert "hello" in f_out.content

    j_out = explore_json(JSONExplorerInput(path="data.json", key_path="a.b"), ctx)
    assert j_out.value == 3

    m_out = read_markdown(MarkdownReaderInput(path="note.md"), ctx)
    assert "Title" in m_out.headings

    c_out = analyze_csv(CSVAnalyzerInput(path="data.csv"), ctx)
    assert c_out.rows == 2
    assert c_out.columns == 2

    d_out = search_documents(DocumentSearchInput(query="hello", directory="."), ctx)
    assert d_out.matches
