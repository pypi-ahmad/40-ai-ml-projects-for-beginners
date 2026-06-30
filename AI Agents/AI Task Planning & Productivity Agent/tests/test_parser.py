from task_planning_agent.ingestion.parser import parse_messy_input


def test_parse_messy_input_extracts_candidates() -> None:
    raw = """
- Finish roadmap by tomorrow 5pm #proj @sara 120min
2. Prepare notes next monday 60min
"""
    candidates = parse_messy_input(raw)
    assert len(candidates) == 2
    assert candidates[0].project == "proj"
    assert "sara" in candidates[0].people
    assert candidates[0].estimated_minutes == 120
