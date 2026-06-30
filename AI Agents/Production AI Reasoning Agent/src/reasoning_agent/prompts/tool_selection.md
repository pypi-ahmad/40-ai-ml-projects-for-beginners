You select the best tool for one plan step.

Query:
{query}

Step:
{step_description}

Available tools:
{available_tools}

Output JSON only:
{"tool_name": "name_or_null", "reason": "short"}

Rules:
- Use only names from available tools.
- Return null when no tool is needed.
- Prefer one specific tool over generic search when possible.
