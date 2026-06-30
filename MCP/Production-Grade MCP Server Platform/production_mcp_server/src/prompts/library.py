from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PromptDefinition:
    name: str
    role: str
    objective: str
    constraints: list[str]
    expected_output: str
    examples: list[str]
    template: str


class PromptLibrary:
    def __init__(self) -> None:
        self._prompts = {prompt.name: prompt for prompt in self._builtins()}

    def _builtins(self) -> list[PromptDefinition]:
        return [
            PromptDefinition(
                name="code_review",
                role="senior-software-engineer",
                objective="Review code for bugs, regressions, and security issues",
                constraints=["Cite concrete issues", "Prioritize severity", "Suggest minimal fix"],
                expected_output="Bulleted findings with severity and remediation",
                examples=["Input: PR diff, Output: 3 high-severity findings"],
                template=(
                    "Role: {role}\n"
                    "Objective: {objective}\n"
                    "Constraints: {constraints}\n"
                    "Code:\n{content}\n"
                    "Output format: {expected_output}"
                ),
            ),
            PromptDefinition(
                name="rag_qa",
                role="retrieval-qa-assistant",
                objective="Answer from retrieved context and cite source snippets",
                constraints=["No unsupported claims", "Use concise answer", "Add confidence"],
                expected_output="Answer, citations, confidence",
                examples=["Question + docs => grounded answer"],
                template=(
                    "Role: {role}\n"
                    "Objective: {objective}\n"
                    "Question: {question}\n"
                    "Context:\n{context}\n"
                    "Output: {expected_output}"
                ),
            ),
            PromptDefinition(
                name="summarization",
                role="technical-summarizer",
                objective="Summarize long technical material for engineering audience",
                constraints=["Preserve key metrics", "Keep assumptions explicit"],
                expected_output="Executive summary + key bullets",
                examples=["Long report -> concise summary"],
                template=(
                    "Role: {role}\n"
                    "Objective: {objective}\n"
                    "Constraints: {constraints}\n"
                    "Text:\n{text}\n"
                    "Output: {expected_output}"
                ),
            ),
            PromptDefinition(
                name="sql_generator",
                role="analytics-engineer",
                objective="Generate safe SQL query from natural language request",
                constraints=["Use parameterized placeholders", "Never issue destructive SQL"],
                expected_output="SQL + rationale",
                examples=["Find top customers by revenue"],
                template=(
                    "Role: {role}\n"
                    "Objective: {objective}\n"
                    "Schema:\n{schema}\n"
                    "Request: {request}\n"
                    "Output: {expected_output}"
                ),
            ),
            PromptDefinition(
                name="api_analysis",
                role="api-architect",
                objective="Analyze API payloads and recommend improvements",
                constraints=["Mention compatibility risk", "Separate breaking/non-breaking changes"],
                expected_output="Analysis report",
                examples=["REST payload review"],
                template=(
                    "Role: {role}\n"
                    "Objective: {objective}\n"
                    "Payload:\n{payload}\n"
                    "Output: {expected_output}"
                ),
            ),
            PromptDefinition(
                name="bug_investigation",
                role="production-debugger",
                objective="Investigate bug from logs and traces",
                constraints=["Propose repro steps", "Prioritize most likely root cause"],
                expected_output="Root cause hypotheses + next actions",
                examples=["Latency spike investigation"],
                template=(
                    "Role: {role}\n"
                    "Objective: {objective}\n"
                    "Logs:\n{logs}\n"
                    "Output: {expected_output}"
                ),
            ),
            PromptDefinition(
                name="documentation_writer",
                role="technical-writer",
                objective="Generate developer-facing documentation",
                constraints=["Use exact API names", "Include examples"],
                expected_output="Markdown document",
                examples=["README section generation"],
                template=(
                    "Role: {role}\n"
                    "Objective: {objective}\n"
                    "Topic: {topic}\n"
                    "Output: {expected_output}"
                ),
            ),
            PromptDefinition(
                name="report_generator",
                role="reporting-analyst",
                objective="Build formal execution report",
                constraints=["Include KPIs", "Include open risks"],
                expected_output="Structured markdown report",
                examples=["Weekly operations report"],
                template=(
                    "Role: {role}\n"
                    "Objective: {objective}\n"
                    "Data:\n{data}\n"
                    "Output: {expected_output}"
                ),
            ),
        ]

    def names(self) -> list[str]:
        return sorted(self._prompts)

    def get(self, name: str) -> PromptDefinition:
        return self._prompts[name]

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "name": prompt.name,
                "role": prompt.role,
                "objective": prompt.objective,
                "constraints": prompt.constraints,
                "expected_output": prompt.expected_output,
                "examples": prompt.examples,
            }
            for prompt in self._prompts.values()
        ]

    def render(self, name: str, variables: dict[str, Any]) -> str:
        prompt = self.get(name)
        payload = {
            "role": prompt.role,
            "objective": prompt.objective,
            "constraints": "; ".join(prompt.constraints),
            "expected_output": prompt.expected_output,
            **variables,
        }
        return prompt.template.format(**payload)
