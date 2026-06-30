"""Graph visualization utilities using NetworkX, Mermaid, and Graphviz DOT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx


def to_networkx(graph_info: dict[str, Any]) -> nx.DiGraph:
    """Convert graph metadata into NetworkX digraph."""

    graph = nx.DiGraph()
    for node in graph_info.get("nodes", []):
        graph.add_node(node)
    for edge in graph_info.get("edges", []):
        graph.add_edge(edge["source"], edge["target"])
    return graph


def to_mermaid(graph_info: dict[str, Any]) -> str:
    """Render graph info to Mermaid format."""

    lines = ["graph TD"]
    for edge in graph_info.get("edges", []):
        lines.append(f"  {edge['source']} --> {edge['target']}")
    return "\n".join(lines)


def to_graphviz_dot(graph_info: dict[str, Any]) -> str:
    """Render graph info to Graphviz DOT format."""

    lines = ["digraph workflow {"]
    for edge in graph_info.get("edges", []):
        lines.append(f'  "{edge["source"]}" -> "{edge["target"]}";')
    lines.append("}")
    return "\n".join(lines)


def export_graph_files(
    graph_info: dict[str, Any], output_dir: str = "artifacts/graphs"
) -> dict[str, str]:
    """Export Mermaid and DOT graph files."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mermaid = out / "workflow.mmd"
    dot = out / "workflow.dot"
    mermaid.write_text(to_mermaid(graph_info), encoding="utf-8")
    dot.write_text(to_graphviz_dot(graph_info), encoding="utf-8")
    return {"mermaid": str(mermaid), "dot": str(dot)}
