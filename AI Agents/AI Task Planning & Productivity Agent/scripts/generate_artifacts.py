"""Generate real-run planning artifacts and visualization screenshots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from task_planning_agent.agent.service import PlanningService
from task_planning_agent.calendar.service import CalendarService
from task_planning_agent.config import load_config
from task_planning_agent.schemas import PriorityStrategy


ARTIFACT_DIR = Path("artifacts")
SCREENSHOT_DIR = ARTIFACT_DIR / "screenshots"
REPORT_DIR = ARTIFACT_DIR / "reports"
LOG_DIR = ARTIFACT_DIR / "logs"

for directory in [SCREENSHOT_DIR, REPORT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def _plan_dataframe(report) -> pd.DataFrame:
    rows = []
    for item in report.schedule:
        rows.append(
            {
                "task": item.task,
                "priority": item.priority,
                "start": pd.to_datetime(item.suggested_start_time),
                "end": pd.to_datetime(item.suggested_end_time),
                "risk": item.risk_level.value,
                "confidence": item.confidence,
            }
        )
    return pd.DataFrame(rows)


def _save_timeline(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    if df.empty:
        ax.text(0.5, 0.5, "No tasks", ha="center", va="center")
    else:
        base = df["start"].min()
        for i, row in df.sort_values("priority", ascending=False).reset_index(drop=True).iterrows():
            start = (row["start"] - base).total_seconds() / 3600
            duration = (row["end"] - row["start"]).total_seconds() / 3600
            ax.barh(row["task"], duration, left=start)
        ax.set_xlabel("Hours from first block start")
        ax.set_title("Planning Timeline")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "timeline.png", dpi=200)
    plt.close(fig)


def _save_gantt(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    if df.empty:
        ax.text(0.5, 0.5, "No tasks", ha="center", va="center")
    else:
        for i, row in df.reset_index(drop=True).iterrows():
            ax.plot([row["start"], row["end"]], [i, i], linewidth=10)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["task"])
        ax.set_title("Gantt Chart")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "gantt.png", dpi=200)
    plt.close(fig)


def _save_kanban(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    columns = ["Backlog", "In Progress", "Done"]
    x_positions = [0, 1, 2]
    for x, col in zip(x_positions, columns, strict=False):
        ax.add_patch(plt.Rectangle((x - 0.35, 0), 0.7, 10, fill=False))
        ax.text(x, 10.2, col, ha="center", va="bottom", fontsize=12, weight="bold")

    if not df.empty:
        sorted_df = df.sort_values("priority", ascending=False).reset_index(drop=True)
        for idx, (_, row) in enumerate(sorted_df.iterrows()):
            col_x = 0 if idx < len(sorted_df) // 2 else 1
            if idx >= len(sorted_df) - 1:
                col_x = 2
            y = 9 - (idx % 5) * 1.8
            ax.text(col_x, y, f"{row['task']}\nP={row['priority']:.1f}", ha="center", va="center")

    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title("Kanban Board")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "kanban.png", dpi=200)
    plt.close(fig)


def _save_dependency_graph(tasks) -> None:
    graph = nx.DiGraph()
    for task in tasks:
        graph.add_node(task.id, label=task.name)
        for dep in task.dependencies:
            graph.add_edge(dep, task.id)

    fig, ax = plt.subplots(figsize=(12, 8))
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No dependencies", ha="center", va="center")
    else:
        pos = nx.spring_layout(graph, seed=7)
        labels = {node: data.get("label", node[:6]) for node, data in graph.nodes(data=True)}
        nx.draw_networkx(graph, pos, labels=labels, ax=ax, node_size=1800, font_size=8)
        ax.set_title("Dependency Graph")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "dependency_graph.png", dpi=200)
    plt.close(fig)


def _save_analytics(report) -> None:
    metrics = report.analytics.model_dump()
    keys = []
    vals = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            keys.append(key)
            vals.append(value)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(keys, vals)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_title("Analytics Dashboard")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "analytics_dashboard.png", dpi=200)
    plt.close(fig)


def _save_memory(report, service: PlanningService) -> None:
    results = service.memory.semantic_search("deadline risk")
    text_lines = ["Memory Search Results", ""]
    for i, item in enumerate(results[:8], start=1):
        doc = str(item.get("document", ""))[:90]
        dist = item.get("distance", "n/a")
        text_lines.append(f"{i}. {doc} | distance={dist}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.text(0.01, 0.99, "\n".join(text_lines), va="top", family="monospace")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "memory_search.png", dpi=200)
    plt.close(fig)


def main() -> None:
    service = PlanningService(load_config())
    raw_input = """
- Finalize quarterly planning deck by tomorrow 5pm #planning @sara 120min
- Review architecture PR before today 7pm 60min
- Sprint backlog refinement meeting friday 11am 45min
- Draft hiring scorecard next monday 90min
- Follow up with design on dashboard UX tomorrow noon 30min
""".strip()

    report = service.plan(
        user_id="demo-user",
        raw_input=raw_input,
        strategy=PriorityStrategy.WSJF,
        timezone="Asia/Kolkata",
    )

    report_path = REPORT_DIR / "real_run_report.json"
    report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")

    history = service.memory.history("demo-user", limit=1)
    if history:
        cal = CalendarService()
        cal.export_ics(str(REPORT_DIR / "real_run_schedule.ics"), history[0].schedule_blocks)

    df = _plan_dataframe(report)
    _save_timeline(df)
    _save_gantt(df)
    _save_kanban(df)
    _save_dependency_graph(service.memory.history("demo-user", limit=1)[0].tasks)
    _save_analytics(report)
    _save_memory(report, service)

    log_path = LOG_DIR / "real_run.log"
    log_path.write_text(
        f"Generated at: {datetime.utcnow().isoformat()}\n"
        f"Plan ID: {report.plan_id}\n"
        f"Tasks: {len(report.schedule)}\n"
        f"Report: {report_path}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
