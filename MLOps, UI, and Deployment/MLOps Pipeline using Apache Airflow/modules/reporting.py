"""HTML reporting for quality, modeling, monitoring, and operations summaries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt_table(df: pd.DataFrame | None, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<p>No data available.</p>"
    return df.head(max_rows).to_html(index=False, border=0, classes="tbl")


def _metrics_cards(metrics: dict[str, Any]) -> str:
    cards = []
    for key in ["rmse", "mae", "mse", "r2", "mape"]:
        if key in metrics:
            cards.append(
                f"<div class='metric'><div class='metric-key'>{key.upper()}</div><div class='metric-value'>{float(metrics[key]):.4f}</div></div>"
            )
    return "".join(cards)


def build_report_html(
    title: str,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    model_leaderboard: pd.DataFrame | None,
    candidate_scoreboard: pd.DataFrame | None,
    feature_ranking: pd.DataFrame | None,
    monitoring_snapshot: dict[str, Any] | None,
    figure_paths: dict[str, str] | None,
) -> str:
    """Build self-contained HTML report string."""
    monitoring_text = "<p>No monitoring snapshot.</p>"
    if monitoring_snapshot:
        alerts = monitoring_snapshot.get("alerts", [])
        drift = monitoring_snapshot.get("drift_report", {})
        runtime = monitoring_snapshot.get("runtime_report", {})
        monitoring_text = (
            f"<p><b>Alerts:</b> {len(alerts)}</p>"
            f"<pre>{pd.Series(drift.get('drifted_columns', [])).to_string(index=False)}</pre>"
            f"<p><b>Total Runtime (s):</b> {runtime.get('total_runtime_seconds', 'n/a')}</p>"
        )

    fig_html = ""
    if figure_paths:
        for name, path in figure_paths.items():
            img_src = Path(path).as_posix()
            fig_html += f"<h4>{name}</h4><img src='{img_src}' alt='{name}' style='max-width:100%;height:auto;' />"

    html = f"""
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#f7f9fb; color:#1f2937; margin:0; padding:20px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .section {{ background:#fff; border-radius:10px; padding:20px; margin-bottom:18px; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
    .metric {{ display:inline-block; min-width:150px; background:#ecf5ff; border-radius:8px; padding:12px; margin:8px; }}
    .metric-key {{ font-size:12px; color:#475569; }}
    .metric-value {{ font-size:24px; font-weight:700; color:#0f4c81; }}
    .tbl {{ width:100%; border-collapse:collapse; }}
    .tbl th, .tbl td {{ border:1px solid #e5e7eb; padding:8px; text-align:left; }}
    .tbl th {{ background:#f1f5f9; }}
  </style>
</head>
<body>
<div class='container'>
  <h1>{title}</h1>
  <p>Generated at {datetime.utcnow().isoformat()} UTC</p>

  <div class='section'>
    <h2>Summary</h2>
    <pre>{pd.Series(summary).to_string()}</pre>
  </div>

  <div class='section'>
    <h2>Evaluation Metrics</h2>
    {_metrics_cards(metrics)}
  </div>

  <div class='section'>
    <h2>Model Leaderboard</h2>
    {_fmt_table(model_leaderboard)}
  </div>

  <div class='section'>
    <h2>Champion Candidate Comparison</h2>
    {_fmt_table(candidate_scoreboard)}
  </div>

  <div class='section'>
    <h2>Feature Rankings</h2>
    {_fmt_table(feature_ranking)}
  </div>

  <div class='section'>
    <h2>Monitoring & Alerts</h2>
    {monitoring_text}
  </div>

  <div class='section'>
    <h2>Diagnostics</h2>
    {fig_html if fig_html else '<p>No figures.</p>'}
  </div>
</div>
</body>
</html>
"""
    return html


def save_report(
    report_html: str,
    output_dir: str | Path,
    filename: str = "pipeline_report.html",
) -> str:
    """Persist HTML report and return path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(report_html, encoding="utf-8")
    return str(path)
