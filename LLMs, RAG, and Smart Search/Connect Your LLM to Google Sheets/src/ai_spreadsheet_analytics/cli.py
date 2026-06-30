"""CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

from ai_spreadsheet_analytics.cleaning import DataCleaner
from ai_spreadsheet_analytics.config import Settings
from ai_spreadsheet_analytics.pipeline import AnalyticsPipeline
from ai_spreadsheet_analytics.schemas import CleaningStrategy

app = typer.Typer(help="AI Spreadsheet Analytics CLI")


@app.command()
def inspect(spreadsheet_id: str) -> None:
    """Inspect spreadsheet worksheets and metadata."""
    load_dotenv()
    settings = Settings()
    pipeline = AnalyticsPipeline(settings)
    details = pipeline.loader.inspect_spreadsheet(spreadsheet_id)
    for row in details:
        typer.echo(row)


@app.command("run-csv")
def run_csv(
    csv_path: Path,
    role: str = "executive",
    model: str | None = None,
    report_title: str = "AI Spreadsheet Analytics Report",
) -> None:
    """Run full pipeline on local CSV (offline development path)."""
    load_dotenv()
    settings = Settings()
    pipeline = AnalyticsPipeline(settings)

    selected_model = model or settings.ollama_primary_model
    df = pd.read_csv(csv_path)
    strategy = CleaningStrategy(missing_value_strategy="median")
    result = pipeline.run(
        dataframe=df,
        cleaning_strategy=strategy,
        role=role,
        model=selected_model,
        report_title=report_title,
    )
    typer.echo(f"Done. Reports: {result['report_artifacts']}")


@app.command("clean-csv")
def clean_csv(
    csv_path: Path,
    output_path: Path,
    missing_strategy: str = "median",
) -> None:
    """Run cleaning engine only."""
    df = pd.read_csv(csv_path)
    cleaner = DataCleaner()
    result = cleaner.clean(
        dataset_key=csv_path.stem,
        df=df,
        strategy=CleaningStrategy(missing_value_strategy=missing_strategy),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.cleaned.to_csv(output_path, index=False)
    typer.echo(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    app()
