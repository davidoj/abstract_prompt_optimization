"""CLI entrypoints for the experiment runner."""

from __future__ import annotations

from pathlib import Path
import typer

from .runner import run_experiment

app = typer.Typer(help="Run prompt-optimization experiments on OTT-QA.")


@app.command()
def run(config: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True)) -> None:
    """Run all strategies defined in the YAML configuration file."""

    reports = run_experiment(config)
    for report in reports:
        typer.echo(
            f"Strategy: {report.name}\n"
            f"  Exact Match: {report.metrics.exact_match:.3f}\n"
            f"  Evidence Precision: {report.metrics.evidence_precision}\n"
            f"  Evidence Recall: {report.metrics.evidence_recall}\n"
            f"  JSON Valid Rate: {report.metrics.json_valid_rate:.3f}\n"
            f"  Avg Latency: {report.metrics.avg_latency}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
