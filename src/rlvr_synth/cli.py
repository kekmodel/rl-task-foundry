"""CLI entry point for rlvr-synth."""

from __future__ import annotations

import typer

app = typer.Typer(name="rlvr-synth", help="RLVR synthetic data generation harness")


@app.command()
def run(
    config: str = typer.Option("rlvr_synth.yaml", help="Path to config file"),
) -> None:
    """Run the RLVR data synthesis pipeline."""
    typer.echo(f"Loading config from {config}...")


@app.command()
def validate_config(
    config: str = typer.Option("rlvr_synth.yaml", help="Path to config file"),
) -> None:
    """Validate the config file without running."""
    typer.echo(f"Validating {config}...")


if __name__ == "__main__":
    app()
