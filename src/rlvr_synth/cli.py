"""CLI entry point for rlvr-synth."""

from __future__ import annotations

import typer
from rich.console import Console

from rlvr_synth.config import load_config

app = typer.Typer(name="rlvr-synth", help="RLVR synthetic data generation harness")
console = Console()


@app.command()
def run(
    config: str = typer.Option("rlvr_synth.yaml", help="Path to config file"),
) -> None:
    """Run the RLVR data synthesis pipeline."""
    try:
        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Config loaded:[/green] {config}")
    console.print(f"  Database: {cfg.database.connection.split('@')[-1]}")
    console.print(f"  Domain: {cfg.domain.scenario}")
    console.print(f"  Providers: {cfg.providers.default}")
    console.print(f"  PKs: {cfg.database.pk_sample_size}")
    console.print(f"  Solvers: {cfg.solver.num_solvers}")
    console.print(f"  Label tier: {cfg.verification.label_tier}")
    console.print()
    console.print("[yellow]Pipeline execution not yet implemented (Plan 5).[/yellow]")


@app.command()
def validate_config(
    config: str = typer.Option("rlvr_synth.yaml", help="Path to config file"),
) -> None:
    """Validate the config file without running."""
    try:
        cfg = load_config(config)
        console.print(f"[green]Config valid:[/green] {config}")

        # Warn about concurrency
        total_solver_slots = 0
        for s in cfg.models.solver:
            prov_name = s.provider
            try:
                prov = cfg.providers.get_provider(prov_name)
                total_solver_slots += prov.max_concurrent
            except KeyError:
                console.print(f"[red]Warning:[/red] Provider '{prov_name}' not found")

        solver_demand = cfg.calibration.max_concurrent_solver_pks * cfg.solver.num_solvers
        if solver_demand > total_solver_slots:
            console.print(
                f"[yellow]Warning:[/yellow] solver demand ({solver_demand}) > "
                f"provider capacity ({total_solver_slots}). Consider reducing "
                f"max_concurrent_solver_pks."
            )

    except Exception as e:
        console.print(f"[red]Config invalid:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
