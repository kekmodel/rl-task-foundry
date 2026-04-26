"""Command-line entrypoints for the rebuilt project skeleton."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from rl_task_foundry.config import load_config
from rl_task_foundry.infra.checkpoint import ensure_checkpoint
from rl_task_foundry.infra.db import smoke_test_connection
from rl_task_foundry.infra.storage import summarize_run
from rl_task_foundry.synthesis.bundle_exporter import TaskBundleExporter
from rl_task_foundry.synthesis.contracts import normalize_topic
from rl_task_foundry.synthesis.coverage_planner import SynthesisCoveragePlanner
from rl_task_foundry.synthesis.harvest import HarvestOutcome, HarvestRunner
from rl_task_foundry.synthesis.proof_environment import run_proof_task
from rl_task_foundry.synthesis.real_db_trial import (
    RealDbTrialRunner,
    RealDbTrialStatus,
)
from rl_task_foundry.synthesis.runner import (
    SynthesisRegistryRunner,
    load_synthesis_registry,
)
from rl_task_foundry.synthesis.task_registry import TaskRegistryWriter

app = typer.Typer(no_args_is_help=True)
console = Console()


def _solver_summary(config) -> str:
    return ", ".join(
        f"{solver.solver_id}={solver.provider}/{solver.model}" for solver in config.models.solvers
    )


@app.command("validate-config")
def validate_config(
    config_path: Path = Path("rl_task_foundry.yaml"),
    composer_provider: str | None = None,
    composer_model: str | None = None,
    solver_provider: str | None = None,
    solver_model: str | None = None,
) -> None:
    """Validate configuration and print the key source-of-truth values."""

    config = load_config(
        config_path,
        composer_provider=composer_provider,
        composer_model=composer_model,
        solver_provider=solver_provider,
        solver_model=solver_model,
    )
    console.print(f"[green]config ok[/green]: {config_path}")
    console.print(f"total_solver_runs={config.models.total_solver_runs}")
    console.print(f"run_db={config.output.run_db_path}")
    console.print(f"composer={config.models.composer.provider}/{config.models.composer.model}")
    console.print(
        "atomic_tools="
        f"max_tools={config.atomic_tools.max_tools},"
        f"bounded_result_limit={config.atomic_tools.bounded_result_limit},"
        f"max_batch_values={config.atomic_tools.max_batch_values},"
        f"float_precision={config.atomic_tools.float_precision}"
    )
    console.print(
        "synthesis_runtime="
        f"max_turns={config.synthesis.runtime.max_turns},"
        f"tracing={config.synthesis.runtime.tracing},"
        f"sdk_sessions_enabled={config.synthesis.runtime.sdk_sessions_enabled},"
        "max_generation_attempts="
        f"{config.synthesis.runtime.max_generation_attempts},"
        "max_consecutive_category_discards="
        f"{config.synthesis.runtime.max_consecutive_category_discards},"
        "category_backoff_duration_s="
        f"{config.synthesis.runtime.category_backoff_duration_s}"
    )
    console.print(f"estimated_total_db_connections={config.estimated_total_db_connections}")
    console.print(
        "dedup="
        f"exact_enabled={config.dedup.exact_enabled},"
        f"near_dup_enabled={config.dedup.near_dup_enabled},"
        f"minhash_threshold={config.dedup.minhash_threshold}"
    )
    console.print(
        "synthesis_coverage="
        "target_count_per_pair="
        f"{config.synthesis.coverage_planner.target_count_per_band}"
    )
    console.print(f"solvers={_solver_summary(config)}")


@app.command("bootstrap-run-db")
def bootstrap_run_db(config_path: Path = Path("rl_task_foundry.yaml")) -> None:
    """Create the durable run database declared in config."""

    config = load_config(config_path)
    snapshot = ensure_checkpoint(config.output.run_db_path)
    console.print(f"[green]run.db ready[/green]: {snapshot.run_db_path}")


@app.command("run-synthesis-registry")
def run_synthesis_registry(
    registry_path: Path,
    max_steps: int = 10,
    checkpoint_namespace: str = "synthesis_registry",
    config_path: Path = Path("rl_task_foundry.yaml"),
) -> None:
    """Run the synthesis multi-db registry loop for a bounded number of steps."""

    async def _run() -> None:
        config = load_config(config_path)
        registry = load_synthesis_registry(registry_path)
        if not registry:
            raise typer.BadParameter("registry file is empty")
        runner = SynthesisRegistryRunner(config)
        try:
            summary = await runner.run_steps(
                registry,
                max_steps=max_steps,
                checkpoint_namespace=checkpoint_namespace,
            )
        finally:
            await runner.close()

        console.print(f"[green]synthesis registry run complete[/green]: {registry_path}")
        console.print(f"outcome={summary.outcome}")
        console.print(f"checkpoint_namespace={summary.checkpoint_namespace}")
        console.print(f"requested_steps={summary.requested_steps}")
        console.print(f"executed_steps={summary.executed_steps}")
        console.print(f"total_entries={summary.total_entries}")
        console.print(f"initially_processed_entries={summary.initially_processed_entries}")
        console.print(f"processed_entries_after_run={summary.processed_entries_after_run}")
        console.print(f"generated_drafts={summary.generated_drafts}")
        console.print(f"quality_accepted_tasks={summary.quality_accepted_tasks}")
        console.print(f"quality_rejected_tasks={summary.quality_rejected_tasks}")
        console.print(f"registry_committed_tasks={summary.registry_committed_tasks}")
        console.print(f"registry_duplicate_tasks={summary.registry_duplicate_tasks}")
        console.print(f"remaining_entries={summary.remaining_entries}")
        if summary.flow_id is not None:
            console.print(f"flow_id={summary.flow_id}")
        if summary.phase_monitor_log_path is not None:
            console.print(f"phase_monitor_log_path={summary.phase_monitor_log_path}")
        if summary.registry_root_dir is not None:
            console.print(f"registry_root_dir={summary.registry_root_dir}")
        if summary.registry_index_db_path is not None:
            console.print(f"registry_index_db_path={summary.registry_index_db_path}")
        if summary.generated_task_ids:
            console.print(f"generated_task_ids={summary.generated_task_ids}")
        if summary.committed_task_ids:
            console.print(f"committed_task_ids={summary.committed_task_ids}")
        if summary.duplicate_task_ids:
            console.print(f"duplicate_task_ids={summary.duplicate_task_ids}")
        if summary.quality_rejected_task_ids:
            console.print(f"quality_rejected_task_ids={summary.quality_rejected_task_ids}")
        if summary.last_decision is not None:
            console.print(f"last_status={summary.last_decision.status}")
            console.print(f"last_db_id={summary.last_decision.db_id}")

    asyncio.run(_run())


@app.command("show-task-registry")
def show_task_registry(
    config_path: Path = Path("rl_task_foundry.yaml"),
    limit: int = 10,
    db_id: str | None = None,
    topic: str | None = typer.Option(None, "--topic", "--category"),
) -> None:
    """Print the current durable synthesis task registry snapshot."""

    config = load_config(config_path)
    registry = TaskRegistryWriter.for_config(config)
    resolved_topic = normalize_topic(topic) if topic is not None else None
    snapshot = registry.snapshot(limit=limit, db_id=db_id, topic=resolved_topic)
    semantic_candidates = registry.semantic_dedup_candidates(
        limit=limit,
        db_id=db_id,
        topic=resolved_topic,
    )

    console.print("[green]synthesis task registry[/green]")
    console.print(f"root_dir={registry.root_dir}")
    console.print(f"index_db_path={registry.index_db_path}")
    console.print(f"task_count={snapshot.task_count}")
    console.print(f"coverage_cells={len(snapshot.coverage)}")
    console.print(f"semantic_candidates={len(semantic_candidates)}")
    for entry in snapshot.coverage:
        console.print(
            f"coverage={entry.db_id}|{entry.topic}|{entry.count}"
        )
    for record in snapshot.recent_tasks:
        console.print(
            "task="
            f"{record.task_id}|{record.db_id}|{record.topic}"
            f"|{record.status.value}"
        )


@app.command("plan-synthesis-coverage")
def plan_synthesis_coverage(
    registry_path: Path,
    limit: int = 10,
    config_path: Path = Path("rl_task_foundry.yaml"),
) -> None:
    """Print the current registry coverage deficit plan against the db/topic inventory."""

    config = load_config(config_path)
    registry = load_synthesis_registry(registry_path)
    if not registry:
        raise typer.BadParameter("registry file is empty")
    task_registry = TaskRegistryWriter.for_config(config)
    planner = SynthesisCoveragePlanner.for_config(config)
    plan = planner.build_plan(registry, task_registry.coverage_entries())

    console.print(f"[green]synthesis coverage plan[/green]: {registry_path}")
    console.print(f"target_count_per_pair={plan.target_count_per_pair}")
    console.print(f"total_pairs={plan.total_pairs}")
    console.print(f"total_cells={plan.total_cells}")
    console.print(f"satisfied_cells={plan.satisfied_cells}")
    console.print(f"deficit_cells={plan.deficit_cells}")
    console.print(f"deficit_pairs={plan.deficit_pairs}")
    console.print(f"total_deficit={plan.total_deficit}")
    for pair in plan.pair_plans[: max(0, limit)]:
        if pair.total_deficit == 0:
            continue
        console.print(
            "pair_gap="
            f"{pair.db_id}|{pair.topic}|deficit={pair.total_deficit}"
            f"|current={pair.total_current_count}|target={pair.total_target_count}"
        )
    for cell in plan.cell_plans[: max(0, limit)]:
        if cell.deficit == 0:
            continue
        console.print(
            "cell_gap="
            f"{cell.db_id}|{cell.topic}"
            f"|current={cell.current_count}|target={cell.target_count}"
            f"|deficit={cell.deficit}"
        )


@app.command("export-bundle")
def export_bundle(
    output_dir: Path,
    config_path: Path = Path("rl_task_foundry.yaml"),
    db_id: str | None = None,
    topic: str | None = typer.Option(None, "--topic", "--category"),
    task_id: str | None = None,
) -> None:
    """Export registered task bundles into the bundle layout."""

    config = load_config(config_path)
    exporter = TaskBundleExporter.for_config(config)
    resolved_topic = normalize_topic(topic) if topic is not None else None
    summary = exporter.export_bundle(
        output_dir,
        db_id=db_id,
        topic=resolved_topic,
        task_id=task_id,
    )
    if summary.task_count == 0:
        raise typer.BadParameter("no registered tasks matched the requested filters")

    console.print(f"[green]bundle exported[/green]: {summary.bundle_root}")
    console.print(f"database_count={summary.database_count}")
    console.print(f"task_count={summary.task_count}")
    if summary.db_ids:
        console.print(f"db_ids={list(summary.db_ids)}")
    if summary.task_ids:
        console.print(f"task_ids={list(summary.task_ids)}")


@app.command("run-proof-task")
def run_proof_task_cli(
    output_dir: Path,
    config_path: Path = Path("rl_task_foundry.yaml"),
) -> None:
    """Run the synthetic proof-task vertical slice and export its bundle."""

    async def _run() -> None:
        config = load_config(config_path)
        summary = await run_proof_task(config, output_root=output_dir)

        console.print(f"[green]proof task run complete[/green]: {output_dir}")
        console.print(f"db_id={summary.db_id}")
        if summary.task_id is not None:
            console.print(f"task_id={summary.task_id}")
        if summary.quality_gate_status is not None:
            console.print(f"quality_gate_status={summary.quality_gate_status}")
        if summary.flow_id is not None:
            console.print(f"flow_id={summary.flow_id}")
        if summary.phase_monitor_log_path is not None:
            console.print(f"phase_monitor_log_path={summary.phase_monitor_log_path}")
        if summary.solver_pass_rate is not None:
            console.print(f"solver_pass_rate={summary.solver_pass_rate}")
        if summary.solver_ci_low is not None:
            console.print(f"solver_ci_low={summary.solver_ci_low}")
        if summary.solver_ci_high is not None:
            console.print(f"solver_ci_high={summary.solver_ci_high}")
        if summary.solver_matched_runs is not None:
            console.print(f"solver_matched_runs={summary.solver_matched_runs}")
        if summary.solver_planned_runs is not None:
            console.print(f"solver_planned_runs={summary.solver_planned_runs}")
        if summary.solver_completed_runs is not None:
            console.print(f"solver_completed_runs={summary.solver_completed_runs}")
        if summary.solver_evaluable_runs is not None:
            console.print(f"solver_evaluable_runs={summary.solver_evaluable_runs}")
        if summary.solver_failed_runs is not None:
            console.print(f"solver_failed_runs={summary.solver_failed_runs}")
        if summary.feedback_events:
            console.print(f"feedback_events={summary.feedback_events}")
        if summary.last_feedback_error_codes:
            console.print(
                f"last_feedback_error_codes={list(summary.last_feedback_error_codes)}"
            )
        if summary.registry_status is not None:
            console.print(f"registry_status={summary.registry_status}")
        if summary.registry_task_id is not None:
            console.print(f"registry_task_id={summary.registry_task_id}")
        if summary.bundle_root is not None:
            console.print(f"bundle_root={summary.bundle_root}")
        if summary.trial_status is not RealDbTrialStatus.ACCEPTED:
            raise typer.Exit(code=1)

    asyncio.run(_run())


@app.command("run-real-db-trial")
def run_real_db_trial(
    db_id: str,
    output_dir: Path,
    topic: str | None = None,
    config_path: Path = Path("rl_task_foundry.yaml"),
    composer_provider: str | None = None,
    composer_model: str | None = None,
    solver_provider: str | None = None,
    solver_model: str | None = None,
) -> None:
    """Run a real-database single-task trial."""

    async def _run() -> None:
        config = load_config(
            config_path,
            composer_provider=composer_provider,
            composer_model=composer_model,
            solver_provider=solver_provider,
            solver_model=solver_model,
        )
        resolved_topic = (
            normalize_topic(topic) if topic else None
        )

        runner = RealDbTrialRunner(config)
        try:
            summary = await runner.run(
                output_dir,
                db_id=db_id,
                topic=resolved_topic,
            )
        finally:
            await runner.close()

        console.print(f"[green]real db trial complete[/green]: {output_dir}")
        console.print(f"trial_status={summary.trial_status}")
        console.print(f"db_id={summary.db_id}")
        console.print(f"requested_topic={summary.requested_topic}")
        if summary.flow_id is not None:
            console.print(f"flow_id={summary.flow_id}")
        if summary.phase_monitor_log_path is not None:
            console.print(f"phase_monitor_log_path={summary.phase_monitor_log_path}")
        if summary.task_id is not None:
            console.print(f"task_id={summary.task_id}")
        if summary.quality_gate_status is not None:
            console.print(f"quality_gate_status={summary.quality_gate_status}")
        if summary.attempt_outcomes:
            console.print(f"attempt_outcomes={list(summary.attempt_outcomes)}")
        if summary.error_codes:
            console.print(f"error_codes={list(summary.error_codes)}")
        if summary.synthesis_error_type is not None:
            console.print(f"synthesis_error_type={summary.synthesis_error_type}")
        if summary.synthesis_error_message is not None:
            console.print(f"synthesis_error_message={summary.synthesis_error_message}")
        if summary.synthesis_phase is not None:
            console.print(f"synthesis_phase={summary.synthesis_phase}")
        if summary.backend_failures:
            console.print(f"backend_failures={list(summary.backend_failures)}")
        if summary.solver_pass_rate is not None:
            console.print(f"solver_pass_rate={summary.solver_pass_rate}")
        if summary.solver_ci_low is not None:
            console.print(f"solver_ci_low={summary.solver_ci_low}")
        if summary.solver_ci_high is not None:
            console.print(f"solver_ci_high={summary.solver_ci_high}")
        if summary.solver_matched_runs is not None:
            console.print(f"solver_matched_runs={summary.solver_matched_runs}")
        if summary.solver_planned_runs is not None:
            console.print(f"solver_planned_runs={summary.solver_planned_runs}")
        if summary.solver_completed_runs is not None:
            console.print(f"solver_completed_runs={summary.solver_completed_runs}")
        if summary.solver_evaluable_runs is not None:
            console.print(f"solver_evaluable_runs={summary.solver_evaluable_runs}")
        if summary.solver_failed_runs is not None:
            console.print(f"solver_failed_runs={summary.solver_failed_runs}")
        if summary.feedback_events:
            console.print(f"feedback_events={summary.feedback_events}")
        if summary.last_feedback_error_codes:
            console.print(
                f"last_feedback_error_codes={list(summary.last_feedback_error_codes)}"
            )
        if summary.registry_status is not None:
            console.print(f"registry_status={summary.registry_status}")
        if summary.registry_task_id is not None:
            console.print(f"registry_task_id={summary.registry_task_id}")
        if summary.bundle_root is not None:
            console.print(f"bundle_root={summary.bundle_root}")
        if summary.debug_root is not None:
            console.print(f"debug_root={summary.debug_root}")
        if summary.debug_traces_dir is not None:
            console.print(f"debug_traces_dir={summary.debug_traces_dir}")
        if summary.synthesis_traces_dir is not None:
            console.print(f"synthesis_traces_dir={summary.synthesis_traces_dir}")
        if summary.solver_traces_dir is not None:
            console.print(f"solver_traces_dir={summary.solver_traces_dir}")
        if summary.trial_status not in (
            RealDbTrialStatus.ACCEPTED,
            RealDbTrialStatus.REGISTRY_DUPLICATE,
        ):
            raise typer.Exit(code=1)

    asyncio.run(_run())


@app.command("harvest")
def harvest(
    db_id: str,
    output_dir: Path,
    target: int = 20,
    stall_timeout_min: float = 10.0,
    workers: int | None = None,
    config_path: Path = Path("rl_task_foundry.yaml"),
) -> None:
    """Run trials in parallel until target accepted+committed tasks reached.

    Stops early if no new commit lands within --stall-timeout-min minutes.
    """

    async def _run() -> None:
        config = load_config(config_path)
        worker_count = workers or config.synthesis.parallel_workers
        runner = HarvestRunner(config)
        try:
            summary = await runner.run(
                output_dir,
                db_id=db_id,
                target_committed=target,
                stall_timeout_seconds=stall_timeout_min * 60.0,
                parallel_workers=worker_count,
            )
        finally:
            await runner.close()

        color = "green" if summary.outcome is HarvestOutcome.TARGET_REACHED else "yellow"
        console.print(f"[{color}]harvest {summary.outcome.value}[/{color}]: {output_dir}")
        console.print(f"db_id={summary.db_id}")
        console.print(f"target_committed={summary.target_committed}")
        console.print(f"committed={summary.committed}")
        console.print(f"attempted={summary.attempted}")
        console.print(f"elapsed_seconds={summary.elapsed_seconds:.1f}")
        console.print(f"flow_id={summary.flow_id}")
        console.print(f"phase_monitor_log_path={summary.phase_monitor_log_path}")
        if summary.accepted_task_ids:
            console.print(f"accepted_task_ids={list(summary.accepted_task_ids)}")
        if summary.outcome is not HarvestOutcome.TARGET_REACHED:
            raise typer.Exit(code=1)

    asyncio.run(_run())


@app.command("check-db")
def check_db(config_path: Path = Path("rl_task_foundry.yaml")) -> None:
    """Run a small PostgreSQL connectivity smoke test."""

    config = load_config(config_path)
    info = asyncio.run(smoke_test_connection(config.database))
    console.print(
        "[green]db ok[/green]: "
        f"db={info['database_name']} user={info['user_name']} schema={info['schema_name']}"
    )


@app.command("show-layout")
def show_layout() -> None:
    """Print the high-level package layout."""

    console.print("config, infra, schema, synthesis, solver, calibration, pipeline")


@app.command("run-summary")
def run_summary(
    run_id: str,
    config_path: Path = Path("rl_task_foundry.yaml"),
) -> None:
    """Reconstruct a small run summary from run.db only."""

    config = load_config(config_path)
    summary = summarize_run(config.output.run_db_path, run_id=run_id)
    console.print(f"run_id={summary.run_id}")
    console.print(f"tasks={summary.total_tasks}")
    console.print(f"accepted={summary.accepted_tasks}")
    console.print(f"rejected={summary.rejected_tasks}")
    console.print(f"skipped={summary.skipped_tasks}")
    console.print(f"verification_results={summary.verification_results}")
    console.print(f"events={summary.event_count}")


if __name__ == "__main__":
    app()
