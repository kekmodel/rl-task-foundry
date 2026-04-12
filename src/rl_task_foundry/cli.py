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
from rl_task_foundry.synthesis.runner import (
    SynthesisRegistryRunner,
    load_synthesis_registry,
)
from rl_task_foundry.synthesis.coverage_planner import SynthesisCoveragePlanner
from rl_task_foundry.synthesis.bundle_exporter import EnvironmentBundleExporter
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.environment_registry import EnvironmentRegistryWriter
from rl_task_foundry.synthesis.proof_environment import ProofEnvironmentRunner
from rl_task_foundry.synthesis.real_db_trial import (
    RealDbTrialRunner,
    RealDbTrialStatus,
)
from rl_task_foundry.synthesis.runtime_policy import build_runtime_isolation_plan

app = typer.Typer(no_args_is_help=True)
console = Console()


def _solver_summary(config) -> str:
    return ", ".join(
        f"{solver.solver_id}={solver.provider}/{solver.model}x{solver.replicas}"
        for solver in config.models.solvers
    )


@app.command("validate-config")
def validate_config(
    config_path: Path = Path("rl_task_foundry.yaml"),
    composer_provider: str | None = None,
    composer_model: str | None = None,
    solver_backbone_provider: str | None = None,
    solver_backbone_model: str | None = None,
    solver_provider: str | None = None,
    solver_model: str | None = None,
) -> None:
    """Validate configuration and print the key source-of-truth values."""

    config = load_config(
        config_path,
        composer_provider=composer_provider,
        composer_model=composer_model,
        solver_backbone_provider=solver_backbone_provider,
        solver_backbone_model=solver_backbone_model,
        solver_provider=solver_provider,
        solver_model=solver_model,
    )
    console.print(f"[green]config ok[/green]: {config_path}")
    console.print(f"solver_replicas={config.models.total_solver_replicas}")
    console.print(f"run_db={config.output.run_db_path}")
    console.print(
        f"composer={config.models.composer.provider}/{config.models.composer.model}"
    )
    console.print(
        "solver_backbone="
        f"{config.models.solver_backbone.provider}/{config.models.solver_backbone.model}"
    )
    console.print(
        "atomic_tools="
        f"max_tool_count={config.atomic_tools.max_tool_count},"
        f"bounded_result_limit={config.atomic_tools.bounded_result_limit},"
        f"max_batch_values={config.atomic_tools.max_batch_values}"
    )
    console.print(f"float_precision={config.verification.float_precision}")
    console.print(f"shadow_sample_rate={config.verification.shadow_sample_rate}")
    console.print(
        "synthesis_runtime="
        f"max_turns={config.synthesis.runtime.max_turns},"
        f"tracing={config.synthesis.runtime.tracing},"
        f"sdk_sessions_enabled={config.synthesis.runtime.sdk_sessions_enabled},"
        f"memory_window={config.synthesis.runtime.explicit_memory_window},"
        "max_self_consistency_iterations="
        f"{config.synthesis.runtime.max_self_consistency_iterations},"
        "max_difficulty_cranks="
        f"{config.synthesis.runtime.max_difficulty_cranks},"
        "max_consecutive_category_discards="
        f"{config.synthesis.runtime.max_consecutive_category_discards},"
        "category_backoff_duration_s="
        f"{config.synthesis.runtime.category_backoff_duration_s}"
    )
    runtime_plan = build_runtime_isolation_plan(config)
    coverage_planner = SynthesisCoveragePlanner.for_config(config)
    console.print(
        "registration_lane="
        f"mode={runtime_plan.registration_lane.worker_mode},"
        f"db_access={runtime_plan.registration_lane.db_access_strategy},"
        f"workers={runtime_plan.registration_lane.worker_count},"
        f"connections_per_worker={runtime_plan.registration_lane.connections_per_worker},"
        f"max_db_connections={runtime_plan.registration_lane.max_db_connections}"
    )
    console.print(
        "solver_lane="
        f"main_process={runtime_plan.production_solver_lane.main_process_execution},"
        f"per_tool_subprocess={runtime_plan.production_solver_lane.per_tool_subprocess_roundtrip}"
    )
    console.print(
        "registration_guards="
        f"timeout_s={runtime_plan.registration_lane.task_timeout_s},"
        f"memory_limit_mb={runtime_plan.registration_lane.memory_limit_mb},"
        f"call_count_limit={runtime_plan.registration_lane.call_count_limit}"
    )
    console.print(f"estimated_total_db_connections={runtime_plan.estimated_total_db_connections}")
    console.print(f"registration_policy_adr={runtime_plan.registration_lane.adr_path}")
    console.print(
        "dedup="
        f"exact_enabled={config.dedup.exact_enabled},"
        f"near_dup_enabled={config.dedup.near_dup_enabled},"
        f"minhash_threshold={config.dedup.minhash_threshold}"
    )
    console.print(
        "synthesis_coverage="
        "target_count_per_band="
        f"{config.synthesis.coverage_planner.target_count_per_band},"
        "include_unset_band="
        f"{config.synthesis.coverage_planner.include_unset_band},"
        "tracked_bands="
        f"{'|'.join(band.value for band in coverage_planner.tracked_bands)}"
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
        console.print(f"total_pairs={summary.total_pairs}")
        console.print(f"initially_processed_pairs={summary.initially_processed_pairs}")
        console.print(f"processed_pairs_after_run={summary.processed_pairs_after_run}")
        console.print(f"generated_drafts={summary.generated_drafts}")
        console.print(f"quality_accepted_envs={summary.quality_accepted_envs}")
        console.print(f"quality_rejected_envs={summary.quality_rejected_envs}")
        console.print(f"registry_committed_envs={summary.registry_committed_envs}")
        console.print(f"registry_duplicate_envs={summary.registry_duplicate_envs}")
        console.print(f"remaining_pairs={summary.remaining_pairs}")
        if summary.flow_id is not None:
            console.print(f"flow_id={summary.flow_id}")
        if summary.event_log_path is not None:
            console.print(f"event_log_path={summary.event_log_path}")
        if summary.registry_root_dir is not None:
            console.print(f"registry_root_dir={summary.registry_root_dir}")
        if summary.registry_index_db_path is not None:
            console.print(f"registry_index_db_path={summary.registry_index_db_path}")
        if summary.generated_env_ids:
            console.print(f"generated_env_ids={summary.generated_env_ids}")
        if summary.committed_env_ids:
            console.print(f"committed_env_ids={summary.committed_env_ids}")
        if summary.duplicate_env_ids:
            console.print(f"duplicate_env_ids={summary.duplicate_env_ids}")
        if summary.quality_rejected_env_ids:
            console.print(f"quality_rejected_env_ids={summary.quality_rejected_env_ids}")
        if summary.last_decision is not None:
            console.print(f"last_status={summary.last_decision.status}")
            console.print(f"last_db_id={summary.last_decision.db_id}")
            console.print(f"last_category={summary.last_decision.category}")

    asyncio.run(_run())


@app.command("show-synthesis-environment-registry")
def show_synthesis_environment_registry(
    config_path: Path = Path("rl_task_foundry.yaml"),
    limit: int = 10,
    db_id: str | None = None,
    category: str | None = None,
) -> None:
    """Print the current durable synthesis environment registry snapshot."""

    config = load_config(config_path)
    registry = EnvironmentRegistryWriter.for_config(config)
    resolved_category = None
    if category is not None:
        try:
            resolved_category = CategoryTaxonomy(category)
        except ValueError as exc:
            raise typer.BadParameter(f"unknown synthesis category: {category}") from exc

    snapshot = registry.snapshot(limit=limit, db_id=db_id, category=resolved_category)
    semantic_candidates = registry.semantic_dedup_candidates(
        limit=limit,
        db_id=db_id,
        category=resolved_category,
    )

    console.print("[green]synthesis environment registry[/green]")
    console.print(f"root_dir={registry.root_dir}")
    console.print(f"index_db_path={registry.index_db_path}")
    console.print(f"environment_count={snapshot.environment_count}")
    console.print(f"coverage_cells={len(snapshot.coverage)}")
    console.print(f"semantic_candidates={len(semantic_candidates)}")
    for entry in snapshot.coverage:
        console.print(
            "coverage="
            f"{entry.db_id}|{entry.category.value}|{entry.difficulty_band.value}|{entry.count}"
        )
    for record in snapshot.recent_environments:
        console.print(
            "env="
            f"{record.env_id}|{record.db_id}|{record.category.value}"
            f"|{record.difficulty_band.value}|{record.status.value}"
        )


@app.command("plan-synthesis-coverage")
def plan_synthesis_coverage(
    registry_path: Path,
    limit: int = 10,
    config_path: Path = Path("rl_task_foundry.yaml"),
) -> None:
    """Print the current registry coverage deficit plan against the db/category inventory."""

    config = load_config(config_path)
    registry = load_synthesis_registry(registry_path)
    if not registry:
        raise typer.BadParameter("registry file is empty")
    environment_registry = EnvironmentRegistryWriter.for_config(config)
    planner = SynthesisCoveragePlanner.for_config(config)
    plan = planner.build_plan(registry, environment_registry.coverage_entries())

    console.print(f"[green]synthesis coverage plan[/green]: {registry_path}")
    console.print(
        f"tracked_bands={'|'.join(band.value for band in plan.tracked_bands)}"
    )
    console.print(f"target_count_per_band={plan.target_count_per_band}")
    console.print(f"total_pairs={plan.total_pairs}")
    console.print(f"total_cells={plan.total_cells}")
    console.print(f"satisfied_cells={plan.satisfied_cells}")
    console.print(f"deficit_cells={plan.deficit_cells}")
    console.print(f"deficit_pairs={plan.deficit_pairs}")
    console.print(f"total_deficit={plan.total_deficit}")
    for pair in plan.pair_plans[: max(0, limit)]:
        if pair.total_deficit == 0:
            continue
        missing_bands = ",".join(band.value for band in pair.missing_bands) or "-"
        console.print(
            "pair_gap="
            f"{pair.db_id}|{pair.category.value}|deficit={pair.total_deficit}"
            f"|current={pair.total_current_count}|target={pair.total_target_count}"
            f"|missing_bands={missing_bands}"
        )
    for cell in plan.cell_plans[: max(0, limit)]:
        if cell.deficit == 0:
            continue
        console.print(
            "cell_gap="
            f"{cell.db_id}|{cell.category.value}|{cell.difficulty_band.value}"
            f"|current={cell.current_count}|target={cell.target_count}"
            f"|deficit={cell.deficit}"
        )


@app.command("export-bundle")
def export_bundle(
    output_dir: Path,
    config_path: Path = Path("rl_task_foundry.yaml"),
    db_id: str | None = None,
    category: str | None = None,
    env_id: str | None = None,
) -> None:
    """Export registered environments into the environment API bundle layout."""

    config = load_config(config_path)
    exporter = EnvironmentBundleExporter.for_config(config)
    resolved_category = None
    if category is not None:
        try:
            resolved_category = CategoryTaxonomy(category)
        except ValueError as exc:
            raise typer.BadParameter(f"unknown synthesis category: {category}") from exc
    summary = exporter.export_bundle(
        output_dir,
        db_id=db_id,
        category=resolved_category,
        env_id=env_id,
    )
    if summary.environment_count == 0:
        raise typer.BadParameter("no registered environments matched the requested filters")

    console.print(f"[green]bundle exported[/green]: {summary.bundle_root}")
    console.print(f"database_count={summary.database_count}")
    console.print(f"environment_count={summary.environment_count}")
    if summary.db_ids:
        console.print(f"db_ids={list(summary.db_ids)}")
    if summary.env_ids:
        console.print(f"env_ids={list(summary.env_ids)}")


@app.command("run-proof-environment")
def run_proof_environment(
    output_dir: Path,
    config_path: Path = Path("rl_task_foundry.yaml"),
) -> None:
    """Run the synthetic proof-environment vertical slice and export its bundle."""

    async def _run() -> None:
        config = load_config(config_path)
        runner = ProofEnvironmentRunner(config)
        try:
            summary = await runner.run(output_dir)
        finally:
            await runner.close()

        console.print(f"[green]proof environment run complete[/green]: {output_dir}")
        console.print(f"db_id={summary.db_id}")
        console.print(f"env_id={summary.env_id}")
        console.print(f"fixture_sql_root={summary.fixture_sql_root}")
        console.print(f"quality_gate_status={summary.quality_gate_status}")
        if summary.flow_id is not None:
            console.print(f"flow_id={summary.flow_id}")
        if summary.event_log_path is not None:
            console.print(f"event_log_path={summary.event_log_path}")
        if summary.cross_instance_error_codes:
            console.print(f"cross_instance_error_codes={list(summary.cross_instance_error_codes)}")
        if summary.solver_pass_rate is not None:
            console.print(f"solver_pass_rate={summary.solver_pass_rate}")
        if summary.solver_ci_low is not None:
            console.print(f"solver_ci_low={summary.solver_ci_low}")
        if summary.solver_ci_high is not None:
            console.print(f"solver_ci_high={summary.solver_ci_high}")
        if summary.registry_status is not None:
            console.print(f"registry_status={summary.registry_status}")
        if summary.registry_env_id is not None:
            console.print(f"registry_env_id={summary.registry_env_id}")
        if summary.bundle_root is not None:
            console.print(f"bundle_root={summary.bundle_root}")
        if summary.quality_gate_status != "accept":
            raise typer.Exit(code=1)

    asyncio.run(_run())


@app.command("run-real-db-trial")
def run_real_db_trial(
    db_id: str,
    category: str,
    output_dir: Path,
    config_path: Path = Path("rl_task_foundry.yaml"),
) -> None:
    """Run a real-database single-environment trial and persist a trial summary."""

    async def _run() -> None:
        config = load_config(config_path)
        try:
            resolved_category = CategoryTaxonomy(category)
        except ValueError as exc:
            raise typer.BadParameter(f"unknown synthesis category: {category}") from exc

        runner = RealDbTrialRunner(config)
        try:
            summary = await runner.run(
                output_dir,
                db_id=db_id,
                category=resolved_category,
            )
        finally:
            await runner.close()

        console.print(f"[green]real db trial complete[/green]: {output_dir}")
        console.print(f"trial_status={summary.trial_status}")
        console.print(f"db_id={summary.db_id}")
        console.print(f"requested_category={summary.requested_category}")
        if summary.flow_id is not None:
            console.print(f"flow_id={summary.flow_id}")
        if summary.event_log_path is not None:
            console.print(f"event_log_path={summary.event_log_path}")
        if summary.env_id is not None:
            console.print(f"env_id={summary.env_id}")
        if summary.quality_gate_status is not None:
            console.print(f"quality_gate_status={summary.quality_gate_status}")
        if summary.attempt_outcomes:
            console.print(f"attempt_outcomes={list(summary.attempt_outcomes)}")
        if summary.error_codes:
            console.print(f"error_codes={list(summary.error_codes)}")
        if summary.cross_instance_error_codes:
            console.print(f"cross_instance_error_codes={list(summary.cross_instance_error_codes)}")
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
        if summary.registry_status is not None:
            console.print(f"registry_status={summary.registry_status}")
        if summary.registry_env_id is not None:
            console.print(f"registry_env_id={summary.registry_env_id}")
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
        console.print(f"summary_path={summary.summary_path}")
        if summary.trial_status not in (
            RealDbTrialStatus.ACCEPTED,
            RealDbTrialStatus.REGISTRY_DUPLICATE,
        ):
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

    console.print(
        "config, infra, schema, synthesis, solver, calibration, pipeline"
    )


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
