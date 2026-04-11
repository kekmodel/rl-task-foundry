"""Command-line entrypoints for the rebuilt project skeleton."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from rl_task_foundry.config import load_config
from rl_task_foundry.infra.checkpoint import ensure_checkpoint
from rl_task_foundry.infra.db import smoke_test_connection
from rl_task_foundry.infra.storage import summarize_run
from rl_task_foundry.pipeline.orchestrator import Orchestrator
from rl_task_foundry.pipeline.review_pack import ReviewPackBuilder
from rl_task_foundry.schema.introspect import PostgresSchemaIntrospector
from rl_task_foundry.schema.path_catalog import build_path_catalog
from rl_task_foundry.synthesis.runner import (
    SynthesisRegistryRunner,
    load_synthesis_registry,
)
from rl_task_foundry.synthesis.coverage_planner import SynthesisCoveragePlanner
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.environment_registry import EnvironmentRegistryWriter
from rl_task_foundry.synthesis.runtime_policy import build_runtime_isolation_plan
from rl_task_foundry.tasks.composer import ComposeRequest, TaskComposer
from rl_task_foundry.tasks.factory import TierATaskFactory
from rl_task_foundry.tasks.models import TaskSpec
from rl_task_foundry.tools.compiler import compile_canonical_tool_bundle, compile_path_tools
from rl_task_foundry.tools.naming_eval import evaluate_tool_bundle_naming
from rl_task_foundry.tools.model_naming import ToolNamingGenerationError, generate_named_tool_bundle
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema

app = typer.Typer(no_args_is_help=True)
console = Console()


def _solver_summary(config) -> str:
    return ", ".join(
        f"{solver.solver_id}={solver.provider}/{solver.model}x{solver.replicas}"
        for solver in config.models.solvers
    )


def _resolve_configured_tool_level(config, requested_level: int | None) -> int:
    if requested_level is None:
        return config.task_composer.selected_tool_level
    if requested_level not in {1, 2}:
        raise typer.BadParameter("tool-level must be 1 or 2")
    return requested_level


def _load_task_specs(path: Path) -> list[TaskSpec]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        payload = json.loads(raw)
        return [TaskSpec.model_validate(item) for item in payload]
    return [
        TaskSpec.model_validate(json.loads(line))
        for line in raw.splitlines()
        if line.strip()
    ]


def _default_review_pack_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("review_packs") / timestamp


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
    console.print(f"label_tier={config.task_composer.label_tier}")
    console.print(
        "selected_tool_level="
        f"{config.task_composer.selected_tool_level}"
    )
    console.print(f"float_precision={config.verification.float_precision}")
    console.print(f"shadow_sample_rate={config.verification.shadow_sample_rate}")
    console.print(f"negative_outcome_ratio={config.task_composer.negative_outcome_ratio}")
    console.print(
        "synthesis_runtime="
        f"max_turns={config.synthesis.runtime.max_turns},"
        f"tracing={config.synthesis.runtime.tracing},"
        f"sdk_sessions_enabled={config.synthesis.runtime.sdk_sessions_enabled},"
        f"memory_window={config.synthesis.runtime.explicit_memory_window},"
        "max_self_consistency_iterations="
        f"{config.synthesis.runtime.max_self_consistency_iterations},"
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
        console.print(f"registry_committed_envs={summary.registry_committed_envs}")
        console.print(f"registry_duplicate_envs={summary.registry_duplicate_envs}")
        console.print(f"remaining_pairs={summary.remaining_pairs}")
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
        "config, infra, schema, tools, tasks, truth, solver, verification, calibration, pipeline"
    )


@app.command("run")
def run_pipeline(
    task_specs_path: Path,
    config_path: Path = Path("rl_task_foundry.yaml"),
) -> None:
    """Run task specs end-to-end through solver, verifier, and run.db persistence."""

    async def _run() -> None:
        config = load_config(config_path)
        task_specs = _load_task_specs(task_specs_path)
        if not task_specs:
            raise typer.BadParameter("task spec file is empty")
        orchestrator = Orchestrator(config)
        summary = await orchestrator.run_tasks(task_specs)
        console.print(f"[green]run complete[/green]: run_id={summary.run_id}")
        console.print(f"tasks={summary.total_tasks}")
        console.print(f"accepted={summary.accepted_tasks}")
        console.print(f"rejected={summary.rejected_tasks}")
        console.print(f"skipped={summary.skipped_tasks}")
        console.print(f"verification_results={summary.verification_results}")
        console.print(f"accepted_jsonl={summary.accepted_jsonl_path}")
        console.print(f"rejected_jsonl={summary.rejected_jsonl_path}")

    asyncio.run(_run())


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


@app.command("generate-task-specs")
def generate_task_specs(
    output_path: Path,
    limit: int = 10,
    config_path: Path = Path("rl_task_foundry.yaml"),
    path_id: list[str] | None = None,
) -> None:
    """Generate Tier A task specs from schema paths and anchor samples."""

    async def _run() -> None:
        config = load_config(config_path)
        introspector = PostgresSchemaIntrospector(
            database=config.database,
            default_visibility=config.privacy.default_visibility,
            visibility_overrides=config.privacy.visibility_overrides,
        )
        graph = await introspector.introspect()
        catalog = build_path_catalog(graph, max_hops=config.tool_compiler.max_hops)
        factory = TierATaskFactory(
            database=config.database,
            domain=config.domain,
            task_config=config.task_composer,
            tool_compiler=config.tool_compiler,
            verification=config.verification,
        )
        task_specs = await factory.generate(graph, catalog, limit=limit, path_ids=path_id or None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for task in task_specs:
                handle.write(json.dumps(task.model_dump(mode="json"), ensure_ascii=False))
                handle.write("\n")
        console.print(f"[green]generated task specs[/green]: {output_path}")
        console.print(f"tasks={len(task_specs)}")
        if task_specs:
            family_counts: dict[str, int] = {}
            outcome_counts: dict[str, int] = {}
            for task in task_specs:
                family_counts[task.question_family] = family_counts.get(task.question_family, 0) + 1
                outcome_counts[task.outcome_type] = outcome_counts.get(task.outcome_type, 0) + 1
            console.print(f"question_families={family_counts}")
            console.print(f"outcomes={outcome_counts}")

    asyncio.run(_run())


@app.command("generate-review-pack")
def generate_review_pack(
    output_dir: Path | None = typer.Argument(None),
    limit: int = 10,
    config_path: Path = Path("rl_task_foundry.yaml"),
    path_id: list[str] | None = None,
    task_specs_path: Path | None = None,
) -> None:
    """Generate a human-review pack with final questions, tool sets, and answer keys."""

    async def _run() -> None:
        config = load_config(config_path)
        review_builder = ReviewPackBuilder(config)
        resolved_output_dir = output_dir or _default_review_pack_dir()
        task_specs = _load_task_specs(task_specs_path) if task_specs_path is not None else None
        entries = await review_builder.build_entries(
            limit=limit,
            path_ids=path_id or None,
            task_specs=task_specs,
        )
        jsonl_path, markdown_path = review_builder.write(resolved_output_dir, entries)
        console.print(f"[green]generated review pack[/green]: {resolved_output_dir}")
        console.print(f"entries={len(entries)}")
        console.print(f"jsonl={jsonl_path}")
        console.print(f"markdown={markdown_path}")

    asyncio.run(_run())


@app.command("preview-tool-bundle")
def preview_tool_bundle(
    path_id: str,
    tool_level: int = 1,
    config_path: Path = Path("rl_task_foundry.yaml"),
    model_generate: bool = True,
) -> None:
    """Preview one compiled tool bundle, optionally with model-generated L2 naming."""

    async def _run() -> None:
        config = load_config(config_path)
        introspector = PostgresSchemaIntrospector(
            database=config.database,
            default_visibility=config.privacy.default_visibility,
            visibility_overrides=config.privacy.visibility_overrides,
        )
        graph = await introspector.introspect()
        catalog = build_path_catalog(graph, max_hops=config.tool_compiler.max_hops)
        path = catalog.get(path_id)
        bundle = compile_path_tools(
            graph,
            path,
            tool_level=tool_level,
            label_tier=config.task_composer.label_tier,
            max_list_cardinality=config.tool_compiler.max_list_cardinality,
            allow_aggregates=config.tool_compiler.allow_aggregates,
            allow_timelines=config.tool_compiler.allow_timelines,
            float_precision=config.verification.float_precision,
            business_alias_overrides=config.tool_compiler.business_alias_overrides,
        )

        if tool_level not in {1, 2}:
            raise typer.BadParameter("tool-level must be 1 or 2")

        if model_generate and tool_level == 2:
            provider = config.providers[config.models.composer.provider]
            try:
                bundle = await generate_named_tool_bundle(
                    provider=provider,
                    model_ref=config.models.composer,
                    domain=config.domain,
                    path=path,
                    bundle=bundle,
                    temperature=config.tool_compiler.naming_temperature_l2,
                )
            except ToolNamingGenerationError as exc:
                console.print(f"[yellow]naming fallback[/yellow]: {exc}")

        console.print(
            f"path={bundle.path_id} level=L{bundle.tool_level} sources="
            f"{sorted({tool.name_source for tool in bundle.tools})}"
        )
        for tool in bundle.tools:
            console.print(
                f"- {tool.kind}: {tool.name} [{tool.name_source}] ({tool.semantic_key})"
            )

    asyncio.run(_run())


@app.command("evaluate-tool-naming")
def evaluate_tool_naming(
    path_id: str,
    tool_level: int = 2,
    config_path: Path = Path("rl_task_foundry.yaml"),
    model_generate: bool = True,
) -> None:
    """Evaluate naming quality for one compiled tool bundle."""

    async def _run() -> None:
        config = load_config(config_path)
        introspector = PostgresSchemaIntrospector(
            database=config.database,
            default_visibility=config.privacy.default_visibility,
            visibility_overrides=config.privacy.visibility_overrides,
        )
        graph = await introspector.introspect()
        catalog = build_path_catalog(graph, max_hops=config.tool_compiler.max_hops)
        path = catalog.get(path_id)
        base_bundle = compile_path_tools(
            graph,
            path,
            tool_level=tool_level,
            label_tier=config.task_composer.label_tier,
            max_list_cardinality=config.tool_compiler.max_list_cardinality,
            allow_aggregates=config.tool_compiler.allow_aggregates,
            allow_timelines=config.tool_compiler.allow_timelines,
            float_precision=config.verification.float_precision,
            business_alias_overrides=config.tool_compiler.business_alias_overrides,
        )

        bundles = [("compiled", base_bundle)]
        if tool_level not in {1, 2}:
            raise typer.BadParameter("tool-level must be 1 or 2")

        if model_generate and tool_level == 2:
            provider = config.providers[config.models.composer.provider]
            try:
                generated_bundle = await generate_named_tool_bundle(
                    provider=provider,
                    model_ref=config.models.composer,
                    domain=config.domain,
                    path=path,
                    bundle=base_bundle,
                    temperature=config.tool_compiler.naming_temperature_l2,
                )
            except ToolNamingGenerationError as exc:
                console.print(f"[yellow]naming fallback[/yellow]: {exc}")
            else:
                bundles.append(("model_generated", generated_bundle))

        for label, bundle in bundles:
            evaluation = evaluate_tool_bundle_naming(graph, path, bundle)
            console.print(
                f"[bold]{label}[/bold] path={bundle.path_id} level=L{bundle.tool_level} "
                f"sources={evaluation.name_sources}"
            )
            console.print(
                "  "
                f"opacity={evaluation.schema_opacity_score:.3f} "
                f"overlap={evaluation.raw_identifier_overlap_ratio:.3f} "
                f"duplicates={evaluation.duplicate_name_count} "
                f"invalid={evaluation.invalid_name_count}"
            )
            if evaluation.policy_violations:
                for violation in evaluation.policy_violations:
                    console.print(f"  [yellow]warning[/yellow]: {violation}")
            for check in evaluation.per_tool:
                console.print(
                    "  "
                    f"- {check.name} "
                    f"tables={check.raw_table_hits or '-'} "
                    f"columns={check.raw_column_hits or '-'}"
                )

    asyncio.run(_run())


@app.command("preview-task-package")
def preview_task_package(
    path_id: str,
    question: str,
    tool_level: int | None = None,
    question_family: str = "status_lookup",
    outcome_type: str = "answer",
    config_path: Path = Path("rl_task_foundry.yaml"),
) -> None:
    """Preview a task-aware presented tool bundle for one path/question."""

    async def _run() -> None:
        config = load_config(config_path)
        selected_tool_level = _resolve_configured_tool_level(config, tool_level)
        introspector = PostgresSchemaIntrospector(
            database=config.database,
            default_visibility=config.privacy.default_visibility,
            visibility_overrides=config.privacy.visibility_overrides,
        )
        graph = await introspector.introspect()
        catalog = build_path_catalog(graph, max_hops=config.tool_compiler.max_hops)
        path = catalog.get(path_id)
        canonical_bundle = compile_canonical_tool_bundle(
            graph,
            path,
            label_tier=config.task_composer.label_tier,
            max_list_cardinality=config.tool_compiler.max_list_cardinality,
            allow_aggregates=config.tool_compiler.allow_aggregates,
            allow_timelines=config.tool_compiler.allow_timelines,
            float_precision=config.verification.float_precision,
            business_alias_overrides=config.tool_compiler.business_alias_overrides,
        )

        fallback_presented_bundle = (
            compile_path_tools(
                graph,
                path,
                tool_level=selected_tool_level,
                label_tier=config.task_composer.label_tier,
                max_list_cardinality=config.tool_compiler.max_list_cardinality,
                allow_aggregates=config.tool_compiler.allow_aggregates,
                allow_timelines=config.tool_compiler.allow_timelines,
                float_precision=config.verification.float_precision,
                business_alias_overrides=config.tool_compiler.business_alias_overrides,
            )
            if selected_tool_level == 2
            else None
        )

        root_table = graph.get_table(path.root_table, schema_name=path.edges[0].source_schema)
        task = TaskSpec(
            task_id=f"preview::{path_id}",
            anchor_table=path.root_table,
            anchor_pk_column=root_table.primary_key[0],
            anchor_pk_value="<preview>",
            domain=config.domain.name,
            language=config.domain.language,
            label_tier=config.task_composer.label_tier,
            question_family=question_family,
            question=question,
            outcome_type=outcome_type,
            answer_schema=AnswerSchema(
                fields=[
                    AnswerField(
                        name="preview_answer",
                        type="string",
                        canonicalizer="lower_trim",
                        description="Preview-only placeholder field.",
                    )
                ]
            ),
            selected_path_id=path_id,
            required_hops=path.hop_count,
            tool_level=selected_tool_level,
            tool_bundle_id=canonical_bundle.bundle_id,
            sensitivity_policy="default",
        )
        provider = config.providers[config.models.composer.provider]
        composer = TaskComposer(
            domain=config.domain,
            provider=provider,
            model_ref=config.models.composer,
            question_temperature=config.task_composer.question_temperature,
            question_validation_temperature=config.task_composer.question_validation_temperature,
            naming_temperature_l2=config.tool_compiler.naming_temperature_l2,
        )
        package = await composer.compose(
            ComposeRequest(
                graph=graph,
                task=task,
                path=path,
                canonical_bundle=canonical_bundle,
                fallback_presented_bundle=fallback_presented_bundle,
            )
        )

        console.print(
            f"task={package.task.task_id} path={path_id} level=L{package.task.tool_level} "
            f"question_family={package.task.question_family}"
        )
        console.print(f"label_tier={package.task.label_tier}")
        console.print(
            "tool_level_config="
            f"selected={config.task_composer.selected_tool_level}"
        )
        console.print(f"available_levels={package.available_tool_levels}")
        console.print(f"question={package.task.question}")
        console.print(f"question_source={package.task.question_source}")
        console.print(
            "presented_bundle="
            f"{package.presented_tool_bundle.bundle_id} "
            f"strategy={package.presented_tool_bundle.generation_metadata.get('presentation_strategy')}"
        )
        if package.presented_tool_bundle.generation_metadata.get("generated_opacity") is not None:
            console.print(
                "generated_metrics="
                f"opacity={package.presented_tool_bundle.generation_metadata.get('generated_opacity')} "
                f"overlap={package.presented_tool_bundle.generation_metadata.get('generated_overlap')} "
                f"violations={package.presented_tool_bundle.generation_metadata.get('generated_policy_violations')}"
            )
        if package.presented_tool_bundle.generation_metadata.get("naming_generation_error"):
            console.print(
                "naming_note="
                f"{package.presented_tool_bundle.generation_metadata.get('naming_generation_error')}"
            )
        for tool in package.presented_tool_bundle.tools:
            console.print(
                f"- {tool.presentation_role}: {tool.kind} {tool.name} "
                f"[{tool.name_source}] ({tool.semantic_key})"
            )

    asyncio.run(_run())


if __name__ == "__main__":
    app()
