import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from rl_task_foundry.cli import app
from rl_task_foundry.infra.storage import (
    bootstrap_run_db,
    connect_run_db,
    record_accepted_example,
    record_event,
    record_run,
    record_task,
    record_verification_result,
)
from rl_task_foundry.synthesis.bundle_exporter import TaskBundleExporter
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, TaskBundleStatus
from rl_task_foundry.synthesis.real_db_trial import (
    RealDbTrialStatus,
    RealDbTrialSummary,
)
from rl_task_foundry.synthesis.runner import (
    SynthesisRegistryRunOutcome,
    SynthesisRegistryRunSummary,
    SynthesisRegistryStepSummary,
)
from rl_task_foundry.synthesis.scheduler import (
    SynthesisSchedulerDecision,
    SynthesisSelectionStatus,
)
from rl_task_foundry.synthesis.task_registry import (
    SemanticDedupCandidate,
    TaskRegistryCommitStatus,
    TaskRegistryCoverageEntry,
    TaskRegistryRecord,
    TaskRegistrySnapshot,
    TaskRegistryWriter,
)
from tests.test_synthesis_task_registry import _sample_draft


def test_cli_validate_config_command():
    result = CliRunner().invoke(app, ["validate-config"])
    normalized = result.stdout.replace("\n", "")
    assert result.exit_code == 0
    assert "total_solver_runs=20" in normalized
    assert "composer=" in normalized
    assert (
        "atomic_tools=max_tools=300,bounded_result_limit=100,max_batch_values=128,float_precision=2"
    ) in normalized
    assert (
        "synthesis_runtime=max_turns=20,tracing=True,sdk_sessions_enabled=False,"
        "max_generation_attempts=5,"
        "max_consecutive_category_discards=3,category_backoff_duration_s=3600"
    ) in normalized
    assert "estimated_total_db_connections=40" in normalized
    assert "dedup=exact_enabled=True,near_dup_enabled=True,minhash_threshold=0.9" in normalized
    assert "synthesis_coverage=target_count_per_pair=3" in normalized


def test_cli_run_synthesis_registry_reports_summary(monkeypatch, tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps([{"db_id": "sakila"}]),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    @dataclass
    class _DummyRunner:
        _config: object

        async def run_steps(self, registry, *, max_steps, checkpoint_namespace):
            captured["registry"] = registry
            captured["max_steps"] = max_steps
            captured["checkpoint_namespace"] = checkpoint_namespace
            return SynthesisRegistryRunSummary(
                outcome=SynthesisRegistryRunOutcome.COMPLETED_ALL,
                checkpoint_namespace=checkpoint_namespace,
                requested_steps=max_steps,
                executed_steps=1,
                total_entries=1,
                initially_processed_entries=0,
                processed_entries_after_run=1,
                generated_drafts=1,
                quality_accepted_tasks=1,
                quality_rejected_tasks=0,
                registry_committed_tasks=1,
                registry_duplicate_tasks=0,
                remaining_entries=0,
                flow_id="flow_registry_test",
                phase_monitor_log_path=Path("artifacts/phase_monitors.jsonl"),
                generated_task_ids=["task_assignment_deadbeef"],
                committed_task_ids=["task_assignment_deadbeef"],
                duplicate_task_ids=[],
                registry_root_dir=Path("artifacts/tasks"),
                registry_index_db_path=Path("artifacts/task_registry.db"),
                steps=[
                    SynthesisRegistryStepSummary(
                        decision=SynthesisSchedulerDecision(
                            status=SynthesisSelectionStatus.READY,
                            db_id="sakila",
                            reason="selected next available db",
                        ),
                        draft_task_id="task_assignment_deadbeef",
                        draft_created_at=datetime(2026, 4, 12, tzinfo=timezone.utc),
                        registry_status=TaskRegistryCommitStatus.COMMITTED,
                        registry_task_id="task_assignment_deadbeef",
                    )
                ],
            )

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr("rl_task_foundry.cli.SynthesisRegistryRunner", _DummyRunner)

    result = CliRunner().invoke(
        app,
        [
            "run-synthesis-registry",
            str(registry_path),
            "--max-steps",
            "2",
            "--checkpoint-namespace",
            "cli_registry_test",
        ],
    )

    assert result.exit_code == 0
    assert "synthesis registry run complete" in result.stdout
    assert "outcome=completed_all" in result.stdout
    assert "checkpoint_namespace=cli_registry_test" in result.stdout
    assert "generated_drafts=1" in result.stdout
    assert "quality_accepted_tasks=1" in result.stdout
    assert "quality_rejected_tasks=0" in result.stdout
    assert "registry_committed_tasks=1" in result.stdout
    assert "registry_duplicate_tasks=0" in result.stdout
    assert "remaining_entries=0" in result.stdout
    assert "flow_id=flow_registry_test" in result.stdout
    assert "phase_monitor_log_path=artifacts/phase_monitors.jsonl" in result.stdout
    assert "last_status=ready" in result.stdout
    assert captured["max_steps"] == 2
    assert captured["checkpoint_namespace"] == "cli_registry_test"
    assert captured["closed"] is True


def test_cli_run_proof_task_reports_summary(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    async def _fake_run_proof_task(
        _config: object,
        *,
        output_root: Path,
        mirror_monitor_path: Path | None = None,
    ) -> RealDbTrialSummary:
        captured["output_root"] = output_root
        captured["mirror_monitor_path"] = mirror_monitor_path
        return RealDbTrialSummary(
            db_id="proof_trip_fixture",
            requested_topic="itinerary",
            trial_status=RealDbTrialStatus.ACCEPTED,
            flow_id="flow_proof_test",
            phase_monitor_log_path=output_root / "debug" / "phase_monitors.jsonl",
            debug_root=output_root / "debug",
            task_id="task_itinerary_fixture",
            quality_gate_status="accept",
            solver_pass_rate=0.5,
            solver_ci_low=0.2,
            solver_ci_high=0.8,
            registry_status=TaskRegistryCommitStatus.COMMITTED,
            registry_task_id="task_itinerary_fixture",
            bundle_root=output_root / "bundle",
        )

    monkeypatch.setattr("rl_task_foundry.cli.run_proof_task", _fake_run_proof_task)

    output_dir = tmp_path / "proof_output"
    result = CliRunner().invoke(app, ["run-proof-task", str(output_dir)])

    assert result.exit_code == 0
    assert "proof task run complete" in result.stdout
    assert "db_id=proof_trip_fixture" in result.stdout
    assert "task_id=task_itinerary_fixture" in result.stdout
    assert "quality_gate_status=accept" in result.stdout
    assert "flow_id=flow_proof_test" in result.stdout
    assert "phase_monitor_log_path=" in result.stdout
    assert "bundle_root=" in result.stdout
    assert captured["output_root"] == output_dir
    assert captured["mirror_monitor_path"] is None


def test_cli_run_real_db_trial_reports_summary(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    @dataclass
    class _DummyTrialRunner:
        _config: object

        def __post_init__(self) -> None:
            captured["config"] = self._config

        async def run(
            self,
            output_dir: Path,
            *,
            db_id: str,
            topic: CategoryTaxonomy,
        ) -> RealDbTrialSummary:
            captured["output_dir"] = output_dir
            captured["db_id"] = db_id
            captured["topic"] = topic
            return RealDbTrialSummary(
                db_id=db_id,
                requested_topic=topic,
                trial_status=RealDbTrialStatus.ACCEPTED,
                flow_id="flow_trial_test",
                phase_monitor_log_path=output_dir / "debug" / "phase_monitors.jsonl",
                task_id="task_real_trial",
                quality_gate_status="accept",
                synthesis_phase=None,
                backend_failures=(),
                solver_pass_rate=0.5,
                solver_ci_low=0.2,
                solver_ci_high=0.8,
                registry_status=TaskRegistryCommitStatus.COMMITTED,
                registry_task_id="task_real_trial",
                bundle_root=output_dir / "bundle",
            )

        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr("rl_task_foundry.cli.RealDbTrialRunner", _DummyTrialRunner)

    output_dir = tmp_path / "real_trial_output"
    result = CliRunner().invoke(
        app,
        [
            "run-real-db-trial",
            "sakila",
            str(output_dir),
            "--topic",
            "assignment",
            "--composer-provider",
            "opencode_zen",
            "--composer-model",
            "minimax-m2.5",
            "--solver-provider",
            "opencode_zen",
            "--solver-model",
            "minimax-m2.5",
        ],
    )

    assert result.exit_code == 0
    assert "real db trial complete" in result.stdout
    assert captured["topic"] == CategoryTaxonomy.ASSIGNMENT
    assert "trial_status=accepted" in result.stdout
    assert "db_id=sakila" in result.stdout
    assert "requested_topic=assignment" in result.stdout
    assert "flow_id=flow_trial_test" in result.stdout
    assert "task_id=task_real_trial" in result.stdout
    assert "phase_monitor_log_path=" in result.stdout
    assert captured["output_dir"] == output_dir
    assert captured["db_id"] == "sakila"
    assert captured["closed"] is True
    config = captured["config"]
    assert config.models.composer.provider == "opencode_zen"
    assert config.models.composer.model == "minimax-m2.5"
    assert {solver.model for solver in config.models.solvers} == {"minimax-m2.5"}


def test_cli_show_task_registry_reports_snapshot(monkeypatch) -> None:
    @dataclass
    class _DummyRegistry:
        root_dir: Path = Path("artifacts/tasks")
        index_db_path: Path = Path("artifacts/task_registry.db")

        def snapshot(self, *, limit, db_id=None, topic=None):
            assert limit == 5
            assert db_id == "sakila"
            assert topic == CategoryTaxonomy.ASSIGNMENT
            return TaskRegistrySnapshot(
                task_count=2,
                coverage=[
                    TaskRegistryCoverageEntry(
                        db_id="sakila",
                        category=CategoryTaxonomy.ASSIGNMENT,
                        count=2,
                    )
                ],
                recent_tasks=[
                    TaskRegistryRecord(
                        task_id="task_assignment_deadbeef",
                        db_id="sakila",
                        domain="service_operations",
                        category=CategoryTaxonomy.ASSIGNMENT,
                        created_at=datetime(2026, 4, 12, tzinfo=timezone.utc),
                        status=TaskBundleStatus.DRAFT,
                        generator_version="milestone-test",
                        exact_signature="sha256:deadbeef",
                        filesystem_path=Path("artifacts/tasks/task_assignment_deadbeef"),
                        question="내 배정을 해줘",
                    )
                ],
            )

        def semantic_dedup_candidates(self, *, limit, db_id=None, topic=None):
            assert limit == 5
            assert db_id == "sakila"
            assert topic == CategoryTaxonomy.ASSIGNMENT
            return [
                SemanticDedupCandidate(
                    task_id="task_assignment_deadbeef",
                    db_id="sakila",
                    domain="service_operations",
                    category=CategoryTaxonomy.ASSIGNMENT,
                    question="내 배정을 해줘",
                    constraint_summaries=("같은 고객을 중복 배정하지 않는다.",),
                    semantic_text="question:내 배정을 해줘",
                    filesystem_path=Path("artifacts/tasks/task_assignment_deadbeef"),
                )
            ]

    monkeypatch.setattr(
        "rl_task_foundry.cli.TaskRegistryWriter.for_config",
        lambda _config: _DummyRegistry(),
    )

    result = CliRunner().invoke(
        app,
        [
            "show-task-registry",
            "--limit",
            "5",
            "--db-id",
            "sakila",
            "--category",
            "assignment",
        ],
    )

    assert result.exit_code == 0
    assert "synthesis task registry" in result.stdout
    assert "task_count=2" in result.stdout
    assert "coverage_cells=1" in result.stdout
    assert "semantic_candidates=1" in result.stdout
    assert "coverage=sakila|assignment|2" in result.stdout
    assert "task=task_assignment_deadbeef|sakila|assignment|draft" in result.stdout


def test_cli_plan_synthesis_coverage_reports_deficits(monkeypatch, tmp_path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps([{"db_id": "sakila"}, {"db_id": "northwind"}]),
        encoding="utf-8",
    )

    @dataclass
    class _DummyRegistry:
        def coverage_entries(self, *, db_id=None, topic=None):
            assert db_id is None
            assert topic is None
            return [
                TaskRegistryCoverageEntry(
                    db_id="sakila",
                    category=CategoryTaxonomy.ASSIGNMENT,
                    count=2,
                )
            ]

    monkeypatch.setattr(
        "rl_task_foundry.cli.TaskRegistryWriter.for_config",
        lambda _config: _DummyRegistry(),
    )

    result = CliRunner().invoke(
        app,
        ["plan-synthesis-coverage", str(registry_path), "--limit", "10"],
    )

    assert result.exit_code == 0
    assert "synthesis coverage plan" in result.stdout
    assert "target_count_per_pair=3" in result.stdout
    assert "total_pairs=2" in result.stdout
    assert "deficit_cells=2" in result.stdout
    assert "deficit_pairs=2" in result.stdout
    assert "total_deficit=4" in result.stdout
    assert "pair_gap=northwind|northwind|deficit=3" in result.stdout
    assert "cell_gap=sakila|sakila|current=2|target=3|deficit=1" in result.stdout


def test_cli_export_bundle_writes_task_layout(monkeypatch, tmp_path: Path) -> None:
    from rl_task_foundry.schema.graph import (
        ColumnProfile,
        SchemaGraph,
        TableProfile,
    )
    from rl_task_foundry.tooling.common import snapshot_from_graph

    writer = TaskRegistryWriter(
        root_dir=tmp_path / "registry" / "tasks",
        index_db_path=tmp_path / "registry" / "task_registry.db",
    )
    draft = _sample_draft()
    writer.commit_draft(draft)
    assert writer.snapshot_materializer is not None
    writer.snapshot_materializer.materialize(
        db_id=draft.db_id,
        snapshot=snapshot_from_graph(
            SchemaGraph(
                tables=[
                    TableProfile(
                        schema_name="public",
                        table_name="customer",
                        columns=[
                            ColumnProfile(
                                schema_name="public",
                                table_name="customer",
                                column_name="customer_id",
                                data_type="integer",
                                ordinal_position=1,
                                is_nullable=False,
                                visibility="user_visible",
                                is_primary_key=True,
                            )
                        ],
                        primary_key=("customer_id",),
                    )
                ],
                edges=[],
            )
        ),
    )
    exporter = TaskBundleExporter(
        registry=writer,
        snapshot_materializer=writer.snapshot_materializer,
    )

    monkeypatch.setattr(
        "rl_task_foundry.cli.TaskBundleExporter.for_config",
        lambda _config: exporter,
    )

    output_dir = tmp_path / "bundle"
    result = CliRunner().invoke(app, ["export-bundle", str(output_dir)])

    assert result.exit_code == 0
    assert "bundle exported" in result.stdout
    assert "database_count=1" in result.stdout
    assert "task_count=1" in result.stdout
    assert (output_dir / "databases" / "sakila" / "schema_snapshot.json").exists()
    assert (output_dir / "databases" / "sakila" / "tooling_version.json").exists()
    task_dir = output_dir / "tasks" / "task_assignment_registrytest"
    assert (task_dir / "task.yaml").exists()
    assert (task_dir / "instance.json").exists()
    assert (task_dir / "canonical_answer.json").exists()


def test_cli_validate_config_applies_runtime_overrides():
    result = CliRunner().invoke(
        app,
        [
            "validate-config",
            "--composer-provider",
            "local_server",
            "--solver-provider",
            "local_server",
            "--solver-model",
            "local-gpt",
        ],
    )
    assert result.exit_code == 0
    assert "composer=local_server/" in result.stdout
    assert "solvers=" in result.stdout
    assert "local_server/local-gpt" in result.stdout


def test_cli_run_summary_reads_run_db(tmp_path):
    config_path = tmp_path / "config.yaml"
    run_db_path = tmp_path / "artifacts" / "run.db"
    config_path.write_text(
        Path("rl_task_foundry.yaml")
        .read_text(encoding="utf-8")
        .replace("./artifacts/run.db", str(run_db_path))
        .replace("./artifacts/accepted.jsonl", str(tmp_path / "artifacts" / "accepted.jsonl"))
        .replace("./artifacts/rejected.jsonl", str(tmp_path / "artifacts" / "rejected.jsonl"))
        .replace("./artifacts/events.jsonl", str(tmp_path / "artifacts" / "events.jsonl"))
        .replace("./artifacts/traces", str(tmp_path / "artifacts" / "traces")),
        encoding="utf-8",
    )

    bootstrap_run_db(run_db_path)
    with connect_run_db(run_db_path) as conn:
        record_run(
            conn, run_id="run_test_456", config_hash="abc", created_at="2026-04-11T00:00:00+00:00"
        )
        record_task(
            conn,
            run_id="run_test_456",
            task_id="task_1",
            status="accepted",
            payload={"task_id": "task_1"},
        )
        record_task(
            conn,
            run_id="run_test_456",
            task_id="task_2",
            status="rejected",
            payload={"task_id": "task_2"},
        )
        record_verification_result(
            conn,
            run_id="run_test_456",
            task_id="task_1",
            solver_id="solver_a",
            payload={"pass_exact": True},
        )
        record_verification_result(
            conn,
            run_id="run_test_456",
            task_id="task_2",
            solver_id="solver_a",
            payload={"pass_exact": False},
        )
        record_accepted_example(
            conn,
            run_id="run_test_456",
            task_id="task_1",
            payload={"task_id": "task_1"},
        )
        record_event(
            conn, run_id="run_test_456", event_type="run_started", payload={"task_count": 2}
        )
        conn.commit()

    result = CliRunner().invoke(
        app,
        ["run-summary", "run_test_456", "--config-path", str(config_path)],
    )

    assert result.exit_code == 0
    assert "run_id=run_test_456" in result.stdout
    assert "tasks=2" in result.stdout
    assert "accepted=1" in result.stdout
    assert "rejected=1" in result.stdout
    assert "skipped=0" in result.stdout
