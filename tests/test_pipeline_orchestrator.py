from __future__ import annotations

import asyncio
import json

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.schema.graph import ColumnProfile, ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.schema.path_catalog import build_path_catalog
from rl_task_foundry.infra.storage import connect_run_db, summarize_run
from rl_task_foundry.pipeline.orchestrator import Orchestrator, ReviewArtifact, TaskExecutionOutcome
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.tasks.models import (
    PresentedToolBundle,
    PresentedToolSpec,
    TaskPackage,
    TaskSpec,
    VerifyResult,
)
from rl_task_foundry.tools.compiler import compile_canonical_tool_bundle
from rl_task_foundry.truth.schemas import AnswerField, AnswerSchema, GroundTruth


def _config(tmp_path):
    config = load_config("rl_task_foundry.yaml")
    output = config.output.model_copy(
        update={
            "run_db_path": tmp_path / "artifacts" / "run.db",
            "accepted_jsonl_path": tmp_path / "artifacts" / "accepted.jsonl",
            "rejected_jsonl_path": tmp_path / "artifacts" / "rejected.jsonl",
            "events_jsonl_path": tmp_path / "artifacts" / "events.jsonl",
            "traces_dir": tmp_path / "artifacts" / "traces",
        }
    )
    return config.model_copy(update={"output": output})


def _task(task_id: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        anchor_table="customer",
        anchor_pk_column="customer_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question_family="status_lookup",
        question="현재 고객의 주소 기준 도시 정보를 확인해줘",
        outcome_type="answer",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="city",
                    type="string",
                    canonicalizer="lower_trim",
                    source_columns=["city.city"],
                )
            ]
        ),
        selected_path_id="customer.address.city",
        required_hops=2,
        tool_level=1,
        tool_bundle_id="customer.address.city.L1",
        sensitivity_policy="default",
    )


def _package(task: TaskSpec) -> TaskPackage:
    presented_bundle = PresentedToolBundle(
        bundle_id=f"{task.tool_bundle_id}::task::{task.task_id}",
        canonical_bundle_id=task.tool_bundle_id,
        path_id=task.selected_path_id,
        tool_level=task.tool_level,
        question_family=task.question_family,
        outcome_type=task.outcome_type,
        generation_metadata={"presentation_strategy": "canonical_rule_based"},
    )
    return TaskPackage(
        task=task.model_copy(update={"presented_tool_bundle_id": presented_bundle.bundle_id}),
        presented_tool_bundle=presented_bundle,
        presentation_options=[presented_bundle],
    )


def _solver_result(task_id: str, *, solver_id: str, replica_index: int, answer: str) -> SolverResult:
    return SolverResult(
        task_id=task_id,
        solver_id=solver_id,
        provider="codex_oauth",
        model="gpt-5.4-mini",
        replica_index=replica_index,
        transcript_ref="memory://transcript",
        tool_trace_ref="memory://tool-trace",
        raw_output_text=answer,
        structured_output={"city": answer},
        status="completed",
        termination_reason="submitted",
        turn_count=2,
    )


def _verification(task_id: str, *, solver_id: str, passed: bool) -> VerifyResult:
    return VerifyResult(
        task_id=task_id,
        solver_id=solver_id,
        pass_exact=passed,
        field_scores={"city": passed},
        provenance_pass=True,
        canonical_prediction={"city": "sasebo" if passed else "other"},
        failure_reason=None if passed else "incorrect_answer",
    )


def test_orchestrator_persists_run_outputs(monkeypatch, tmp_path):
    config = _config(tmp_path)
    accepted_task = _task("task_accept")
    rejected_task = _task("task_reject")

    async def _fake_execute_task(self, task: TaskSpec) -> TaskExecutionOutcome:
        package = _package(task)
        truth = GroundTruth(
            task_id=task.task_id,
            verification_sql="SELECT 1",
            canonical_answer={"city": "sasebo"},
            answer_schema_version="v1",
        )
        solver_results = [
            _solver_result(task.task_id, solver_id="solver_a", replica_index=0, answer="sasebo"),
            _solver_result(task.task_id, solver_id="solver_b", replica_index=1, answer="other"),
        ]
        if task.task_id == "task_accept":
            verification_results = [
                _verification(task.task_id, solver_id="solver_a", passed=True),
                _verification(task.task_id, solver_id="solver_b", passed=True),
            ]
            pass_rate = 1.0
            accepted = True
        else:
            verification_results = [
                _verification(task.task_id, solver_id="solver_a", passed=False),
                _verification(task.task_id, solver_id="solver_b", passed=False),
            ]
            pass_rate = 0.0
            accepted = False
        return TaskExecutionOutcome(
            task=package.task,
            package=package,
            ground_truth=truth,
            solver_results=solver_results,
            verification_results=verification_results,
            pass_rate=pass_rate,
            accepted=accepted,
            calibration_band=(0.2, 0.8),
            calibration_decision="accept" if accepted else "reject_too_hard",
            executed_solver_replicas=len(solver_results),
            planned_solver_replicas=len(solver_results),
        )

    monkeypatch.setattr(Orchestrator, "_execute_task", _fake_execute_task)

    summary = asyncio.run(Orchestrator(config).run_tasks([accepted_task, rejected_task]))

    assert summary.total_tasks == 2
    assert summary.accepted_tasks == 1
    assert summary.rejected_tasks == 1
    assert summary.skipped_tasks == 0
    assert summary.verification_results == 4

    accepted_lines = config.output.accepted_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    rejected_lines = config.output.rejected_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(accepted_lines) == 1
    assert len(rejected_lines) == 1

    accepted_payload = json.loads(accepted_lines[0])
    assert accepted_payload["training_metadata"]["mean_correct_solver_turns_rounded"] == 2
    assert accepted_payload["task"]["question_source"] == "seed_rule_based"
    assert accepted_payload["task"]["question_generation_metadata"] == {}
    assert accepted_payload["export_payload"]["question_source"] == "seed_rule_based"
    assert accepted_payload["export_payload"]["question_generation_metadata"] == {}

    run_summary = summarize_run(config.output.run_db_path, run_id=summary.run_id)
    assert run_summary.total_tasks == 2
    assert run_summary.accepted_tasks == 1
    assert run_summary.rejected_tasks == 1
    assert run_summary.skipped_tasks == 0
    assert run_summary.verification_results == 4
    assert run_summary.event_count == 4

    with connect_run_db(config.output.run_db_path) as conn:
        task_rows = conn.execute(
            "SELECT run_id, status FROM tasks ORDER BY task_id"
        ).fetchall()
    assert [tuple(row) for row in task_rows] == [
        (summary.run_id, "accepted"),
        (summary.run_id, "rejected"),
    ]


def test_orchestrator_skips_checkpointed_tasks(monkeypatch, tmp_path):
    config = _config(tmp_path)
    task = _task("task_checkpoint")
    call_counter = {"count": 0}

    async def _fake_execute_task(self, task: TaskSpec) -> TaskExecutionOutcome:
        call_counter["count"] += 1
        package = _package(task)
        truth = GroundTruth(
            task_id=task.task_id,
            verification_sql="SELECT 1",
            canonical_answer={"city": "sasebo"},
            answer_schema_version="v1",
        )
        verification_results = [_verification(task.task_id, solver_id="solver_a", passed=True)]
        solver_results = [_solver_result(task.task_id, solver_id="solver_a", replica_index=0, answer="sasebo")]
        return TaskExecutionOutcome(
            task=package.task,
            package=package,
            ground_truth=truth,
            solver_results=solver_results,
            verification_results=verification_results,
            pass_rate=1.0,
            accepted=True,
            calibration_band=(0.2, 0.8),
            calibration_decision="accept",
            executed_solver_replicas=len(solver_results),
            planned_solver_replicas=len(solver_results),
        )

    monkeypatch.setattr(Orchestrator, "_execute_task", _fake_execute_task)

    first_summary = asyncio.run(Orchestrator(config).run_tasks([task]))
    second_summary = asyncio.run(Orchestrator(config).run_tasks([task]))

    assert first_summary.accepted_tasks == 1
    assert second_summary.accepted_tasks == 0
    assert second_summary.skipped_tasks == 1
    assert call_counter["count"] == 1

    run_summary = summarize_run(config.output.run_db_path, run_id=second_summary.run_id)
    assert run_summary.skipped_tasks == 1


def test_orchestrator_blocks_when_budget_reservation_fails(monkeypatch, tmp_path):
    config = _config(tmp_path).model_copy(
        update={
            "budget": _config(tmp_path).budget.model_copy(
                update={
                    "max_run_usd": 1.0,
                    "compose_phase_usd": 4.0,
                    "solve_phase_usd": 4.0,
                }
            )
        }
    )
    task = _task("task_budget")

    async def _fail_if_called(self, task: TaskSpec) -> TaskExecutionOutcome:
        raise AssertionError("task execution should not start when reservation fails")

    monkeypatch.setattr(Orchestrator, "_execute_task", _fail_if_called)

    summary = asyncio.run(Orchestrator(config).run_tasks([task]))

    assert summary.total_tasks == 1
    assert summary.accepted_tasks == 0
    assert summary.rejected_tasks == 0
    assert summary.skipped_tasks == 1

    run_summary = summarize_run(config.output.run_db_path, run_id=summary.run_id)
    assert run_summary.skipped_tasks == 1

    with connect_run_db(config.output.run_db_path) as conn:
        row = conn.execute(
            "SELECT status FROM tasks WHERE run_id = ? AND task_id = ?",
            (summary.run_id, task.task_id),
        ).fetchone()
    assert row["status"] == "skipped_budget"


def test_orchestrator_skips_seed_fallback_tasks(monkeypatch, tmp_path):
    config = _config(tmp_path)
    task = _task("task_seed_fallback")

    async def _fake_execute_task(self, task: TaskSpec) -> TaskExecutionOutcome:
        package_task = task.model_copy(
            update={
                "question_source": "seed_fallback",
                "question_generation_metadata": {"status": "fallback"},
            }
        )
        package = _package(package_task)
        truth = GroundTruth(
            task_id=task.task_id,
            verification_sql="SELECT 1",
            canonical_answer={"city": "sasebo"},
            answer_schema_version="v1",
        )
        return TaskExecutionOutcome(
            task=package.task,
            package=package,
            ground_truth=truth,
            solver_results=[],
            verification_results=[],
            pass_rate=0.0,
            accepted=False,
            calibration_band=(0.2, 0.8),
            calibration_decision="skipped_seed_fallback",
            executed_solver_replicas=0,
            planned_solver_replicas=0,
        )

    monkeypatch.setattr(Orchestrator, "_execute_task", _fake_execute_task)

    summary = asyncio.run(Orchestrator(config).run_tasks([task]))

    assert summary.total_tasks == 1
    assert summary.accepted_tasks == 0
    assert summary.rejected_tasks == 0
    assert summary.skipped_tasks == 1

    run_summary = summarize_run(config.output.run_db_path, run_id=summary.run_id)
    assert run_summary.skipped_tasks == 1

    with connect_run_db(config.output.run_db_path) as conn:
        row = conn.execute(
            "SELECT status, payload_json FROM tasks WHERE run_id = ? AND task_id = ?",
            (summary.run_id, task.task_id),
        ).fetchone()
    assert row["status"] == "skipped_seed_fallback"
    payload = json.loads(row["payload_json"])
    assert payload["question_source"] == "seed_fallback"


@pytest.mark.asyncio
async def test_execute_task_attempt_stops_before_solver_for_seed_fallback(monkeypatch, tmp_path):
    config = _config(tmp_path)
    orchestrator = Orchestrator(config)
    task = _task("task_seed_gate")
    graph = SchemaGraph(tables=[], edges=[])
    catalog = build_path_catalog(graph, max_hops=1)
    package_task = task.model_copy(
        update={
            "question_source": "seed_fallback",
            "question_generation_metadata": {"status": "fallback"},
        }
    )
    package = _package(package_task)
    truth = GroundTruth(
        task_id=task.task_id,
        verification_sql="SELECT 1",
        canonical_answer={"city": "sasebo"},
        answer_schema_version="v1",
    )

    async def _fake_build_review_artifact(self, task, *, graph=None, catalog=None):
        del task, graph, catalog
        return ReviewArtifact(
            task=package_task,
            path=type("P", (), {"path_id": package_task.selected_path_id})(),
            package=package,
            canonical_bundle=None,  # type: ignore[arg-type]
            ground_truth=truth,
            question_context={},
        )

    async def _fail_if_solver_runs(*_args, **_kwargs):
        raise AssertionError("solver execution should not run for seed_fallback tasks")

    monkeypatch.setattr(Orchestrator, "build_review_artifact", _fake_build_review_artifact)
    monkeypatch.setattr(Orchestrator, "_run_calibrated_solvers", _fail_if_solver_runs)

    outcome = await orchestrator._execute_task_attempt(task, graph, catalog)

    assert outcome.calibration_decision == "skipped_seed_fallback"
    assert outcome.executed_solver_replicas == 0
    assert outcome.planned_solver_replicas == 0


def test_task_difficulty_score_uses_configured_family_and_outcome_order(tmp_path):
    config = _config(tmp_path).model_copy(
        update={
            "calibration": _config(tmp_path).calibration.model_copy(
                update={
                    "difficulty_family_order": {
                        "status_lookup": 5,
                        "aggregate_verification": 0,
                    },
                    "difficulty_outcome_order": {
                        "answer": 7,
                        "no_result": 0,
                    },
                }
            )
        }
    )
    orchestrator = Orchestrator(config)
    answer_task = _task("task_rank_answer")
    no_result_task = answer_task.model_copy(
        update={
            "task_id": "task_rank_no_result",
            "question_family": "aggregate_verification",
            "outcome_type": "no_result",
        }
    )

    assert orchestrator._task_difficulty_score(answer_task) > orchestrator._task_difficulty_score(
        no_result_task
    )


def test_orchestrator_runs_tasks_with_rolling_inflight_execution(monkeypatch, tmp_path):
    config = _config(tmp_path)
    tasks = [_task("task_parallel_1"), _task("task_parallel_2")]
    first_started = asyncio.Event()
    release = asyncio.Event()
    started_count = {"value": 0}

    async def _fake_execute_task(self, task: TaskSpec) -> TaskExecutionOutcome:
        started_count["value"] += 1
        first_started.set()
        await release.wait()
        package = _package(task)
        truth = GroundTruth(
            task_id=task.task_id,
            verification_sql="SELECT 1",
            canonical_answer={"city": "sasebo"},
            answer_schema_version="v1",
        )
        return TaskExecutionOutcome(
            task=package.task,
            package=package,
            ground_truth=truth,
            solver_results=[_solver_result(task.task_id, solver_id="solver_a", replica_index=0, answer="sasebo")],
            verification_results=[_verification(task.task_id, solver_id="solver_a", passed=True)],
            pass_rate=1.0,
            accepted=True,
            calibration_band=(0.2, 0.8),
            calibration_decision="accept",
            executed_solver_replicas=1,
            planned_solver_replicas=1,
        )

    monkeypatch.setattr(Orchestrator, "_execute_task", _fake_execute_task)

    async def _run() -> int:
        orchestrator_task = asyncio.create_task(Orchestrator(config).run_tasks(tasks))
        await first_started.wait()
        await asyncio.sleep(0.05)
        concurrent_starts = started_count["value"]
        release.set()
        await orchestrator_task
        return concurrent_starts

    concurrent_starts = asyncio.run(_run())
    assert concurrent_starts == 2


def test_run_calibrated_solvers_stops_before_full_limit_on_hard_task(monkeypatch, tmp_path):
    config = _config(tmp_path).model_copy(
        update={
            "calibration": _config(tmp_path).calibration.model_copy(
                update={
                    "canary_replica_count": 2,
                    "post_canary_batch_size": 2,
                    "full_replica_limit": 6,
                }
            )
        }
    )
    orchestrator = Orchestrator(config)
    task = _task("task_hard")
    truth = GroundTruth(
        task_id=task.task_id,
        verification_sql="SELECT 1",
        canonical_answer={"city": "sasebo"},
        answer_schema_version="v1",
    )

    solver_cfg = config.models.solvers[0]
    provider_cfg = config.providers[solver_cfg.provider]
    assignments = [(solver_cfg, provider_cfg, index) for index in range(6)]

    monkeypatch.setattr(Orchestrator, "_replica_assignments", lambda self: assignments)

    batch_sizes: list[int] = []

    async def _fake_run_batch(self, _task, batch_assignments, _tool_specs, _tool_executors):
        batch_sizes.append(len(batch_assignments))
        return [
            _solver_result(task.task_id, solver_id=f"solver_{replica_index}", replica_index=replica_index, answer="fail")
            for _, _, replica_index in batch_assignments
        ]

    monkeypatch.setattr(Orchestrator, "_run_solver_assignment_batch", _fake_run_batch)
    monkeypatch.setattr(
        Orchestrator,
        "_verify",
        lambda self, _task, _truth, solver_results: [
            _verification(_task.task_id, solver_id=result.solver_id, passed=False)
            for result in solver_results
        ],
    )

    solver_results, verification_results, decision, planned = asyncio.run(
        orchestrator._run_calibrated_solvers(task, truth, [], {})
    )

    assert planned == 6
    assert batch_sizes == [2, 2, 2]
    assert len(solver_results) == 6
    assert len(verification_results) == 6
    assert decision == "reject_too_hard"


def test_run_solver_replica_converts_provider_errors_and_opens_circuit(tmp_path):
    config = _config(tmp_path)
    orchestrator = Orchestrator(config)
    task = _task("task_provider_error")
    solver_cfg = config.models.solvers[0]
    provider_cfg = config.providers[solver_cfg.provider]
    call_counter = {"count": 0}

    class _FailingRuntime:
        async def run(self, _task: TaskSpec, *, replica_index: int) -> SolverResult:
            call_counter["count"] += 1
            raise TimeoutError(f"timed out replica={replica_index}")

    async def _run_all() -> tuple[SolverResult, SolverResult, SolverResult]:
        semaphore = asyncio.Semaphore(1)
        first = await orchestrator._run_solver_replica(
            runtime=_FailingRuntime(),
            solver_config=solver_cfg,
            provider_config=provider_cfg,
            provider_semaphore=semaphore,
            task=task,
            replica_index=0,
        )
        second = await orchestrator._run_solver_replica(
            runtime=_FailingRuntime(),
            solver_config=solver_cfg,
            provider_config=provider_cfg,
            provider_semaphore=semaphore,
            task=task,
            replica_index=1,
        )
        third = await orchestrator._run_solver_replica(
            runtime=_FailingRuntime(),
            solver_config=solver_cfg,
            provider_config=provider_cfg,
            provider_semaphore=semaphore,
            task=task,
            replica_index=2,
        )
        return first, second, third

    first, second, third = asyncio.run(_run_all())

    assert first.status == "provider_timeout"
    assert second.status == "provider_timeout"
    assert third.status == "provider_unavailable"
    assert third.termination_reason == "provider_circuit_open"
    assert call_counter["count"] == 2


def test_rebalance_assignments_moves_quota_to_healthy_provider(tmp_path):
    base_config = _config(tmp_path)
    primary_solver = base_config.models.solvers[0].model_copy(
        update={"solver_id": "primary_openai", "provider": "codex_oauth", "replicas": 2}
    )
    backup_solver = base_config.models.solvers[0].model_copy(
        update={"solver_id": "backup_local", "provider": "local_server", "replicas": 1}
    )
    config = base_config.model_copy(
        update={
            "models": base_config.models.model_copy(
                update={"solvers": [primary_solver, backup_solver]}
            )
        }
    )
    orchestrator = Orchestrator(config)
    orchestrator._provider_breakers()["codex_oauth"].force_open(cooldown_s=60)

    assignments = orchestrator._replica_assignments()
    next_replica_indices = orchestrator._next_replica_indices()
    rebalanced = orchestrator._rebalance_assignments(
        [assignments[0]],
        next_replica_indices=next_replica_indices,
    )

    assert len(rebalanced) == 1
    assert rebalanced[0][0].provider == "local_server"
    assert rebalanced[0][0].solver_id == "backup_local"
    assert rebalanced[0][2] == 1


@pytest.mark.parametrize("question_family", ["status_lookup", "causal_chain", "timeline_resolution"])
def test_execute_task_retries_with_harder_compatible_path(monkeypatch, tmp_path, question_family):
    config = _config(tmp_path).model_copy(
        update={
            "task_composer": _config(tmp_path).task_composer.model_copy(
                update={
                    "max_attempts_per_anchor": 3,
                    "family_min_required_hops": {
                        "status_lookup": 1,
                        "aggregate_verification": 1,
                    },
                }
            )
        }
    )
    orchestrator = Orchestrator(config)
    graph = SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="orders",
                primary_key=("order_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="order_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="address_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="customer_id",
                        data_type="int4",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                ],
            ),
            TableProfile(
                schema_name="public",
                table_name="customers",
                primary_key=("customer_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="customer_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="customers",
                        column_name="address_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                ],
            ),
            TableProfile(
                schema_name="public",
                table_name="addresses",
                primary_key=("address_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="addresses",
                        column_name="address_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="addresses",
                        column_name="city",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_address_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("address_id",),
                target_schema="public",
                target_table="addresses",
                target_columns=("address_id",),
                source_is_unique=False,
                fanout_estimate=1.0,
            ),
            ForeignKeyEdge(
                constraint_name="orders_customer_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("customer_id",),
                target_schema="public",
                target_table="customers",
                target_columns=("customer_id",),
                source_is_unique=False,
                fanout_estimate=2.0,
            ),
            ForeignKeyEdge(
                constraint_name="customers_address_fk",
                source_schema="public",
                source_table="customers",
                source_columns=("address_id",),
                target_schema="public",
                target_table="addresses",
                target_columns=("address_id",),
                source_is_unique=False,
                fanout_estimate=2.0,
            ),
        ],
    )
    orchestrator._graph = graph
    orchestrator._catalog = build_path_catalog(graph, max_hops=3)

    task = TaskSpec(
        task_id="task_adaptive",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question_family=question_family,
        question="현재 주문의 배송 도시를 확인해줘",
        outcome_type="answer",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="city",
                    type="string",
                    canonicalizer="lower_trim",
                    source_columns=["addresses.city"],
                )
            ]
        ),
        selected_path_id="orders.addresses",
        required_hops=1,
        tool_level=1,
        tool_bundle_id="orders.addresses.L1",
        sensitivity_policy="default",
    )
    attempted_paths: list[str] = []

    class _FakeGenerator:
        async def generate(self, task: TaskSpec) -> GroundTruth:
            return GroundTruth(
                task_id=task.task_id,
                verification_sql="SELECT 1",
                canonical_answer={"city": "sasebo"},
                answer_schema_version="v1",
            )

    def _fake_ground_truth_generator(self, _graph, _catalog):
        return _FakeGenerator()

    async def _fake_execute_task_attempt(self, task: TaskSpec, _graph, _catalog):
        attempted_paths.append(task.selected_path_id)
        package = _package(task)
        truth = GroundTruth(
            task_id=task.task_id,
            verification_sql="SELECT 1",
            canonical_answer={"city": "sasebo"},
            answer_schema_version="v1",
        )
        accepted = task.selected_path_id == "orders.customers.addresses"
        return TaskExecutionOutcome(
            task=package.task,
            package=package,
            ground_truth=truth,
            solver_results=[],
            verification_results=[],
            pass_rate=0.9 if not accepted else 0.5,
            accepted=accepted,
            calibration_band=(0.2, 0.8),
            calibration_decision="reject_too_easy" if not accepted else "accept",
            executed_solver_replicas=2,
            planned_solver_replicas=4,
        )

    monkeypatch.setattr(Orchestrator, "_ground_truth_generator", _fake_ground_truth_generator)
    monkeypatch.setattr(Orchestrator, "_execute_task_attempt", _fake_execute_task_attempt)

    outcome = asyncio.run(orchestrator._execute_task(task))

    assert attempted_paths == ["orders.addresses", "orders.customers.addresses"]
    assert outcome.accepted is True
    assert outcome.task.selected_path_id == "orders.customers.addresses"
    assert outcome.calibration_attempts == 2
    assert len(outcome.difficulty_history) == 2


def test_execute_task_returns_best_so_far_when_no_compatible_adjustment(monkeypatch, tmp_path):
    config = _config(tmp_path)
    orchestrator = Orchestrator(config)
    graph = SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="orders",
                primary_key=("order_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="order_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="address_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                ],
            ),
            TableProfile(
                schema_name="public",
                table_name="addresses",
                primary_key=("address_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="addresses",
                        column_name="address_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="addresses",
                        column_name="city",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_address_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("address_id",),
                target_schema="public",
                target_table="addresses",
                target_columns=("address_id",),
                source_is_unique=False,
                fanout_estimate=1.0,
            )
        ],
    )
    orchestrator._graph = graph
    orchestrator._catalog = build_path_catalog(graph, max_hops=2)
    task = TaskSpec(
        task_id="task_best_so_far",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question_family="status_lookup",
        question="현재 주문의 배송 도시를 확인해줘",
        outcome_type="answer",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="city",
                    type="string",
                    canonicalizer="lower_trim",
                    source_columns=["addresses.city"],
                )
            ]
        ),
        selected_path_id="orders.addresses",
        required_hops=1,
        tool_level=1,
        tool_bundle_id="orders.addresses.L1",
        sensitivity_policy="default",
    )

    async def _fake_execute_task_attempt(self, task: TaskSpec, _graph, _catalog):
        package = _package(task)
        truth = GroundTruth(
            task_id=task.task_id,
            verification_sql="SELECT 1",
            canonical_answer={"city": "sasebo"},
            answer_schema_version="v1",
        )
        return TaskExecutionOutcome(
            task=package.task,
            package=package,
            ground_truth=truth,
            solver_results=[],
            verification_results=[],
            pass_rate=0.95,
            accepted=False,
            calibration_band=(0.2, 0.8),
            calibration_decision="reject_too_easy",
            executed_solver_replicas=2,
            planned_solver_replicas=4,
        )

    monkeypatch.setattr(Orchestrator, "_execute_task_attempt", _fake_execute_task_attempt)

    outcome = asyncio.run(orchestrator._execute_task(task))

    assert outcome.accepted is False
    assert outcome.calibration_decision == "best_so_far::reject_too_easy"
    assert outcome.calibration_attempts == 1


def test_execute_task_rewrites_contract_on_same_path_when_needed(monkeypatch, tmp_path):
    config = _config(tmp_path).model_copy(
        update={
            "task_composer": _config(tmp_path).task_composer.model_copy(
                update={
                    "max_attempts_per_anchor": 3,
                    "family_min_required_hops": {
                        "status_lookup": 1,
                        "aggregate_verification": 1,
                    },
                }
            )
        }
    )
    orchestrator = Orchestrator(config)
    graph = SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="orders",
                primary_key=("order_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="order_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="address_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                ],
            ),
            TableProfile(
                schema_name="public",
                table_name="addresses",
                primary_key=("address_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="addresses",
                        column_name="address_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="addresses",
                        column_name="city",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_address_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("address_id",),
                target_schema="public",
                target_table="addresses",
                target_columns=("address_id",),
                source_is_unique=False,
                fanout_estimate=1.0,
            )
        ],
    )
    orchestrator._graph = graph
    orchestrator._catalog = build_path_catalog(graph, max_hops=2)
    task = TaskSpec(
        task_id="task_contract_rewrite",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question_family="status_lookup",
        question="현재 주문의 배송 도시를 확인해줘",
        outcome_type="answer",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="city",
                    type="string",
                    canonicalizer="lower_trim",
                    source_columns=["addresses.city"],
                )
            ]
        ),
        selected_path_id="orders.addresses",
        required_hops=1,
        tool_level=1,
        tool_bundle_id="orders.addresses.L1",
        sensitivity_policy="default",
    )
    attempted_variants: list[tuple[str, str]] = []

    class _FakeGenerator:
        async def generate(self, task: TaskSpec) -> GroundTruth:
            return GroundTruth(
                task_id=task.task_id,
                verification_sql="SELECT 1",
                canonical_answer={"city": "sasebo"},
                answer_schema_version="v1",
            )

    def _fake_ground_truth_generator(self, _graph, _catalog):
        return _FakeGenerator()

    async def _fake_execute_task_attempt(self, task: TaskSpec, _graph, _catalog):
        attempted_variants.append((task.selected_path_id, task.question_family))
        package = _package(task)
        truth = GroundTruth(
            task_id=task.task_id,
            verification_sql="SELECT 1",
            canonical_answer={"city": "sasebo"},
            answer_schema_version="v1",
        )
        accepted = task.question_family == "aggregate_verification"
        return TaskExecutionOutcome(
            task=package.task,
            package=package,
            ground_truth=truth,
            solver_results=[],
            verification_results=[],
            pass_rate=0.95 if not accepted else 0.5,
            accepted=accepted,
            calibration_band=(0.2, 0.8),
            calibration_decision="reject_too_easy" if not accepted else "accept",
            executed_solver_replicas=2,
            planned_solver_replicas=4,
        )

    monkeypatch.setattr(Orchestrator, "_ground_truth_generator", _fake_ground_truth_generator)
    monkeypatch.setattr(Orchestrator, "_execute_task_attempt", _fake_execute_task_attempt)

    outcome = asyncio.run(orchestrator._execute_task(task))

    assert attempted_variants == [
        ("orders.addresses", "status_lookup"),
        ("orders.addresses", "aggregate_verification"),
    ]
    assert outcome.accepted is True
    assert outcome.task.selected_path_id == "orders.addresses"
    assert outcome.task.question_family == "aggregate_verification"
    assert outcome.difficulty_history[-1]["question_family"] == "aggregate_verification"


def test_execute_task_does_not_rewrite_path_for_aggregate_verification(monkeypatch, tmp_path):
    config = _config(tmp_path)
    orchestrator = Orchestrator(config)
    graph = SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="orders",
                primary_key=("order_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="order_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="address_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                ],
            ),
            TableProfile(
                schema_name="public",
                table_name="addresses",
                primary_key=("address_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="addresses",
                        column_name="address_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="addresses",
                        column_name="city",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_address_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("address_id",),
                target_schema="public",
                target_table="addresses",
                target_columns=("address_id",),
                source_is_unique=False,
                fanout_estimate=1.0,
            )
        ],
    )
    orchestrator._graph = graph
    orchestrator._catalog = build_path_catalog(graph, max_hops=2)
    task = TaskSpec(
        task_id="task_aggregate_family",
        anchor_table="orders",
        anchor_pk_column="order_id",
        anchor_pk_value="1",
        domain="customer_support",
        language="ko",
        label_tier="A",
        question_family="aggregate_verification",
        question="현재 주문의 배송 도시를 확인해줘",
        outcome_type="answer",
        answer_schema=AnswerSchema(
            fields=[
                AnswerField(
                    name="city",
                    type="string",
                    canonicalizer="lower_trim",
                    source_columns=["addresses.city"],
                )
            ]
        ),
        selected_path_id="orders.addresses",
        required_hops=1,
        tool_level=1,
        tool_bundle_id="orders.addresses.L1",
        sensitivity_policy="default",
    )
    attempted_paths: list[str] = []

    async def _fake_execute_task_attempt(self, task: TaskSpec, _graph, _catalog):
        attempted_paths.append(task.selected_path_id)
        package = _package(task)
        truth = GroundTruth(
            task_id=task.task_id,
            verification_sql="SELECT 1",
            canonical_answer={"city": "sasebo"},
            answer_schema_version="v1",
        )
        return TaskExecutionOutcome(
            task=package.task,
            package=package,
            ground_truth=truth,
            solver_results=[],
            verification_results=[],
            pass_rate=0.95,
            accepted=False,
            calibration_band=(0.2, 0.8),
            calibration_decision="reject_too_easy",
            executed_solver_replicas=2,
            planned_solver_replicas=4,
        )

    monkeypatch.setattr(Orchestrator, "_execute_task_attempt", _fake_execute_task_attempt)

    outcome = asyncio.run(orchestrator._execute_task(task))

    assert attempted_paths == ["orders.addresses"]
    assert outcome.calibration_decision == "best_so_far::reject_too_easy"


def test_materialize_presented_tool_specs_supports_distractor_paths(tmp_path):
    config = _config(tmp_path)
    graph = SchemaGraph(
        tables=[
            TableProfile(
                schema_name="public",
                table_name="orders",
                primary_key=("order_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="order_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="shipment_id",
                        data_type="int4",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="orders",
                        column_name="invoice_id",
                        data_type="int4",
                        ordinal_position=3,
                        is_nullable=False,
                        visibility="blocked",
                        is_foreign_key=True,
                    ),
                ],
            ),
            TableProfile(
                schema_name="public",
                table_name="shipments",
                primary_key=("shipment_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="shipment_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="shipments",
                        column_name="status",
                        data_type="text",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
            ),
            TableProfile(
                schema_name="public",
                table_name="invoices",
                primary_key=("invoice_id",),
                columns=[
                    ColumnProfile(
                        schema_name="public",
                        table_name="invoices",
                        column_name="invoice_id",
                        data_type="int4",
                        ordinal_position=1,
                        is_nullable=False,
                        visibility="blocked",
                        is_primary_key=True,
                        is_unique=True,
                    ),
                    ColumnProfile(
                        schema_name="public",
                        table_name="invoices",
                        column_name="total",
                        data_type="numeric",
                        ordinal_position=2,
                        is_nullable=False,
                        visibility="user_visible",
                    ),
                ],
            ),
        ],
        edges=[
            ForeignKeyEdge(
                constraint_name="orders_shipments_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("shipment_id",),
                target_schema="public",
                target_table="shipments",
                target_columns=("shipment_id",),
            ),
            ForeignKeyEdge(
                constraint_name="orders_invoices_fk",
                source_schema="public",
                source_table="orders",
                source_columns=("invoice_id",),
                target_schema="public",
                target_table="invoices",
                target_columns=("invoice_id",),
            ),
        ],
    )
    catalog = build_path_catalog(graph, max_hops=2)
    core_path = catalog.get("orders.shipments")
    distractor_path = catalog.get("orders.invoices")
    core_bundle = compile_canonical_tool_bundle(graph, core_path, label_tier="A")
    distractor_bundle = compile_canonical_tool_bundle(graph, distractor_path, label_tier="A")
    presented_bundle = PresentedToolBundle(
        bundle_id="bundle-presented",
        canonical_bundle_id=core_bundle.bundle_id,
        path_id=core_path.path_id,
        tool_level=1,
        question_family="status_lookup",
        outcome_type="answer",
        tools=[
            PresentedToolSpec(
                name=core_bundle.tools[0].name,
                description=core_bundle.tools[0].description,
                semantic_key=core_bundle.tools[0].semantic_key,
                kind=core_bundle.tools[0].kind,
                parameter_names=[parameter.name for parameter in core_bundle.tools[0].parameters],
                output_fields=list(core_bundle.tools[0].output_fields),
                name_source="rule_based",
                presentation_role="core",
            ),
            PresentedToolSpec(
                name=distractor_bundle.tools[0].name,
                description=distractor_bundle.tools[0].description,
                semantic_key=distractor_bundle.tools[0].semantic_key,
                kind=distractor_bundle.tools[0].kind,
                parameter_names=[parameter.name for parameter in distractor_bundle.tools[0].parameters],
                output_fields=list(distractor_bundle.tools[0].output_fields),
                name_source="rule_based",
                presentation_role="distractor",
            ),
        ],
    )

    materialized = Orchestrator(config)._materialize_presented_tool_specs(
        presented_bundle,
        core_bundle,
        graph=graph,
        catalog=catalog,
        label_tier="A",
    )

    assert len(materialized) == 2
    assert {tool.path_id for tool in materialized} == {"orders.shipments", "orders.invoices"}
