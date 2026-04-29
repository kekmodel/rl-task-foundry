from __future__ import annotations

import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig, SolverModelConfig
from rl_task_foundry.pipeline.solver_orchestrator import (
    SolverOrchestrator,
    TaskQualityGateStatus,
    TaskRolloutSummary,
    TaskSolverRun,
    evaluate_rollout_summary,
)
from rl_task_foundry.solver.backend_openai_agents import OpenAIAgentsSolverBackend
from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.solver.runtime import SolverEpisodeInput
from rl_task_foundry.synthesis.canonicalize import RewardResult
from tests.test_synthesis_task_registry import _sample_draft
from tests.test_tooling_atomic_tool_factory import _snapshot, _StubConnection


async def _empty_sdk_tools(_task_bundle: object) -> list[object]:
    return []


def _config(tmp_path: Path):
    config = load_config("rl_task_foundry.yaml")
    output = OutputConfig(
        run_db_path=tmp_path / "run.db",
        traces_dir=tmp_path / "traces",
    )
    models = config.models.model_copy(
        update={
            "solvers": [
                SolverModelConfig(
                    solver_id="solver_a",
                    provider="codex_oauth",
                    model="gpt-5.4-mini",
                )
            ]
        }
    )
    calibration = config.calibration.model_copy(update={"max_solver_runs": 1})
    return config.model_copy(
        update={"output": output, "models": models, "calibration": calibration},
        deep=True,
    )


class _FakeRuntime:
    def __init__(self, raw_output_text: str, seen_max_turns: list[int]) -> None:
        self.raw_output_text = raw_output_text
        self.seen_max_turns = seen_max_turns

    async def run(self, episode) -> SolverResult:
        self.seen_max_turns.append(episode.task_bundle.rollout_constraints.max_turns)
        return SolverResult(
            task_id=episode.task_id,
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
            raw_output_text=self.raw_output_text,
            status="completed",
        )


class _InvokingAtomicRuntime:
    def __init__(self, sdk_tools: list[object]) -> None:
        self.sdk_tools = sdk_tools

    async def run(self, episode) -> SolverResult:
        tools = {
            getattr(tool, "name"): tool
            for tool in self.sdk_tools
        }
        await tools["create_record_set"].on_invoke_tool(  # pyright: ignore[reportAttributeAccessIssue]
            None,
            '{"table":"customer"}',
        )
        return SolverResult(
            task_id=episode.task_id,
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
            raw_output_text='{"customer":"Alice","day":"2026-04-12"}',
            status="completed",
            termination_metadata={"run_items": []},
        )


class _FakeDatabasePools:
    @asynccontextmanager
    async def solver_connection(self):
        yield _StubConnection()

    async def close(self) -> None:
        return None


class _FakeEventLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def log_sync(self, **event: object) -> None:
        self.events.append(event)


def test_solver_orchestrator_rejects_provider_concurrency_below_solver_batch(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(
        update={"solver_batch_size": 4, "max_solver_runs": 8}
    )
    config.providers["codex_oauth"] = config.providers["codex_oauth"].model_copy(
        update={"max_concurrency": 1}
    )
    config.models.solvers = [
        SolverModelConfig(
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        )
    ]

    with pytest.raises(ValueError, match="max_concurrency >= 4"):
        SolverOrchestrator(
            config,
            runtime_factory=lambda *_args: _FakeRuntime(
                '{"customer":"Alice","day":"2026-04-12"}',
                [],
            ),
            sdk_tools_factory=_empty_sdk_tools,
        )


@pytest.mark.asyncio
async def test_solver_orchestrator_runs_solver_batch_concurrently(
    tmp_path: Path,
) -> None:
    class _Tracker:
        active = 0
        peak = 0

    class _TrackingRuntime:
        def __init__(self, solver_id: str) -> None:
            self.solver_id = solver_id

        async def run(self, episode) -> SolverResult:
            _Tracker.active += 1
            _Tracker.peak = max(_Tracker.peak, _Tracker.active)
            try:
                await asyncio.sleep(0.01)
                return SolverResult(
                    task_id=episode.task_id,
                    solver_id=self.solver_id,
                    provider="codex_oauth",
                    model="gpt-5.4-mini",
                    raw_output_text='{"customer":"Alice","day":"2026-04-12"}',
                    status="completed",
                )
            finally:
                _Tracker.active -= 1

    draft = _sample_draft()
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(
        update={"solver_batch_size": 4, "max_solver_runs": 4}
    )
    config.providers["codex_oauth"] = config.providers["codex_oauth"].model_copy(
        update={"max_concurrency": 4}
    )
    config.models.solvers = [
        SolverModelConfig(
            solver_id=f"solver_{index}",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        )
        for index in range(4)
    ]

    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda solver_config, *_args: _TrackingRuntime(
            solver_config.solver_id
        ),
        sdk_tools_factory=_empty_sdk_tools,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.total_solver_runs == 4
    assert _Tracker.peak == 4


@pytest.mark.asyncio
async def test_solver_orchestrator_scores_against_canonical_answer(tmp_path: Path) -> None:
    draft = _sample_draft()
    seen_max_turns: list[int] = []
    orchestrator = SolverOrchestrator(
        _config(tmp_path),
        runtime_factory=lambda *_args: _FakeRuntime(
            '{"customer":"Alice","day":"2026-04-12"}',
            seen_max_turns,
        ),
        sdk_tools_factory=_empty_sdk_tools,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.total_solver_runs == 1
    assert summary.matched_solver_runs == 1
    assert summary.pass_rate == 1.0
    assert seen_max_turns == [draft.task_bundle.rollout_constraints.max_turns]


@pytest.mark.asyncio
async def test_solver_orchestrator_reuses_single_solver_model_for_configured_run_count(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(update={"max_solver_runs": 3})
    seen_solver_ids: list[str] = []

    def _runtime_factory(solver_config, *_args):
        seen_solver_ids.append(solver_config.solver_id)
        return _FakeRuntime('{"customer":"Alice","day":"2026-04-12"}', [])

    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=_runtime_factory,
        sdk_tools_factory=_empty_sdk_tools,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.planned_solver_runs == 3
    assert summary.total_solver_runs == 3
    assert seen_solver_ids == [
        "gpt-5.4-mini_00",
        "gpt-5.4-mini_01",
        "gpt-5.4-mini_02",
    ]


@pytest.mark.asyncio
async def test_solver_orchestrator_invokes_sdk_tools_factory_per_solver_run(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(update={"max_solver_runs": 2})
    config.models.solvers = [
        SolverModelConfig(
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        ),
        SolverModelConfig(
            solver_id="solver_b",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        ),
    ]
    factory_calls: list[str] = []

    async def _recording_factory(task_bundle: object) -> list[object]:
        factory_calls.append(getattr(task_bundle, "task_id", "?"))
        return []

    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: _FakeRuntime(
            '{"customer":"Alice","day":"2026-04-12"}', []
        ),
        sdk_tools_factory=_recording_factory,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.total_solver_runs == 2
    # One factory invocation per solver run (tools are built fresh each run
    # so the cursor store / atomic session stays scoped to that run).
    assert len(factory_calls) == 2


@pytest.mark.asyncio
async def test_solver_orchestrator_attaches_atomic_trace_metadata(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    config = _config(tmp_path)
    solver_config = config.models.solvers[0]
    provider_config = config.providers[solver_config.provider]
    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: _InvokingAtomicRuntime(_args[3]),
        database_pools=_FakeDatabasePools(),  # type: ignore[arg-type]
    )
    orchestrator._schema_snapshot_cache[draft.task_bundle.db_id] = _snapshot()
    episode = SolverEpisodeInput(
        task_bundle=draft.task_bundle,
        rendered_user_prompt=draft.rendered_user_prompt,
    )

    try:
        result = await orchestrator._run_with_tools(
            solver_config=solver_config,
            provider_config=provider_config,
            bundle=draft,
            episode=episode,
        )
    finally:
        await orchestrator.close()

    assert result.termination_metadata["atomic_trace_version"] == (
        "atomic-resource-api-v6.trace.v1"
    )
    events = result.termination_metadata["atomic_trace_events"]
    assert isinstance(events, list)
    assert events[0]["operation"] == "create_record_set"
    output_resource = events[0]["output_resource"]
    assert output_resource["id"] == "record_set_1"
    assert output_resource["type"] == "record_set"
    assert output_resource["table"] == "customer"
    assert "columns" not in output_resource
    assert "relations" not in output_resource
    assert result.termination_metadata["run_items"] == []


@pytest.mark.asyncio
async def test_solver_orchestrator_close_clears_snapshot_cache(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    orchestrator = SolverOrchestrator(
        _config(tmp_path),
        runtime_factory=lambda *_args: _FakeRuntime(
            '{"customer":"Alice","day":"2026-04-12"}',
            [],
        ),
        sdk_tools_factory=_empty_sdk_tools,
    )

    await orchestrator.run_draft(draft)

    orchestrator._schema_snapshot_cache["sakila"] = object()  # type: ignore[assignment]
    await orchestrator.close()

    assert orchestrator._schema_snapshot_cache == {}


@pytest.mark.asyncio
async def test_solver_orchestrator_close_clears_solver_model_cache(
    tmp_path: Path,
) -> None:
    orchestrator = SolverOrchestrator(
        _config(tmp_path),
        runtime_factory=lambda *_args: _FakeRuntime('{"customer":"Alice","day":"2026-04-12"}', []),
        sdk_tools_factory=_empty_sdk_tools,
    )
    OpenAIAgentsSolverBackend._shared_models[
        (1, 2, "openai_compatible", None, "dummy", 30.0, "gpt-5.4-mini")
    ] = object()

    await orchestrator.close()

    assert OpenAIAgentsSolverBackend._shared_models == {}


def test_evaluate_rollout_summary_accepts_in_band_results(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = TaskRolloutSummary(
        task_id="task_assignment_registrytest",
        db_id="sakila",
        planned_solver_runs=4,
        total_solver_runs=4,
        matched_solver_runs=2,
        runs=(),
    )

    gate = evaluate_rollout_summary(config, summary)

    assert gate.status is TaskQualityGateStatus.ACCEPT


def test_evaluate_rollout_summary_marks_out_of_band_ci_overlap_inconclusive(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    summary = TaskRolloutSummary(
        task_id="task_assignment_registrytest",
        db_id="sakila",
        planned_solver_runs=30,
        total_solver_runs=30,
        matched_solver_runs=29,
        runs=(),
    )

    gate = evaluate_rollout_summary(config, summary)

    assert gate.status is TaskQualityGateStatus.CALIBRATION_INCONCLUSIVE
    assert gate.pass_rate == pytest.approx(29 / 30)
    assert gate.ci_lower < config.calibration.upper_pass_rate < gate.ci_upper


def test_evaluate_rollout_summary_rejects_incomplete_evaluable_denominator(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    summary = TaskRolloutSummary(
        task_id="task_assignment_registrytest",
        db_id="sakila",
        planned_solver_runs=3,
        total_solver_runs=3,
        matched_solver_runs=2,
        runs=(),
        evaluable_solver_runs=2,
        failed_solver_runs=1,
    )

    with pytest.raises(ValueError, match="target evaluable solver denominator"):
        evaluate_rollout_summary(config, summary)


def _make_run(*, solver_id: str, raw: str, matched: bool) -> TaskSolverRun:
    return TaskSolverRun(
        task_id="task_divergence_fixture",
        solver_id=solver_id,
        solver_index=0,
        solver_result=SolverResult(
            task_id="task_divergence_fixture",
            solver_id=solver_id,
            provider="codex_oauth",
            model="gpt-5.4-mini",
            raw_output_text=raw,
            status="completed",
        ),
        reward_result=RewardResult(
            reward=1.0 if matched else 0.0,
            status="matched" if matched else "em_mismatch",
        ),
    )


def test_evaluate_rollout_summary_survives_malformed_json_raw_output(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    runs = (
        _make_run(solver_id="s_a", raw='{"answer":"alice"}', matched=True),
        _make_run(solver_id="s_b", raw="{not-json", matched=False),
        _make_run(solver_id="s_c", raw='[1, 2, 3', matched=False),
        _make_run(solver_id="s_d", raw="alice", matched=True),
    )
    summary = TaskRolloutSummary(
        task_id="task_divergence_fixture",
        db_id="sakila",
        planned_solver_runs=4,
        total_solver_runs=4,
        matched_solver_runs=2,
        runs=runs,
    )

    gate = evaluate_rollout_summary(config, summary)

    assert gate.unique_answers == 4
    assert gate.divergence_ratio == 1.0
    assert gate.total_solver_runs == 4


class _FailingRuntime:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def run(self, episode):
        raise self._exc


@pytest.mark.asyncio
async def test_solver_orchestrator_excludes_failed_runs_from_pass_rate(
    tmp_path: Path,
) -> None:
    class RateLimitError(RuntimeError):
        pass

    draft = _sample_draft()
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(update={"max_solver_runs": 3})
    config.models.solvers = [
        SolverModelConfig(
            solver_id=f"solver_{i}",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        )
        for i in range(3)
    ]
    seen: list[int] = []
    runtimes = iter([
        _FakeRuntime('{"customer":"Alice","day":"2026-04-12"}', seen),
        _FailingRuntime(RateLimitError("simulated rate limit")),
        _FakeRuntime('{"customer":"Alice","day":"2026-04-12"}', seen),
        _FakeRuntime('{"customer":"Alice","day":"2026-04-12"}', seen),
    ])
    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: next(runtimes),
        sdk_tools_factory=_empty_sdk_tools,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.planned_solver_runs == 3
    assert summary.total_solver_runs == 4
    assert summary.evaluable_solver_runs == 3
    assert summary.matched_solver_runs == 3
    assert summary.failed_solver_runs == 1
    assert summary.pass_rate == 1.0
    assert len(summary.runs) == 4
    gate = evaluate_rollout_summary(config, summary)
    assert gate.total_solver_runs == 4
    assert gate.evaluable_solver_runs == 3
    assert gate.failed_solver_runs == 1


@pytest.mark.asyncio
async def test_solver_orchestrator_counts_wrong_answers_as_evaluable_without_topup(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(update={"max_solver_runs": 3})
    config.models.solvers = [
        SolverModelConfig(
            solver_id=f"solver_{i}",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        )
        for i in range(3)
    ]
    runtimes = iter([
        _FakeRuntime('{"customer":"Alice","day":"2026-04-12"}', []),
        _FakeRuntime('{"customer":"Bob","day":"2026-04-12"}', []),
        _FakeRuntime('{"customer":"Carol","day":"2026-04-12"}', []),
    ])
    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: next(runtimes),
        sdk_tools_factory=_empty_sdk_tools,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.planned_solver_runs == 3
    assert summary.total_solver_runs == 3
    assert summary.evaluable_solver_runs == 3
    assert summary.matched_solver_runs == 1
    assert summary.failed_solver_runs == 0
    assert summary.pass_rate == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_solver_orchestrator_can_reject_too_hard_before_full_denominator(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(
        update={
            "solver_batch_size": 4,
            "max_solver_runs": 20,
            "safe_early_termination": True,
        }
    )
    config.models.solvers = [
        SolverModelConfig(
            solver_id=f"solver_{i}",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        )
        for i in range(20)
    ]
    runtimes = iter([
        _FakeRuntime('{"customer":"wrong","day":"2026-04-12"}', [])
        for _ in range(20)
    ])
    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: next(runtimes),
        sdk_tools_factory=_empty_sdk_tools,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.planned_solver_runs == 20
    # With the default lower bound 0.5, the first all-miss batch can no longer
    # recover into the accepted band.
    assert summary.total_solver_runs == 4
    assert summary.evaluable_solver_runs == 4
    assert summary.matched_solver_runs == 0
    assert summary.early_stop_decision == "reject_too_hard"


@pytest.mark.asyncio
async def test_solver_orchestrator_applies_exact_ci_before_full_denominator(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(
        update={
            "solver_batch_size": 1,
            "max_solver_runs": 40,
            "safe_early_termination": True,
        }
    )
    config.models.solvers = [
        SolverModelConfig(
            solver_id=f"solver_{i}",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        )
        for i in range(40)
    ]
    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: _FakeRuntime(
            '{"customer":"Alice","day":"2026-04-12"}', []
        ),
        sdk_tools_factory=_empty_sdk_tools,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.planned_solver_runs == 40
    # The one-sided exact lower bound first exceeds 0.9 at 22/22.
    assert summary.total_solver_runs == 22
    assert summary.evaluable_solver_runs == 22
    assert summary.matched_solver_runs == 22
    assert summary.early_stop_decision == "reject_too_easy"


@pytest.mark.asyncio
async def test_solver_orchestrator_accepts_when_one_sided_bounds_enter_band(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(
        update={
            "solver_batch_size": 4,
            "max_solver_runs": 20,
            "safe_early_termination": True,
        }
    )
    config.models.solvers = [
        SolverModelConfig(
            solver_id=f"solver_{i}",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        )
        for i in range(20)
    ]
    runtimes = iter(
        [_FakeRuntime('{"customer":"Alice","day":"2026-04-12"}', []) for _ in range(12)]
        + [_FakeRuntime('{"customer":"wrong","day":"2026-04-12"}', []) for _ in range(8)]
    )
    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: next(runtimes),
        sdk_tools_factory=_empty_sdk_tools,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.planned_solver_runs == 20
    # At 12/16, both one-sided exact bounds are inside [0.5, 0.9].
    assert summary.total_solver_runs == 16
    assert summary.evaluable_solver_runs == 16
    assert summary.matched_solver_runs == 12
    assert summary.early_stop_decision == "accept"


@pytest.mark.asyncio
async def test_solver_orchestrator_counts_user_error_as_evaluable_actor_failure(
    tmp_path: Path,
) -> None:
    class UserError(RuntimeError):
        pass

    draft = _sample_draft()
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(update={"max_solver_runs": 3})
    config.models.solvers = [
        SolverModelConfig(
            solver_id=f"solver_{i}",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        )
        for i in range(3)
    ]
    runtimes = iter([
        _FakeRuntime('{"customer":"Alice","day":"2026-04-12"}', []),
        _FailingRuntime(UserError("model stopped after tool calls")),
        _FailingRuntime(UserError("model stopped after tool calls")),
    ])
    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: next(runtimes),
        sdk_tools_factory=_empty_sdk_tools,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.total_solver_runs == 3
    assert summary.evaluable_solver_runs == 3
    assert summary.matched_solver_runs == 1
    assert summary.failed_solver_runs == 0
    assert summary.pass_rate == pytest.approx(1 / 3)
    gate = evaluate_rollout_summary(config, summary)
    assert gate.status is TaskQualityGateStatus.CALIBRATION_INCONCLUSIVE
    assert gate.total_solver_runs == 3
    assert gate.evaluable_solver_runs == 3


@pytest.mark.asyncio
async def test_solver_event_log_marks_user_error_as_evaluable(
    tmp_path: Path,
) -> None:
    class UserError(RuntimeError):
        pass

    draft = _sample_draft()
    event_logger = _FakeEventLogger()
    orchestrator = SolverOrchestrator(
        _config(tmp_path),
        runtime_factory=lambda *_args: _FailingRuntime(
            UserError("model stopped after tool calls")
        ),
        sdk_tools_factory=_empty_sdk_tools,
        event_logger=event_logger,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.evaluable_solver_runs == 1
    assert summary.failed_solver_runs == 0
    payload = event_logger.events[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["termination_reason"] == "UserError"
    assert payload["excluded_from_pass_rate"] is False
    assert payload["failure_class"] == "evaluable"
    assert payload["failure_detail"] == "model stopped after tool calls"


@pytest.mark.asyncio
async def test_solver_orchestrator_counts_max_turns_as_evaluable_actor_failure(
    tmp_path: Path,
) -> None:
    class MaxTurnsExceeded(RuntimeError):
        pass

    draft = _sample_draft()
    config = _config(tmp_path)
    config.calibration = config.calibration.model_copy(update={"max_solver_runs": 3})
    config.models.solvers = [
        SolverModelConfig(
            solver_id=f"solver_{i}",
            provider="codex_oauth",
            model="gpt-5.4-mini",
        )
        for i in range(3)
    ]
    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda *_args: _FailingRuntime(
            MaxTurnsExceeded("max turns exceeded")
        ),
        sdk_tools_factory=_empty_sdk_tools,
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.total_solver_runs == 3
    assert summary.evaluable_solver_runs == 3
    assert summary.matched_solver_runs == 0
    assert summary.failed_solver_runs == 0
    gate = evaluate_rollout_summary(config, summary)
    assert gate.status is TaskQualityGateStatus.CALIBRATION_INCONCLUSIVE


def test_solver_orchestrator_module_has_no_legacy_imports() -> None:
    module_path = Path("src/rl_task_foundry/pipeline/solver_orchestrator.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    banned_prefixes = (
        "rl_task_foundry.tools",
        "rl_task_foundry.tasks",
        "rl_task_foundry.truth",
        "rl_task_foundry.verification",
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(banned_prefixes)
