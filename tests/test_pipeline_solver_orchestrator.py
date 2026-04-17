from __future__ import annotations

import ast
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
from rl_task_foundry.synthesis.canonicalize import RewardResult
from tests.test_synthesis_task_registry import _sample_draft


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
    return config.model_copy(update={"output": output, "models": models}, deep=True)


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
            transcript_ref="memory://transcript",
            tool_trace_ref="memory://tools",
            raw_output_text=self.raw_output_text,
            status="completed",
        )


class _SeedCapturingRuntime:
    def __init__(
        self,
        *,
        raw_output_text: str,
        executor: object,
        seen_payloads: list[dict[str, object]],
    ) -> None:
        self.raw_output_text = raw_output_text
        self.executor = executor
        self.seen_payloads = seen_payloads

    async def run(self, episode) -> SolverResult:
        result = self.executor({"customer_id": 1})
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, dict):
            self.seen_payloads.append(result)
        return SolverResult(
            task_id=episode.task_id,
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
            transcript_ref="memory://transcript",
            tool_trace_ref="memory://tools",
            raw_output_text=self.raw_output_text,
            status="completed",
        )


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
        tool_executor_factory=lambda _bundle: {},
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
async def test_solver_orchestrator_injects_distinct_shuffle_seed_per_solver_run(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    config = _config(tmp_path)
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
    recorded_kwargs: list[dict[str, object]] = []

    async def _recording_executor(kwargs: dict[str, object]) -> dict[str, object]:
        recorded_kwargs.append(dict(kwargs))
        return dict(kwargs)

    orchestrator = SolverOrchestrator(
        config,
        runtime_factory=lambda _solver, _provider, _task, _defs, tool_executors: (
            _SeedCapturingRuntime(
                raw_output_text='{"customer":"Alice","day":"2026-04-12"}',
                executor=next(iter(tool_executors.values())),
                seen_payloads=[],
            )
        ),
        tool_executor_factory=lambda _bundle: {"get_customer_by_id": _recording_executor},
    )

    try:
        summary = await orchestrator.run_draft(draft)
    finally:
        await orchestrator.close()

    assert summary.total_solver_runs == 2
    shuffle_seeds = [str(kwargs["_shuffle_seed"]) for kwargs in recorded_kwargs]
    assert len(shuffle_seeds) == 2
    assert len(set(shuffle_seeds)) == 2


@pytest.mark.asyncio
async def test_solver_orchestrator_close_clears_cached_tool_executors(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    orchestrator = SolverOrchestrator(
        _config(tmp_path),
        runtime_factory=lambda *_args: _FakeRuntime(
            '{"customer":"Alice","day":"2026-04-12"}',
            [],
        ),
        tool_executor_factory=lambda _bundle: {"noop": lambda _kwargs: {}},
    )

    await orchestrator.run_draft(draft)
    assert orchestrator._tool_executor_cache

    await orchestrator.close()

    assert orchestrator._tool_executor_cache == {}


@pytest.mark.asyncio
async def test_solver_orchestrator_close_clears_solver_model_cache(
    tmp_path: Path,
) -> None:
    orchestrator = SolverOrchestrator(
        _config(tmp_path),
        runtime_factory=lambda *_args: _FakeRuntime('{"customer":"Alice","day":"2026-04-12"}', []),
        tool_executor_factory=lambda _bundle: {"noop": lambda _kwargs: {}},
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
            transcript_ref="memory://transcript",
            tool_trace_ref="memory://tools",
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
