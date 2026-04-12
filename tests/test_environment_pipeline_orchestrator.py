from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.config.models import OutputConfig, SolverModelConfig
from rl_task_foundry.pipeline.environment_orchestrator import (
    EnvironmentOrchestrator,
    EnvironmentQualityGateStatus,
    EnvironmentRolloutSummary,
    evaluate_rollout_summary,
)
from rl_task_foundry.solver.models import SolverResult
from tests.test_synthesis_environment_registry import _sample_draft


def _config(tmp_path: Path):
    config = load_config("rl_task_foundry.yaml")
    output = OutputConfig(
        run_db_path=tmp_path / "run.db",
        accepted_jsonl_path=tmp_path / "accepted.jsonl",
        rejected_jsonl_path=tmp_path / "rejected.jsonl",
        events_jsonl_path=tmp_path / "events.jsonl",
        traces_dir=tmp_path / "traces",
    )
    models = config.models.model_copy(
        update={
            "solvers": [
                SolverModelConfig(
                    solver_id="solver_a",
                    provider="codex_oauth",
                    model="gpt-5.4-mini",
                    replicas=1,
                )
            ]
        }
    )
    return config.model_copy(update={"output": output, "models": models}, deep=True)


class _FakeRuntime:
    def __init__(self, raw_output_text: str, seen_max_turns: list[int]) -> None:
        self.raw_output_text = raw_output_text
        self.seen_max_turns = seen_max_turns

    async def run(self, episode, replica_index: int = 0) -> SolverResult:
        self.seen_max_turns.append(episode.environment.rollout_constraints.max_turns)
        return SolverResult(
            task_id=episode.task_id,
            solver_id="solver_a",
            provider="codex_oauth",
            model="gpt-5.4-mini",
            replica_index=replica_index,
            transcript_ref="memory://transcript",
            tool_trace_ref="memory://tools",
            raw_output_text=self.raw_output_text,
            status="completed",
        )


@pytest.mark.asyncio
async def test_environment_orchestrator_scores_against_canonical_answer(tmp_path: Path) -> None:
    draft = _sample_draft()
    seen_max_turns: list[int] = []
    orchestrator = EnvironmentOrchestrator(
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
    assert seen_max_turns == [draft.environment.rollout_constraints.max_turns]


def test_evaluate_rollout_summary_accepts_in_band_results(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = EnvironmentRolloutSummary(
        env_id="env_assignment_registrytest",
        db_id="sakila",
        planned_solver_runs=4,
        total_instances=1,
        total_solver_runs=4,
        matched_solver_runs=2,
        runs=(),
    )

    gate = evaluate_rollout_summary(config, summary)

    assert gate.status is EnvironmentQualityGateStatus.ACCEPT


def test_environment_orchestrator_module_has_no_legacy_imports() -> None:
    module_path = Path("src/rl_task_foundry/pipeline/environment_orchestrator.py")
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
