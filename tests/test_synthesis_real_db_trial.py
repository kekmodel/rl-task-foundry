from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.real_db_trial import (
    RealDbTrialRunner,
    RealDbTrialStatus,
)
from rl_task_foundry.synthesis.runtime import (
    SynthesisArtifactDiagnostics,
    SynthesisArtifactGenerationError,
    SynthesisGenerationAttempt,
    SynthesisGenerationOutcome,
)
from rl_task_foundry.synthesis.task_registry import (
    TaskRegistryCommitResult,
    TaskRegistryCommitStatus,
)
from tests.test_synthesis_task_registry import _sample_draft


@dataclass(slots=True)
class _FakeSynthesisRuntime:
    draft: object | None = None
    exc: Exception | None = None
    delay_s: float = 0.0
    calls: list[dict[str, object]] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_topic: str | None,
    ) -> object:
        self.calls.append({"db_id": db_id, "requested_topic": requested_topic})
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.exc is not None:
            raise self.exc
        assert self.draft is not None
        return self.draft

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _FakeRegistry:
    result: TaskRegistryCommitResult
    committed_drafts: list[object] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.committed_drafts is None:
            self.committed_drafts = []

    def commit_draft(self, draft: object) -> TaskRegistryCommitResult:
        self.committed_drafts.append(draft)
        return self.result

    def close(self) -> None:
        self.closed = True


def _config_with_tmp_traces(
    tmp_path: Path,
    *,
    trial_timeout_s: float | None = None,
):
    config = load_config(Path("rl_task_foundry.yaml"))
    output = config.output.model_copy(
        update={
            "traces_dir": tmp_path / "traces",
            "run_db_path": tmp_path / "run.db",
        },
        deep=True,
    )
    updates: dict[str, object] = {"output": output}
    if trial_timeout_s is not None:
        runtime = config.synthesis.runtime.model_copy(
            update={"trial_timeout_s": trial_timeout_s},
            deep=True,
        )
        updates["synthesis"] = config.synthesis.model_copy(
            update={"runtime": runtime},
            deep=True,
        )
    return config.model_copy(update=updates, deep=True)


@pytest.mark.asyncio
async def test_real_db_trial_runner_commits_without_exporting_bundle(
    tmp_path: Path,
) -> None:
    base_draft = _sample_draft(tmp_task_id="task_real_trial")
    accepted_draft = base_draft.model_copy(
        update={
            "task_bundle": base_draft.task_bundle.model_copy(
                update={
                    "status": "accepted",
                    "quality_metrics": base_draft.task_bundle.quality_metrics.model_copy(
                        update={
                            "solver_pass_rate": 0.5,
                            "solver_ci_low": 0.1,
                            "solver_ci_high": 0.9,
                            "solver_matched_runs": 10,
                            "solver_planned_runs": 20,
                            "solver_completed_runs": 20,
                            "solver_evaluable_runs": 20,
                            "solver_failed_runs": 0,
                        }
                    ),
                }
            )
        }
    )
    runtime = _FakeSynthesisRuntime(draft=accepted_draft)
    registry = _FakeRegistry(
        result=TaskRegistryCommitResult(
            status=TaskRegistryCommitStatus.COMMITTED,
            task_id=accepted_draft.task_bundle.task_id,
            exact_signature="sha256:trial",
            filesystem_path=tmp_path / "tasks" / accepted_draft.task_bundle.task_id,
        )
    )
    runner = RealDbTrialRunner(
        _config_with_tmp_traces(tmp_path),
        synthesis_runtime=runtime,
        registry=registry,
    )
    output_root = tmp_path / "real_trial"

    try:
        summary = await runner.run(
            output_root,
            db_id="sakila",
            topic="assignment",
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.ACCEPTED
    assert summary.quality_gate_status == "accept"
    assert summary.registry_status is TaskRegistryCommitStatus.COMMITTED
    assert summary.bundle_root is None
    assert not (output_root / "bundle").exists()
    assert not (output_root / "debug" / "databases").exists()
    assert not (output_root / "trial_summary.json").exists()
    assert summary.task_id == "task_real_trial"
    assert summary.solver_matched_runs == 10
    assert summary.solver_planned_runs == 20
    assert summary.solver_completed_runs == 20
    assert summary.solver_evaluable_runs == 20
    assert summary.solver_failed_runs == 0
    assert summary.elapsed_seconds is not None
    assert summary.analysis_log_path == output_root / "debug" / "analysis.jsonl"
    assert summary.analysis_log_path.exists()
    assert not (output_root / "debug" / "phase_monitors.jsonl").exists()
    assert not (output_root / "debug" / "trial_events.jsonl").exists()
    analysis_lines = [
        json.loads(line)
        for line in summary.analysis_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {line["actor"] for line in analysis_lines} >= {"runner", "phase"}
    assert registry.closed is False  # injected registry stays open; caller owns lifecycle


@pytest.mark.asyncio
async def test_real_db_trial_runner_enforces_wall_clock_timeout(tmp_path: Path) -> None:
    registry = _FakeRegistry(
        result=TaskRegistryCommitResult(
            status=TaskRegistryCommitStatus.COMMITTED,
            task_id="unused",
            exact_signature="sha256:unused",
            filesystem_path=tmp_path / "unused",
        )
    )
    runner = RealDbTrialRunner(
        _config_with_tmp_traces(tmp_path, trial_timeout_s=0.01),
        synthesis_runtime=_FakeSynthesisRuntime(
            delay_s=0.1,
            draft=_sample_draft(tmp_task_id="task_never_committed"),
        ),
        registry=registry,
    )

    try:
        summary = await runner.run(
            tmp_path / "real_trial",
            db_id="sakila",
            topic="assignment",
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.SYNTHESIS_FAILED
    assert summary.synthesis_error_type == "TrialWallClockTimeout"
    assert summary.synthesis_phase == "trial_timeout"
    assert summary.error_codes == ("trial_wall_clock_timeout",)
    assert summary.elapsed_seconds is not None
    assert summary.elapsed_seconds < 1.0
    assert registry.committed_drafts == []


@pytest.mark.asyncio
async def test_real_db_trial_runner_surfaces_generation_failure(tmp_path: Path) -> None:
    exc = SynthesisArtifactGenerationError(
        "single-agent synthesis exhausted retries",
        attempts=[
            SynthesisGenerationAttempt(
                attempt_index=1,
                outcome=SynthesisGenerationOutcome.ARTIFACT_INVALID,
                provider="codex_oauth",
                model="gpt-5.4-mini",
                memory_summary="attempt failed",
                artifact_diagnostics=SynthesisArtifactDiagnostics(
                    error_codes=["reject_too_easy"],
                ),
                solver_pass_rate=0.95,
                solver_ci_low=0.8,
                solver_ci_high=1.0,
                solver_matched_runs=19,
                solver_planned_runs=20,
                solver_completed_runs=20,
                solver_evaluable_runs=20,
                solver_failed_runs=0,
            )
        ],
        last_artifact_diagnostics=SynthesisArtifactDiagnostics(
            error_codes=["reject_too_easy"],
            feedback_events=2,
            last_feedback_error_codes=["answer_contract_evidence_mismatch"],
        ),
    )
    runner = RealDbTrialRunner(
        _config_with_tmp_traces(tmp_path),
        synthesis_runtime=_FakeSynthesisRuntime(exc=exc),
        registry=_FakeRegistry(
            result=TaskRegistryCommitResult(
                status=TaskRegistryCommitStatus.COMMITTED,
                task_id="unused",
                exact_signature="sha256:unused",
                filesystem_path=tmp_path / "unused",
            )
        ),
    )

    try:
        summary = await runner.run(
            tmp_path / "real_trial",
            db_id="sakila",
            topic="assignment",
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.SYNTHESIS_FAILED
    assert summary.synthesis_error_type == "SynthesisArtifactGenerationError"
    assert summary.attempt_outcomes == ("artifact_invalid",)
    assert summary.error_codes == ("reject_too_easy",)
    assert summary.solver_pass_rate == 0.95
    assert summary.solver_ci_low == 0.8
    assert summary.solver_ci_high == 1.0
    assert summary.solver_matched_runs == 19
    assert summary.solver_planned_runs == 20
    assert summary.solver_completed_runs == 20
    assert summary.solver_evaluable_runs == 20
    assert summary.solver_failed_runs == 0
    assert summary.feedback_events == 2
    assert summary.last_feedback_error_codes == ("answer_contract_evidence_mismatch",)
    assert summary.elapsed_seconds is not None
    assert runner.registry.closed is False  # injected registry stays open
