from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.harvest import (
    HarvestOutcome,
    HarvestRunner,
)
from rl_task_foundry.synthesis.real_db_trial import (
    RealDbTrialRunner,
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
class _ScriptedSynthesisRuntime:
    """Yields a queue of outcomes per call: each item is either a draft or an exception."""

    outcomes: list[object]
    calls: list[dict[str, object]] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_topic: str | None = None,
    ) -> object:
        self.calls.append({"db_id": db_id, "requested_topic": requested_topic})
        if not self.outcomes:
            raise RuntimeError("scripted runtime exhausted")
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _RecordingRegistry:
    next_status: TaskRegistryCommitStatus = TaskRegistryCommitStatus.COMMITTED
    drafts: list[object] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.drafts is None:
            self.drafts = []

    def commit_draft(self, draft: object) -> TaskRegistryCommitResult:
        self.drafts.append(draft)
        task_id = draft.task_bundle.task_id
        return TaskRegistryCommitResult(
            status=self.next_status,
            task_id=task_id,
            exact_signature=f"sha256:{task_id}",
            filesystem_path=Path("/tmp") / "tasks" / task_id,
        )

    def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _NoopExporter:
    calls: list[tuple[Path, str | None]] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def export_bundle(
        self,
        bundle_root: Path,
        *,
        task_id: str | None = None,
        **_: object,
    ) -> object:
        self.calls.append((bundle_root, task_id))
        bundle_root.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(bundle_root=bundle_root)


def _config_with_tmp_traces(tmp_path: Path):
    config = load_config(Path("rl_task_foundry.yaml"))
    output = config.output.model_copy(
        update={
            "traces_dir": tmp_path / "traces",
            "run_db_path": tmp_path / "run.db",
        },
        deep=True,
    )
    return config.model_copy(update={"output": output}, deep=True)


def _accepted_draft(task_id: str):
    base = _sample_draft(tmp_task_id=task_id)
    return base.model_copy(
        update={
            "task_bundle": base.task_bundle.model_copy(
                update={
                    "status": "accepted",
                    "quality_metrics": base.task_bundle.quality_metrics.model_copy(
                        update={
                            "solver_pass_rate": 0.5,
                            "solver_ci_low": 0.3,
                            "solver_ci_high": 0.7,
                        }
                    ),
                }
            )
        }
    )


def _too_hard_failure() -> SynthesisArtifactGenerationError:
    return SynthesisArtifactGenerationError(
        "discarded: too_hard",
        attempts=[
            SynthesisGenerationAttempt(
                attempt_index=1,
                outcome=SynthesisGenerationOutcome.ARTIFACT_INVALID,
                provider="codex_oauth",
                model="gpt-5.4-mini",
                memory_summary="too_hard",
                artifact_diagnostics=SynthesisArtifactDiagnostics(
                    error_codes=["reject_too_hard"],
                ),
            )
        ],
        last_artifact_diagnostics=SynthesisArtifactDiagnostics(
            error_codes=["reject_too_hard"],
        ),
    )


def _build_factory(config, registry, exporter, runtime_outcomes_per_trial):
    """Returns a factory that pops one runtime per trial from a shared script."""

    runtimes_iter = iter(runtime_outcomes_per_trial)

    def factory():
        runtime = next(runtimes_iter)
        return RealDbTrialRunner(
            config,
            synthesis_runtime=runtime,
            registry=registry,
            exporter=exporter,
        )

    return factory


@pytest.mark.asyncio
async def test_harvest_runner_reaches_target(tmp_path: Path) -> None:
    config = _config_with_tmp_traces(tmp_path)
    registry = _RecordingRegistry()
    exporter = _NoopExporter()
    drafts = [_accepted_draft(f"task_h_{i:02d}") for i in range(3)]
    runtimes = [_ScriptedSynthesisRuntime(outcomes=[d]) for d in drafts]

    runner = HarvestRunner(
        config,
        registry=registry,
        exporter=exporter,
        trial_runner_factory=_build_factory(config, registry, exporter, runtimes),
    )
    out = tmp_path / "harvest_target"
    summary = await runner.run(
        out,
        db_id="sakila",
        target_committed=3,
        stall_timeout_seconds=60.0,
        parallel_workers=1,
    )
    await runner.close()

    assert summary.outcome is HarvestOutcome.TARGET_REACHED
    assert summary.committed == 3
    assert summary.attempted == 3
    assert summary.accepted_task_ids == ("task_h_00", "task_h_01", "task_h_02")
    assert (out / "phase_monitors.jsonl").exists()
    assert (out / "trials" / "trial_0001" / "debug" / "phase_monitors.jsonl").exists()
    # mirror aggregation includes harvest events + every trial's events
    aggregate = (out / "phase_monitors.jsonl").read_text().strip().splitlines()
    flow_kinds = {json.loads(line)["flow_kind"] for line in aggregate}
    assert "harvest" in flow_kinds
    assert "real_db_trial" in flow_kinds


@pytest.mark.asyncio
async def test_harvest_runner_stalls_when_no_commits(tmp_path: Path) -> None:
    config = _config_with_tmp_traces(tmp_path)
    registry = _RecordingRegistry()
    exporter = _NoopExporter()
    runtimes = [
        _ScriptedSynthesisRuntime(outcomes=[_too_hard_failure()]) for _ in range(50)
    ]

    runner = HarvestRunner(
        config,
        registry=registry,
        exporter=exporter,
        trial_runner_factory=_build_factory(config, registry, exporter, runtimes),
    )
    out = tmp_path / "harvest_stall"

    summary = await runner.run(
        out,
        db_id="sakila",
        target_committed=5,
        stall_timeout_seconds=2.0,
        parallel_workers=1,
    )
    await runner.close()

    assert summary.outcome is HarvestOutcome.STALLED
    assert summary.committed == 0
    assert summary.attempted >= 1
    assert summary.accepted_task_ids == ()


@dataclass(slots=True)
class _SharedConcurrencyRuntime:
    """One runtime shared across all trials; tracks peak in-flight trials."""

    outcomes: list[object]
    step_delay: float = 0.01
    in_flight: int = 0
    max_in_flight: int = 0
    total_calls: int = 0
    closed_count: int = 0

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_topic: str | None = None,
    ) -> object:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.total_calls += 1
        try:
            await asyncio.sleep(self.step_delay)
            if not self.outcomes:
                raise RuntimeError("scripted runtime exhausted")
            item = self.outcomes.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        finally:
            self.in_flight -= 1

    async def close(self) -> None:
        self.closed_count += 1


@pytest.mark.asyncio
async def test_harvest_runner_executes_trials_concurrently_with_four_workers(
    tmp_path: Path,
) -> None:
    config = _config_with_tmp_traces(tmp_path)
    registry = _RecordingRegistry()
    exporter = _NoopExporter()
    target = 8
    drafts: list[object] = [_accepted_draft(f"task_par_{i:02d}") for i in range(target * 2)]
    shared_runtime = _SharedConcurrencyRuntime(outcomes=drafts, step_delay=0.01)

    def factory():
        return RealDbTrialRunner(
            config,
            synthesis_runtime=shared_runtime,
            registry=registry,
            exporter=exporter,
        )

    runner = HarvestRunner(
        config,
        registry=registry,
        exporter=exporter,
        trial_runner_factory=factory,
    )
    out = tmp_path / "harvest_parallel4"

    summary = await runner.run(
        out,
        db_id="sakila",
        target_committed=target,
        stall_timeout_seconds=10.0,
        parallel_workers=4,
    )
    await runner.close()

    assert summary.outcome is HarvestOutcome.TARGET_REACHED
    assert summary.committed == target
    assert shared_runtime.max_in_flight >= 2, (
        "parallel_workers=4 harvest must overlap trials; "
        f"observed max_in_flight={shared_runtime.max_in_flight}"
    )
    # all four workers should have fetched at least one runtime call
    assert shared_runtime.total_calls >= target


@pytest.mark.asyncio
async def test_harvest_runner_skips_duplicates(tmp_path: Path) -> None:
    config = _config_with_tmp_traces(tmp_path)
    registry = _RecordingRegistry(next_status=TaskRegistryCommitStatus.DUPLICATE)
    exporter = _NoopExporter()
    runtimes = [
        _ScriptedSynthesisRuntime(outcomes=[_accepted_draft(f"task_dup_{i}")])
        for i in range(20)
    ]

    runner = HarvestRunner(
        config,
        registry=registry,
        exporter=exporter,
        trial_runner_factory=_build_factory(config, registry, exporter, runtimes),
    )
    out = tmp_path / "harvest_dup"

    summary = await runner.run(
        out,
        db_id="sakila",
        target_committed=2,
        stall_timeout_seconds=2.0,
        parallel_workers=1,
    )
    await runner.close()

    assert summary.outcome is HarvestOutcome.STALLED
    assert summary.committed == 0
    assert summary.attempted >= 1
