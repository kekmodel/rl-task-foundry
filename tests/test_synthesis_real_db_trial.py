from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
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
    calls: list[dict[str, object]] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_topic: str,
    ) -> object:
        self.calls.append({"db_id": db_id, "requested_topic": requested_topic})
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


@dataclass(slots=True)
class _FakeExporter:
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


@pytest.mark.asyncio
async def test_real_db_trial_runner_commits_and_exports_bundle(tmp_path: Path) -> None:
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
    exporter = _FakeExporter()
    runner = RealDbTrialRunner(
        _config_with_tmp_traces(tmp_path),
        synthesis_runtime=runtime,
        registry=registry,
        exporter=exporter,
    )
    output_root = tmp_path / "real_trial"

    try:
        summary = await runner.run(
            output_root,
            db_id="sakila",
            topic=CategoryTaxonomy.ASSIGNMENT,
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.ACCEPTED
    assert summary.quality_gate_status == "accept"
    assert summary.registry_status is TaskRegistryCommitStatus.COMMITTED
    assert summary.bundle_root == output_root / "bundle"
    assert exporter.calls == [(output_root / "bundle", accepted_draft.task_bundle.task_id)]
    assert not (output_root / "trial_summary.json").exists()
    assert summary.task_id == "task_real_trial"
    assert summary.phase_monitor_log_path == output_root / "debug" / "phase_monitors.jsonl"
    assert registry.closed is False  # injected registry stays open; caller owns lifecycle


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
            )
        ],
        last_artifact_diagnostics=SynthesisArtifactDiagnostics(
            error_codes=["reject_too_easy"],
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
        exporter=_FakeExporter(),
    )

    try:
        summary = await runner.run(
            tmp_path / "real_trial",
            db_id="sakila",
            topic=CategoryTaxonomy.ASSIGNMENT,
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.SYNTHESIS_FAILED
    assert summary.synthesis_error_type == "SynthesisArtifactGenerationError"
    assert summary.attempt_outcomes == ("artifact_invalid",)
    assert summary.error_codes == ("reject_too_easy",)
    assert runner.registry.closed is False  # injected registry stays open
