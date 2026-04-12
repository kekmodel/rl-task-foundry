from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.pipeline.environment_orchestrator import EnvironmentRolloutSummary
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.environment_registry import (
    DifficultyBand,
    EnvironmentRegistryCommitResult,
    EnvironmentRegistryCommitStatus,
)
from rl_task_foundry.synthesis.real_db_trial import (
    RealDbTrialRunner,
    RealDbTrialStatus,
)
from rl_task_foundry.synthesis.runtime import (
    SynthesisSelfConsistencyAttempt,
    SynthesisSelfConsistencyDiagnostics,
    SynthesisSelfConsistencyError,
    SynthesisSelfConsistencyOutcome,
)
from tests.test_synthesis_environment_registry import _sample_draft


@dataclass(slots=True)
class _FakeSynthesisRuntime:
    draft: object | None = None
    exc: Exception | None = None
    calls: list[tuple[str, CategoryTaxonomy]] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    async def synthesize_environment_draft(
        self,
        *,
        db_id: str,
        requested_category: CategoryTaxonomy,
    ) -> object:
        self.calls.append((db_id, requested_category))
        if self.exc is not None:
            raise self.exc
        assert self.draft is not None
        return self.draft

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _FakeEnvironmentOrchestrator:
    summary: EnvironmentRolloutSummary
    calls: list[object] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    async def run_draft(self, draft: object) -> EnvironmentRolloutSummary:
        self.calls.append(draft)
        return self.summary

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _FakeRegistry:
    result: EnvironmentRegistryCommitResult
    committed_drafts: list[object] | None = None

    def __post_init__(self) -> None:
        if self.committed_drafts is None:
            self.committed_drafts = []

    def commit_draft(self, draft: object) -> EnvironmentRegistryCommitResult:
        self.committed_drafts.append(draft)
        return self.result


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
        env_id: str | None = None,
        **_: object,
    ) -> object:
        self.calls.append((bundle_root, env_id))
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


def _rollout_summary(*, env_id: str, matched: int, total: int) -> EnvironmentRolloutSummary:
    runs = tuple(
        SimpleNamespace(
            reward_result=SimpleNamespace(
                status="matched" if index < matched else "em_mismatch"
            )
        )
        for index in range(total)
    )
    return EnvironmentRolloutSummary(
        env_id=env_id,
        db_id="sakila",
        planned_solver_runs=total,
        total_instances=1,
        total_solver_runs=total,
        matched_solver_runs=matched,
        runs=runs,
    )


@pytest.mark.asyncio
async def test_real_db_trial_runner_commits_and_exports_bundle(tmp_path: Path) -> None:
    draft = _sample_draft(tmp_env_id="env_real_trial")
    runtime = _FakeSynthesisRuntime(draft=draft)
    orchestrator = _FakeEnvironmentOrchestrator(
        summary=_rollout_summary(env_id=draft.environment.env_id, matched=1, total=2)
    )
    registry = _FakeRegistry(
        result=EnvironmentRegistryCommitResult(
            status=EnvironmentRegistryCommitStatus.COMMITTED,
            env_id=draft.environment.env_id,
            exact_signature="sha256:trial",
            difficulty_band=DifficultyBand.UNSET,
            filesystem_path=tmp_path / "environments" / draft.environment.env_id,
        )
    )
    exporter = _FakeExporter()
    runner = RealDbTrialRunner(
        _config_with_tmp_traces(tmp_path),
        synthesis_runtime=runtime,
        environment_orchestrator=orchestrator,
        registry=registry,
        exporter=exporter,
    )
    output_root = tmp_path / "real_trial"

    try:
        summary = await runner.run(
            output_root,
            db_id="sakila",
            category=CategoryTaxonomy.ASSIGNMENT,
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.ACCEPTED
    assert summary.quality_gate_status == "accept"
    assert summary.registry_status is EnvironmentRegistryCommitStatus.COMMITTED
    assert summary.registry_env_id == draft.environment.env_id
    assert summary.bundle_root == output_root / "bundle"
    assert summary.summary_path.exists()
    assert exporter.calls == [(output_root / "bundle", draft.environment.env_id)]
    assert len(registry.committed_drafts) == 1
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert payload["trial_status"] == "accepted"
    assert payload["env_id"] == "env_real_trial"
    assert payload["quality_gate_status"] == "accept"


@pytest.mark.asyncio
async def test_real_db_trial_runner_captures_synthesis_failure_taxonomy(
    tmp_path: Path,
) -> None:
    draft = _sample_draft()
    registration_diagnostics = draft.registration_diagnostics.model_copy(
        update={"error_codes": ["shadow_verifier_not_independent"]}
    )
    self_consistency_diagnostics = SynthesisSelfConsistencyDiagnostics(
        passed=False,
        error_codes=["canonical_answer_schema_mismatch"],
    )
    exc = SynthesisSelfConsistencyError(
        "artifact generation exhausted self-consistency budget after verifier failures",
        attempts=[
            SynthesisSelfConsistencyAttempt(
                attempt_index=1,
                outcome=SynthesisSelfConsistencyOutcome.REGISTRATION_FAILED,
                provider="codex_oauth",
                model="gpt-5.4-mini",
                memory_summary="retry",
                registration_diagnostics=registration_diagnostics,
                self_consistency_diagnostics=self_consistency_diagnostics,
            )
        ],
        last_registration_diagnostics=registration_diagnostics,
        last_self_consistency_diagnostics=self_consistency_diagnostics,
    )
    runtime = _FakeSynthesisRuntime(exc=exc)
    orchestrator = _FakeEnvironmentOrchestrator(
        summary=_rollout_summary(env_id="unused", matched=1, total=2)
    )
    registry = _FakeRegistry(
        result=EnvironmentRegistryCommitResult(
            status=EnvironmentRegistryCommitStatus.COMMITTED,
            env_id="unused",
            exact_signature="sha256:unused",
            difficulty_band=DifficultyBand.UNSET,
            filesystem_path=tmp_path / "unused",
        )
    )
    exporter = _FakeExporter()
    runner = RealDbTrialRunner(
        _config_with_tmp_traces(tmp_path),
        synthesis_runtime=runtime,
        environment_orchestrator=orchestrator,
        registry=registry,
        exporter=exporter,
    )

    try:
        summary = await runner.run(
            tmp_path / "real_trial_failure",
            db_id="sakila",
            category=CategoryTaxonomy.ASSIGNMENT,
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.SYNTHESIS_FAILED
    assert summary.synthesis_error_type == "SynthesisSelfConsistencyError"
    assert summary.attempt_outcomes == ("registration_failed",)
    assert summary.error_codes == (
        "shadow_verifier_not_independent",
        "canonical_answer_schema_mismatch",
    )
    assert not registry.committed_drafts
    assert not exporter.calls
    assert not orchestrator.calls


@pytest.mark.asyncio
async def test_real_db_trial_runner_rejects_cross_instance_before_rollout(
    tmp_path: Path,
) -> None:
    draft = _sample_draft(tmp_env_id="env_cross_instance_reject")
    invalid_cross_instance = draft.environment.cross_instance_set.model_copy(
        update={"minimum_required": 2}
    )
    draft = draft.model_copy(
        update={
            "environment": draft.environment.model_copy(
                update={"cross_instance_set": invalid_cross_instance}
            )
        }
    )
    runtime = _FakeSynthesisRuntime(draft=draft)
    orchestrator = _FakeEnvironmentOrchestrator(
        summary=_rollout_summary(env_id=draft.environment.env_id, matched=1, total=2)
    )
    registry = _FakeRegistry(
        result=EnvironmentRegistryCommitResult(
            status=EnvironmentRegistryCommitStatus.COMMITTED,
            env_id=draft.environment.env_id,
            exact_signature="sha256:unused",
            difficulty_band=DifficultyBand.UNSET,
            filesystem_path=tmp_path / "unused",
        )
    )
    exporter = _FakeExporter()
    runner = RealDbTrialRunner(
        _config_with_tmp_traces(tmp_path),
        synthesis_runtime=runtime,
        environment_orchestrator=orchestrator,
        registry=registry,
        exporter=exporter,
    )

    try:
        summary = await runner.run(
            tmp_path / "real_trial_cross_instance",
            db_id="sakila",
            category=CategoryTaxonomy.ASSIGNMENT,
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.REJECT_CROSS_INSTANCE
    assert "insufficient_instances" in summary.cross_instance_error_codes
    assert not orchestrator.calls
    assert not registry.committed_drafts
    assert not exporter.calls


def test_synthesis_real_db_trial_module_has_zero_legacy_imports() -> None:
    import rl_task_foundry.synthesis.real_db_trial as trial_module

    module_source = Path(trial_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(module_source)
    imported_modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert all(not name.startswith("rl_task_foundry.tools") for name in imported_modules)
    assert all(not name.startswith("rl_task_foundry.tasks") for name in imported_modules)
    assert all(not name.startswith("rl_task_foundry.truth") for name in imported_modules)
