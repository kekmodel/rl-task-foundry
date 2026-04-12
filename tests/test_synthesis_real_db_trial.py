from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.pipeline.environment_orchestrator import EnvironmentRolloutSummary
from rl_task_foundry.synthesis.contracts import CategoryTaxonomy, build_difficulty_vector
from rl_task_foundry.synthesis.environment_registry import (
    DifficultyBand,
    EnvironmentRegistryCommitResult,
    EnvironmentRegistryCommitStatus,
)
from rl_task_foundry.synthesis.real_db_trial import (
    RealDbTrialRunner,
    RealDbTrialStatus,
    _config_with_trial_traces_dir,
)
from rl_task_foundry.synthesis.runtime import (
    SynthesisDifficultyRetrySeed,
    SynthesisBackendFailure,
    SynthesisPhase,
    SynthesisPhaseExecutionError,
    SynthesisSelfConsistencyAttempt,
    SynthesisSelfConsistencyDiagnostics,
    SynthesisSelfConsistencyError,
    SynthesisSelfConsistencyOutcome,
)
from tests.test_synthesis_environment_registry import _sample_draft


@dataclass(slots=True)
class _FakeSynthesisRuntime:
    draft: object | None = None
    drafts: list[object] | None = None
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
        requested_category: CategoryTaxonomy,
        retry_seed: SynthesisDifficultyRetrySeed | None = None,
    ) -> object:
        self.calls.append(
            {
                "db_id": db_id,
                "requested_category": requested_category,
                "retry_seed": retry_seed,
            }
        )
        if self.exc is not None:
            raise self.exc
        if self.drafts:
            return self.drafts.pop(0)
        assert self.draft is not None
        return self.draft

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _FakeEnvironmentOrchestrator:
    summary: EnvironmentRolloutSummary | None = None
    summaries: list[EnvironmentRolloutSummary] | None = None
    calls: list[object] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []
        if self.summaries is None:
            self.summaries = []

    async def run_draft(self, draft: object) -> EnvironmentRolloutSummary:
        self.calls.append(draft)
        if self.summaries:
            return self.summaries.pop(0)
        assert self.summary is not None
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
            "events_jsonl_path": tmp_path / "events.jsonl",
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
    assert summary.debug_root == output_root / "debug"
    assert summary.flow_id is not None
    assert summary.phase_monitor_log_path == output_root / "debug" / "phase_monitors.jsonl"
    assert summary.debug_traces_dir == output_root / "debug" / "traces"
    assert summary.synthesis_traces_dir == output_root / "debug" / "traces" / "synthesis"
    assert summary.solver_traces_dir == output_root / "debug" / "traces"
    assert summary.synthesis_session_db_path == output_root / "debug" / "traces" / "synthesis_sessions.sqlite"
    assert summary.solver_session_db_path == output_root / "debug" / "traces" / "sessions.sqlite"
    assert summary.bundle_root == output_root / "bundle"
    assert summary.summary_path.exists()
    assert exporter.calls == [(output_root / "bundle", draft.environment.env_id)]
    assert len(registry.committed_drafts) == 1
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert payload["trial_status"] == "accepted"
    assert payload["env_id"] == "env_real_trial"
    assert payload["quality_gate_status"] == "accept"
    assert payload["phase_monitor_log_path"] == str(output_root / "debug" / "phase_monitors.jsonl")
    assert payload["debug_root"] == str(output_root / "debug")
    assert payload["debug_traces_dir"] == str(output_root / "debug" / "traces")
    phase_monitor_lines = [
        json.loads(line)
        for line in summary.phase_monitor_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["phase"] for line in phase_monitor_lines] == [
        "cross_instance",
        "rollout",
        "quality_gate",
        "registry_commit",
        "bundle_export",
    ]


@pytest.mark.asyncio
async def test_real_db_trial_runner_retries_too_easy_with_harder_synthesis(
    tmp_path: Path,
) -> None:
    first_draft = _sample_draft(
        tmp_env_id="env_too_easy",
        difficulty_vector=build_difficulty_vector(search_cost=1.0),
        question="가장 이른 고객의 store를 반환하세요.",
    )
    second_draft = _sample_draft(
        tmp_env_id="env_harder",
        difficulty_vector=build_difficulty_vector(search_cost=2.0),
        question="두 hop 탐색이 필요한 더 어려운 배정 질문입니다.",
    )
    second_draft = second_draft.model_copy(
        update={
            "instances": [
                second_draft.instances[0].model_copy(
                    update={
                        "rendered_user_prompt": "두 hop 탐색이 필요한 더 어려운 배정 질문입니다.\n\n# Submit Result Format\n{}"
                    }
                )
            ]
        }
    )
    runtime = _FakeSynthesisRuntime(drafts=[first_draft, second_draft])
    orchestrator = _FakeEnvironmentOrchestrator(
        summaries=[
            _rollout_summary(env_id=first_draft.environment.env_id, matched=2, total=2),
            _rollout_summary(env_id=second_draft.environment.env_id, matched=1, total=2),
        ]
    )
    registry = _FakeRegistry(
        result=EnvironmentRegistryCommitResult(
            status=EnvironmentRegistryCommitStatus.COMMITTED,
            env_id=second_draft.environment.env_id,
            exact_signature="sha256:trial",
            difficulty_band=DifficultyBand.UNSET,
            filesystem_path=tmp_path / "environments" / second_draft.environment.env_id,
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
    output_root = tmp_path / "real_trial_retry"

    try:
        summary = await runner.run(
            output_root,
            db_id="sakila",
            category=CategoryTaxonomy.ASSIGNMENT,
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.ACCEPTED
    assert summary.quality_retry_count == 1
    assert summary.quality_retry_history == ("search_cost",)
    assert len(runtime.calls) == 2
    assert runtime.calls[0]["retry_seed"] is None
    retry_seed = runtime.calls[1]["retry_seed"]
    assert isinstance(retry_seed, SynthesisDifficultyRetrySeed)
    assert retry_seed.retry_requires_harder is True
    assert retry_seed.latest_quality_gate_feedback is not None
    assert retry_seed.latest_quality_gate_feedback.status == "reject_too_easy"
    assert retry_seed.latest_quality_gate_feedback.pass_rate == 1.0
    assert retry_seed.latest_quality_gate_feedback.matched_solver_runs == 2
    assert retry_seed.latest_quality_gate_feedback.total_solver_runs == 2
    assert retry_seed.latest_quality_gate_feedback.previous_env_id == "env_too_easy"
    assert retry_seed.latest_quality_gate_feedback.previous_question == "가장 이른 고객의 store를 반환하세요."
    assert (
        retry_seed.latest_quality_gate_feedback.previous_rendered_user_prompt
        == first_draft.instances[0].rendered_user_prompt
    )
    phase_monitor_lines = [
        json.loads(line)
        for line in summary.phase_monitor_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["phase"] for line in phase_monitor_lines] == [
        "cross_instance",
        "rollout",
        "quality_gate",
        "quality_retry",
        "cross_instance",
        "rollout",
        "quality_gate",
        "registry_commit",
        "bundle_export",
    ]


@pytest.mark.asyncio
async def test_real_db_trial_runner_rejects_semantically_unchanged_harder_retry(
    tmp_path: Path,
) -> None:
    first_draft = _sample_draft(
        tmp_env_id="env_too_easy_repeat",
        difficulty_vector=build_difficulty_vector(search_cost=1.0),
        question="가장 이른 고객의 store를 반환하세요.",
    )
    first_draft = first_draft.model_copy(
        update={
            "instances": [
                first_draft.instances[0].model_copy(
                    update={
                        "rendered_user_prompt": "가장 이른 고객의 store를 반환하세요.\n\n# Submit Result Format\n{}"
                    }
                )
            ]
        }
    )
    repeated_draft = _sample_draft(
        tmp_env_id="env_repeat_same",
        difficulty_vector=build_difficulty_vector(search_cost=1.0),
        question="가장 이른 고객의 store를 반환하세요.",
    )
    repeated_draft = repeated_draft.model_copy(
        update={
            "instances": [
                repeated_draft.instances[0].model_copy(
                    update={
                        "rendered_user_prompt": "가장 이른 고객의 store를 반환하세요.\n\n# Submit Result Format\n{}"
                    }
                )
            ],
            "canonical_answers": [
                repeated_draft.canonical_answers[0].model_copy(
                    update={"solution_fingerprint": "sha256:answer"}
                )
            ],
        }
    )
    harder_draft = _sample_draft(
        tmp_env_id="env_harder_after_repeat",
        difficulty_vector=build_difficulty_vector(search_cost=2.0),
        question="가장 이른 고객과 마지막 고객의 store를 비교해 더 작은 store_id를 반환하세요.",
        task_signature="sha256:harder",
    )
    harder_draft = harder_draft.model_copy(
        update={
            "instances": [
                harder_draft.instances[0].model_copy(
                    update={
                        "rendered_user_prompt": "가장 이른 고객과 마지막 고객의 store를 비교해 더 작은 store_id를 반환하세요.\n\n# Submit Result Format\n{}"
                    }
                )
            ],
            "canonical_answers": [
                harder_draft.canonical_answers[0].model_copy(
                    update={
                        "canonical_answer_json": json.dumps(
                            {"customer": "Bob", "day": "2026-04-13"},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "solution_fingerprint": "sha256:harder-answer",
                    }
                )
            ],
            "environment": harder_draft.environment.model_copy(
                update={
                    "cross_instance_set": harder_draft.environment.cross_instance_set.model_copy(
                        update={
                            "instances": [
                                harder_draft.environment.cross_instance_set.instances[
                                    0
                                ].model_copy(
                                    update={
                                        "expected_solution_fingerprint": "sha256:harder-answer"
                                    }
                                )
                            ]
                        }
                    )
                }
            ),
        }
    )
    runtime = _FakeSynthesisRuntime(drafts=[first_draft, repeated_draft, harder_draft])
    orchestrator = _FakeEnvironmentOrchestrator(
        summaries=[
            _rollout_summary(env_id=first_draft.environment.env_id, matched=2, total=2),
            _rollout_summary(env_id=harder_draft.environment.env_id, matched=1, total=2),
        ]
    )
    registry = _FakeRegistry(
        result=EnvironmentRegistryCommitResult(
            status=EnvironmentRegistryCommitStatus.COMMITTED,
            env_id=harder_draft.environment.env_id,
            exact_signature="sha256:trial",
            difficulty_band=DifficultyBand.UNSET,
            filesystem_path=tmp_path / "environments" / harder_draft.environment.env_id,
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
            tmp_path / "real_trial_semantic_retry",
            db_id="sakila",
            category=CategoryTaxonomy.ASSIGNMENT,
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.ACCEPTED
    assert len(runtime.calls) == 3
    assert len(orchestrator.calls) == 2
    phase_monitor_lines = [
        json.loads(line)
        for line in summary.phase_monitor_log_path.read_text(encoding="utf-8").splitlines()
    ]
    quality_retry_statuses = [
        line["status"] for line in phase_monitor_lines if line["phase"] == "quality_retry"
    ]
    assert "semantic_change_rejected" in quality_retry_statuses
    semantic_rejection = next(
        line
        for line in phase_monitor_lines
        if line["phase"] == "quality_retry" and line["status"] == "semantic_change_rejected"
    )
    assert semantic_rejection["checks"]["same_question"] is True
    assert semantic_rejection["checks"]["same_rendered_user_prompt"] is True
    assert semantic_rejection["checks"]["same_solution_fingerprints"] is True


@pytest.mark.asyncio
async def test_real_db_trial_runner_captures_phase_execution_taxonomy(
    tmp_path: Path,
) -> None:
    exc = SynthesisPhaseExecutionError(
        "synthesis phase schema_exploration failed across candidate providers: "
        "codex_oauth/gpt-5.4-mini: ModelBehaviorError",
        phase=SynthesisPhase.SCHEMA_EXPLORATION,
        backend_failures=[
            SynthesisBackendFailure(
                provider="codex_oauth",
                model="gpt-5.4-mini",
                error_type="ModelBehaviorError",
            )
        ],
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
            tmp_path / "real_trial_phase_failure",
            db_id="sakila",
            category=CategoryTaxonomy.ASSIGNMENT,
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.SYNTHESIS_FAILED
    assert summary.synthesis_error_type == "SynthesisPhaseExecutionError"
    assert summary.synthesis_phase == "schema_exploration"
    assert summary.backend_failures == ("codex_oauth/gpt-5.4-mini:ModelBehaviorError",)
    assert summary.phase_monitor_log_path is not None
    assert not registry.committed_drafts
    assert not exporter.calls
    assert not orchestrator.calls


def test_config_with_trial_traces_dir_rebinds_only_trace_root(tmp_path: Path) -> None:
    config = _config_with_tmp_traces(tmp_path)
    trial_traces_dir = tmp_path / "real_trial" / "debug" / "traces"

    updated = _config_with_trial_traces_dir(config, trial_traces_dir)

    assert updated.output.traces_dir == trial_traces_dir
    assert updated.output.run_db_path == config.output.run_db_path


@pytest.mark.asyncio
async def test_real_db_trial_runner_builds_default_runtime_with_run_scoped_debug_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_synthesis_traces: list[Path] = []
    captured_solver_traces: list[Path] = []

    draft = _sample_draft(tmp_env_id="env_real_trial_debug")

    class FakeConstructedRuntime(_FakeSynthesisRuntime):
        def __init__(self, config):
            captured_synthesis_traces.append(config.output.traces_dir)
            super().__init__(draft=draft)

    class FakeConstructedOrchestrator(_FakeEnvironmentOrchestrator):
        def __init__(self, config):
            captured_solver_traces.append(config.output.traces_dir)
            super().__init__(
                summary=_rollout_summary(env_id=draft.environment.env_id, matched=1, total=2)
            )

    from rl_task_foundry.synthesis import real_db_trial as real_db_trial_module

    monkeypatch.setattr(real_db_trial_module, "SynthesisAgentRuntime", FakeConstructedRuntime)
    monkeypatch.setattr(
        real_db_trial_module,
        "EnvironmentOrchestrator",
        FakeConstructedOrchestrator,
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
        registry=registry,
        exporter=exporter,
    )
    output_root = tmp_path / "real_trial_debug"

    try:
        summary = await runner.run(
            output_root,
            db_id="sakila",
            category=CategoryTaxonomy.ASSIGNMENT,
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.ACCEPTED
    assert captured_synthesis_traces == [output_root / "debug" / "traces"]
    assert captured_solver_traces == [output_root / "debug" / "traces"]


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
