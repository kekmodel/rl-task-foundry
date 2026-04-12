from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_task_foundry.config import load_config
from rl_task_foundry.pipeline.environment_orchestrator import EnvironmentRolloutSummary
from rl_task_foundry.synthesis.contracts import (
    CategoryTaxonomy,
    build_difficulty_vector,
)
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
    SynthesisArtifactDiagnostics,
    SynthesisArtifactGenerationError,
    SynthesisBackendFailure,
    SynthesisDifficultyRetrySeed,
    SynthesisGenerationAttempt,
    SynthesisGenerationOutcome,
    SynthesisPhase,
    SynthesisPhaseExecutionError,
    MaterializedCanonicalAnswerRecord,
    MaterializedInstanceRecord,
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
        requested_topic: str,
        retry_seed: SynthesisDifficultyRetrySeed | None = None,
    ) -> object:
        self.calls.append(
            {
                "db_id": db_id,
                "requested_topic": requested_topic,
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
            topic=CategoryTaxonomy.ASSIGNMENT,
        )
    finally:
        await runner.close()

    assert summary.trial_status is RealDbTrialStatus.ACCEPTED
    assert summary.quality_gate_status == "accept"
    assert summary.registry_status is EnvironmentRegistryCommitStatus.COMMITTED
    assert summary.bundle_root == output_root / "bundle"
    assert exporter.calls == [(output_root / "bundle", draft.environment.env_id)]
    payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert payload["trial_status"] == "accepted"
    assert payload["env_id"] == "env_real_trial"


@pytest.mark.asyncio
async def test_real_db_trial_runner_retries_too_easy_with_harder_synthesis(
    tmp_path: Path,
) -> None:
    first_draft = _sample_draft(
        tmp_env_id="env_too_easy",
        difficulty_vector=build_difficulty_vector(search_cost=1.0),
        question="가장 이른 고객의 store를 반환하세요.",
    )
    base_second_draft = _sample_draft(
        tmp_env_id="env_harder",
        difficulty_vector=build_difficulty_vector(search_cost=2.0),
        question="두 hop 탐색이 필요한 더 어려운 배정 질문입니다.",
        task_signature="sha256:harder-task",
    )
    second_draft = base_second_draft.model_copy(
        update={
            "environment": base_second_draft.environment.model_copy(
                update={
                    "cross_instance_set": base_second_draft.environment.cross_instance_set.model_copy(
                        update={"minimum_required": 1}
                    )
                }
            ),
            "instances": [
                MaterializedInstanceRecord(
                    instance_id="instance_0001",
                    rendered_user_prompt=(
                        "<entity>\n"
                        "{\"customer_id\": 2}\n"
                        "</entity>\n\n"
                        "두 hop 탐색이 필요한 더 어려운 배정 질문입니다.\n\n"
                        "# Submit Result Format\n"
                        "{\"properties\":{\"customer\":{\"title\":\"Customer\",\"type\":\"string\"},"
                        "\"day\":{\"format\":\"date\",\"title\":\"Day\",\"type\":\"string\"}},"
                        "\"required\":[\"customer\",\"day\"],\"title\":\"AnswerSchema\",\"type\":\"object\"}"
                    ),
                    params={"customer_id": 2},
                    anchor_values={"customer_id": 2},
                )
            ],
            "canonical_answers": [
                MaterializedCanonicalAnswerRecord(
                    instance_id="instance_0001",
                    canonical_answer={"customer": "Bob", "day": "2026-04-13"},
                    canonical_answer_json='{"customer":"Bob","day":"2026-04-13"}',
                    label_signature="sha256:harder-answer",
                )
            ],
        }
    )
    runtime = _FakeSynthesisRuntime(drafts=[first_draft, second_draft])
    orchestrator = _FakeEnvironmentOrchestrator(
        summaries=[
            _rollout_summary(env_id=first_draft.environment.env_id, matched=4, total=4),
            _rollout_summary(env_id=second_draft.environment.env_id, matched=2, total=4),
        ]
    )
    registry = _FakeRegistry(
        result=EnvironmentRegistryCommitResult(
            status=EnvironmentRegistryCommitStatus.COMMITTED,
            env_id=second_draft.environment.env_id,
            exact_signature="sha256:trial",
            difficulty_band=DifficultyBand.MEDIUM,
            filesystem_path=tmp_path / "environments" / second_draft.environment.env_id,
        )
    )
    runner = RealDbTrialRunner(
        _config_with_tmp_traces(tmp_path),
        synthesis_runtime=runtime,
        environment_orchestrator=orchestrator,
        registry=registry,
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

    assert summary.trial_status is RealDbTrialStatus.ACCEPTED
    assert len(runtime.calls) == 2
    assert runtime.calls[1]["retry_seed"] is not None
    assert runtime.calls[1]["retry_seed"].retry_requires_harder is True


@pytest.mark.asyncio
async def test_real_db_trial_runner_surfaces_artifact_generation_error_codes(
    tmp_path: Path,
) -> None:
    exc = SynthesisArtifactGenerationError(
        "label-first generation exhausted retries",
        attempts=[
            SynthesisGenerationAttempt(
                attempt_index=1,
                outcome=SynthesisGenerationOutcome.ARTIFACT_INVALID,
                provider="codex_oauth",
                model="gpt-5.4-mini",
                memory_summary="attempt failed",
                artifact_diagnostics=SynthesisArtifactDiagnostics(
                    error_codes=["canonical_answer_schema_mismatch"],
                    payload_repair_codes=["artifact_key_remapped"],
                ),
            )
        ],
        last_artifact_diagnostics=SynthesisArtifactDiagnostics(
            error_codes=["canonical_answer_schema_mismatch"],
            payload_repair_codes=["artifact_key_remapped"],
        ),
    )
    runner = RealDbTrialRunner(
        _config_with_tmp_traces(tmp_path),
        synthesis_runtime=_FakeSynthesisRuntime(exc=exc),
        environment_orchestrator=_FakeEnvironmentOrchestrator(
            summary=_rollout_summary(env_id="unused", matched=1, total=1)
        ),
        registry=_FakeRegistry(
            result=EnvironmentRegistryCommitResult(
                status=EnvironmentRegistryCommitStatus.COMMITTED,
                env_id="unused",
                exact_signature="sha256:x",
                difficulty_band=DifficultyBand.UNSET,
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
    assert summary.attempt_outcomes == ("artifact_invalid",)
    assert summary.error_codes == ("canonical_answer_schema_mismatch",)


def test_config_with_trial_traces_dir_rebinds_trace_outputs(tmp_path: Path) -> None:
    config = load_config(Path("rl_task_foundry.yaml"))

    updated = _config_with_trial_traces_dir(config, tmp_path / "trial_debug" / "traces")

    assert updated.output.traces_dir == tmp_path / "trial_debug" / "traces"
    assert updated.output.traces_dir != config.output.traces_dir


def test_real_db_trial_module_has_no_legacy_imports() -> None:
    module_path = Path("src/rl_task_foundry/synthesis/real_db_trial.py")
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


def test_synthesis_phase_execution_error_backend_failures_round_trip() -> None:
    exc = SynthesisPhaseExecutionError(
        "schema exploration failed",
        phase=SynthesisPhase.SCHEMA_EXPLORATION,
        backend_failures=[
            SynthesisBackendFailure(
                provider="codex_oauth",
                model="gpt-5.4-mini",
                error_type="ValidationError",
            )
        ],
    )

    assert exc.phase is SynthesisPhase.SCHEMA_EXPLORATION
    assert exc.backend_failures[0].error_type == "ValidationError"
