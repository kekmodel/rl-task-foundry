"""High-level registration runner for generated synthesis artifacts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.contracts import MaterializedFactsSchema
from rl_task_foundry.synthesis.registration_policy import (
    ArtifactKind,
    RegistrationError,
    VerifierHybridAnalysis,
    analyze_verifier_module,
    validate_generated_module,
)
from rl_task_foundry.synthesis.subprocess_pool import (
    RegistrationExecutionResult,
    RegistrationSubprocessPool,
    RegistrationVerifierProbeResult,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegistrationArtifactName(StrEnum):
    TOOL = "tool"
    TOOL_SELF_TEST = "tool_self_test"
    SOLUTION = "solution"
    VERIFIER = "verifier"
    SHADOW_VERIFIER = "shadow_verifier"


class RegistrationBundleStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class GeneratedArtifactBundle(StrictModel):
    tool_source: str
    tool_self_test_source: str
    solution_source: str
    verifier_source: str
    shadow_verifier_source: str


class VerifierProbeSpec(StrictModel):
    answer_sample: object
    facts_schema: MaterializedFactsSchema


class ArtifactRegistrationResult(StrictModel):
    artifact_name: RegistrationArtifactName
    artifact_kind: ArtifactKind
    static_errors: list[RegistrationError] = Field(default_factory=list)
    verifier_hybrid_analysis: VerifierHybridAnalysis | None = None
    execution_required: bool = False
    executed: bool = False
    execution_errors: list[RegistrationError] = Field(default_factory=list)
    execution_call_count: int | None = None
    execution_return_value: object | None = None
    probe_required: bool = False
    probe_executed: bool = False
    probe_errors: list[RegistrationError] = Field(default_factory=list)
    verifier_execution_probe: RegistrationVerifierProbeResult | None = None

    @property
    def passed(self) -> bool:
        return self.static_passed and self.runtime_passed and self.probe_passed

    @property
    def static_passed(self) -> bool:
        return not self.static_errors

    @property
    def runtime_passed(self) -> bool:
        if not self.execution_required:
            return True
        return self.executed and not self.execution_errors

    @property
    def probe_passed(self) -> bool:
        if not self.probe_required:
            return True
        return self.probe_executed and not self.probe_errors


class RegistrationArtifactDiagnostics(StrictModel):
    artifact_name: RegistrationArtifactName
    passed: bool
    static_passed: bool
    runtime_passed: bool
    probe_passed: bool
    execution_required: bool = False
    executed: bool = False
    probe_required: bool = False
    probe_executed: bool = False
    static_error_codes: list[str] = Field(default_factory=list)
    execution_error_codes: list[str] = Field(default_factory=list)
    probe_error_codes: list[str] = Field(default_factory=list)
    weak_signal_codes: list[str] = Field(default_factory=list)
    verify_stage_calls: dict[str, int] = Field(default_factory=dict)
    fetch_facts_tool_calls: int | None = None
    fetch_facts_answer_references: int | None = None
    facts_match_answer_references: int | None = None
    facts_match_facts_references: int | None = None
    check_constraints_facts_references: int | None = None
    facts_match_constant_boolean_return: bool | None = None
    check_constraints_constant_boolean_return: bool | None = None
    probe_fetch_facts_return_keys: list[str] = Field(default_factory=list)
    probe_expected_fact_keys: list[str] = Field(default_factory=list)
    probe_missing_fact_keys: list[str] = Field(default_factory=list)
    probe_extra_fact_keys: list[str] = Field(default_factory=list)
    probe_fetch_facts_tool_calls: int | None = None
    probe_fetch_facts_answer_reads: int | None = None
    probe_facts_match_answer_reads: int | None = None
    probe_facts_match_facts_reads: int | None = None
    probe_check_constraints_facts_reads: int | None = None
    probe_verify_tool_calls: int | None = None
    probe_facts_match_result: bool | None = None
    probe_check_constraints_result: bool | None = None
    probe_verify_result: bool | None = None


class RegistrationBundleDiagnostics(StrictModel):
    status: RegistrationBundleStatus
    failing_artifacts: list[RegistrationArtifactName] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    weak_signal_codes: list[str] = Field(default_factory=list)
    tool: RegistrationArtifactDiagnostics
    tool_self_test: RegistrationArtifactDiagnostics
    solution: RegistrationArtifactDiagnostics
    verifier: RegistrationArtifactDiagnostics
    shadow_verifier: RegistrationArtifactDiagnostics


class RegistrationBundleReport(StrictModel):
    status: RegistrationBundleStatus
    tool: ArtifactRegistrationResult
    tool_self_test: ArtifactRegistrationResult
    solution: ArtifactRegistrationResult
    verifier: ArtifactRegistrationResult
    shadow_verifier: ArtifactRegistrationResult


def build_registration_diagnostics(
    report: RegistrationBundleReport,
) -> RegistrationBundleDiagnostics:
    tool = _artifact_diagnostics(report.tool)
    tool_self_test = _artifact_diagnostics(report.tool_self_test)
    solution = _artifact_diagnostics(report.solution)
    verifier = _artifact_diagnostics(report.verifier)
    shadow_verifier = _artifact_diagnostics(report.shadow_verifier)
    artifacts = [tool, tool_self_test, solution, verifier, shadow_verifier]
    return RegistrationBundleDiagnostics(
        status=report.status,
        failing_artifacts=[artifact.artifact_name for artifact in artifacts if not artifact.passed],
        error_codes=_dedupe_preserving_order(
            [
                *tool.static_error_codes,
                *tool.execution_error_codes,
                *tool.probe_error_codes,
                *tool_self_test.static_error_codes,
                *tool_self_test.execution_error_codes,
                *tool_self_test.probe_error_codes,
                *solution.static_error_codes,
                *solution.execution_error_codes,
                *solution.probe_error_codes,
                *verifier.static_error_codes,
                *verifier.execution_error_codes,
                *verifier.probe_error_codes,
                *shadow_verifier.static_error_codes,
                *shadow_verifier.execution_error_codes,
                *shadow_verifier.probe_error_codes,
            ]
        ),
        weak_signal_codes=_dedupe_preserving_order(
            [
                *tool.weak_signal_codes,
                *tool_self_test.weak_signal_codes,
                *solution.weak_signal_codes,
                *verifier.weak_signal_codes,
                *shadow_verifier.weak_signal_codes,
            ]
        ),
        tool=tool,
        tool_self_test=tool_self_test,
        solution=solution,
        verifier=verifier,
        shadow_verifier=shadow_verifier,
    )


def _artifact_diagnostics(result: ArtifactRegistrationResult) -> RegistrationArtifactDiagnostics:
    analysis = result.verifier_hybrid_analysis
    probe = result.verifier_execution_probe
    return RegistrationArtifactDiagnostics(
        artifact_name=result.artifact_name,
        passed=result.passed,
        static_passed=result.static_passed,
        runtime_passed=result.runtime_passed,
        probe_passed=result.probe_passed,
        execution_required=result.execution_required,
        executed=result.executed,
        probe_required=result.probe_required,
        probe_executed=result.probe_executed,
        static_error_codes=[error.code for error in result.static_errors],
        execution_error_codes=[error.code for error in result.execution_errors],
        probe_error_codes=[error.code for error in result.probe_errors],
        weak_signal_codes=_weak_signal_codes(result),
        verify_stage_calls=dict(analysis.verify_stage_calls) if analysis is not None else {},
        fetch_facts_tool_calls=analysis.fetch_facts_tool_calls if analysis is not None else None,
        fetch_facts_answer_references=analysis.fetch_facts_answer_references
        if analysis is not None
        else None,
        facts_match_answer_references=analysis.facts_match_answer_references
        if analysis is not None
        else None,
        facts_match_facts_references=analysis.facts_match_facts_references
        if analysis is not None
        else None,
        check_constraints_facts_references=analysis.check_constraints_facts_references
        if analysis is not None
        else None,
        facts_match_constant_boolean_return=analysis.facts_match_constant_boolean_return
        if analysis is not None
        else None,
        check_constraints_constant_boolean_return=analysis.check_constraints_constant_boolean_return
        if analysis is not None
        else None,
        probe_fetch_facts_return_keys=list(probe.fetch_facts_return_keys)
        if probe is not None
        else [],
        probe_expected_fact_keys=list(probe.expected_fact_keys) if probe is not None else [],
        probe_missing_fact_keys=list(probe.missing_fact_keys) if probe is not None else [],
        probe_extra_fact_keys=list(probe.extra_fact_keys) if probe is not None else [],
        probe_fetch_facts_tool_calls=probe.fetch_facts_tool_calls if probe is not None else None,
        probe_fetch_facts_answer_reads=probe.fetch_facts_answer_reads
        if probe is not None
        else None,
        probe_facts_match_answer_reads=probe.facts_match_answer_reads
        if probe is not None
        else None,
        probe_facts_match_facts_reads=probe.facts_match_facts_reads
        if probe is not None
        else None,
        probe_check_constraints_facts_reads=probe.check_constraints_facts_reads
        if probe is not None
        else None,
        probe_verify_tool_calls=probe.verify_tool_calls if probe is not None else None,
        probe_facts_match_result=probe.facts_match_result if probe is not None else None,
        probe_check_constraints_result=probe.check_constraints_result if probe is not None else None,
        probe_verify_result=probe.verify_result if probe is not None else None,
    )


def _weak_signal_codes(result: ArtifactRegistrationResult) -> list[str]:
    analysis = result.verifier_hybrid_analysis
    probe = result.verifier_execution_probe
    weak_signals: list[str] = []
    if analysis is not None:
        if analysis.fetch_facts_tool_calls == 0:
            weak_signals.append("fetch_facts_missing_tool_usage")
        if analysis.fetch_facts_answer_references == 0:
            weak_signals.append("fetch_facts_missing_answer_usage")
        if analysis.facts_match_answer_references == 0:
            weak_signals.append("facts_match_missing_answer_usage")
        if analysis.facts_match_facts_references == 0:
            weak_signals.append("facts_match_missing_facts_usage")
        if analysis.check_constraints_facts_references == 0:
            weak_signals.append("check_constraints_missing_facts_usage")
        if analysis.facts_match_constant_boolean_return:
            weak_signals.append("facts_match_constant_boolean")
        if analysis.check_constraints_constant_boolean_return:
            weak_signals.append("check_constraints_constant_boolean")
    if probe is not None:
        if probe.fetch_facts_tool_calls == 0:
            weak_signals.append("probe_fetch_facts_missing_tool_usage")
        if probe.fetch_facts_answer_reads == 0:
            weak_signals.append("probe_fetch_facts_missing_answer_usage")
        if probe.facts_match_answer_reads == 0:
            weak_signals.append("probe_facts_match_missing_answer_usage")
        if probe.facts_match_facts_reads == 0:
            weak_signals.append("probe_facts_match_missing_facts_usage")
        if probe.check_constraints_facts_reads == 0:
            weak_signals.append("probe_check_constraints_missing_facts_usage")
        if probe.missing_fact_keys or probe.extra_fact_keys:
            weak_signals.append("probe_facts_schema_key_drift")
    return _dedupe_preserving_order(weak_signals)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


async def run_registration_bundle(
    *,
    config: AppConfig,
    bundle: GeneratedArtifactBundle,
    pool: RegistrationSubprocessPool | None = None,
    verifier_probe_specs: dict[RegistrationArtifactName, VerifierProbeSpec] | None = None,
) -> RegistrationBundleReport:
    """Run Milestone 2 registration checks for a generated artifact bundle.

    Passing an existing pool is the normal production path. Letting the runner create
    its own pool is primarily a convenience for tests and one-off local checks.
    """

    policy = config.synthesis.registration_policy
    tool = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.TOOL,
        artifact_kind=ArtifactKind.TOOL_MODULE,
        static_errors=validate_generated_module(
            bundle.tool_source,
            kind=ArtifactKind.TOOL_MODULE,
            policy=policy,
        ),
    )
    tool_self_test = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.TOOL_SELF_TEST,
        artifact_kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
        execution_required=True,
        static_errors=validate_generated_module(
            bundle.tool_self_test_source,
            kind=ArtifactKind.TOOL_SELF_TEST_MODULE,
            policy=policy,
        ),
    )
    solution = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.SOLUTION,
        artifact_kind=ArtifactKind.SOLUTION_MODULE,
        static_errors=validate_generated_module(
            bundle.solution_source,
            kind=ArtifactKind.SOLUTION_MODULE,
            policy=policy,
        ),
    )
    verifier = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.VERIFIER,
        artifact_kind=ArtifactKind.VERIFIER_MODULE,
        probe_required=verifier_probe_specs is not None
        and RegistrationArtifactName.VERIFIER in verifier_probe_specs,
        static_errors=validate_generated_module(
            bundle.verifier_source,
            kind=ArtifactKind.VERIFIER_MODULE,
            policy=policy,
        ),
        verifier_hybrid_analysis=analyze_verifier_module(
            bundle.verifier_source,
            kind=ArtifactKind.VERIFIER_MODULE,
        ),
    )
    shadow_verifier = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.SHADOW_VERIFIER,
        artifact_kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
        probe_required=verifier_probe_specs is not None
        and RegistrationArtifactName.SHADOW_VERIFIER in verifier_probe_specs,
        static_errors=validate_generated_module(
            bundle.shadow_verifier_source,
            kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
            policy=policy,
        ),
        verifier_hybrid_analysis=analyze_verifier_module(
            bundle.shadow_verifier_source,
            kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
        ),
    )

    owns_pool = pool is None
    if tool.static_passed and tool_self_test.static_passed:
        pool = pool or await RegistrationSubprocessPool.start(config)
        execution_result = await pool.run_tool_self_test(
            tool_source=bundle.tool_source,
            self_test_source=bundle.tool_self_test_source,
        )
        tool_self_test.executed = True
        tool_self_test.execution_errors = execution_result.errors
        tool_self_test.execution_call_count = execution_result.call_count
        tool_self_test.execution_return_value = execution_result.return_value

    if verifier_probe_specs is not None and tool.static_passed:
        pool = pool or await RegistrationSubprocessPool.start(config)
        for artifact_result, artifact_name, verifier_source, artifact_kind in (
            (
                verifier,
                RegistrationArtifactName.VERIFIER,
                bundle.verifier_source,
                ArtifactKind.VERIFIER_MODULE,
            ),
            (
                shadow_verifier,
                RegistrationArtifactName.SHADOW_VERIFIER,
                bundle.shadow_verifier_source,
                ArtifactKind.SHADOW_VERIFIER_MODULE,
            ),
        ):
            probe_spec = verifier_probe_specs.get(artifact_name)
            if probe_spec is None or not artifact_result.static_passed:
                continue
            probe_result = await pool.probe_verifier_module(
                tool_source=bundle.tool_source,
                verifier_source=verifier_source,
                artifact_kind=artifact_kind,
                answer_sample=probe_spec.answer_sample,
                expected_fact_keys=[fact.key for fact in probe_spec.facts_schema.facts],
            )
            artifact_result.probe_executed = True
            artifact_result.probe_errors = probe_result.errors
            artifact_result.verifier_execution_probe = probe_result

    if owns_pool and pool is not None:
        await pool.close()

    status = (
        RegistrationBundleStatus.PASSED
        if all(
            result.passed
            for result in [tool, tool_self_test, solution, verifier, shadow_verifier]
        )
        else RegistrationBundleStatus.FAILED
    )
    return RegistrationBundleReport(
        status=status,
        tool=tool,
        tool_self_test=tool_self_test,
        solution=solution,
        verifier=verifier,
        shadow_verifier=shadow_verifier,
    )
