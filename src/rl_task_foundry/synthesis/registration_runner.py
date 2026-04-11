"""High-level registration runner for generated synthesis artifacts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rl_task_foundry.config.models import AppConfig
from rl_task_foundry.synthesis.registration_policy import (
    ArtifactKind,
    RegistrationError,
    validate_generated_module,
)
from rl_task_foundry.synthesis.subprocess_pool import (
    RegistrationExecutionResult,
    RegistrationSubprocessPool,
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


class ArtifactRegistrationResult(StrictModel):
    artifact_name: RegistrationArtifactName
    artifact_kind: ArtifactKind
    static_errors: list[RegistrationError] = Field(default_factory=list)
    execution_required: bool = False
    executed: bool = False
    execution_errors: list[RegistrationError] = Field(default_factory=list)
    execution_call_count: int | None = None
    execution_return_value: object | None = None

    @property
    def passed(self) -> bool:
        return self.static_passed and self.runtime_passed

    @property
    def static_passed(self) -> bool:
        return not self.static_errors

    @property
    def runtime_passed(self) -> bool:
        if not self.execution_required:
            return True
        return self.executed and not self.execution_errors


class RegistrationBundleReport(StrictModel):
    status: RegistrationBundleStatus
    tool: ArtifactRegistrationResult
    tool_self_test: ArtifactRegistrationResult
    solution: ArtifactRegistrationResult
    verifier: ArtifactRegistrationResult
    shadow_verifier: ArtifactRegistrationResult


async def run_registration_bundle(
    *,
    config: AppConfig,
    bundle: GeneratedArtifactBundle,
    pool: RegistrationSubprocessPool | None = None,
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
        static_errors=validate_generated_module(
            bundle.verifier_source,
            kind=ArtifactKind.VERIFIER_MODULE,
            policy=policy,
        ),
    )
    shadow_verifier = ArtifactRegistrationResult(
        artifact_name=RegistrationArtifactName.SHADOW_VERIFIER,
        artifact_kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
        static_errors=validate_generated_module(
            bundle.shadow_verifier_source,
            kind=ArtifactKind.SHADOW_VERIFIER_MODULE,
            policy=policy,
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
