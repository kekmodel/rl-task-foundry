"""Verification engine."""

from __future__ import annotations

from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.tasks.models import TaskSpec, VerifyResult
from rl_task_foundry.tasks.provenance import validate_provenance
from rl_task_foundry.truth.schemas import GroundTruth
from rl_task_foundry.verification.compare import compare_structured_output
from rl_task_foundry.verification.policies import VerificationPolicy
from rl_task_foundry.verification.shadow import (
    run_shadow_verifier,
    should_sample_shadow_verifier,
)


class VerificationEngine:
    """Deterministic verifier for solver results."""

    def __init__(self, policy: VerificationPolicy | None = None) -> None:
        self.policy = policy or VerificationPolicy()

    def verify(self, task: TaskSpec, ground_truth: GroundTruth, solver_result: SolverResult) -> VerifyResult:
        provenance = validate_provenance(task, solver_result)
        primary = compare_structured_output(
            task_id=task.task_id,
            solver_id=solver_result.solver_id,
            answer_schema=task.answer_schema,
            ground_truth=ground_truth,
            structured_output=solver_result.structured_output,
            provenance_pass=provenance.passed or not self.policy.require_provenance,
            fail_on_internal_field_leak=self.policy.fail_on_internal_field_leak,
            float_precision=self.policy.float_precision,
            missing_output_reason=solver_result.termination_reason,
        )
        if not should_sample_shadow_verifier(
            task_id=task.task_id,
            solver_id=solver_result.solver_id,
            sample_rate=self.policy.shadow_sample_rate,
        ):
            return primary
        shadow = run_shadow_verifier(
            task=task,
            ground_truth=ground_truth,
            solver_result=solver_result,
            primary_verification=primary,
            provenance_pass=provenance.passed or not self.policy.require_provenance,
            fail_on_internal_field_leak=self.policy.fail_on_internal_field_leak,
            float_precision=self.policy.float_precision,
        )
        return primary.model_copy(
            update={
                "shadow_verifier_status": shadow.status,
                "shadow_pass_exact": shadow.pass_exact,
                "shadow_failure_reason": shadow.failure_reason,
            }
        )
