"""Canonical answer comparison."""

from __future__ import annotations

from rl_task_foundry.tasks.models import VerifyResult
from rl_task_foundry.truth.canonicalize import canonicalize_answer
from rl_task_foundry.truth.schemas import AnswerSchema, GroundTruth


def compare_structured_output(
    *,
    task_id: str,
    solver_id: str,
    answer_schema: AnswerSchema,
    ground_truth: GroundTruth,
    structured_output: dict[str, object] | None,
    provenance_pass: bool,
    fail_on_internal_field_leak: bool = True,
    float_precision: int = 6,
    missing_output_reason: str | None = None,
) -> VerifyResult:
    """Compare a structured prediction to the canonical ground truth."""

    if structured_output is None:
        return VerifyResult(
            task_id=task_id,
            solver_id=solver_id,
            pass_exact=False,
            provenance_pass=provenance_pass,
            failure_reason=missing_output_reason or "missing_structured_output",
        )
    leaked_internal_fields = _internal_fields_in_output(answer_schema, structured_output)
    canonical_prediction = canonicalize_answer(
        answer_schema,
        structured_output,
        float_precision=float_precision,
    )
    field_scores = {
        key: canonical_prediction.get(key) == value for key, value in ground_truth.canonical_answer.items()
    }
    has_internal_field_leak = fail_on_internal_field_leak and bool(leaked_internal_fields)
    pass_exact = all(field_scores.values()) and provenance_pass and not has_internal_field_leak
    if has_internal_field_leak:
        failure_reason = "internal_field_leak"
    elif all(field_scores.values()):
        failure_reason = None
    else:
        failure_reason = "field_mismatch"
    return VerifyResult(
        task_id=task_id,
        solver_id=solver_id,
        pass_exact=pass_exact,
        field_scores=field_scores,
        provenance_pass=provenance_pass,
        canonical_prediction=canonical_prediction,
        failure_reason=failure_reason,
    )


def _internal_fields_in_output(
    answer_schema: AnswerSchema,
    structured_output: dict[str, object],
) -> list[str]:
    return [
        field.name
        for field in answer_schema.fields
        if field.visibility == "internal" and field.name in structured_output
    ]
