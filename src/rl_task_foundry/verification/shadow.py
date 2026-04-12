"""Sampled shadow verification for judge disagreement monitoring."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

from rl_task_foundry.solver.models import SolverResult
from rl_task_foundry.tasks.models import TaskSpec, VerifyResult
from rl_task_foundry.truth.canonicalize import canonicalize_answer
from rl_task_foundry.truth.schemas import AnswerField, GroundTruth
from rl_task_foundry.verification.compare import compare_structured_output


@dataclass(slots=True)
class ShadowVerificationOutcome:
    status: Literal["not_run", "match", "disagree"]
    pass_exact: bool | None = None
    failure_reason: str | None = None


def should_sample_shadow_verifier(
    *,
    task_id: str,
    solver_id: str,
    sample_rate: float,
) -> bool:
    """Deterministically sample task/solver pairs for shadow verification."""

    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    digest = hashlib.blake2s(f"{task_id}:{solver_id}".encode("utf-8"), digest_size=8).digest()
    sample_value = int.from_bytes(digest, "big") / float(2**64)
    return sample_value < sample_rate


def run_shadow_verifier(
    *,
    task: TaskSpec,
    ground_truth: GroundTruth,
    solver_result: SolverResult,
    primary_verification: VerifyResult,
    provenance_pass: bool,
    fail_on_internal_field_leak: bool,
    float_precision: int,
) -> ShadowVerificationOutcome:
    """Run a secondary verification path from row_context-derived expectations."""

    try:
        shadow_expected = _reconstruct_expected_answer(
            task=task,
            ground_truth=ground_truth,
            float_precision=float_precision,
        )
    except Exception as exc:
        return ShadowVerificationOutcome(
            status="disagree",
            failure_reason=f"shadow_reconstruction_error:{type(exc).__name__}",
        )

    shadow_truth = ground_truth.model_copy(update={"canonical_answer": shadow_expected})
    shadow_verification = compare_structured_output(
        task_id=task.task_id,
        solver_id=solver_result.solver_id,
        answer_schema=task.answer_schema,
        ground_truth=shadow_truth,
        structured_output=solver_result.structured_output,
        provenance_pass=provenance_pass,
        fail_on_internal_field_leak=fail_on_internal_field_leak,
        float_precision=float_precision,
        missing_output_reason=solver_result.termination_reason,
    )
    if _shadow_matches_primary(primary_verification, shadow_verification):
        return ShadowVerificationOutcome(
            status="match",
            pass_exact=shadow_verification.pass_exact,
            failure_reason=shadow_verification.failure_reason,
        )
    return ShadowVerificationOutcome(
        status="disagree",
        pass_exact=shadow_verification.pass_exact,
        failure_reason=shadow_verification.failure_reason,
    )


def _shadow_matches_primary(
    primary_verification: VerifyResult,
    shadow_verification: VerifyResult,
) -> bool:
    if primary_verification.pass_exact != shadow_verification.pass_exact:
        return False
    if primary_verification.pass_exact and shadow_verification.pass_exact:
        return True
    return primary_verification.failure_reason == shadow_verification.failure_reason


def _reconstruct_expected_answer(
    *,
    task: TaskSpec,
    ground_truth: GroundTruth,
    float_precision: int,
) -> dict[str, object]:
    visible_fields = [
        field for field in task.answer_schema.fields if field.visibility == "user_visible"
    ]
    if not visible_fields:
        raise ValueError("shadow verifier requires at least one user_visible answer field")
    rows = ground_truth.row_context
    meta_markers = {_field_marker(field) for field in visible_fields}
    if meta_markers == {"meta:count"}:
        if len(visible_fields) != 1:
            raise ValueError("count shadow verification requires exactly one visible field")
        raw_answer = {visible_fields[0].name: rows[0]["count"] if rows else 0}
        return canonicalize_answer(
            task.answer_schema,
            raw_answer,
            float_precision=float_precision,
        )
    if meta_markers == {"meta:exists"}:
        if len(visible_fields) != 1:
            raise ValueError("exists shadow verification requires exactly one visible field")
        raw_answer = {visible_fields[0].name: rows[0]["exists"] if rows else False}
        return canonicalize_answer(
            task.answer_schema,
            raw_answer,
            float_precision=float_precision,
        )
    if all(field.type.startswith("list[") for field in visible_fields):
        raw_answer = {
            field.name: [row.get(field.name) for row in rows]
            for field in visible_fields
        }
        return canonicalize_answer(
            task.answer_schema,
            raw_answer,
            float_precision=float_precision,
        )
    if not rows:
        raw_answer = {field.name: None for field in visible_fields}
        return canonicalize_answer(
            task.answer_schema,
            raw_answer,
            float_precision=float_precision,
        )
    if len(rows) > 1:
        raise ValueError("shadow verifier scalar reconstruction is ambiguous")
    raw_answer = {field.name: rows[0].get(field.name) for field in visible_fields}
    return canonicalize_answer(
        task.answer_schema,
        raw_answer,
        float_precision=float_precision,
    )


def _field_marker(field: AnswerField) -> str | None:
    if not field.source_columns:
        return None
    marker = field.source_columns[0].strip().lower()
    if marker in {"meta:count", "meta:exists"}:
        return marker
    return None
