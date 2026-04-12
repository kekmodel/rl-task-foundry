"""Cross-instance consistency checks for synthesized environments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rl_task_foundry.synthesis.contracts import CrossInstanceSet
from rl_task_foundry.synthesis.runtime import (
    MaterializedCanonicalAnswerRecord,
    MaterializedInstanceRecord,
    SynthesisEnvironmentDraft,
)


class CrossInstanceConsistencyStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CrossInstanceConsistencySummary:
    status: CrossInstanceConsistencyStatus
    minimum_required: int
    instance_count: int
    canonical_answer_count: int
    error_codes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status is CrossInstanceConsistencyStatus.PASSED


def evaluate_cross_instance_draft(
    draft: SynthesisEnvironmentDraft,
) -> CrossInstanceConsistencySummary:
    return evaluate_cross_instance_consistency(
        cross_instance_set=draft.environment.cross_instance_set,
        instances=draft.instances,
        canonical_answers=draft.canonical_answers,
    )


def evaluate_cross_instance_consistency(
    *,
    cross_instance_set: CrossInstanceSet,
    instances: list[MaterializedInstanceRecord],
    canonical_answers: list[MaterializedCanonicalAnswerRecord],
) -> CrossInstanceConsistencySummary:
    error_codes: list[str] = []
    instance_count = len(instances)
    canonical_answer_count = len(canonical_answers)
    minimum_required = cross_instance_set.minimum_required

    instance_ids = [record.instance_id for record in instances]
    canonical_ids = [record.instance_id for record in canonical_answers]
    expected_ids = [record.instance_id for record in cross_instance_set.instances]
    expected_fingerprints = {
        record.instance_id: record.expected_solution_fingerprint
        for record in cross_instance_set.instances
    }
    actual_fingerprints = {
        record.instance_id: record.solution_fingerprint for record in canonical_answers
    }

    if instance_count < minimum_required:
        error_codes.append("insufficient_instances")
    if canonical_answer_count < minimum_required:
        error_codes.append("insufficient_canonical_answers")
    if len(set(instance_ids)) != len(instance_ids):
        error_codes.append("duplicate_instance_ids")
    if len(set(canonical_ids)) != len(canonical_ids):
        error_codes.append("duplicate_canonical_answer_ids")
    if set(expected_ids) != set(instance_ids) or set(expected_ids) != set(canonical_ids):
        error_codes.append("cross_instance_membership_mismatch")
    for instance_id, expected_fingerprint in expected_fingerprints.items():
        if expected_fingerprint is None:
            error_codes.append("missing_expected_solution_fingerprint")
            continue
        actual_fingerprint = actual_fingerprints.get(instance_id)
        if actual_fingerprint is None:
            continue
        if actual_fingerprint != expected_fingerprint:
            error_codes.append("cross_instance_solution_fingerprint_mismatch")
    if cross_instance_set.require_distinct_solution_fingerprints:
        fingerprints = list(actual_fingerprints.values())
        if len(set(fingerprints)) != len(fingerprints):
            error_codes.append("duplicate_solution_fingerprints")

    deduped_error_codes = tuple(dict.fromkeys(error_codes))
    return CrossInstanceConsistencySummary(
        status=(
            CrossInstanceConsistencyStatus.PASSED
            if not deduped_error_codes
            else CrossInstanceConsistencyStatus.FAILED
        ),
        minimum_required=minimum_required,
        instance_count=instance_count,
        canonical_answer_count=canonical_answer_count,
        error_codes=deduped_error_codes,
    )
