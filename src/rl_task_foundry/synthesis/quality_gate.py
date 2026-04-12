"""Shared helpers for applying solver-side quality gate results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl_task_foundry.synthesis.contracts import (
    EnvironmentContract,
    EnvironmentStatus,
)
from rl_task_foundry.synthesis.runtime import (
    SynthesisEnvironmentDraft,
    SynthesisSelfConsistencyDiagnostics,
)

if TYPE_CHECKING:
    from rl_task_foundry.pipeline.environment_orchestrator import EnvironmentQualityGateSummary


def shadow_disagreement_rate_for_draft(
    draft: SynthesisEnvironmentDraft,
) -> float | None:
    diagnostics = [
        attempt.self_consistency_diagnostics
        for attempt in draft.self_consistency_attempts
        if attempt.self_consistency_diagnostics is not None
    ]
    comparable = _comparable_shadow_diagnostics(diagnostics)
    if not comparable and draft.self_consistency_diagnostics is not None:
        comparable = _comparable_shadow_diagnostics([draft.self_consistency_diagnostics])
    if not comparable:
        return None
    disagreements = sum(
        1
        for diagnostic in comparable
        if diagnostic.verify_result != diagnostic.shadow_verify_result
    )
    return disagreements / len(comparable)


def accepted_draft_with_quality_metrics(
    draft: SynthesisEnvironmentDraft,
    *,
    quality_gate_summary: EnvironmentQualityGateSummary,
) -> SynthesisEnvironmentDraft:
    """Promote a draft to ACCEPTED and persist solver rollout metrics on it."""

    quality_metrics_payload = draft.environment.quality_metrics.model_dump(mode="python")
    quality_metrics_payload.update(
        {
            "shadow_disagreement_rate": shadow_disagreement_rate_for_draft(draft),
            "solver_pass_rate": quality_gate_summary.pass_rate,
            "solver_ci_low": quality_gate_summary.ci_lower,
            "solver_ci_high": quality_gate_summary.ci_upper,
        }
    )
    environment_payload = draft.environment.model_dump(mode="python")
    environment_payload.update(
        {
            "status": EnvironmentStatus.ACCEPTED,
            "quality_metrics": quality_metrics_payload,
        }
    )
    accepted_environment = EnvironmentContract.model_validate(environment_payload)
    return draft.model_copy(update={"environment": accepted_environment})


def _comparable_shadow_diagnostics(
    diagnostics: list[SynthesisSelfConsistencyDiagnostics],
) -> list[SynthesisSelfConsistencyDiagnostics]:
    return [
        diagnostic
        for diagnostic in diagnostics
        if isinstance(diagnostic.verify_result, bool)
        and isinstance(diagnostic.shadow_verify_result, bool)
    ]
