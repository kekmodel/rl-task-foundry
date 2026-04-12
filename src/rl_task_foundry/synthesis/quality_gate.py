"""Shared helpers for applying solver-side quality gate results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl_task_foundry.synthesis.contracts import (
    EnvironmentContract,
    EnvironmentStatus,
)
from rl_task_foundry.synthesis.runtime import SynthesisEnvironmentDraft

if TYPE_CHECKING:
    from rl_task_foundry.pipeline.environment_orchestrator import EnvironmentQualityGateSummary


def accepted_draft_with_quality_metrics(
    draft: SynthesisEnvironmentDraft,
    *,
    quality_gate_summary: EnvironmentQualityGateSummary,
) -> SynthesisEnvironmentDraft:
    """Promote a draft to ACCEPTED and persist solver rollout metrics on it."""

    quality_metrics_payload = draft.environment.quality_metrics.model_dump(mode="python")
    quality_metrics_payload.update(
        {
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
