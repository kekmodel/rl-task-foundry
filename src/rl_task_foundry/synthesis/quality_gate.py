"""Shared helpers for applying solver-side quality gate results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl_task_foundry.synthesis.contracts import (
    TaskBundleContract,
    TaskBundleStatus,
)
from rl_task_foundry.synthesis.runtime import SynthesisTaskDraft

if TYPE_CHECKING:
    from rl_task_foundry.pipeline.solver_orchestrator import TaskQualityGateSummary


def accepted_draft_with_quality_metrics(
    draft: SynthesisTaskDraft,
    *,
    quality_gate_summary: TaskQualityGateSummary,
) -> SynthesisTaskDraft:
    """Promote a draft to ACCEPTED and persist solver rollout metrics on it."""

    quality_metrics_payload = draft.task_bundle.quality_metrics.model_dump(mode="python")
    quality_metrics_payload.update(
        {
            "solver_pass_rate": quality_gate_summary.pass_rate,
            "solver_ci_low": quality_gate_summary.ci_lower,
            "solver_ci_high": quality_gate_summary.ci_upper,
        }
    )
    task_bundle_payload = draft.task_bundle.model_dump(mode="python")
    task_bundle_payload.update(
        {
            "status": TaskBundleStatus.ACCEPTED,
            "quality_metrics": quality_metrics_payload,
        }
    )
    accepted_task_bundle = TaskBundleContract.model_validate(task_bundle_payload)
    return draft.model_copy(update={"task_bundle": accepted_task_bundle})
