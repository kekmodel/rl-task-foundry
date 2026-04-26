from __future__ import annotations

from rl_task_foundry.pipeline.solver_orchestrator import (
    TaskQualityGateStatus,
    TaskQualityGateSummary,
)
from rl_task_foundry.synthesis.quality_gate import accepted_draft_with_quality_metrics
from tests.test_synthesis_task_registry import _sample_draft


def _quality_gate_summary() -> TaskQualityGateSummary:
    return TaskQualityGateSummary(
        status=TaskQualityGateStatus.ACCEPT,
        pass_rate=0.5,
        matched_solver_runs=1,
        total_solver_runs=2,
        planned_solver_runs=3,
        evaluable_solver_runs=2,
        failed_solver_runs=1,
        ci_lower=0.1,
        ci_upper=0.9,
        band_lower=0.5,
        band_upper=0.9,
    )


def test_accepted_draft_with_quality_metrics_persists_solver_metrics() -> None:
    accepted = accepted_draft_with_quality_metrics(
        _sample_draft(),
        quality_gate_summary=_quality_gate_summary(),
    )

    assert accepted.task_bundle.quality_metrics.solver_pass_rate == 0.5
    assert accepted.task_bundle.quality_metrics.solver_ci_low == 0.1
    assert accepted.task_bundle.quality_metrics.solver_ci_high == 0.9
    assert accepted.task_bundle.quality_metrics.solver_planned_runs == 3
    assert accepted.task_bundle.quality_metrics.solver_completed_runs == 2
    assert accepted.task_bundle.quality_metrics.solver_evaluable_runs == 2
    assert accepted.task_bundle.quality_metrics.solver_failed_runs == 1
