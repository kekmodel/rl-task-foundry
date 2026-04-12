from __future__ import annotations

from rl_task_foundry.pipeline.environment_orchestrator import (
    EnvironmentQualityGateStatus,
    EnvironmentQualityGateSummary,
)
from rl_task_foundry.synthesis.quality_gate import (
    accepted_draft_with_quality_metrics,
    shadow_disagreement_rate_for_draft,
)
from rl_task_foundry.synthesis.runtime import (
    SynthesisSelfConsistencyAttempt,
    SynthesisSelfConsistencyDiagnostics,
    SynthesisSelfConsistencyOutcome,
)
from tests.test_synthesis_environment_registry import _sample_draft


def _quality_gate_summary() -> EnvironmentQualityGateSummary:
    return EnvironmentQualityGateSummary(
        status=EnvironmentQualityGateStatus.ACCEPT,
        pass_rate=0.5,
        matched_solver_runs=1,
        total_solver_runs=2,
        ci_lower=0.1,
        ci_upper=0.9,
        band_lower=0.25,
        band_upper=0.75,
    )


def _attempt(
    *,
    attempt_index: int,
    verify_result: bool,
    shadow_verify_result: bool,
) -> SynthesisSelfConsistencyAttempt:
    return SynthesisSelfConsistencyAttempt(
        attempt_index=attempt_index,
        outcome=SynthesisSelfConsistencyOutcome.PASSED,
        provider="codex_oauth",
        model="gpt-5.4-mini",
        memory_summary="attempt",
        self_consistency_diagnostics=SynthesisSelfConsistencyDiagnostics(
            passed=verify_result and shadow_verify_result,
            verify_result=verify_result,
            shadow_verify_result=shadow_verify_result,
        ),
    )


def test_shadow_disagreement_rate_for_draft_uses_attempt_history() -> None:
    draft = _sample_draft().model_copy(
        update={
            "self_consistency_attempts": [
                _attempt(attempt_index=1, verify_result=True, shadow_verify_result=False),
                _attempt(attempt_index=2, verify_result=True, shadow_verify_result=True),
            ]
        }
    )

    assert shadow_disagreement_rate_for_draft(draft) == 0.5


def test_accepted_draft_with_quality_metrics_persists_shadow_disagreement_rate() -> None:
    draft = _sample_draft().model_copy(
        update={
            "self_consistency_attempts": [
                _attempt(attempt_index=1, verify_result=True, shadow_verify_result=True),
            ]
        }
    )

    accepted = accepted_draft_with_quality_metrics(
        draft,
        quality_gate_summary=_quality_gate_summary(),
    )

    assert accepted.environment.quality_metrics.shadow_disagreement_rate == 0.0
    assert accepted.environment.quality_metrics.solver_pass_rate == 0.5
