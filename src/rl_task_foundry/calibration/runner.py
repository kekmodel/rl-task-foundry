"""Calibration decisions from verification outcomes."""

from __future__ import annotations

from rl_task_foundry.calibration.banding import PassRateBand, clopper_pearson_interval
from rl_task_foundry.calibration.early_stop import EarlyStopDecision, safe_early_stop
from rl_task_foundry.tasks.models import VerifyResult


def compute_pass_rate(results: list[VerifyResult]) -> float:
    """Compute exact pass rate from verification results."""

    if not results:
        return 0.0
    return sum(1 for result in results if result.pass_exact) / len(results)


def accepted_by_band(results: list[VerifyResult], band: PassRateBand) -> bool:
    """Return whether a task should be accepted by pass-rate band."""

    return band.contains(compute_pass_rate(results))


def calibration_decision(
    *,
    total_replicas: int,
    results: list[VerifyResult],
    band: PassRateBand,
    ci_alpha: float,
) -> EarlyStopDecision:
    """Combine deterministic bounds and CI-based banding."""

    completed_replicas = len(results)
    passes_so_far = sum(1 for result in results if result.pass_exact)
    deterministic = safe_early_stop(
        total_replicas=total_replicas,
        completed_replicas=completed_replicas,
        passes_so_far=passes_so_far,
        lower_bound=band.lower,
        upper_bound=band.upper,
    )
    if deterministic != "continue":
        return deterministic
    interval = clopper_pearson_interval(
        successes=passes_so_far,
        trials=completed_replicas,
        alpha=ci_alpha,
    )
    if interval.below(band):
        return "reject_too_hard"
    if interval.above(band):
        return "reject_too_easy"
    if interval.inside(band):
        return "accept"
    return "continue"
