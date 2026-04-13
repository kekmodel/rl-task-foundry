"""Calibration decisions from verification outcomes."""

from __future__ import annotations

from typing import Protocol

from rl_task_foundry.calibration.banding import PassRateBand, clopper_pearson_interval
from rl_task_foundry.calibration.early_stop import EarlyStopDecision, safe_early_stop


class PassExactResult(Protocol):
    pass_exact: bool


def _passed(result: bool | PassExactResult) -> bool:
    if isinstance(result, bool):
        return result
    return result.pass_exact


def compute_pass_rate(results: list[bool | PassExactResult]) -> float:
    """Compute exact pass rate from verification results."""

    if not results:
        return 0.0
    return sum(1 for result in results if _passed(result)) / len(results)


def accepted_by_band(results: list[bool | PassExactResult], band: PassRateBand) -> bool:
    """Return whether a task should be accepted by pass-rate band."""

    return band.contains(compute_pass_rate(results))


def calibration_decision(
    *,
    total_solver_runs: int,
    results: list[bool | PassExactResult],
    band: PassRateBand,
    ci_alpha: float,
) -> EarlyStopDecision:
    """Combine deterministic bounds and CI-based banding."""

    completed_solver_runs = len(results)
    passes_so_far = sum(1 for result in results if _passed(result))
    deterministic = safe_early_stop(
        total_solver_runs=total_solver_runs,
        completed_solver_runs=completed_solver_runs,
        passes_so_far=passes_so_far,
        lower_bound=band.lower,
        upper_bound=band.upper,
    )
    if deterministic != "continue":
        return deterministic
    interval = clopper_pearson_interval(
        successes=passes_so_far,
        trials=completed_solver_runs,
        alpha=ci_alpha,
    )
    if interval.below(band):
        return "reject_too_hard"
    if interval.above(band):
        return "reject_too_easy"
    if interval.inside(band):
        return "accept"
    return "continue"
