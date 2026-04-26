"""Calibration decisions from verification outcomes."""

from __future__ import annotations

from typing import Protocol

from rl_task_foundry.calibration.banding import (
    PassRateBand,
    clopper_pearson_one_sided_bounds,
)
from rl_task_foundry.calibration.early_stop import EarlyStopDecision


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
    """Return a decision only when exact binomial bounds are decisive."""

    completed_solver_runs = len(results)
    passes_so_far = sum(1 for result in results if _passed(result))
    return calibration_decision_from_counts(
        total_solver_runs=total_solver_runs,
        completed_solver_runs=completed_solver_runs,
        passes_so_far=passes_so_far,
        band=band,
        ci_alpha=ci_alpha,
    )


def calibration_decision_from_counts(
    *,
    total_solver_runs: int,
    completed_solver_runs: int,
    passes_so_far: int,
    band: PassRateBand,
    ci_alpha: float,
) -> EarlyStopDecision:
    """Return an exact-CI calibration decision from aggregate counts.

    ``total_solver_runs`` is retained for call-site clarity: callers may plan
    more solver samples, but confidence is based only on completed evaluable
    Bernoulli trials.
    """

    if completed_solver_runs <= 0:
        return "continue"
    interval = clopper_pearson_one_sided_bounds(
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
