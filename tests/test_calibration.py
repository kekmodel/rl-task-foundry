from rl_task_foundry.calibration.banding import PassRateBand, clopper_pearson_interval
from rl_task_foundry.calibration.early_stop import safe_early_stop
from rl_task_foundry.calibration.runner import (
    calibration_decision,
    calibration_decision_from_counts,
)


def test_safe_early_stop_respects_upper_and_lower_bounds():
    assert (
        safe_early_stop(
            total_solver_runs=6,
            completed_solver_runs=5,
            passes_so_far=0,
            lower_bound=0.2,
            upper_bound=0.8,
        )
        == "reject_too_hard"
    )
    assert (
        safe_early_stop(
            total_solver_runs=6,
            completed_solver_runs=5,
            passes_so_far=5,
            lower_bound=0.2,
            upper_bound=0.8,
        )
        == "reject_too_easy"
    )
    assert (
        safe_early_stop(
            total_solver_runs=4,
            completed_solver_runs=4,
            passes_so_far=2,
            lower_bound=0.2,
            upper_bound=0.8,
        )
        == "accept"
    )


def test_clopper_pearson_interval_and_ci_decision():
    interval = clopper_pearson_interval(successes=2, trials=2, alpha=0.1)
    assert 0.0 <= interval.lower <= interval.upper <= 1.0

    band = PassRateBand(lower=0.2, upper=0.8)
    results = [False] * 14
    assert (
        calibration_decision(
            total_solver_runs=30,
            results=results,
            band=band,
            ci_alpha=0.1,
        )
        == "reject_too_hard"
    )
    assert (
        calibration_decision(
            total_solver_runs=30,
            results=[True] * 12,
            band=PassRateBand(lower=0.25, upper=0.75),
            ci_alpha=0.1,
        )
        == "reject_too_easy"
    )
    assert (
        calibration_decision_from_counts(
            total_solver_runs=30,
            completed_solver_runs=30,
            passes_so_far=24,
            band=PassRateBand(lower=0.25, upper=0.75),
            ci_alpha=0.1,
        )
        == "continue"
    )
