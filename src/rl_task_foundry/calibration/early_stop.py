"""Safe early termination rules."""

from __future__ import annotations

from typing import Literal


EarlyStopDecision = Literal["continue", "accept", "reject_too_easy", "reject_too_hard"]


def safe_early_stop(
    *,
    total_replicas: int,
    completed_replicas: int,
    passes_so_far: int,
    lower_bound: float,
    upper_bound: float,
) -> EarlyStopDecision:
    """Return a safe early-stop decision that respects both bounds."""

    remaining = total_replicas - completed_replicas
    min_final = passes_so_far / total_replicas
    max_final = (passes_so_far + remaining) / total_replicas
    if max_final < lower_bound:
        return "reject_too_hard"
    if min_final > upper_bound:
        return "reject_too_easy"
    if min_final >= lower_bound and max_final <= upper_bound:
        return "accept"
    return "continue"
