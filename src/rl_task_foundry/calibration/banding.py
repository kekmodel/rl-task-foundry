"""Pass-rate band helpers."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta


@dataclass(slots=True)
class PassRateBand:
    lower: float
    upper: float

    def contains(self, pass_rate: float) -> bool:
        return self.lower <= pass_rate <= self.upper


@dataclass(slots=True)
class BinomialConfidenceInterval:
    lower: float
    upper: float

    def inside(self, band: PassRateBand) -> bool:
        return self.lower >= band.lower and self.upper <= band.upper

    def below(self, band: PassRateBand) -> bool:
        return self.upper < band.lower

    def above(self, band: PassRateBand) -> bool:
        return self.lower > band.upper


def clopper_pearson_interval(
    *,
    successes: int,
    trials: int,
    alpha: float,
) -> BinomialConfidenceInterval:
    """Return a conservative binomial confidence interval."""

    if trials <= 0:
        return BinomialConfidenceInterval(lower=0.0, upper=1.0)
    if successes <= 0:
        lower = 0.0
    else:
        lower = float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    if successes >= trials:
        upper = 1.0
    else:
        upper = float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return BinomialConfidenceInterval(lower=lower, upper=upper)


def clopper_pearson_one_sided_bounds(
    *,
    successes: int,
    trials: int,
    alpha: float,
) -> BinomialConfidenceInterval:
    """Return exact one-sided Clopper-Pearson bounds.

    The lower and upper values are directional confidence bounds at
    confidence ``1 - alpha``. They are used for sequential early-stop
    decisions, while the two-sided interval remains the reporting metric.
    """

    if trials <= 0:
        return BinomialConfidenceInterval(lower=0.0, upper=1.0)
    if successes <= 0:
        lower = 0.0
    else:
        lower = float(beta.ppf(alpha, successes, trials - successes + 1))
    if successes >= trials:
        upper = 1.0
    else:
        upper = float(beta.ppf(1 - alpha, successes + 1, trials - successes))
    return BinomialConfidenceInterval(lower=lower, upper=upper)
