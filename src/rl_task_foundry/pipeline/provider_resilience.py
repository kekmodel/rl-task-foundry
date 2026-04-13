"""Provider health tracking and simple circuit-breaker utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import monotonic


@dataclass(slots=True)
class ProviderCircuitSnapshot:
    total_requests: int
    failures: int
    error_rate: float
    available: bool
    cooldown_remaining_s: float


@dataclass(slots=True)
class ProviderCircuitBreaker:
    """Small rolling-window circuit breaker for provider health."""

    provider_name: str
    window_s: int
    threshold: float
    probe_interval_s: int
    minimum_request_count: int
    _history: deque[tuple[float, bool]] = field(default_factory=deque, init=False, repr=False)
    _open_until_monotonic: float | None = field(default=None, init=False, repr=False)

    def _prune(self, now: float) -> None:
        while self._history and now - self._history[0][0] > self.window_s:
            self._history.popleft()

    def _stats(self, now: float) -> tuple[int, int]:
        self._prune(now)
        total_requests = len(self._history)
        failures = sum(1 for _, success in self._history if not success)
        return total_requests, failures

    def snapshot(self, *, now: float | None = None) -> ProviderCircuitSnapshot:
        observed_at = monotonic() if now is None else now
        total_requests, failures = self._stats(observed_at)
        error_rate = (failures / total_requests) if total_requests else 0.0
        cooldown_remaining_s = 0.0
        if self._open_until_monotonic is not None and observed_at < self._open_until_monotonic:
            cooldown_remaining_s = self._open_until_monotonic - observed_at
        return ProviderCircuitSnapshot(
            total_requests=total_requests,
            failures=failures,
            error_rate=error_rate,
            available=self.is_available(now=observed_at),
            cooldown_remaining_s=cooldown_remaining_s,
        )

    def is_available(self, *, now: float | None = None) -> bool:
        observed_at = monotonic() if now is None else now
        self._prune(observed_at)
        if self._open_until_monotonic is None:
            return True
        return observed_at >= self._open_until_monotonic

    def record_success(self, *, now: float | None = None) -> None:
        observed_at = monotonic() if now is None else now
        self._history.append((observed_at, True))
        self._prune(observed_at)
        if self._open_until_monotonic is not None and observed_at >= self._open_until_monotonic:
            self._open_until_monotonic = None

    def record_failure(self, *, now: float | None = None) -> bool:
        observed_at = monotonic() if now is None else now
        self._history.append((observed_at, False))
        total_requests, failures = self._stats(observed_at)
        if total_requests < self.minimum_request_count:
            return False
        error_rate = failures / total_requests if total_requests else 0.0
        if error_rate >= self.threshold:
            self._open_until_monotonic = observed_at + self.probe_interval_s
            return True
        return False

    def force_open(self, *, cooldown_s: int | None = None, now: float | None = None) -> None:
        observed_at = monotonic() if now is None else now
        duration_s = self.probe_interval_s if cooldown_s is None else cooldown_s
        self._open_until_monotonic = observed_at + duration_s
