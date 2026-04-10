"""Coverage counters."""

from __future__ import annotations

from collections import Counter


class CoverageTracker:
    """Simple labeled coverage counter."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def record(self, key: str) -> None:
        self._counts[key] += 1

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)
