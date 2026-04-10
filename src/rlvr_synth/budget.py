"""Budget tracker with hybrid mode (API cost + GPU hours) and reservation pattern."""

from __future__ import annotations

import logging
from typing import Literal
from uuid import uuid4

log = logging.getLogger(__name__)


class BudgetTracker:
    """Tracks API spend and GPU hours with reservation support."""

    def __init__(
        self,
        *,
        mode: Literal["hybrid", "token_cost", "gpu_hours"],
        max_api_usd: float,
        max_gpu_hours: float,
    ) -> None:
        self._mode = mode
        self._max_api_usd = max_api_usd
        self._max_gpu_hours = max_gpu_hours

        self.spent_api_usd: float = 0.0
        self.spent_gpu_hours: float = 0.0

        # Reservations: id -> (api_usd, gpu_hours)
        self._reservations: dict[str, tuple[float, float]] = {}

        # Accept rate tracking
        self._total_processed: int = 0
        self._total_accepted: int = 0

    @property
    def committed_api_usd(self) -> float:
        return sum(r[0] for r in self._reservations.values())

    @property
    def committed_gpu_hours(self) -> float:
        return sum(r[1] for r in self._reservations.values())

    def exceeded(self) -> bool:
        """Check if budget (spent + reserved) exceeds limits."""
        total_api = self.spent_api_usd + self.committed_api_usd
        total_gpu = self.spent_gpu_hours + self.committed_gpu_hours

        if self._mode in ("hybrid", "token_cost"):
            if total_api > self._max_api_usd:
                return True
        if self._mode in ("hybrid", "gpu_hours"):
            if total_gpu > self._max_gpu_hours:
                return True
        return False

    def record_api_cost(self, usd: float) -> None:
        self.spent_api_usd += usd

    def record_gpu_time(self, hours: float) -> None:
        self.spent_gpu_hours += hours

    def reserve(self, api_usd: float = 0.0, gpu_hours: float = 0.0) -> str:
        """Reserve estimated budget. Returns reservation ID."""
        rid = uuid4().hex[:8]
        self._reservations[rid] = (api_usd, gpu_hours)
        return rid

    def settle(
        self, reservation_id: str, actual_api_usd: float = 0.0, actual_gpu_hours: float = 0.0
    ) -> None:
        """Settle a reservation with actual usage."""
        self._reservations.pop(reservation_id, None)
        self.spent_api_usd += actual_api_usd
        self.spent_gpu_hours += actual_gpu_hours

    def record_processed(self, accepted: bool) -> None:
        self._total_processed += 1
        if accepted:
            self._total_accepted += 1

    def accept_rate_too_low(self, threshold: float) -> bool:
        if self._total_processed == 0:
            return False
        return (self._total_accepted / self._total_processed) < threshold

    def summary(self) -> dict[str, float]:
        return {
            "spent_api_usd": self.spent_api_usd,
            "committed_api_usd": self.committed_api_usd,
            "spent_gpu_hours": self.spent_gpu_hours,
            "committed_gpu_hours": self.committed_gpu_hours,
            "total_processed": self._total_processed,
            "total_accepted": self._total_accepted,
        }
