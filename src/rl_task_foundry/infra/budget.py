"""Phase-specific budget accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class PhaseReservation:
    reservation_id: str
    compose_api_usd: float = 0.0
    solve_api_usd: float = 0.0
    gpu_hours: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


class BudgetLedger:
    """Track reserved and settled spend by phase."""

    def __init__(self, max_run_usd: float, *, max_gpu_hours: float | None = None) -> None:
        self.max_run_usd = max_run_usd
        self.max_gpu_hours = max_gpu_hours
        self.spent_compose_usd = 0.0
        self.spent_solve_usd = 0.0
        self.spent_gpu_hours = 0.0
        self._reservations: dict[str, PhaseReservation] = {}

    @property
    def reserved_compose_usd(self) -> float:
        return sum(reservation.compose_api_usd for reservation in self._reservations.values())

    @property
    def reserved_solve_usd(self) -> float:
        return sum(reservation.solve_api_usd for reservation in self._reservations.values())

    @property
    def reserved_gpu_hours(self) -> float:
        return sum(reservation.gpu_hours for reservation in self._reservations.values())

    def reserve(
        self,
        *,
        compose_api_usd: float = 0.0,
        solve_api_usd: float = 0.0,
        gpu_hours: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> str:
        projected = (
            self.spent_compose_usd
            + self.spent_solve_usd
            + self.reserved_compose_usd
            + self.reserved_solve_usd
            + compose_api_usd
            + solve_api_usd
        )
        if projected > self.max_run_usd:
            raise ValueError("budget exceeded")
        if self.max_gpu_hours is not None and (
            self.spent_gpu_hours + self.reserved_gpu_hours + gpu_hours
        ) > self.max_gpu_hours:
            raise ValueError("gpu budget exceeded")
        reservation_id = str(uuid4())
        self._reservations[reservation_id] = PhaseReservation(
            reservation_id=reservation_id,
            compose_api_usd=compose_api_usd,
            solve_api_usd=solve_api_usd,
            gpu_hours=gpu_hours,
            metadata=dict(metadata or {}),
        )
        return reservation_id

    def settle(
        self,
        reservation_id: str,
        *,
        compose_api_usd: float | None = None,
        solve_api_usd: float | None = None,
        gpu_hours: float | None = None,
    ) -> PhaseReservation:
        reservation = self._reservations.pop(reservation_id)
        compose_amount = reservation.compose_api_usd if compose_api_usd is None else compose_api_usd
        solve_amount = reservation.solve_api_usd if solve_api_usd is None else solve_api_usd
        gpu_amount = reservation.gpu_hours if gpu_hours is None else gpu_hours
        self.spent_compose_usd += compose_amount
        self.spent_solve_usd += solve_amount
        self.spent_gpu_hours += gpu_amount
        return PhaseReservation(
            reservation_id=reservation.reservation_id,
            compose_api_usd=compose_amount,
            solve_api_usd=solve_amount,
            gpu_hours=gpu_amount,
            metadata=reservation.metadata,
        )

    def release(self, reservation_id: str) -> PhaseReservation:
        return self._reservations.pop(reservation_id)

    def abort_if_accept_rate_below(
        self,
        *,
        accepted_examples: int,
        attempted_tasks: int,
        minimum_accept_rate: float,
        min_attempts: int,
    ) -> bool:
        if attempted_tasks < min_attempts or attempted_tasks == 0:
            return False
        return (accepted_examples / attempted_tasks) < minimum_accept_rate
