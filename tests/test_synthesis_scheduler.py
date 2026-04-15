from __future__ import annotations

from datetime import datetime, timezone

from rl_task_foundry.synthesis.scheduler import (
    SynthesisDbSnapshot,
    SynthesisDomainScheduler,
    SynthesisSelectionStatus,
)


def test_synthesis_domain_scheduler_selects_round_robin_ready_db() -> None:
    scheduler = SynthesisDomainScheduler()
    snapshots = [
        SynthesisDbSnapshot(db_id="sakila"),
        SynthesisDbSnapshot(db_id="northwind"),
    ]

    first = scheduler.choose_next(snapshots)
    second = scheduler.choose_next(snapshots)
    third = scheduler.choose_next(snapshots)

    assert first.status == SynthesisSelectionStatus.READY
    assert first.db_id == "sakila"
    assert second.status == SynthesisSelectionStatus.READY
    assert second.db_id == "northwind"
    assert third.status == SynthesisSelectionStatus.READY
    assert third.db_id == "sakila"


def test_synthesis_domain_scheduler_preserves_db_turn_when_snapshot_list_mutates() -> None:
    scheduler = SynthesisDomainScheduler()
    initial = [
        SynthesisDbSnapshot(db_id="sakila"),
        SynthesisDbSnapshot(db_id="northwind"),
        SynthesisDbSnapshot(db_id="chinook"),
    ]
    mutated = [
        SynthesisDbSnapshot(db_id="sakila"),
        SynthesisDbSnapshot(db_id="chinook"),
    ]

    first = scheduler.choose_next(initial)
    second = scheduler.choose_next(mutated)

    assert first.db_id == "sakila"
    assert second.db_id == "chinook"


def test_synthesis_domain_scheduler_returns_empty_when_no_snapshots() -> None:
    scheduler = SynthesisDomainScheduler()

    decision = scheduler.choose_next([])

    assert decision.status == SynthesisSelectionStatus.EMPTY
    assert decision.db_id is None
