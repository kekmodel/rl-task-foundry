from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from rl_task_foundry.synthesis.contracts import CategoryTaxonomy
from rl_task_foundry.synthesis.runtime import (
    SynthesisCategoryStatus,
    SynthesisGenerationOutcome,
)
from rl_task_foundry.synthesis.scheduler import (
    SynthesisDbSnapshot,
    SynthesisDomainScheduler,
    SynthesisSelectionStatus,
)


def _backed_off_status(
    *,
    db_id: str,
    category: CategoryTaxonomy,
    now: datetime,
    seconds: int,
) -> SynthesisCategoryStatus:
    return SynthesisCategoryStatus(
        db_id=db_id,
        category=category,
        consecutive_discards=2,
        backed_off=True,
        backoff_until=now + timedelta(seconds=seconds),
        backoff_remaining_s=float(seconds),
        last_outcome=SynthesisGenerationOutcome.ARTIFACT_INVALID,
        last_error_codes=["solution_source_missing"],
        last_updated_at=now,
    )


def test_synthesis_domain_scheduler_selects_round_robin_ready_pair() -> None:
    scheduler = SynthesisDomainScheduler()
    snapshots = [
        SynthesisDbSnapshot(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT, CategoryTaxonomy.ITINERARY],
        ),
        SynthesisDbSnapshot(
            db_id="northwind",
            categories=[CategoryTaxonomy.ASSIGNMENT],
        ),
    ]

    first = scheduler.choose_next(snapshots)
    second = scheduler.choose_next(snapshots)
    third = scheduler.choose_next(snapshots)

    assert first.status == SynthesisSelectionStatus.READY
    assert (first.db_id, first.category) == ("sakila", CategoryTaxonomy.ASSIGNMENT)
    assert second.status == SynthesisSelectionStatus.READY
    assert (second.db_id, second.category) == ("northwind", CategoryTaxonomy.ASSIGNMENT)
    assert third.status == SynthesisSelectionStatus.READY
    assert (third.db_id, third.category) == ("sakila", CategoryTaxonomy.ITINERARY)


def test_synthesis_domain_scheduler_preserves_db_turn_when_snapshot_list_mutates() -> None:
    scheduler = SynthesisDomainScheduler()
    initial = [
        SynthesisDbSnapshot(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT],
        ),
        SynthesisDbSnapshot(
            db_id="northwind",
            categories=[CategoryTaxonomy.ASSIGNMENT],
        ),
        SynthesisDbSnapshot(
            db_id="chinook",
            categories=[CategoryTaxonomy.ITINERARY],
        ),
    ]
    mutated = [
        SynthesisDbSnapshot(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT],
        ),
        SynthesisDbSnapshot(
            db_id="chinook",
            categories=[CategoryTaxonomy.ITINERARY],
        ),
    ]

    first = scheduler.choose_next(initial)
    second = scheduler.choose_next(mutated)

    assert (first.db_id, first.category) == ("sakila", CategoryTaxonomy.ASSIGNMENT)
    assert (second.db_id, second.category) == ("chinook", CategoryTaxonomy.ITINERARY)


def test_synthesis_domain_scheduler_preserves_category_order_when_snapshot_shrinks() -> None:
    scheduler = SynthesisDomainScheduler()
    initial = [
        SynthesisDbSnapshot(
            db_id="sakila",
            categories=[
                CategoryTaxonomy.ASSIGNMENT,
                CategoryTaxonomy.ITINERARY,
                CategoryTaxonomy.BUNDLE_SELECTION,
            ],
        )
    ]
    after_first = [
        SynthesisDbSnapshot(
            db_id="sakila",
            categories=[
                CategoryTaxonomy.ITINERARY,
                CategoryTaxonomy.BUNDLE_SELECTION,
            ],
        )
    ]
    after_second = [
        SynthesisDbSnapshot(
            db_id="sakila",
            categories=[CategoryTaxonomy.BUNDLE_SELECTION],
        )
    ]

    first = scheduler.choose_next(initial)
    second = scheduler.choose_next(after_first)
    third = scheduler.choose_next(after_second)

    assert (first.db_id, first.category) == ("sakila", CategoryTaxonomy.ASSIGNMENT)
    assert (second.db_id, second.category) == ("sakila", CategoryTaxonomy.ITINERARY)
    assert (third.db_id, third.category) == ("sakila", CategoryTaxonomy.BUNDLE_SELECTION)


def test_synthesis_domain_scheduler_skips_backed_off_category_within_db() -> None:
    scheduler = SynthesisDomainScheduler()
    now = datetime(2026, 4, 12, tzinfo=timezone.utc)
    snapshots = [
        SynthesisDbSnapshot(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT, CategoryTaxonomy.ITINERARY],
            category_status={
                CategoryTaxonomy.ASSIGNMENT: _backed_off_status(
                    db_id="sakila",
                    category=CategoryTaxonomy.ASSIGNMENT,
                    now=now,
                    seconds=60,
                )
            },
        )
    ]

    decision = scheduler.choose_next(snapshots, now=now)

    assert decision.status == SynthesisSelectionStatus.READY
    assert decision.db_id == "sakila"
    assert decision.category == CategoryTaxonomy.ITINERARY


def test_synthesis_domain_scheduler_falls_through_to_next_db() -> None:
    scheduler = SynthesisDomainScheduler()
    now = datetime(2026, 4, 12, tzinfo=timezone.utc)
    snapshots = [
        SynthesisDbSnapshot(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT],
            category_status={
                CategoryTaxonomy.ASSIGNMENT: _backed_off_status(
                    db_id="sakila",
                    category=CategoryTaxonomy.ASSIGNMENT,
                    now=now,
                    seconds=60,
                )
            },
        ),
        SynthesisDbSnapshot(
            db_id="northwind",
            categories=[CategoryTaxonomy.ITINERARY],
        ),
    ]

    decision = scheduler.choose_next(snapshots, now=now)

    assert decision.status == SynthesisSelectionStatus.READY
    assert decision.db_id == "northwind"
    assert decision.category == CategoryTaxonomy.ITINERARY


def test_synthesis_domain_scheduler_returns_earliest_backoff_when_all_blocked() -> None:
    scheduler = SynthesisDomainScheduler()
    now = datetime(2026, 4, 12, tzinfo=timezone.utc)
    snapshots = [
        SynthesisDbSnapshot(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT],
            category_status={
                CategoryTaxonomy.ASSIGNMENT: _backed_off_status(
                    db_id="sakila",
                    category=CategoryTaxonomy.ASSIGNMENT,
                    now=now,
                    seconds=120,
                )
            },
        ),
        SynthesisDbSnapshot(
            db_id="northwind",
            categories=[CategoryTaxonomy.ITINERARY],
            category_status={
                CategoryTaxonomy.ITINERARY: _backed_off_status(
                    db_id="northwind",
                    category=CategoryTaxonomy.ITINERARY,
                    now=now,
                    seconds=30,
                )
            },
        ),
    ]

    decision = scheduler.choose_next(snapshots, now=now)

    assert decision.status == SynthesisSelectionStatus.BACKOFF
    assert decision.wait_until == now + timedelta(seconds=30)
    assert decision.wait_seconds == 30.0


def test_synthesis_domain_scheduler_returns_empty_when_no_snapshots() -> None:
    scheduler = SynthesisDomainScheduler()

    decision = scheduler.choose_next([])

    assert decision.status == SynthesisSelectionStatus.EMPTY
    assert decision.db_id is None
    assert decision.category is None


def test_synthesis_db_snapshot_rejects_mismatched_category_status() -> None:
    now = datetime(2026, 4, 12, tzinfo=timezone.utc)

    with pytest.raises(ValidationError):
        SynthesisDbSnapshot(
            db_id="sakila",
            categories=[CategoryTaxonomy.ASSIGNMENT],
            category_status={
                CategoryTaxonomy.ASSIGNMENT: _backed_off_status(
                    db_id="northwind",
                    category=CategoryTaxonomy.ASSIGNMENT,
                    now=now,
                    seconds=60,
                )
            },
        )


def test_synthesis_category_status_requires_timezone_aware_datetimes() -> None:
    with pytest.raises(ValidationError):
        SynthesisCategoryStatus(
            db_id="sakila",
            category=CategoryTaxonomy.ASSIGNMENT,
            consecutive_discards=1,
            backed_off=True,
            backoff_until=datetime(2026, 4, 12),
        )
