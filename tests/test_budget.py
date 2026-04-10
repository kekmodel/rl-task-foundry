"""Tests for the budget tracker."""

from __future__ import annotations

import pytest

from rlvr_synth.budget import BudgetTracker


def test_hybrid_mode_not_exceeded() -> None:
    bt = BudgetTracker(mode="hybrid", max_api_usd=100.0, max_gpu_hours=2.0)
    assert not bt.exceeded()
    bt.record_api_cost(50.0)
    bt.record_gpu_time(1.0)
    assert not bt.exceeded()


def test_hybrid_mode_api_exceeded() -> None:
    bt = BudgetTracker(mode="hybrid", max_api_usd=100.0, max_gpu_hours=2.0)
    bt.record_api_cost(101.0)
    assert bt.exceeded()


def test_hybrid_mode_gpu_exceeded() -> None:
    bt = BudgetTracker(mode="hybrid", max_api_usd=100.0, max_gpu_hours=2.0)
    bt.record_gpu_time(2.1)
    assert bt.exceeded()


def test_reservation_and_settle() -> None:
    bt = BudgetTracker(mode="hybrid", max_api_usd=100.0, max_gpu_hours=2.0)
    reservation_id = bt.reserve(api_usd=30.0, gpu_hours=0.5)
    # Reserved budget counts toward total
    assert bt.committed_api_usd == 30.0

    # Settle with actual (less than reserved)
    bt.settle(reservation_id, actual_api_usd=20.0, actual_gpu_hours=0.3)
    assert bt.spent_api_usd == 20.0
    assert bt.committed_api_usd == 0.0  # reservation cleared


def test_accept_rate_check() -> None:
    bt = BudgetTracker(mode="hybrid", max_api_usd=100.0, max_gpu_hours=2.0)
    assert not bt.accept_rate_too_low(threshold=0.05)

    bt.record_processed(accepted=False)
    bt.record_processed(accepted=False)
    bt.record_processed(accepted=False)
    # 0/3 = 0% < 5%
    assert bt.accept_rate_too_low(threshold=0.05)

    bt.record_processed(accepted=True)
    # 1/4 = 25% > 5%
    assert not bt.accept_rate_too_low(threshold=0.05)
