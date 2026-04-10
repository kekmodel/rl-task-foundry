from rl_task_foundry.infra.budget import BudgetLedger


def test_budget_ledger_reservation_ids_and_settlement():
    ledger = BudgetLedger(max_run_usd=10.0, max_gpu_hours=2.0)

    reservation_id = ledger.reserve(
        compose_api_usd=1.0,
        solve_api_usd=2.5,
        gpu_hours=0.5,
        metadata={"task_id": "task_1"},
    )

    assert isinstance(reservation_id, str)
    assert ledger.reserved_compose_usd == 1.0
    assert ledger.reserved_solve_usd == 2.5
    assert ledger.reserved_gpu_hours == 0.5

    settled = ledger.settle(reservation_id)

    assert settled.metadata == {"task_id": "task_1"}
    assert ledger.spent_compose_usd == 1.0
    assert ledger.spent_solve_usd == 2.5
    assert ledger.spent_gpu_hours == 0.5
    assert ledger.reserved_compose_usd == 0.0
    assert ledger.reserved_solve_usd == 0.0


def test_budget_ledger_abort_guard_and_budget_limit():
    ledger = BudgetLedger(max_run_usd=5.0)

    reservation_id = ledger.reserve(compose_api_usd=1.0, solve_api_usd=2.0)
    assert ledger.abort_if_accept_rate_below(
        accepted_examples=1,
        attempted_tasks=10,
        minimum_accept_rate=0.2,
    )

    ledger.release(reservation_id)

    try:
        ledger.reserve(compose_api_usd=4.0, solve_api_usd=2.0)
    except ValueError as exc:
        assert "budget exceeded" in str(exc)
    else:
        raise AssertionError("expected reserve to fail when projected budget is exceeded")
