"""Unit tests for CursorPlan construction, hashing, and the cursor store."""

from __future__ import annotations

import pytest

from rl_task_foundry.tooling.atomic.cursor import (
    CursorStore,
    OrderNode,
    WhereNode,
    hash_plan,
    order_by,
    plan_target_table,
    plan_to_dict,
)


def _where(value: object = 45) -> WhereNode:
    return WhereNode(
        table="rental",
        column="customer_id",
        op="eq",
        value=value,
    )


def test_hash_plan_is_stable_across_equal_plans():
    a = _where(45)
    b = _where(45)
    assert hash_plan(a) == hash_plan(b)


def test_hash_plan_changes_with_value():
    assert hash_plan(_where(45)) != hash_plan(_where(46))


def test_hash_plan_changes_with_order_annotation():
    base = _where()
    annotated = OrderNode(source=base, column="rental_date", direction="desc")
    assert hash_plan(base) != hash_plan(annotated)


def test_plan_target_table_follows_outer_node():
    base = _where()
    ordered = OrderNode(source=base, column="rental_date", direction="asc")
    assert plan_target_table(base) == "rental"
    assert plan_target_table(ordered) == "rental"


def test_plan_to_dict_nests_source_plans_in_order():
    base = _where()
    ordered = OrderNode(source=base, column="rental_date", direction="asc")
    payload = plan_to_dict(ordered)
    assert payload["kind"] == "order"
    assert payload["column"] == "rental_date"
    assert payload["source"]["kind"] == "where"
    assert payload["source"]["table"] == "rental"


def test_cursor_store_interns_identical_plans_to_same_id():
    store = CursorStore()
    id_a = store.intern(_where(45))
    id_b = store.intern(_where(45))
    assert id_a == id_b
    assert len(store) == 1


def test_cursor_store_resolves_to_original_plan():
    store = CursorStore()
    plan = _where(45)
    cursor_id = store.intern(plan)
    assert store.resolve(cursor_id) == plan


def test_cursor_store_raises_when_capacity_exceeded():
    store = CursorStore(max_entries=2)
    store.intern(_where(1))
    store.intern(_where(2))
    with pytest.raises(RuntimeError):
        store.intern(_where(3))


def test_order_by_wraps_cursor_and_produces_new_id():
    store = CursorStore()
    base_id = store.intern(_where())
    ordered_id = order_by(store, base_id, "rental_date", "desc")
    assert ordered_id != base_id
    resolved = store.resolve(ordered_id)
    assert isinstance(resolved, OrderNode)
    assert resolved.column == "rental_date"
    assert resolved.direction == "desc"


def test_order_by_rejects_invalid_direction():
    store = CursorStore()
    cursor_id = store.intern(_where())
    with pytest.raises(ValueError):
        order_by(store, cursor_id, "rental_date", "up")  # type: ignore[arg-type]
