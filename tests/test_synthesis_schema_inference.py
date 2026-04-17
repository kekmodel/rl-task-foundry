from __future__ import annotations

import json

from rl_task_foundry.synthesis.canonicalize import compute_reward
from rl_task_foundry.synthesis.contracts import OutputFieldType
from rl_task_foundry.synthesis.schema_inference import (
    extract_output_schema_from_canonical,
)


def test_list_of_primitives_defaults_to_unordered_without_sort_key() -> None:
    schema = extract_output_schema_from_canonical(["seoul", "busan", "jeju"])

    assert schema.root.type is OutputFieldType.LIST
    assert schema.root.ordered is False
    assert schema.root.sort_key is None
    assert schema.primary_output_format == "json_array"


def test_list_of_objects_defaults_to_unordered_with_auto_sort_key() -> None:
    schema = extract_output_schema_from_canonical(
        [
            {"customer_id": 1, "customer_name": "alice"},
            {"customer_id": 2, "customer_name": "bob"},
        ]
    )

    assert schema.root.type is OutputFieldType.LIST
    assert schema.root.ordered is False
    assert schema.root.sort_key == ("customer_id", "customer_name")


def test_nested_list_inside_object_also_defaults_unordered() -> None:
    schema = extract_output_schema_from_canonical(
        {
            "anchor_customer": "alice",
            "top_rentals": [
                {"film": "matrix", "days": 3},
                {"film": "alien", "days": 5},
            ],
        }
    )

    assert schema.root.type is OutputFieldType.OBJECT
    list_field = next(child for child in schema.root.fields if child.name == "top_rentals")
    assert list_field.type is OutputFieldType.LIST
    assert list_field.ordered is False
    assert list_field.sort_key == ("film", "days")


def test_empty_list_defaults_to_unordered_with_fallback_item() -> None:
    schema = extract_output_schema_from_canonical([])

    assert schema.root.type is OutputFieldType.LIST
    assert schema.root.ordered is False
    assert schema.root.sort_key is None


def test_list_of_objects_without_primitive_fields_falls_back_to_ordered() -> None:
    schema = extract_output_schema_from_canonical(
        [
            {"anchor": {"id": 1, "name": "alice"}},
            {"anchor": {"id": 2, "name": "bob"}},
        ]
    )

    assert schema.root.type is OutputFieldType.LIST
    assert schema.root.ordered is True
    assert schema.root.sort_key is None


def test_reward_matches_same_items_in_different_order() -> None:
    canonical_answer = [
        {"customer_id": 1, "customer_name": "alice"},
        {"customer_id": 2, "customer_name": "bob"},
        {"customer_id": 3, "customer_name": "charlie"},
    ]
    schema = extract_output_schema_from_canonical(canonical_answer)

    shuffled = [
        {"customer_id": 3, "customer_name": "charlie"},
        {"customer_id": 1, "customer_name": "alice"},
        {"customer_id": 2, "customer_name": "bob"},
    ]
    result = compute_reward(
        submitted_answer_text=json.dumps(shuffled),
        canonical_answer=canonical_answer,
        output_schema=schema,
    )

    assert result.status == "matched"
    assert result.reward == 1.0


def test_reward_matches_list_of_primitives_in_different_order() -> None:
    canonical_answer = ["alice", "bob", "charlie"]
    schema = extract_output_schema_from_canonical(canonical_answer)

    result = compute_reward(
        submitted_answer_text=json.dumps(["charlie", "alice", "bob"]),
        canonical_answer=canonical_answer,
        output_schema=schema,
    )

    assert result.status == "matched"
    assert result.reward == 1.0
