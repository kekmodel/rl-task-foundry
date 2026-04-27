from __future__ import annotations

import json

from rl_task_foundry.synthesis.canonicalize import compute_reward
from rl_task_foundry.synthesis.contracts import OutputFieldType
from rl_task_foundry.synthesis.schema_inference import (
    extract_output_schema_from_canonical,
    extract_prompt_schema_from_canonical,
)


def test_list_of_primitives_defaults_to_ordered_without_sort_key() -> None:
    schema = extract_output_schema_from_canonical(["seoul", "busan", "jeju"])

    assert schema.root.type is OutputFieldType.LIST
    assert schema.root.ordered is True
    assert schema.root.sort_key is None
    assert schema.root.length == 3
    assert schema.primary_output_format == "json_array"


def test_list_of_objects_defaults_to_ordered_without_auto_sort_key() -> None:
    schema = extract_output_schema_from_canonical(
        [
            {"customer_id": 1, "customer_name": "alice"},
            {"customer_id": 2, "customer_name": "bob"},
        ]
    )

    assert schema.root.type is OutputFieldType.LIST
    assert schema.root.ordered is True
    assert schema.root.sort_key is None
    assert schema.root.length == 2


def test_nested_list_inside_object_also_defaults_ordered() -> None:
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
    assert list_field.ordered is True
    assert list_field.sort_key is None
    assert list_field.length == 2


def test_empty_list_defaults_to_ordered_with_fallback_item() -> None:
    schema = extract_output_schema_from_canonical([])

    assert schema.root.type is OutputFieldType.LIST
    assert schema.root.ordered is True
    assert schema.root.sort_key is None
    assert schema.root.length == 0


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


def test_reward_rejects_same_objects_in_different_order() -> None:
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

    assert result.status == "em_mismatch"
    assert result.reward == 0.0


def test_reward_rejects_primitive_list_in_different_order() -> None:
    canonical_answer = ["alice", "bob", "charlie"]
    schema = extract_output_schema_from_canonical(canonical_answer)

    result = compute_reward(
        submitted_answer_text=json.dumps(["charlie", "alice", "bob"]),
        canonical_answer=canonical_answer,
        output_schema=schema,
    )

    assert result.status == "em_mismatch"
    assert result.reward == 0.0


def test_schema_inference_marks_null_fields_nullable_across_list_items() -> None:
    canonical_answer = [
        {"medication": "A", "rate": 10.5, "rate_unit": "mL/hour"},
        {"medication": "B", "rate": None, "rate_unit": None},
    ]

    schema = extract_output_schema_from_canonical(canonical_answer)

    assert schema.root.items is not None
    item_fields = {field.name: field for field in schema.root.items.fields}
    assert item_fields["rate"].type is OutputFieldType.FLOAT
    assert item_fields["rate"].nullable is True
    assert item_fields["rate_unit"].type is OutputFieldType.STRING
    assert item_fields["rate_unit"].nullable is True

    result = compute_reward(
        submitted_answer_text=json.dumps(canonical_answer),
        canonical_answer=canonical_answer,
        output_schema=schema,
    )

    assert result.status == "matched"
    assert result.reward == 1.0


def test_prompt_schema_allows_null_for_nullable_canonical_fields() -> None:
    prompt_schema = extract_prompt_schema_from_canonical(
        [
            {"item": "A", "rate": None},
            {"item": "B", "rate": 4.25},
        ]
    )

    rate_schema = prompt_schema["$defs"]["AnswerSchemaItem"]["properties"]["rate"]

    assert {"type": "null"} in rate_schema["anyOf"]
    assert {"type": "number"} in rate_schema["anyOf"]
