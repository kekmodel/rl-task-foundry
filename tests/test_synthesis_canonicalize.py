from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rl_task_foundry.synthesis import canonicalize as canonicalize_module
from rl_task_foundry.synthesis.canonicalize import (
    CanonicalizationError,
    canonical_json,
    canonicalize_output,
    compute_reward,
)
from rl_task_foundry.synthesis.contracts import (
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
)


def _list_of_objects_schema(
    *,
    ordered: bool,
    sort_key: tuple[str, ...] | None = None,
) -> OutputSchemaContract:
    return OutputSchemaContract(
        root=OutputFieldContract(
            name="items",
            type=OutputFieldType.LIST,
            ordered=ordered,
            sort_key=sort_key,
            items=OutputFieldContract(
                name="item",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="id", type=OutputFieldType.INT),
                    OutputFieldContract(name="city", type=OutputFieldType.STRING),
                ],
            ),
        ),
        primary_output_format="json_array",
    )


def test_canonicalize_unordered_objects_reverses_input_via_sort_key() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="itinerary",
            type=OutputFieldType.LIST,
            ordered=False,
            sort_key=("time",),
            items=OutputFieldContract(
                name="day",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="time", type=OutputFieldType.DATE),
                    OutputFieldContract(name="city", type=OutputFieldType.STRING),
                ],
            ),
        ),
        primary_output_format="json_array",
    )

    canonical = canonicalize_output(
        schema,
        [
            {"time": "2026-10-03", "city": "Busan"},
            {"time": "2026-10-01", "city": "Seoul"},
            {"time": "2026-10-02", "city": "Jeju"},
        ],
    )

    assert canonical == [
        {"time": "2026-10-01", "city": "Seoul"},
        {"time": "2026-10-02", "city": "Jeju"},
        {"time": "2026-10-03", "city": "Busan"},
    ]


def test_canonicalize_output_preserves_ordered_list_order() -> None:
    schema = _list_of_objects_schema(ordered=True)

    canonical = canonicalize_output(
        schema,
        [
            {"id": 3, "city": "Busan"},
            {"id": 1, "city": "Seoul"},
        ],
    )

    assert canonical == [
        {"id": 3, "city": "Busan"},
        {"id": 1, "city": "Seoul"},
    ]


def test_sort_key_composite_sorts_by_multiple_components() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="entries",
            type=OutputFieldType.LIST,
            ordered=False,
            sort_key=("year", "month"),
            items=OutputFieldContract(
                name="entry",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="year", type=OutputFieldType.INT),
                    OutputFieldContract(name="month", type=OutputFieldType.INT),
                    OutputFieldContract(name="note", type=OutputFieldType.STRING),
                ],
            ),
        ),
        primary_output_format="json_array",
    )

    canonical = canonicalize_output(
        schema,
        [
            {"year": 2025, "month": 12, "note": "C"},
            {"year": 2026, "month": 1, "note": "A"},
            {"year": 2025, "month": 6, "note": "B"},
        ],
    )

    assert canonical == [
        {"year": 2025, "month": 6, "note": "B"},
        {"year": 2025, "month": 12, "note": "C"},
        {"year": 2026, "month": 1, "note": "A"},
    ]


def test_sort_key_tie_breaks_with_canonical_json() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="items",
            type=OutputFieldType.LIST,
            ordered=False,
            sort_key=("group",),
            items=OutputFieldContract(
                name="item",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="group", type=OutputFieldType.STRING),
                    OutputFieldContract(name="detail", type=OutputFieldType.STRING),
                ],
            ),
        ),
        primary_output_format="json_array",
    )

    canonical = canonicalize_output(
        schema,
        [
            {"group": "A", "detail": "zeta"},
            {"group": "A", "detail": "alpha"},
        ],
    )

    assert canonical == [
        {"group": "A", "detail": "alpha"},
        {"group": "A", "detail": "zeta"},
    ]


def test_unique_elements_removes_duplicate_objects() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="tags",
            type=OutputFieldType.LIST,
            ordered=False,
            unique_elements=True,
            items=OutputFieldContract(name="tag", type=OutputFieldType.STRING),
        ),
        primary_output_format="json_array",
    )

    canonical = canonicalize_output(
        schema,
        ["scenic", "cultural", "scenic", "cultural", "modern"],
    )

    assert canonical == ["cultural", "modern", "scenic"]


def test_schema_rejects_unordered_object_list_without_sort_key() -> None:
    with pytest.raises(ValidationError, match="unordered list of objects must declare sort_key"):
        OutputFieldContract(
            name="itinerary",
            type=OutputFieldType.LIST,
            ordered=False,
            items=OutputFieldContract(
                name="day",
                type=OutputFieldType.OBJECT,
                fields=[OutputFieldContract(name="time", type=OutputFieldType.DATE)],
            ),
        )


def test_schema_rejects_sort_key_referencing_nonexistent_field() -> None:
    with pytest.raises(ValidationError, match="sort_key references unknown field"):
        OutputFieldContract(
            name="entries",
            type=OutputFieldType.LIST,
            ordered=False,
            sort_key=("nonexistent_field",),
            items=OutputFieldContract(
                name="entry",
                type=OutputFieldType.OBJECT,
                fields=[OutputFieldContract(name="time", type=OutputFieldType.DATE)],
            ),
        )


def test_schema_rejects_sort_key_with_non_primitive_component() -> None:
    with pytest.raises(ValidationError, match="sort_key components must reference primitive"):
        OutputFieldContract(
            name="entries",
            type=OutputFieldType.LIST,
            ordered=False,
            sort_key=("nested",),
            items=OutputFieldContract(
                name="entry",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(
                        name="nested",
                        type=OutputFieldType.OBJECT,
                        fields=[OutputFieldContract(name="x", type=OutputFieldType.INT)],
                    )
                ],
            ),
        )


def test_schema_rejects_sort_key_with_nullable_component() -> None:
    with pytest.raises(ValidationError, match="sort_key components must reference non-nullable"):
        OutputFieldContract(
            name="entries",
            type=OutputFieldType.LIST,
            ordered=False,
            sort_key=("time",),
            items=OutputFieldContract(
                name="entry",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(
                        name="time",
                        type=OutputFieldType.DATE,
                        nullable=True,
                    )
                ],
            ),
        )


def test_schema_rejects_sort_key_on_ordered_list() -> None:
    with pytest.raises(ValidationError, match="cannot declare sort_key"):
        OutputFieldContract(
            name="entries",
            type=OutputFieldType.LIST,
            ordered=True,
            sort_key=("id",),
            items=OutputFieldContract(
                name="entry",
                type=OutputFieldType.OBJECT,
                fields=[OutputFieldContract(name="id", type=OutputFieldType.INT)],
            ),
        )


def test_schema_rejects_unique_elements_on_ordered_list() -> None:
    with pytest.raises(ValidationError, match="cannot declare unique_elements"):
        OutputFieldContract(
            name="entries",
            type=OutputFieldType.LIST,
            ordered=True,
            unique_elements=True,
            items=OutputFieldContract(name="entry", type=OutputFieldType.STRING),
        )


def test_canonicalize_int_rejects_boolean() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[OutputFieldContract(name="count", type=OutputFieldType.INT)],
        ),
        primary_output_format="json_object",
    )

    with pytest.raises(CanonicalizationError):
        canonicalize_output(schema, {"count": True})


def test_canonicalize_output_normalizes_date_and_datetime_to_iso() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[
                OutputFieldContract(name="ship_date", type=OutputFieldType.DATE),
                OutputFieldContract(name="shipped_at", type=OutputFieldType.DATETIME),
            ],
        ),
        primary_output_format="json_object",
    )

    canonical = canonicalize_output(
        schema,
        {
            "ship_date": "2026-04-12T08:10:00+09:00",
            "shipped_at": "2026-04-12T08:10:05+09:00",
        },
    )

    assert canonical == {
        "ship_date": "2026-04-12",
        "shipped_at": "2026-04-12T08:10:05+09:00",
    }


def test_canonicalize_output_rejects_unexpected_keys() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[OutputFieldContract(name="city", type=OutputFieldType.STRING)],
        ),
        primary_output_format="json_object",
    )

    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_output(schema, {"city": "Seoul", "budget": 10})

    assert "unexpected object keys" in str(exc_info.value)


def test_canonicalize_output_rejects_missing_required_field() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[
                OutputFieldContract(name="city", type=OutputFieldType.STRING),
                OutputFieldContract(name="budget", type=OutputFieldType.INT),
            ],
        ),
        primary_output_format="json_object",
    )

    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize_output(schema, {"city": "Seoul"})

    assert "$.budget" in str(exc_info.value)


def test_canonicalize_output_keeps_string_bytes_exact() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[OutputFieldContract(name="city", type=OutputFieldType.STRING)],
        ),
        primary_output_format="json_object",
    )

    canonical = canonicalize_output(schema, {"city": "  Seoul  "})

    assert canonical == {"city": "  Seoul  "}


def test_canonical_json_is_stable_for_exact_match() -> None:
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_compute_reward_returns_matched_for_equal_canonical() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[OutputFieldContract(name="city", type=OutputFieldType.STRING)],
        ),
        primary_output_format="json_object",
    )

    result = compute_reward(
        submitted_answer_text='{"city":"Seoul"}',
        canonical_answer={"city": "Seoul"},
        output_schema=schema,
    )

    assert result.reward == 1.0
    assert result.status == "matched"
    assert result.detail is None


def test_compute_reward_returns_json_decode_failed_for_malformed_json() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[OutputFieldContract(name="city", type=OutputFieldType.STRING)],
        ),
        primary_output_format="json_object",
    )

    result = compute_reward(
        submitted_answer_text="not valid json",
        canonical_answer={"city": "Seoul"},
        output_schema=schema,
    )

    assert result.reward == 0.0
    assert result.status == "json_decode_failed"
    assert result.detail is not None


def test_compute_reward_returns_schema_mismatch_for_invalid_shape() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[OutputFieldContract(name="city", type=OutputFieldType.STRING)],
        ),
        primary_output_format="json_object",
    )

    result = compute_reward(
        submitted_answer_text='{"wrong_field":"Seoul"}',
        canonical_answer={"city": "Seoul"},
        output_schema=schema,
    )

    assert result.reward == 0.0
    assert result.status == "schema_mismatch"


def test_compute_reward_returns_schema_mismatch_for_invalid_canonical_answer() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[OutputFieldContract(name="city", type=OutputFieldType.STRING)],
        ),
        primary_output_format="json_object",
    )

    result = compute_reward(
        submitted_answer_text='{"city":"Seoul"}',
        canonical_answer={"wrong_field": "Seoul"},
        output_schema=schema,
    )

    assert result.reward == 0.0
    assert result.status == "schema_mismatch"
    assert result.detail is not None
    assert "canonical answer failed validation" in result.detail


def test_compute_reward_returns_em_mismatch_for_different_canonical() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[OutputFieldContract(name="city", type=OutputFieldType.STRING)],
        ),
        primary_output_format="json_object",
    )

    result = compute_reward(
        submitted_answer_text='{"city":"Busan"}',
        canonical_answer={"city": "Seoul"},
        output_schema=schema,
    )

    assert result.reward == 0.0
    assert result.status == "em_mismatch"


def test_compute_reward_is_order_insensitive_for_unordered_list_with_sort_key() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="itinerary",
            type=OutputFieldType.LIST,
            ordered=False,
            sort_key=("time",),
            items=OutputFieldContract(
                name="day",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="time", type=OutputFieldType.DATE),
                    OutputFieldContract(name="city", type=OutputFieldType.STRING),
                ],
            ),
        ),
        primary_output_format="json_array",
    )

    submitted_text = json.dumps(
        [
            {"time": "2026-10-03", "city": "Busan"},
            {"time": "2026-10-01", "city": "Seoul"},
            {"time": "2026-10-02", "city": "Jeju"},
        ]
    )
    canonical_answer = [
        {"time": "2026-10-01", "city": "Seoul"},
        {"time": "2026-10-02", "city": "Jeju"},
        {"time": "2026-10-03", "city": "Busan"},
    ]

    result = compute_reward(
        submitted_answer_text=submitted_text,
        canonical_answer=canonical_answer,
        output_schema=schema,
    )

    assert result.reward == 1.0
    assert result.status == "matched"


def test_compute_reward_is_pure_function() -> None:
    schema = OutputSchemaContract(
        root=OutputFieldContract(
            name="answer",
            type=OutputFieldType.OBJECT,
            fields=[OutputFieldContract(name="x", type=OutputFieldType.INT)],
        ),
        primary_output_format="json_object",
    )

    first = compute_reward(
        submitted_answer_text='{"x":1}',
        canonical_answer={"x": 1},
        output_schema=schema,
    )
    second = compute_reward(
        submitted_answer_text='{"x":1}',
        canonical_answer={"x": 1},
        output_schema=schema,
    )

    assert first == second


def test_synthesis_canonicalize_module_has_zero_legacy_imports() -> None:
    module_source = Path(canonicalize_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(module_source)
    imported_modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert all(not name.startswith("rl_task_foundry.truth") for name in imported_modules)
    assert all(not name.startswith("rl_task_foundry.tools") for name in imported_modules)
    assert all(not name.startswith("rl_task_foundry.tasks") for name in imported_modules)
