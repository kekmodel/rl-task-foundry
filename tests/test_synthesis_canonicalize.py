from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rl_task_foundry.synthesis import canonicalize as canonicalize_module
from rl_task_foundry.synthesis.canonicalize import (
    CanonicalizationError,
    canonical_json,
    canonicalize_output,
)
from rl_task_foundry.synthesis.contracts import (
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
)


def _list_of_objects_schema(*, ordered: bool) -> OutputSchemaContract:
    return OutputSchemaContract(
        root=OutputFieldContract(
            name="items",
            type=OutputFieldType.LIST,
            ordered=ordered,
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


def test_canonicalize_output_normalizes_nested_unordered_objects() -> None:
    schema = _list_of_objects_schema(ordered=False)

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
