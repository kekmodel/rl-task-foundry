from __future__ import annotations

import ast
from pathlib import Path

from rl_task_foundry.synthesis.contracts import (
    CategoryTaxonomy,
    ConstraintKind,
    ConstraintSummaryItem,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    TaskContract,
)
from rl_task_foundry.synthesis.rendered_prompt_builder import (
    build_output_schema_prompt_payload,
    build_rendered_user_prompt,
)


def test_build_output_schema_prompt_payload_includes_ordering_metadata() -> None:
    field = OutputFieldContract(
        name="itinerary",
        type=OutputFieldType.LIST,
        ordered=False,
        sort_key=("day",),
        unique_elements=True,
        items=OutputFieldContract(
            name="entry",
            type=OutputFieldType.OBJECT,
            fields=[
                OutputFieldContract(name="day", type=OutputFieldType.INT),
                OutputFieldContract(name="city", type=OutputFieldType.STRING),
            ],
        ),
    )

    payload = build_output_schema_prompt_payload(field)

    assert payload["type"] == "list"
    assert payload["ordered"] is False
    assert payload["sort_key"] == ["day"]
    assert payload["unique_elements"] is True
    assert payload["items"]["properties"]["day"]["type"] == "int"


def test_build_rendered_user_prompt_includes_constraints_and_instance_parameters() -> None:
    task = TaskContract(
        question="3일 itinerary를 만들어 주세요.",
        category=CategoryTaxonomy.ITINERARY,
        output_schema=OutputSchemaContract(
            root=OutputFieldContract(
                name="itinerary",
                type=OutputFieldType.LIST,
                ordered=False,
                sort_key=("day",),
                items=OutputFieldContract(
                    name="entry",
                    type=OutputFieldType.OBJECT,
                    fields=[
                        OutputFieldContract(name="day", type=OutputFieldType.INT),
                        OutputFieldContract(name="city", type=OutputFieldType.STRING),
                    ],
                ),
            ),
            primary_output_format="json_array",
        ),
        constraint_summary=[
            ConstraintSummaryItem(
                key="unique_city",
                kind=ConstraintKind.UNIQUENESS,
                summary="city는 중복되면 안 된다.",
            )
        ],
        instance_parameters={"season": "spring", "day_count": 3},
    )

    prompt = build_rendered_user_prompt(task)

    assert "3일 itinerary를 만들어 주세요." in prompt
    assert "Constraints:" in prompt
    assert "- city는 중복되면 안 된다." in prompt
    assert "Instance Parameters:" in prompt
    assert '- season: "spring"' in prompt
    assert "- day_count: 3" in prompt
    assert "Submit Result Format:" in prompt
    assert '"sort_key": [' in prompt
    assert "Call submit_result with a JSON string" in prompt


def test_synthesis_rendered_prompt_builder_module_has_zero_legacy_imports() -> None:
    from rl_task_foundry.synthesis import rendered_prompt_builder as builder_module

    module_source = Path(builder_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(module_source)
    imported_modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert all(not name.startswith("rl_task_foundry.tools") for name in imported_modules)
    assert all(not name.startswith("rl_task_foundry.tasks") for name in imported_modules)
    assert all(not name.startswith("rl_task_foundry.truth") for name in imported_modules)
