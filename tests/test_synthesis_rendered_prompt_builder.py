from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rl_task_foundry.synthesis.contracts import (
    ConstraintKind,
    ConstraintSummaryItem,
    OutputFieldContract,
    OutputFieldType,
    OutputSchemaContract,
    TaskContract,
)
from rl_task_foundry.synthesis.rendered_prompt_builder import build_rendered_user_prompt


def test_build_rendered_user_prompt_uses_only_entity_and_user_request() -> None:
    task = TaskContract(
        question="3일 itinerary를 만들어 주세요.",
        topic="itinerary",
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

    prompt = build_rendered_user_prompt(
        task,
        anchor_entity={"customer_id": 148},
    )

    assert prompt.startswith("<entity>\n")
    assert '{"customer_id": 148}' in prompt
    assert "</entity>\n\n3일 itinerary를 만들어 주세요." in prompt
    assert "# Submit Result Format" not in prompt
    assert '"type": "array"' not in prompt
    assert '"properties"' not in prompt
    assert "Constraints:" not in prompt
    assert "Instance Parameters:" not in prompt
    assert "submit_result as a JSON string matching this schema" not in prompt
    for authoring_term in (
        "curriculum",
        "specificity",
        "Cardinality",
        "Item-complexity",
        "pass rate",
        "quality gate",
        "solver",
        "actor",
        "training",
        "feedback",
    ):
        assert authoring_term not in prompt


def test_build_rendered_user_prompt_rejects_prebuilt_entity_block() -> None:
    task = TaskContract(
        question='<entity>\n{"customer_id": 148}\n</entity>\n\n고객의 배정 상태를 알려 주세요.',
        topic="assignment",
        output_schema=OutputSchemaContract(
            root=OutputFieldContract(
                name="answer",
                type=OutputFieldType.OBJECT,
                fields=[
                    OutputFieldContract(name="store_id", type=OutputFieldType.INT),
                ],
            ),
            primary_output_format="json_object",
        ),
    )

    with pytest.raises(ValueError, match="must not include an entity block"):
        build_rendered_user_prompt(
            task,
            anchor_entity={"customer_id": 148},
        )


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
