from __future__ import annotations

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.prompts import (
    build_synthesis_agent_instructions,
    build_synthesis_input,
)


def _composer_surface() -> dict[str, object]:
    return {
        "tool_count": 5,
        "tools": [
            {
                "name": "schema_map",
                "description": "Schema graph slice with hub/bridge tags.",
            },
            {
                "name": "profile",
                "description": "Row/distinct/null counts + top-k frequency.",
            },
            {
                "name": "sample",
                "description": "Up to n representative rows with optional seed.",
            },
            {
                "name": "neighborhood",
                "description": "Anchor row + per-edge sample IDs.",
            },
            {
                "name": "query",
                "description": "JSON DSL compiler for canonical answers.",
            },
        ],
        "solver_primitives": {
            "set_producing": ["rows_where", "rows_via", "intersect"],
            "set_annotating": ["order_by"],
            "set_materializing": ["take", "count", "aggregate", "group_top"],
            "row_reading": ["read"],
        },
    }


def test_synthesis_input_renders_composer_and_solver_tool_surface() -> None:
    config = load_config("rl_task_foundry.yaml")
    prompt = build_synthesis_input(
        domain_name="service_operations",
        scenario_description="end-user support requests over a business database",
        requested_topic="record_history",
        task_language="ko",
        runtime_config=config.synthesis.runtime,
        schema_summary={
            "table_count": 2,
            "edge_count": 1,
            "tables": [
                {
                    "qualified_name": "public.customer",
                    "column_names": ["customer_id", "store_id", "first_name"],
                },
                {
                    "qualified_name": "public.rental",
                    "column_names": ["rental_id", "customer_id", "inventory_id"],
                },
            ],
        },
        tool_surface_summary=_composer_surface(),
    )

    # structural sections
    assert "# Session Context" in prompt
    assert "# Environment" in prompt
    assert "# Tools Available" in prompt

    # session context
    assert "Domain: service_operations" in prompt
    assert "Scenario: end-user support" in prompt
    assert "record_history" in prompt
    assert "Korean" in prompt

    # schema
    assert "public.customer" in prompt
    assert "public.rental" in prompt

    # composer tools surface
    assert "Composer tools: 5" in prompt
    assert "schema_map — Schema graph slice" in prompt
    assert "query — JSON DSL compiler" in prompt

    # solver primitive inventory
    assert "set-producing: rows_where, rows_via, intersect" in prompt
    assert "set-materializing: take, count, aggregate, group_top" in prompt
    assert "row-reading: read" in prompt

    # no legacy atomic-bundle vocabulary
    assert "Total atomic tools" not in prompt
    assert "family_counts" not in prompt
    assert "entity_surfaces" not in prompt

    # navigation hint still present
    assert "navigation only" in prompt

    # submit_draft format lives in the system prompt, not the per-request input
    assert "# submit_draft" not in prompt


def test_synthesis_agent_instructions_describe_composer_workflow() -> None:
    instructions = build_synthesis_agent_instructions(
        load_config("rl_task_foundry.yaml").synthesis.runtime
    )

    # identity
    assert "task-synthesis agent" in instructions
    assert "9-primitive atomic calculus" in instructions

    # commit rule with tool-call vocabulary
    assert "# Commit Rule" in instructions
    assert "tool calls" in instructions
    assert "submit_draft" in instructions

    # composer toolset section with each primitive
    assert "# Composer Tools" in instructions
    for tool in ("schema_map", "neighborhood", "profile", "sample", "query"):
        assert tool in instructions

    # solver context section enumerating atomic calculus primitives
    assert "# Solver Context" in instructions
    for primitive in (
        "rows_where",
        "rows_via",
        "intersect",
        "order_by",
        "take",
        "count",
        "aggregate",
        "group_top",
        "read",
    ):
        assert primitive in instructions
    assert "2 ≤ n ≤ 5" in instructions

    # workflow now names composer tools instead of "atomic calls"
    assert "# Workflow" in instructions
    assert "schema_map" in instructions
    assert "query(spec)" in instructions
    assert "too_easy" in instructions
    assert "too_hard" in instructions

    # escalation axes (structural, no fixed rungs)
    assert "# Escalation Axes" in instructions
    for axis in ("Width", "Filter", "Cardinality", "Cross-item", "Composite"):
        assert axis in instructions

    # label rules
    assert "# Label Rules" in instructions
    assert "verbatim" in instructions

    # determinism
    assert "# Deterministic Answers" in instructions
    assert "only correct answer" in instructions

    # prohibitions
    assert "# Never" in instructions
    assert "Never write SQL" in instructions
    assert "Never weaken" in instructions

    # submit_draft format lives in the system prompt now
    assert "# submit_draft" in instructions
    assert "topic = " in instructions

    # no legacy atomic-tool language
    assert "atomic calls" not in instructions
    assert "# Prohibitions" not in instructions


def test_synthesis_input_includes_anchor_hint() -> None:
    config = load_config("rl_task_foundry.yaml")
    prompt = build_synthesis_input(
        domain_name="ops",
        scenario_description="test",
        requested_topic=None,
        task_language="ko",
        runtime_config=config.synthesis.runtime,
        schema_summary={"table_count": 1, "tables": []},
        tool_surface_summary=_composer_surface(),
        anchor_hint={"film_id": 42},
    )

    assert "# Starting Entity" in prompt
    assert '"film_id": 42' in prompt
    assert prompt.index("# Starting Entity") < prompt.index("# Session Context")
