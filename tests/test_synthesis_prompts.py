from __future__ import annotations

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.prompts import (
    build_synthesis_agent_instructions,
    build_synthesis_input,
)


def test_synthesis_input_is_minimal_and_schema_oriented() -> None:
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
        tool_surface_summary={
            "tool_count": 4,
            "family_counts": {"get": 2, "find": 1, "calc": 1},
            "entity_surfaces": [
                {
                    "tool_name": "get_customer",
                    "readable_fields": ["first_name", "last_name"],
                },
                {
                    "tool_name": "get_staff",
                    "readable_fields": [],
                },
                {
                    "tool_name": "find_order_by_customer_id",
                    "readable_fields": ["status", "total_amount"],
                },
            ],
        },
    )

    # structural sections
    assert "# Session Context" in prompt
    assert "# Environment" in prompt

    # session context
    assert "Domain: service_operations" in prompt
    assert "Scenario: end-user support" in prompt
    assert "record_history" in prompt
    assert "Korean" in prompt

    # schema
    assert "public.customer" in prompt
    assert "public.rental" in prompt
    assert "Total atomic tools: 4" in prompt
    assert "get: 2 tools" in prompt

    # entity surfaces
    assert "get_customer: readable fields=" in prompt
    assert "get_staff: readable fields=[] (id-only)" in prompt
    assert "navigation only" in prompt

    # submit_draft format lives in the system prompt, not the per-request input
    assert "# submit_draft" not in prompt


def test_synthesis_agent_instructions_describe_single_conversation_loop() -> None:
    instructions = build_synthesis_agent_instructions(
        load_config("rl_task_foundry.yaml").synthesis.runtime
    )

    # identity
    assert "task-synthesis agent" in instructions

    # commit rule with tool-call vocabulary
    assert "# Commit Rule" in instructions
    assert "tool calls" in instructions
    assert "submit_draft" in instructions

    # workflow: gradual escalation
    assert "# Workflow" in instructions
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

    # no legacy
    assert "# Prohibitions" not in instructions
    assert "Too-hard" not in instructions
    assert "Budget exhausted" not in instructions


def test_synthesis_input_includes_anchor_hint() -> None:
    config = load_config("rl_task_foundry.yaml")
    prompt = build_synthesis_input(
        domain_name="ops",
        scenario_description="test",
        requested_topic=None,
        task_language="ko",
        runtime_config=config.synthesis.runtime,
        schema_summary={"table_count": 1, "tables": []},
        tool_surface_summary={"tool_count": 0, "family_counts": {}, "entity_surfaces": []},
        anchor_hint={"film_id": 42},
    )

    assert "# Starting Entity" in prompt
    assert '"film_id": 42' in prompt
    assert prompt.index("# Starting Entity") < prompt.index("# Session Context")
