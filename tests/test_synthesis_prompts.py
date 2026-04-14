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

    # structural sections present
    assert "# BOUNDARY" in prompt
    assert "# Session Context" in prompt
    assert "# Environment and State" in prompt

    # session context populated
    assert "- Domain: service_operations" in prompt
    assert "- Scenario: end-user support requests over a business database" in prompt
    assert "- Requested topic hint: record_history" in prompt
    assert "Coverage hint: record history" in prompt
    assert "soft hint" in prompt
    assert "Korean" in prompt

    # schema orientation
    assert "public.customer" in prompt
    assert "public.rental" in prompt
    assert "Total atomic tools: 4" in prompt
    assert "get: 2 tools" in prompt
    assert "find: 1 tools" in prompt
    assert "calc: 1 tools" in prompt

    # entity surfaces
    assert "get_customer: readable fields=['first_name', 'last_name']" in prompt
    assert "get_staff: readable fields=[] (id-only)" in prompt
    assert "find_order_by_customer_id: readable fields=['status', 'total_amount']" in prompt
    assert "navigation only" in prompt

    # no legacy sections
    assert "Previous Phase Outputs" not in prompt
    assert "Grounded Evidence" not in prompt
    assert "Recent Memory" not in prompt
    assert "One-Shot Example" not in prompt
    assert "Required Output Contract" not in prompt


def test_synthesis_agent_instructions_describe_single_conversation_loop() -> None:
    instructions = build_synthesis_agent_instructions(
        load_config("rl_task_foundry.yaml").synthesis.runtime
    )

    # preamble (no heading)
    assert instructions.startswith("You explore a database")
    assert "canonical answer" in instructions
    assert "knows nothing about the schema" in instructions

    # sections present
    assert "# Workflow" in instructions
    assert "# Label Rules" in instructions
    assert "# Prohibitions" in instructions

    # removed sections
    assert "# Role" not in instructions
    assert "# Principles" not in instructions

    # workflow steps
    assert "Research" in instructions
    assert "Compare" in instructions
    assert "Render" in instructions
    assert "Crank" in instructions

    # label rules
    assert "directly observed tool result" in instructions
    assert "anchor_entity must be a flat JSON object" in instructions
    assert "<entity>" in instructions

    # difficulty guidance removed from system prompt (delivered via feedback)
    assert "search_cost:" not in instructions
    assert "solution_space:" not in instructions
    assert "constraint_density:" not in instructions

    # prohibitions
    assert "Do not submit single-call labels" in instructions
    assert "Accepted or Budget exhausted" in instructions

    # no DB-specific examples
    assert "customer-name" not in instructions
    assert "assigned staff" not in instructions


def test_synthesis_input_humanizes_requested_topic_without_topic_specific_rules() -> None:
    config = load_config("rl_task_foundry.yaml")
    prompt = build_synthesis_input(
        domain_name="service_operations",
        scenario_description="end-user support requests over a business database",
        requested_topic="payment_history",
        task_language="ko",
        runtime_config=config.synthesis.runtime,
        schema_summary={
            "table_count": 2,
            "edge_count": 1,
            "tables": [
                {
                    "qualified_name": "public.payment",
                    "column_names": ["payment_id", "customer_id", "amount", "payment_date"],
                }
            ],
        },
        tool_surface_summary={"tool_count": 0, "family_counts": {}, "entity_surfaces": []},
    )

    assert "# BOUNDARY" in prompt
    assert "# Session Context" in prompt
    assert "# Environment and State" in prompt
    assert "Coverage hint: payment history" in prompt
    assert "soft hint" in prompt
    assert "grounded" in prompt
