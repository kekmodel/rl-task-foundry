from __future__ import annotations

from rl_task_foundry.synthesis.prompts import (
    build_synthesis_agent_instructions,
    build_synthesis_input,
)


def test_synthesis_input_is_minimal_and_schema_oriented() -> None:
    prompt = build_synthesis_input(
        domain_name="service_operations",
        scenario_description="end-user support requests over a business database",
        requested_topic="assignment",
        task_language="ko",
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
    )

    assert "# Domain" in prompt
    assert "# Requested Topic" in prompt
    assert "# Schema Orientation" in prompt
    assert "# User-Facing Language" in prompt
    assert "public.customer" in prompt
    assert "public.rental" in prompt
    assert "Korean" in prompt
    assert "Previous Phase Outputs" not in prompt
    assert "Grounded Evidence" not in prompt
    assert "Recent Memory" not in prompt
    assert "One-Shot Example" not in prompt
    assert "Required Output Contract" not in prompt


def test_synthesis_agent_instructions_describe_single_conversation_loop() -> None:
    instructions = build_synthesis_agent_instructions()

    assert "synthesis agent" in instructions
    assert "requested topic is fixed" in instructions
    assert "Before every submit_draft call" in instructions
    assert "Single-call labels are forbidden." in instructions
    assert "requires combining at least two distinct grounded observations" in instructions
    assert "Do not repeat raw identifier field names" in instructions
    assert "only chains of internal *_id fields" in instructions
    assert "Do not mention raw table names" in instructions
    assert "keep working inside the same conversation" in instructions
    assert "smallest grounded step" in instructions
    assert "When submit_draft returns Accepted, stop." in instructions
    assert "Do not emit markdown fences" in instructions
