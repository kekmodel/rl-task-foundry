from __future__ import annotations

from rl_task_foundry.synthesis.prompts import (
    build_synthesis_agent_instructions,
    build_synthesis_input,
)


def test_synthesis_input_is_minimal_and_schema_oriented() -> None:
    prompt = build_synthesis_input(
        domain_name="service_operations",
        scenario_description="end-user support requests over a business database",
        requested_topic="record_history",
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
        tool_surface_summary={
            "entity_surfaces": [
                {
                    "tool_name": "get_customer_by_id",
                    "readable_fields": ["first_name", "last_name"],
                },
                {
                    "tool_name": "get_staff_by_id",
                    "readable_fields": [],
                },
                {
                    "tool_name": "traverse_customer_to_order_by_customer_id",
                    "readable_fields": ["status", "total_amount"],
                },
            ]
        },
    )

    assert "# Domain" in prompt
    assert "# Topic Hint" in prompt
    assert "# Topic Semantics" in prompt
    assert "# Schema Orientation" in prompt
    assert "# Tool Surface Hints" in prompt
    assert "# User-Facing Language" in prompt
    assert "public.customer" in prompt
    assert "public.rental" in prompt
    assert "get_customer_by_id: readable fields=['first_name', 'last_name']" in prompt
    assert "get_staff_by_id: readable fields=[] (id-only surface)" in prompt
    assert "traverse_customer_to_order_by_customer_id: readable fields=['status', 'total_amount']" in prompt
    assert "Use text answer fields only from surfaces that already expose readable non-identifier fields" in prompt
    assert "Korean" in prompt
    assert "Coverage hint: record history" in prompt
    assert "Treat this as a soft planning hint" in prompt
    assert "choose the topic string that best describes that label" in prompt
    assert "ignore the hint and choose a better grounded topic" in prompt
    assert "Previous Phase Outputs" not in prompt
    assert "Grounded Evidence" not in prompt
    assert "Recent Memory" not in prompt
    assert "One-Shot Example" not in prompt
    assert "Required Output Contract" not in prompt


def test_synthesis_agent_instructions_describe_single_conversation_loop() -> None:
    instructions = build_synthesis_agent_instructions()

    assert "synthesis agent" in instructions
    assert "Build the grounded label first" in instructions
    assert "requested topic is only a soft coverage hint" in instructions
    assert "If the hint would force an id-only, trivial, or weak label" in instructions
    assert "Before every submit_draft call" in instructions
    assert "label_summary" in instructions
    assert "explicitly includes the selected topic phrase" in instructions
    assert "anchor_entity must be a flat JSON object" in instructions
    assert "question must already be the full user-facing prompt in this exact shape" in instructions
    assert "The JSON inside the <entity> block must exactly match anchor_entity" in instructions
    assert "Only use names, titles, labels, statuses" in instructions
    assert "Do not submit blank or placeholder string fields in the canonical answer" in instructions
    assert "Before choosing text answer fields such as names, titles, labels, or statuses" in instructions
    assert "If the observed surface is id-only" in instructions
    assert "Do not copy anchor_entity fields into the canonical answer" in instructions
    assert "prefer local grounded orderings inside the anchored scope" in instructions
    assert "Single-call labels are forbidden." in instructions
    assert "requires combining at least two distinct grounded observations" in instructions
    assert "Do not base the label on whichever related row happened to appear first" in instructions
    assert "Do not repeat the raw anchor entity key or raw anchor entity id" in instructions
    assert "Do not repeat raw identifier field names" in instructions
    assert "only chains of internal *_id fields" in instructions
    assert "Do not mention raw table names" in instructions
    assert "When you strengthen search_cost" in instructions
    assert "When you strengthen solution_space" in instructions
    assert "When you strengthen constraint_density" in instructions
    assert "ways to change the label itself" in instructions
    assert "do not keep the same label and only rewrite the question" in instructions
    assert "keep working inside the same conversation" in instructions
    assert "Calling submit_draft without anchor_entity is always wrong." in instructions
    assert "A rejection is not the end of the task." in instructions
    assert "smallest grounded step" in instructions
    assert "When submit_draft returns Accepted, stop." in instructions
    assert "Do not emit markdown fences" in instructions


def test_synthesis_input_humanizes_requested_topic_without_topic_specific_rules() -> None:
    prompt = build_synthesis_input(
        domain_name="service_operations",
        scenario_description="end-user support requests over a business database",
        requested_topic="payment_history",
        task_language="ko",
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
        tool_surface_summary={"entity_surfaces": []},
    )

    assert "# Topic Semantics" in prompt
    assert "Coverage hint: payment history" in prompt
    assert "Treat this as a soft planning hint" in prompt
    assert "choose the topic string that best describes that label" in prompt
    assert "ignore the hint and choose a better grounded topic" in prompt
