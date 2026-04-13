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
            "self_anchor_surfaces": ["get_customer", "get_staff"],
        },
    )

    assert "# Domain" in prompt
    assert "# Topic Hint" in prompt
    assert "# Topic Semantics" in prompt
    assert "# Schema Orientation" in prompt
    assert "# Tool Surface Hints" in prompt
    assert "# Self Anchor Hints" in prompt
    assert "# User-Facing Language" in prompt
    assert "public.customer" in prompt
    assert "public.rental" in prompt
    assert "Total atomic tools: 4" in prompt
    assert "get: 2 tools available; use these to retrieve one entry by ID." in prompt
    assert "find: 1 tools available; use these to find entries that match a condition." in prompt
    assert "calc: 1 tools available; use these to compute one statistic over matching entries." in prompt
    assert "get_customer: readable fields=['first_name', 'last_name']" in prompt
    assert "get_staff: readable fields=[] (id-only surface)" in prompt
    assert "find_order_by_customer_id: readable fields=['status', 'total_amount']" in prompt
    assert "Person-like self anchor surfaces are available: get_customer, get_staff." in prompt
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
    instructions = build_synthesis_agent_instructions(load_config("rl_task_foundry.yaml").synthesis.runtime)

    assert "Role" in instructions
    assert "Workflow" in instructions
    assert "IMPORTANT" in instructions
    assert "DO NOT" in instructions
    assert "Example" in instructions
    assert "GOOD" in instructions
    assert "BAD" in instructions
    assert "synthesis agent" in instructions
    assert "user knows nothing about the database schema" in instructions
    assert "normal business request from that user's perspective" in instructions
    assert "Treat anchor_entity as the requesting user's own entity by default." in instructions
    assert "prefer that self entity as anchor_entity instead of anchoring on a content object" in instructions
    assert "Start from a believable first-person need of that anchored user" in instructions
    assert "requested topic is only a soft coverage hint" in instructions
    assert "If the hint would force an id-only, trivial, or weak label" in instructions
    assert "Research the database broadly before drafting anything." in instructions
    assert "Map the database relationships first." in instructions
    assert "trace many relationships and interesting grounded data paths across the database" in instructions
    assert "understand the exposed relationships across the database" in instructions
    assert "Analyze the anchored user's reachable surfaces." in instructions
    assert "Compare candidate paths before you choose a label." in instructions
    assert "choose one path before you draft" in instructions
    assert "Choose the label first, then derive topic and anchor framing from it." in instructions
    assert "Retry intelligently after feedback." in instructions
    assert "Stop only on Accepted or Budget exhausted." in instructions
    assert "After you submit a draft with a valid self anchor, keep that same anchor_entity across retries." in instructions
    assert "Do research and analysis first." in instructions
    assert "Submit only when you fully understand the anchored user, the relevant evidence path" in instructions
    assert "If you are still unsure whether a label field is grounded, readable, anchor-scoped, or necessary" in instructions
    assert "Before the first judged submit_draft call, stay in exploration mode until you have gathered at least" in instructions
    assert "Use that research phase to classify nearby paths as readable, id-only, local-only, countable, aggregate-capable, or dead ends" in instructions
    assert "distinct tool names" in instructions
    assert "anchor-scoped observations whose parameters depend on anchor_entity" in instructions
    assert "identify multiple grounded label candidates" in instructions
    assert "pick one path to turn into the final label" in instructions
    assert "Calling submit_draft without anchor_entity is always wrong." not in instructions
    assert "Do not call submit_draft without anchor_entity." in instructions
    assert "Do not write SQL, draft SQL, or include SQL queries in the submission." in instructions
    assert "label_summary" in instructions
    assert "explicitly includes the selected topic phrase" in instructions
    assert "anchor_entity must be a flat JSON object" in instructions
    assert "question must already be the full user-facing prompt in this exact shape" in instructions
    assert "The JSON inside the <entity> block must exactly match anchor_entity" in instructions
    assert "Only use names, titles, labels, statuses" in instructions
    assert "Do not use opaque identifiers such as UUIDs, hashes, encrypted tokens" in instructions
    assert "Do not submit blank or placeholder string fields in the canonical answer" in instructions
    assert "Do not manufacture readable labels by wrapping an id in generic words" in instructions
    assert "If the label returns a count, ground that count with an explicit count or aggregate observation." in instructions
    assert "Do not copy anchor_entity fields into the canonical answer" in instructions
    assert "Do not start with a multi-item set, top-few list, or paired bundle" in instructions
    assert "Do not submit single-call labels." in instructions
    assert "Do not reveal internal tool paths" in instructions
    assert "Do not repeat the raw anchor entity key or raw anchor entity id" in instructions
    assert "Do not reveal internal tool paths, raw table names" in instructions
    assert "Jumping to a global count for a self-scoped request." in instructions
    assert 'Returning a label such as {"store_id": 1}, {"customer_id": 42}' in instructions
    assert "Returning *_id fields, UUIDs, hashes, tokens" in instructions
    assert "Writing SQL or describing the answer path as a SQL query" in instructions
    assert "Which of my recent requests is still open, and when was it created?" in instructions
    assert "When you strengthen search_cost" in instructions
    assert "When you strengthen solution_space" in instructions
    assert "When you strengthen constraint_density" in instructions
    assert "ways to change the label itself" in instructions
    assert "do not keep the same label and only rewrite the question" in instructions
    assert "Do not reset to a different topic, a different anchor, or a simpler scalar count" in instructions
    assert "use anchor-scoped count evidence rather than a global database total" in instructions
    assert "A rejection is not the end of the task." in instructions
    assert "Continue until submit_draft returns Accepted or Budget exhausted." in instructions


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

    assert "# Topic Semantics" in prompt
    assert "Coverage hint: payment history" in prompt
    assert "Treat this as a soft planning hint" in prompt
    assert "choose the topic string that best describes that label" in prompt
    assert "ignore the hint and choose a better grounded topic" in prompt
