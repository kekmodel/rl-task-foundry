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

    assert "BOUNDARY" in prompt
    assert "Session Context" in prompt
    assert "Environment and State" in prompt
    assert "- Domain: service_operations" in prompt
    assert "- Scenario: end-user support requests over a business database" in prompt
    assert "- Requested topic hint: record_history" in prompt
    assert "- Topic semantics: Coverage hint: record history" in prompt
    assert "Treat this as a soft planning hint" in prompt
    assert "choose the topic string that best describes that label" in prompt
    assert "ignore the hint and choose a better grounded topic" in prompt
    assert "- User-facing language: Generate the user-facing question in Korean." in prompt
    assert "public.customer" in prompt
    assert "public.rental" in prompt
    assert "Total atomic tools: 4" in prompt
    assert "get: 2 tools available; use these to retrieve one entry by ID." in prompt
    assert "find: 1 tools available; use these to find entries that match a condition." in prompt
    assert "calc: 1 tools available; use these to compute one statistic over matching entries." in prompt
    assert "get_customer: readable fields=['first_name', 'last_name']" in prompt
    assert "get_staff: readable fields=[] (id-only surface)" in prompt
    assert "find_order_by_customer_id: readable fields=['status', 'total_amount']" in prompt
    assert "Use text answer fields only from surfaces that already expose readable non-identifier fields" in prompt
    assert "Korean" in prompt
    assert "Previous Phase Outputs" not in prompt
    assert "Grounded Evidence" not in prompt
    assert "Recent Memory" not in prompt
    assert "One-Shot Example" not in prompt
    assert "Required Output Contract" not in prompt


def test_synthesis_agent_instructions_describe_single_conversation_loop() -> None:
    instructions = build_synthesis_agent_instructions(load_config("rl_task_foundry.yaml").synthesis.runtime)

    assert "Identity" in instructions
    assert "Meta Rules" in instructions
    assert "Core Behavior" in instructions
    assert "Safety and Constraints" in instructions
    assert "Means" in instructions
    assert "Expression" in instructions
    assert "synthesis agent" in instructions
    assert "discover a grounded, verifiable label and then render that label as a natural user request" in instructions
    assert "user knows nothing about the database schema" in instructions
    assert "normal business request from that user's perspective" in instructions
    assert "ALWAYS build the label before you write the user-facing request." in instructions
    assert "WHY: the request is a rendering of the label" in instructions
    assert "Treat the requested topic as a SOFT hint" in instructions
    assert "Feedback and tool errors are working signals, not terminal states." in instructions
    assert "Research first." in instructions
    assert "Expand the connected neighborhood." in instructions
    assert "Compare candidate paths." in instructions
    assert "Build the label from one chosen path." in instructions
    assert "Render the request from the label." in instructions
    assert "Retry after feedback." in instructions
    assert "Do research and analysis first." in instructions
    assert "Submit only when you fully understand the anchored user, the relevant evidence path" in instructions
    assert "If you are still unsure whether a label field is grounded, readable, anchor-scoped, or necessary" in instructions
    assert "IMPORTANT: Build the label first, then write a user-facing request that explicitly asks for EVERY non-anchor answer slot in that label." in instructions
    assert "IMPORTANT: The user-facing request MUST cover the whole label" in instructions
    assert "IMPORTANT: If an answer slot would sound unnatural, redundant, or hard to ask for in the request, remove that slot from the label" in instructions
    assert "IMPORTANT: The <entity> block already identifies the subject." in instructions
    assert "DO NOT add subject-name slots to the label unless the request explicitly asks for that subject's name." in instructions
    assert "Do not jump to a disconnected table just because it happens to expose readable fields." in instructions
    assert "After a too-easy result, keep the current good readable path when possible" in instructions
    assert "drop any slot that no longer belongs in the request" in instructions
    assert "make the smallest connected strengthening step on that same anchored relation map" in instructions
    assert "Calling submit_draft without anchor_entity is always wrong." not in instructions
    assert "ALWAYS include anchor_entity with at least one real primary-key value from the current database." in instructions
    assert "anchor_entity must be a flat JSON object" in instructions
    assert "question must already be the full user-facing prompt in this exact shape" in instructions
    assert "The JSON inside the <entity> block must exactly match anchor_entity" in instructions
    assert "ALWAYS use names, titles, labels, statuses, dates, or other business strings exactly as you directly observed them" in instructions
    assert "WHY: even small rewrites break grounding." in instructions
    assert "If the label returns a count, ground that count with an explicit count or aggregate observation." in instructions
    assert "Use tool results as evidence, not inspiration." in instructions
    assert "GOOD: A request asks for a recent item's title, date, and assigned staff" in instructions
    assert "BAD: A request asks only for a recent item's title, date, and assigned staff, but the label still includes extra customer-name slots." in instructions
    assert "BAD: Combining first_name and last_name into one full-name slot" in instructions
    assert "Write the user-facing request as a normal business request" in instructions
    assert "DO NOT reveal internal tool paths, raw table names" in instructions
    assert "DO NOT repeat the raw anchor entity key or raw anchor entity id" in instructions
    assert "DO NOT shorten, paraphrase, partially copy, or reformat observed string or date values." in instructions
    assert "DO NOT merge separate observed fields into a new readable value" in instructions
    assert "DO NOT ask for unreadable text fields from an id-only surface." in instructions
    assert "DO NOT manufacture readable labels by wrapping an id in generic words" in instructions
    assert "DO NOT submit single-call labels." in instructions
    assert "DO NOT write SQL, draft SQL, or include SQL queries in the submission." in instructions
    assert "DO NOT submit a label with non-anchor answer slots that the user-facing request does not explicitly ask for." in instructions
    assert "DO NOT keep extra subject-name or anchor-descriptive slots in the label" in instructions
    assert "When you strengthen search_cost" in instructions
    assert "When you strengthen solution_space" in instructions
    assert "When you strengthen constraint_density" in instructions
    assert "preserve that readable path when possible and deepen it by one connected anchored hop" in instructions
    assert "ways to change the label itself" in instructions
    assert "do not keep the same label and only rewrite the question" in instructions
    assert "DO NOT reset to a different topic, a different anchor, or a simpler scalar count" in instructions
    assert "use anchor-scoped count evidence rather than a global database total" in instructions
    assert "A rejection is not the end of the task." in instructions
    assert "Continue until submit_draft returns Accepted or Budget exhausted." in instructions
    assert "request does not cover the full label" in instructions


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

    assert "BOUNDARY" in prompt
    assert "Session Context" in prompt
    assert "Environment and State" in prompt
    assert "Coverage hint: payment history" in prompt
    assert "Treat this as a soft planning hint" in prompt
    assert "choose the topic string that best describes that label" in prompt
    assert "ignore the hint and choose a better grounded topic" in prompt
