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

    assert "# Domain" in prompt
    assert "# Topic Hint" in prompt
    assert "# Topic Semantics" in prompt
    assert "# Schema Orientation" in prompt
    assert "# Tool Surface Hints" in prompt
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
    assert "Start from a believable user need" in instructions
    assert "requested topic is only a soft coverage hint" in instructions
    assert "If the hint would force an id-only, trivial, or weak label" in instructions
    assert "Research the database broadly before drafting anything." in instructions
    assert "Build a relation map before drafting anything." in instructions
    assert "Expand the anchored neighborhood systematically." in instructions
    assert "inspect multiple one-hop and two-hop paths" in instructions
    assert "Keep each new tool call attached to the current relation map" in instructions
    assert "Classify paths before you draft." in instructions
    assert "Choose one path and build the label from that path." in instructions
    assert "Build the label first, then derive the selected topic string and the user-facing framing from that label." in instructions
    assert "Retry intelligently after feedback." in instructions
    assert "Stop only on Accepted or Budget exhausted." in instructions
    assert "Do research and analysis first." in instructions
    assert "Submit only when you fully understand the anchored user, the relevant evidence path" in instructions
    assert "If you are still unsure whether a label field is grounded, readable, anchor-scoped, or necessary" in instructions
    assert "Use that research phase to build a small relation map around the anchored user" in instructions
    assert "classify nearby paths as readable, id-only, local-only, countable, orderable, aggregate-capable, or dead ends" in instructions
    assert "Do not jump to a disconnected table just because it happens to expose readable fields." in instructions
    assert "After a too-easy result, keep the current good readable path when possible" in instructions
    assert "make the smallest connected strengthening step on that same anchored relation map" in instructions
    assert "identify multiple grounded label candidates" in instructions
    assert "pick one path to turn into the final label" in instructions
    assert "Calling submit_draft without anchor_entity is always wrong." not in instructions
    assert "Do not call submit_draft without anchor_entity." in instructions
    assert "Do not write SQL, draft SQL, or include SQL queries in the submission." in instructions
    assert "label_summary" not in instructions
    assert "anchor_entity must be a flat JSON object" in instructions
    assert "question must already be the full user-facing prompt in this exact shape" in instructions
    assert "The JSON inside the <entity> block must exactly match anchor_entity" in instructions
    assert "Only use names, titles, labels, statuses, dates" in instructions
    assert "using the exact values and formatting you actually saw there" in instructions
    assert "Do not submit blank or placeholder string fields in the canonical answer" in instructions
    assert "Do not shorten, paraphrase, partially copy, or reformat observed string or date values" in instructions
    assert "Do not merge separate observed fields into a new readable value" in instructions
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
    assert "Returning *_id fields or other internal references" in instructions
    assert "Writing SQL or describing the answer path as a SQL query" in instructions
    assert "Returning 'Bob' when the tool response showed 'Jon Stephens'" in instructions
    assert "Which of my recent requests is still open, and when was it created?" in instructions
    assert "Inspecting two or more nearby paths around the same anchored" in instructions
    assert "After a too-easy result, keeping the current readable path and adding one more connected grounded fact" in instructions
    assert "When you strengthen search_cost" in instructions
    assert "When you strengthen solution_space" in instructions
    assert "When you strengthen constraint_density" in instructions
    assert "preserve that readable path when possible and deepen it by one connected anchored hop" in instructions
    assert "ways to change the label itself" in instructions
    assert "do not keep the same label and only rewrite the question" in instructions
    assert "Do not reset to a different topic, a different anchor, or a simpler scalar count" in instructions
    assert "use anchor-scoped count evidence rather than a global database total" in instructions
    assert "A rejection is not the end of the task." in instructions
    assert "Continue until submit_draft returns Accepted or Budget exhausted." in instructions
    assert "Submitting a label from the first path you happened to inspect before you understand the nearby relationships" in instructions
    assert "Seeing that one nearby path only returns identifiers, but drafting from it anyway" in instructions
    assert "Jumping to an unrelated entry type that is not yet connected to the anchored neighborhood" in instructions
    assert "throwing away a good readable path and replacing it with a disconnected path, an id-only fallback, or a simpler global count" in instructions


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
