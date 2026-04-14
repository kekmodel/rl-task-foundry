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
            "introspection_rules": [
                "Composite-key tables exist: public.customer_audit. Keep every primary-key field together when anchoring or identifying a row.",
                "Bridge-like or reference-heavy tables exist: public.rental(refs=['customer_id', 'inventory_id']). These tables often require another verified hop before a user-facing value appears.",
                "Columns with stored defaults exist: public.film.rating. A defaulted value may be schema-supplied rather than evidence of a meaningful event.",
            ],
            "tables": [
                {
                    "qualified_name": "public.customer",
                    "primary_key": ["customer_id"],
                    "column_names": ["customer_id", "store_id", "first_name"],
                    "outbound_edges": ["public.customer->public.store"],
                    "inbound_edges": ["public.rental->public.customer"],
                },
                {
                    "qualified_name": "public.rental",
                    "primary_key": ["rental_id"],
                    "column_names": ["rental_id", "customer_id", "inventory_id"],
                    "outbound_edges": [
                        "public.rental->public.customer",
                        "public.rental->public.inventory",
                    ],
                    "inbound_edges": [],
                },
            ],
        },
        tool_surface_summary={
            "tool_count": 4,
            "family_counts": {"get": 2, "find": 1, "calc": 1},
            "business_ready_get_surface_count": 1,
            "reference_heavy_get_surface_count": 1,
            "entity_surfaces": [
                {
                    "tool_name": "surface_a",
                    "readable_fields": ["first_name", "last_name"],
                },
                {
                    "tool_name": "surface_b",
                    "readable_fields": [],
                },
            ],
        },
    )

    assert "# Session Context" in prompt
    assert "# Environment and State" in prompt
    assert "# Introspection Rules" in prompt
    assert "# Boundary" not in prompt
    assert "- Domain: service_operations" in prompt
    assert "- Scenario: end-user support requests over a business database" in prompt
    assert "- Requested topic hint: record_history" in prompt
    assert "- Topic semantics: Coverage hint: record history" in prompt
    assert "Use it only if the observed label naturally fits" in prompt
    assert "- User-facing language: Generate the user-facing question in Korean." in prompt
    assert "Table count: 2" in prompt
    assert "Foreign-key edge count: 1" in prompt
    assert "public.customer: columns=['customer_id', 'store_id', 'first_name']; primary_key=['customer_id']; outbound=['public.customer->public.store']; inbound=['public.rental->public.customer']" in prompt
    assert "public.rental: columns=['rental_id', 'customer_id', 'inventory_id']; primary_key=['rental_id']; outbound=['public.rental->public.customer', 'public.rental->public.inventory']" in prompt
    assert "Composite-key tables exist: public.customer_audit." in prompt
    assert "Bridge-like or reference-heavy tables exist: public.rental(refs=['customer_id', 'inventory_id'])." in prompt
    assert "Columns with stored defaults exist: public.film.rating." in prompt
    assert "Total atomic tools" not in prompt
    assert "Tool families" not in prompt
    assert "GET surface mix" not in prompt
    assert "get_customer" not in prompt
    assert "surface_a" not in prompt
    assert "Schema map is for orientation only." in prompt
    assert "Verify reachable evidence with tool calls before using downstream business fields." in prompt
    assert "Korean" in prompt
    assert "Previous Phase Outputs" not in prompt
    assert "Grounded Evidence" not in prompt
    assert "Recent Memory" not in prompt
    assert "One-Shot Example" not in prompt
    assert "Required Output Contract" not in prompt


def test_synthesis_agent_instructions_stay_general_and_compact() -> None:
    instructions = build_synthesis_agent_instructions(load_config("rl_task_foundry.yaml").synthesis.runtime)

    assert "Identity" not in instructions
    assert "Workflow" in instructions
    assert "Rules" in instructions
    assert "Difficulty" in instructions
    assert "You are a synthesis agent for grounded database task generation" in instructions
    assert "Choose the smallest non-trivial label" in instructions
    assert "Prefer an initial draft that uses a connected multi-step evidence chain" in instructions
    assert "Explore the anchor and nearby connected paths before drafting anything." in instructions
    assert "Avoid direct or one-hop drafts when a grounded two-hop path is available." in instructions
    assert "resolve that chain step by step before using downstream business fields" in instructions
    assert "Treat the requested topic as a soft hint." in instructions
    assert "Use only tool-observed evidence." in instructions
    assert "Do not assume that a reference value from one row identifies a different entity" in instructions
    assert "If a path stays reference-only or id-only, dereference it or choose a different path." in instructions
    assert "Prefer a non-trivial connected label over a trivial direct lookup" in instructions
    assert "Ground first/latest/earliest/count claims with local ordering or aggregate evidence." in instructions
    assert "The question must start with the exact literal tag pair <entity> and </entity>" in instructions
    assert "Never replace that tag with <customer>, <film>, or any entity-specific variant." in instructions
    assert "The request must ask for every non-anchor answer slot and nothing else." in instructions
    assert "Do not leak table names, SQL, hidden joins, or raw identifiers in the request." in instructions
    assert "Difficulty changes must change the label itself" in instructions
    assert "Start from the simplest grounded multi-step label" in instructions
    assert "change exactly one axis, and stop once the draft is plausibly in an acceptable difficulty band." in instructions
    assert "Continue until Accepted or Budget exhausted." in instructions
    assert "Calling submit_draft without anchor_entity is always wrong." not in instructions
    assert len(instructions.split()) < 1000


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
            "introspection_rules": [
                "Nullable reference columns exist: public.payment.customer_id. Verify that the downstream row exists before using target-side fields.",
                "High-null columns exist: public.payment.refund_reason. Check that these values are actually present before building them into a label.",
            ],
            "tables": [
                {
                    "qualified_name": "public.payment",
                    "column_names": ["payment_id", "customer_id", "amount", "payment_date"],
                }
            ],
        },
        tool_surface_summary={"tool_count": 0, "family_counts": {}, "entity_surfaces": []},
    )

    assert "# Session Context" in prompt
    assert "# Environment and State" in prompt
    assert "# Introspection Rules" in prompt
    assert "# Boundary" not in prompt
    assert "Coverage hint: payment history" in prompt
    assert "Use it only if the observed label naturally fits" in prompt
    assert "High-null columns exist: public.payment.refund_reason." in prompt


def test_synthesis_input_allows_schema_map_but_hides_tool_placeholders() -> None:
    config = load_config("rl_task_foundry.yaml")
    prompt = build_synthesis_input(
        domain_name="ops_analytics",
        scenario_description="ad hoc support over a relational database",
        requested_topic="assignment",
        task_language="ko",
        runtime_config=config.synthesis.runtime,
        schema_summary={
            "table_count": 99,
            "edge_count": 123,
            "introspection_rules": [
                "High-inbound hub tables exist: tenant_42.customers_dim(inbound=7). Multiple anchored paths can converge on them, so do not skip intermediate evidence."
            ],
            "tables": [
                {
                    "qualified_name": "tenant_42.orders_archive_2025",
                    "primary_key": ["tenant_order_id"],
                    "column_names": [
                        "tenant_order_id",
                        "customer_fk_42",
                        "internal_status_code",
                    ],
                    "outbound_edges": [
                        "tenant_42.orders_archive_2025->tenant_42.customers_dim"
                    ],
                    "inbound_edges": [
                        "tenant_42.shipments_fact->tenant_42.orders_archive_2025"
                    ],
                }
            ],
        },
        tool_surface_summary={
            "tool_count": 321,
            "family_counts": {"get": 12, "find": 34, "calc": 5, "rank": 2},
            "business_ready_get_surface_count": 7,
            "reference_heavy_get_surface_count": 5,
            "entity_surfaces": [
                {
                    "tool_name": "get_orders_archive_2025",
                    "readable_fields": ["internal_status_code"],
                },
                {
                    "tool_name": "find_customer_fk_42_by_tenant_order_id",
                    "readable_fields": [],
                },
            ],
        },
    )

    assert "tenant_42.orders_archive_2025" in prompt
    assert "tenant_42.customers_dim" in prompt
    assert "tenant_42.shipments_fact" in prompt
    assert "tenant_order_id" in prompt
    assert "customer_fk_42" in prompt
    assert "internal_status_code" in prompt
    assert "# Introspection Rules" in prompt
    assert "High-inbound hub tables exist: tenant_42.customers_dim(inbound=7)." in prompt
    assert "get_orders_archive_2025" not in prompt
    assert "find_customer_fk_42_by_tenant_order_id" not in prompt
    assert "321" not in prompt
    assert "Tool families" not in prompt
    assert "GET surface mix" not in prompt
