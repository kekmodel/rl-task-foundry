from __future__ import annotations

import re
from types import SimpleNamespace

from rl_task_foundry.config import load_config
from rl_task_foundry.synthesis.composer_tools import summarize_composer_tool_surface
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
            "resource_creating": [
                "create_record_set",
                "filter_record_set",
                "filter_record_set_by_values",
                "filter_record_set_by_pattern",
                "filter_record_set_by_related",
                "follow_relation",
            ],
            "resource_combining": ["intersect_record_sets", "sort_record_set"],
            "data_fetching": [
                "list_record_refs",
                "list_records",
                "count_records",
                "aggregate_records",
            ],
            "row_reading": ["get_record"],
        },
    }


def test_composer_surface_summary_exposes_only_callable_tools() -> None:
    summary = summarize_composer_tool_surface(
        [
            SimpleNamespace(name="schema_map", description="schema"),
            SimpleNamespace(name="query", description="query"),
        ]
    )

    assert summary == {
        "tool_count": 2,
        "tools": [
            {"name": "schema_map", "description": "schema"},
            {"name": "query", "description": "query"},
        ],
    }
    assert "solver_primitives" not in summary


def test_synthesis_input_renders_only_callable_composer_tool_surface() -> None:
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
        affordance_map={
            "table_affordances": [
                {
                    "table": "public.customer",
                    "structure": "hub",
                    "affordances": ["label_surface", "anchor_candidate"],
                    "readable": ["first_name"],
                    "categorical_filters": [],
                    "numeric_metrics": [],
                    "time_columns": [],
                }
            ],
            "path_affordances": [
                {
                    "path": "public.customer -> public.rental",
                    "fanout": 12.0,
                    "supports": ["ordered_list", "count_or_cardinality"],
                    "readable": ["rental_date"],
                    "filters": ["rental_date"],
                    "metrics": [],
                }
            ],
        },
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

    # data tools surface
    assert "Data tools: 5" in prompt
    assert "schema_map — Schema graph slice" in prompt
    assert "query — JSON DSL compiler" in prompt
    assert "DB-native topic choice" not in prompt
    assert "feasible strengthening directions" not in prompt
    assert "# DB Affordance Map" in prompt
    assert "Complete rule-based navigation map" in prompt
    assert "## Complete table index" in prompt
    assert "## Complete relationship index" in prompt
    assert "High-signal" not in prompt
    assert "public.customer: structure=hub" in prompt
    assert "public.customer -> public.rental" in prompt
    assert "Topic affordance cards" not in prompt

    # hidden evaluator/runtime tool inventory is not exposed
    for leaked in (
        "solver",
        "Solver",
        "create_record_set",
        "filter_record_set",
        "filter_record_set_by_values",
        "filter_record_set_by_pattern",
        "filter_record_set_by_related",
        "follow_relation",
        "intersect_record_sets",
        "sort_record_set",
        "list_record_refs",
        "list_records",
        "count_records",
        "aggregate_records",
        "get_record",
    ):
        assert leaked not in prompt

    # no legacy atomic-bundle vocabulary
    assert "Total atomic tools" not in prompt
    assert "family_counts" not in prompt
    assert "entity_surfaces" not in prompt

    # no priority/topic scaffolding in the DB context
    assert "High-signal" not in prompt
    assert "Topic affordance cards" not in prompt

    # submit_draft format lives in the tool schema, not the per-request input
    assert "# submit_draft" not in prompt


def test_synthesis_agent_instructions_describe_composer_workflow() -> None:
    instructions = build_synthesis_agent_instructions(
        load_config("rl_task_foundry.yaml").synthesis.runtime
    )

    # Minimal role, no pipeline/training meta.
    assert len(instructions) < 8000
    assert "grounded customer-facing task drafts" in instructions
    assert "synthetic" not in instructions
    assert "dataset" not in instructions
    assert "RLVR" not in instructions
    assert "actor" not in instructions
    for primitive in (
        "solver",
        "Solver",
        "create_record_set",
        "filter_record_set",
        "filter_record_set_by_values",
        "filter_record_set_by_pattern",
        "filter_record_set_by_related",
        "follow_relation",
        "intersect_record_sets",
        "sort_record_set",
        "list_record_refs",
        "list_records",
        "count_records",
        "aggregate_records",
        "get_record",
        "evaluation rollouts",
        "evaluator",
        "pass_rate",
        "quality gate",
        "quality threshold",
        "runtime",
        "difficulty_crank_invalid",
        "crank_invalid",
        "agent submits",
        "composing tool/API responses",
    ):
        assert primitive not in instructions

    # Tool-call budget and tool surface are explicit.
    assert "# Commit Rule" in instructions
    assert "tool calls" in instructions
    assert "submit_draft" in instructions
    assert "# Tools" in instructions
    assert "# Composer Tools" not in instructions
    assert "composer DSL" not in instructions
    for tool in (
        "schema_map",
        "neighborhood",
        "profile",
        "sample",
        "query",
    ):
        assert tool in instructions

    # Workflow is short and each major instruction includes a reason.
    assert "# Workflow" in instructions
    assert instructions.count("Why:") >= 8
    assert "the DB decides the domain" in instructions
    assert "hidden entity values must be grounded" in instructions
    assert "the label must be copied from the latest query evidence" in instructions
    assert "must not be a global answer with a decorative entity attached" in instructions
    assert "specificity rejection" in instructions
    assert "overconstrained/terminal feedback" in instructions

    # User-facing language and ID guidance stay general and English.
    assert "# User Request" in instructions
    assert "configured target language" in instructions
    assert "customer does not know DB tables" in instructions
    assert "Use first-person ownership only" in instructions
    assert "non-user subject" in instructions
    assert "Bad patterns" not in instructions
    assert "'<entity type> 38'" not in instructions
    assert "'<table>_id=38'" not in instructions
    assert "'number 38 <entity>'" not in instructions
    assert "hidden structural handle" in instructions
    assert "hidden current subject/context" in instructions
    assert "Do not attach `entity` to a global report" in instructions
    assert "latest query evidence is scoped" in instructions
    assert "do not expose that handle" in instructions
    assert "multiple relationship roles" in instructions
    assert not re.search(r"[가-힣]", instructions)

    # Label and contract rules preserve exact verifiability.
    assert "# Label And Contract" in instructions
    assert "structured result" in instructions
    assert "not final prose" in instructions
    assert "Use `answer_contract.kind='scalar'` only for aggregate answers" in instructions
    assert "Use `answer_contract.kind='list'` for selected rows" in instructions
    assert "Copy label values from the latest successful `query(spec)` result" in instructions
    assert "Do not reformat" in instructions
    assert "unrelated global answer is invalid" in instructions
    assert "Prefer user-visible non-handle values" in instructions
    assert "include only fields that should appear in the submitted label" in instructions
    assert "every selected field becomes part of the exact submitted answer" in instructions
    assert "combines facts from the same event or record" in instructions
    assert "Avoid independent sibling joins" in instructions
    assert "Do not make a raw handle the main selected answer" in instructions
    assert "current query evidence marks it user-visible" in instructions
    assert "`answer_contract` is only a request-binding surface" in instructions
    assert "Do not restate tables, columns, operators, or SQL" in instructions
    assert "latest query result supplies structural evidence" in instructions
    assert "Every contract phrase must be an exact substring of `user_request`" in instructions
    assert "exactly one correct structured result" in instructions
    assert "fix membership, order, limit, and tie-breaks" in instructions
    assert "exact timestamp" in instructions

    # Task shape is domain-agnostic and concise.
    assert "# Task Shape" in instructions
    assert "real data-service tasks" in instructions
    assert "plan-like list" in instructions
    assert "arbitrary good DBs" in instructions
    assert "Scalar tasks" in instructions
    assert "List tasks" in instructions
    assert "avoid trivial 0/1 results" in instructions
    assert "homogeneous ordered list" in instructions
    assert "at least one non-handle visible field" in instructions
    assert "Open-ended recommendations" in instructions
    assert "filters, thresholds, ordering, limit, and tie-breaks" in instructions

    # Common system instructions stay database-neutral. Concrete table/column
    # names may still appear in per-request schema summaries, not here.
    for sample_specific in (
        "customer → rental",
        "film_actor",
        "rental_date",
        "payment_date",
        "staff_id",
        "film_id=473",
        "제 대여",
    ):
        assert sample_specific not in instructions

    # Callable shape lives in the submit_draft tool schema, not duplicated here.
    assert "# submit_draft" not in instructions
    assert "topic = " not in instructions
    assert "follow that tool's schema exactly" in instructions

    # no legacy atomic-tool language
    assert "atomic calls" not in instructions
    assert "atomic chain" not in instructions
    assert "2 ≤ n ≤ 5" not in instructions


def test_synthesis_input_can_include_tool_only_anchor_hint() -> None:
    config = load_config("rl_task_foundry.yaml")
    prompt = build_synthesis_input(
        domain_name="ops",
        scenario_description="test",
        requested_topic=None,
        task_language="ko",
        runtime_config=config.synthesis.runtime,
        schema_summary={"table_count": 1, "tables": []},
        tool_surface_summary=_composer_surface(),
        anchor_hint={
            "table": "film",
            "pk_column": "film_id",
            "row_id": 42,
            "entity": {"film_id": 42},
        },
    )

    assert "# Starting Entity" in prompt
    assert '"film_id": 42' in prompt
    assert "Anchor table: film" in prompt
    assert "Anchor primary key: film_id = 42" in prompt
    assert 'submit_draft.entity_json: {"film_id": 42}' in prompt
    assert "never pass `row_id: null`" in prompt
    assert prompt.index("# Starting Entity") < prompt.index("# Session Context")


def test_synthesis_input_can_include_candidate_anchor_pool() -> None:
    config = load_config("rl_task_foundry.yaml")
    prompt = build_synthesis_input(
        domain_name="ops",
        scenario_description="test",
        requested_topic=None,
        task_language="ko",
        runtime_config=config.synthesis.runtime,
        schema_summary={"table_count": 1, "tables": []},
        tool_surface_summary=_composer_surface(),
        anchor_hint={
            "candidate_entities": [
                {
                    "table": "customer",
                    "qualified_table": "public.customer",
                    "pk_columns": ["customer_id"],
                    "row_id": 284,
                    "entity": {"customer_id": 284},
                    "preview": {"display_name": "Mary Smith"},
                    "relationship_summary": [],
                }
            ]
        },
    )

    assert "# Candidate Starting Points" in prompt
    assert "# Starting Entity" not in prompt
    assert "Candidate starting points:" in prompt
    assert '"row_id": 284' in prompt
    assert "not answer hints or required topics" in prompt
    assert "hidden current subject/context" in prompt
    assert "do not attach a candidate to an otherwise global task" in prompt
    assert "smallest id" in prompt
    assert "first call `neighborhood`" in prompt
    assert "`preview` and `relationship_summary` are orientation context" in prompt
    assert "not final label evidence" in prompt
    assert "must never expose raw primary-key or row_id values" in prompt


def test_synthesis_input_defaults_to_schema_map_entity_selection() -> None:
    config = load_config("rl_task_foundry.yaml")
    instructions = build_synthesis_agent_instructions(config.synthesis.runtime)
    prompt = build_synthesis_input(
        domain_name="ops",
        scenario_description="test",
        requested_topic=None,
        task_language="ko",
        runtime_config=config.synthesis.runtime,
        schema_summary={"table_count": 1, "tables": []},
        tool_surface_summary=_composer_surface(),
    )

    assert "# Starting Entity" not in prompt
    assert "choose a plausible root table from the current DB" in instructions
    assert "observe a real entity and candidate values" in instructions
    assert "follow that tool's schema exactly" in instructions
