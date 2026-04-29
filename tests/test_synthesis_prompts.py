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


def test_synthesis_input_does_not_mirror_sdk_tool_surface() -> None:
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

    # structural context envelope
    assert prompt.startswith("<environment_context>\n")
    assert prompt.endswith("\n</environment_context>")
    assert "<session_context>" in prompt
    assert "<database_context>" in prompt
    assert "<tools_available>" not in prompt

    # session context
    assert "<domain_name>\nservice_operations\n</domain_name>" in prompt
    assert "<scenario_description>" in prompt
    assert "end-user support" in prompt
    assert "record_history" in prompt
    assert "<topic_experiment_hint>\nrecord_history\n</topic_experiment_hint>" in prompt
    assert "<requested_topic>" not in prompt
    assert "Edge-case experiment hint: record_history" in prompt
    assert "not a required topic or coverage target" in prompt
    assert "soft hint" not in prompt
    assert "Korean" in prompt

    # schema
    assert "public.customer" in prompt
    assert "public.rental" in prompt

    # callable shape lives in the SDK tool payload, not per-request context
    assert "<tool_count>" not in prompt
    assert '"name": "schema_map"' not in prompt
    assert '"description": "Schema graph slice with hub/bridge tags."' not in prompt
    assert '"name": "query"' not in prompt
    assert '"description": "JSON DSL compiler for canonical answers."' not in prompt
    assert "DB-native topic choice" not in prompt
    assert "feasible strengthening directions" not in prompt
    assert "<db_affordance_map>" in prompt
    assert "Complete rule-based navigation map" in prompt
    assert "<table_affordances>" in prompt
    assert "<path_affordances>" in prompt
    assert "High-signal" not in prompt
    assert '"table": "public.customer"' in prompt
    assert '"structure": "hub"' in prompt
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
    assert "# Role" in instructions
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

    # Tool-call budget is explicit; callable shape lives in SDK tool schemas.
    assert "# Draft Submission Budget" in instructions
    assert "tool calls" in instructions
    assert "submit_draft" in instructions
    assert "Plain text is invalid" in instructions
    assert "# Tools" not in instructions
    assert "# Composer Tools" not in instructions
    assert "composer DSL" not in instructions
    assert "`schema_map(root_table?, depth?)`" not in instructions
    assert "`sample(target)`" not in instructions
    assert "`profile(target)`" not in instructions
    assert "`neighborhood(table, row_id, max_per_edge?)`" not in instructions
    assert "{as, column}" not in instructions

    # Workflow is short and keeps rationale in prose, not flag-like markers.
    assert "# Workflow" in instructions
    assert "Why:" not in instructions
    assert "the DB decides the domain" in instructions
    assert "hidden entity values must be grounded" in instructions
    assert "Build a requestable label candidate" in instructions
    assert "interesting, unique, verifiable, scoped" in instructions
    assert "Check requestability" in instructions
    assert "realistic customer" in instructions
    assert "without technical or awkward controls" in instructions
    assert "formatting quirks" in instructions
    assert "choose another label" in instructions
    assert "Exact substring binding never justifies broken wording" in instructions
    assert "invented terms, diagnostic phrases" in instructions
    assert "one natural request" in instructions
    assert "Derive `user_request` and `topic` from that label" in instructions
    assert "exactly the label fields and row controls" in instructions
    assert "supplies copied label JSON" in instructions
    assert "if diagnostics do not block it, submit immediately" in instructions
    assert "no decorative global answer" in instructions
    assert "Difficulty-Up Policy" in instructions
    assert "smallest grounded related/derived dimension" in instructions
    assert "there is no fixed ladder" in instructions
    assert "baseline is the last evaluated draft" in instructions
    assert "preserve anchor/target" in instructions
    assert "Lists preserve row set/order/limit" in instructions
    assert "scalar aggregates may add group/compare" in instructions
    assert "smallest grounded related/derived dimension" in instructions
    assert "Real strengthening changes answer work" in instructions
    assert "not passive same-row fields" in instructions
    assert "If just tried, switch work" in instructions
    assert "do not add narrowing filters" in instructions
    assert "remove filters" in instructions
    assert "output fields/source meanings/phrases" in instructions
    assert "Add one dimension that keeps the same rows" in instructions
    assert "changes lookup, comparison, visible ordering, or related-row reasoning" in instructions
    assert "ask for it in user_request/answer_contract" in instructions
    assert "shrink the fixed list" in instructions
    assert "replace fields" in instructions
    assert "narrowing filter changes row set" in instructions
    assert "difficulty jump" in instructions
    assert "# Difficulty-Up Examples" in instructions
    assert "<draft_before>list R fields A,B" in instructions
    assert "<commentary>Good: changes what must be found" in instructions
    assert "<commentary>Bad: passive width" in instructions
    assert "If `submit_draft` says the conversation is terminated" in instructions

    # Durable policies are named so feedback/tool descriptions can reference
    # them without restating broad strategy.
    assert "# Core Definitions" in instructions
    assert "Source surface" in instructions
    assert "user wording, label fields, query path" in instructions
    assert "ambiguous across reachable surfaces" in instructions
    assert "request/contract must name chosen source role" in instructions
    assert "broad nouns invalid" in instructions
    assert "label/output_schema cannot disambiguate" in instructions
    assert "ordinary wording points elsewhere" in instructions
    assert "use that source or name the role" in instructions
    assert "visible cannot replace hidden" in instructions
    assert "If no primary key" in instructions
    assert "primary-key-backed path/aggregate" in instructions
    assert "hidden path guessing" in instructions
    assert "Timestamp/date surfaces are distinct too" in instructions
    assert "event time, stored/entered time, scheduled time" in instructions
    assert "generic time/date wording is invalid" in instructions
    assert "# Feedback And Difficulty-Up Policy" in instructions
    assert "not a new durable instruction source" in instructions
    assert "pointer to an existing named policy" in instructions
    assert "Preserve anchor/language" in instructions
    assert "preserve target for repair/difficulty-up" in instructions
    assert "switch target when policy says another label/scope" in instructions
    assert "phrase repair: rewrite one clean fluent request" in instructions
    assert "phrase/binding-only: no exploration" in instructions
    assert "malformed fragments" in instructions

    # User-facing language and ID guidance stay general and English.
    assert "# Request Contract" in instructions
    assert "Target language" in instructions
    assert "No DB/table/column jargon" in instructions
    assert "domain roles" in instructions
    assert "technical sequences" in instructions
    assert "Hidden filter ids go in `entity`" in instructions
    assert "Use first-person only" in instructions
    assert "otherwise use neutral wording" in instructions
    assert "Bad patterns" not in instructions
    assert "'<entity type> 38'" not in instructions
    assert "'<table>_id=38'" not in instructions
    assert "'number 38 <entity>'" not in instructions
    assert "hidden handle" in instructions
    assert "hidden current subject/context" in instructions
    assert "Bind modifiers/filters to exact object/scope" in instructions
    assert "Non-null/status/type filters need row-set wording" in instructions
    assert "matching query.where" in instructions
    assert "not output names" in instructions
    assert "If only returned, ask records plus field" in instructions
    assert "Match hidden scope" in instructions
    assert "parent/list/history requests query that scope" in instructions
    assert "not one child event/record unless asked" in instructions
    assert "current-record handle lookup" in instructions
    assert "child->parent->sibling rows" in instructions
    assert "entity needs the parent/current-subject key" in instructions
    assert "# Scope Examples" in instructions
    assert '"user_request":"show R"' in instructions
    assert '"query":"S_event.R"' in instructions
    assert "hidden lifecycle source" in instructions
    assert "S_order.R also fits" in instructions
    assert '"user_request":"show S_event R"' in instructions
    assert "source role visible" in instructions
    assert "entity={C.pk}; query C->P->siblings" in instructions
    assert "rewording as C-related is not enough" in instructions
    assert "entity={P.pk}; query P->siblings" in instructions
    assert "answer row set share parent scope" in instructions
    assert "Do not attach `entity` to a global report" in instructions
    assert "Copy scoped evidence values exactly" in instructions
    assert "scoped evidence" in instructions
    assert "do not translate/transliterate them" in instructions
    assert "Keep request compact" in instructions
    assert "Exact substring binding never justifies broken wording" in instructions
    assert "misleading column/key translations" in instructions
    assert "Field keys stay in JSON" in instructions
    assert "fields/source roles/tie-breaks cannot fit one natural request" in (
        instructions
    )
    assert "do not expose it" in instructions
    assert "exact object/scope/source" in instructions
    assert not re.search(r"[가-힣]", instructions)

    # Label and contract rules preserve exact verifiability.
    assert "# Label Contract" in instructions
    assert "Structured label" in instructions
    assert "not prose" in instructions
    assert "Verifiable label: final `query(spec)` exactly reproduces" in instructions
    assert "Unique label: one correct structured answer" in instructions
    assert "never rely on hidden ids/order/filters" in instructions
    assert "`answer_contract.kind='scalar'` only for aggregate answers" in instructions
    assert "`list` for selected rows" in instructions
    assert "Label Grounding Policy" in instructions
    assert "copy label values from latest successful `query(spec)` result" in instructions
    assert "Do not reformat" in instructions
    assert "no null fields" in instructions
    assert "unrelated global answer is invalid" in instructions
    assert "semantic API-style field names" in instructions
    assert "not raw DB aliases" in instructions
    assert "values stay user-visible non-handles" in instructions
    assert "Bind answer representation exactly" in instructions
    assert "no code/reference-to-display" in instructions
    assert "source-field-to-related-field upgrade" in instructions
    assert "user_request names role/representation" in instructions
    assert "multiple answer surfaces are valid" in instructions
    assert "Source-sensitive result/status/type fields/filters" in instructions
    assert "query-path source role" in instructions
    assert "request must name it" in instructions
    assert "Source status text is recorded" in instructions
    assert "not current/derived state" in instructions
    assert "no normalized choices" in instructions
    assert "Note/comment text needs that surface" in instructions
    assert "not broad result/value" in instructions
    assert "Distinguish source sequence from display rank" in instructions
    assert "never add sequence/order output only to make duplicate rows unique" in (
        instructions
    )
    assert "Keep output names faithful" in instructions
    assert "broad/vague words are invalid" in instructions
    assert "several reachable sources fit" in instructions
    assert "`query.select` includes only returned label fields" in instructions
    assert "every selected field becomes exact answer" in instructions
    assert "distinguishable through requested output fields" in instructions
    assert "duplicate projected answer rows" in instructions
    assert "if only sequence/reference/record-order can distinguish them" in (
        instructions
    )
    assert "Do not select helper values unless" in instructions
    assert "answer item combines facts from the same event/record" in instructions
    assert "Avoid independent sibling joins" in instructions
    assert "never make a raw handle the main answer" in instructions
    assert "handle refs need query/user visibility" in instructions
    assert "`answer_contract` only binds request phrases" in instructions
    assert "output/order bindings" in instructions
    assert "No tables, columns, operators, or SQL" in instructions
    assert "every phrase must be an exact substring of `user_request`" in instructions
    assert "Binding phrases name returned/order roles" in instructions
    assert "order roles" in instructions
    assert "direction/recency/tie-break wording" in instructions
    assert "not the bare output noun" in instructions
    assert "display-only wording is not enough" in instructions
    assert "Multi-key order needs distinct request phrases" in instructions
    assert "never bind one vague phrase to multiple concepts" in instructions
    assert "List Determinism Policy" in instructions
    assert "exact result: membership, order, limit, tie-breaks" in instructions
    assert "Row-set controls must be in entity/request/contract" in instructions
    assert "boundary words and direction agree" in instructions
    assert "newest/latest vs oldest/earliest" in instructions
    assert "One list order" in instructions
    assert "selection/display split" in instructions
    assert "order leaves distinct-answer ties" in instructions
    assert "ask for a natural visible tie-break before" in instructions
    assert "if two keys still tie, switch label" in instructions
    assert "Use sequence/rank only for named source record order" in instructions
    assert "not generated rank" in instructions
    assert "artificial technical sequence/id/order wording" in instructions
    assert "Max two order keys" in instructions
    assert "choose unique ordering, or return tied rows" in instructions
    assert "Do not shrink limits" in instructions
    assert "or add hidden handles" in instructions

    # Task shape is domain-agnostic and concise.
    assert "# Task Shapes" in instructions
    assert "Scalar: count/min/max/sum/avg" in instructions
    assert "avoid trivial 0/1" in instructions
    assert "Avoid single-record detail lookup" in instructions
    assert "List: ordered 3-5 homogeneous rows" in instructions
    assert "if query count is outside range, switch target/scalar" in instructions
    assert "natural order <=1 visible tie-break" in instructions
    assert "avoid all matching when count >5" in instructions
    assert "Row/list labels" in instructions
    assert "prefer 3-4 fields" in instructions
    assert "<=5 before feedback" in instructions
    assert "meaningful dimension" in instructions

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

    assert "<starting_entity>" in prompt
    assert "</starting_entity>" in prompt
    assert "# Starting Entity" not in prompt
    assert '"film_id": 42' in prompt
    assert "<anchor_table>\nfilm\n</anchor_table>" in prompt
    assert "<anchor_primary_key_column>\nfilm_id\n</anchor_primary_key_column>" in prompt
    assert "<submit_draft_entity_json>" in prompt
    assert "never pass `row_id: null`" in prompt
    assert prompt.index("<starting_entity>") < prompt.index("<session_context>")


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

    assert "<candidate_starting_points>" in prompt
    assert "</candidate_starting_points>" in prompt
    assert "<starting_entity>" not in prompt
    assert "<candidate_entities>" in prompt
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

    assert "<starting_entity>" not in prompt
    assert "<candidate_starting_points>" not in prompt
    assert "<topic_experiment_hint>" not in prompt
    assert "<requested_topic>" not in prompt
    assert "choose a plausible root" in instructions
    assert "hidden entity values must be grounded" in instructions
    assert "follow that tool's schema exactly" in instructions
