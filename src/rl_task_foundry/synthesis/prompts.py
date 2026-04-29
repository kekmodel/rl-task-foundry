"""Prompt builders for the single-agent synthesis loop."""

from __future__ import annotations

import json as _json

from rl_task_foundry.config.models import ExamplePack, SynthesisRuntimeConfig
from rl_task_foundry.schema.profiler import DataProfile
from rl_task_foundry.synthesis.turn_budget import build_tool_call_budget_instruction

LANGUAGE_NAMES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
}


def _render_anchor_hint(anchor_hint: dict[str, object]) -> str:
    candidate_entities = anchor_hint.get("candidate_entities")
    if isinstance(candidate_entities, list):
        return "\n\n".join([
            _render_context_json("candidate_entities", candidate_entities),
            _render_context_block(
                "candidate_instruction",
                "These are optional random observed rows for initial "
                "orientation, not answer hints or required topics. Use one if "
                "it helps avoid restarting from the first or smallest id; "
                "ignore them and inspect the DB if none fit. Choose a "
                "candidate only when it can serve as the hidden current "
                "subject/context of the customer's request; do not attach a "
                "candidate to an otherwise global task.",
            ),
            _render_context_block(
                "candidate_tool_instruction",
                "If you choose one, first call `neighborhood` with that "
                "candidate's `table` and `row_id`, then use data tools and a "
                "final `query(spec)` to produce the exact label. Copy only the "
                "chosen candidate's `entity` object, encoded as JSON, into "
                "`submit_draft.entity_json`.",
            ),
            _render_context_block(
                "candidate_visibility_instruction",
                "`preview` and `relationship_summary` are orientation context, "
                "not final label evidence. The customer request may use "
                "visible preview values when natural, but must never expose "
                "raw primary-key or row_id values.",
            ),
        ])
    table = anchor_hint.get("table")
    pk_column = anchor_hint.get("pk_column")
    row_id = anchor_hint.get("row_id")
    if not isinstance(table, str) or not isinstance(pk_column, str):
        return _render_context_json("anchor", anchor_hint)
    entity = anchor_hint.get("entity")
    if not isinstance(entity, dict):
        entity = {pk_column: row_id}
    return "\n\n".join([
        _render_context_json("anchor", anchor_hint),
        _render_context_value("anchor_table", table),
        _render_context_value("anchor_primary_key_column", pk_column),
        _render_context_json("anchor_row_id", row_id),
        _render_context_json("submit_draft_entity_json", entity),
        _render_context_block(
            "anchor_tool_instruction",
            f"When calling `neighborhood`, use `table` = {table!r} and "
            f"`row_id` = {_json.dumps(row_id, ensure_ascii=False)}; "
            "never pass `row_id: null`.",
        ),
    ])


def _topic_hint_semantics_instruction(
    topic_hint: str,
) -> str | None:
    normalized = topic_hint.strip()
    if not normalized:
        return None
    return (
        f"Edge-case experiment hint: {normalized}. "
        "Normal runs omit this hint. Use it only as an optional exploration "
        "seed; it is not a required topic or coverage target. The submitted "
        "topic must be your own natural summary of the final grounded "
        "user_request, query path, and label. Ignore the hint if it pulls the "
        "draft away from a good grounded task."
    )


def _render_context_block(tag: str, content: str) -> str:
    return f"<{tag}>\n{content.strip()}\n</{tag}>"


def _render_context_value(tag: str, value: object) -> str:
    return _render_context_block(tag, str(value))


def _render_context_json(tag: str, value: object) -> str:
    return _render_context_block(
        tag,
        _json.dumps(value, ensure_ascii=False, indent=2),
    )


def build_synthesis_agent_instructions(
    runtime_config: SynthesisRuntimeConfig,
) -> str:
    return "\n\n".join([
        "# Role\n"
        "Make grounded customer-facing task drafts; use "
        "`submit_draft`",

        build_tool_call_budget_instruction(
            max_tool_calls=runtime_config.max_turns,
        ),

        "# Workflow\n"
        "1. Map; choose a plausible root; the DB decides the domain\n"
        "2. Observe via tools; "
        "Do not invent ids; hidden entity values must be grounded.\n"
        "3. Build a requestable label candidate: interesting, unique, "
        "verifiable, scoped.\n"
        "4. Check requestability: realistic customer can ask "
        "fields/row controls without technical or awkward controls or "
        "formatting quirks; "
        "else choose another label.\n"
        "5. Derive `user_request` and `topic` from that label; request "
        "exactly the label fields and row controls.\n"
        "6. Final `query(spec)` supplies copied label JSON before "
        "`submit_draft`; no decorative global answer.",

        "# Core Definitions\n"
        "- Verifiable label: final `query(spec)` exactly reproduces submitted "
        "JSON; values copied.\n"
        "- Unique label: one correct structured answer for hidden entity/request. "
        "Return, distinguish, aggregate, or avoid visible ties; never rely on "
        "hidden ids/order/filters.\n"
        "- Source surface: user wording, label fields, query path name same "
        "role. If one phrase can map to several reachable surfaces, "
        "request/contract must name chosen source role; label/output_schema "
        "names cannot disambiguate; broad nouns invalid. If no primary key, "
        "use primary-key-backed path/aggregate; no hidden path guessing.",

        "# Request Contract\n"
        "Use configured target language:\n"
        "- Use first-person only for requester/account/session/order/records; "
        "otherwise use neutral wording.\n"
        "- customer does not know DB tables, ids, technical sequences/refs, or "
        "hidden handles. Hidden filter ids go in `entity`, never raw wording.\n"
        "- Match hidden scope: parent/list/history requests query that scope, "
        "not one child event/record unless asked. For child->parent->sibling "
        "rows, entity needs the parent/current-subject key; rewording as "
        "child-related is not enough. current-record handle lookup asks "
        "record facts. Do not attach `entity` to a global report.\n"
        "- Hidden-handle-only entity/filter: do not expose it; choose another "
        "path. Copy scoped evidence values exactly; do not "
        "translate/transliterate them.\n"
        "- Keep request compact; use ordinary target-language words, "
        "no malformed terms. Field keys stay in JSON; no aliases/choices in "
        "parentheses. If long tie-breaks/mechanical field lists are needed, "
        "choose another label.\n"
        "- Bind modifiers/filters to exact object/scope/source. Non-null/status/"
        "type filters need row-set wording and matching query.where, not output "
        "names. If only returned, ask records plus field. If "
        "parent/child rows both fit, name surface; child-row-filter wording "
        "needs query use.",

        "# Label Contract\n"
        "Label is structured result, not final prose.\n"
        "- `answer_contract.kind='scalar'` only for aggregate answers; "
        "`list` for selected rows.\n"
        "- Label Grounding Policy: copy label values from latest successful "
        "`query(spec)` result; Do not reformat; no null fields.\n"
        "- Scope label to hidden current subject/context; "
        "unrelated global answer is invalid.\n"
        "- Use semantic API-style field names, not raw DB aliases; values stay "
        "user-visible non-handles; handle refs need query/user visibility; "
        "never make a raw handle the main answer.\n"
        "- Bind answer representation exactly: no code/reference-to-display or "
        "source-field-to-related-field upgrade unless user_request names "
        "role/representation; multiple answer surfaces are valid.\n"
        "- Source-sensitive result/status/type fields/filters need same source "
        "role used by query path; request must name it; no normalized choices. "
        "Distinguish "
        "source sequence from display rank. Keep output names faithful; broad/"
        "vague words are invalid when several reachable sources fit.\n"
        "- `query.select` includes only returned label fields; every selected field "
        "becomes exact answer. Do not select helper values unless user asks.\n"
        "- When one answer item combines facts from the same event/record, follow "
        "that path. Avoid independent sibling joins.\n"
        "- `answer_contract` only binds request phrases to label/order: kind, "
        "answer_phrase, constraints, limit, output/order bindings. No tables, "
        "columns, operators, or SQL; every phrase must be an exact substring "
        "of `user_request`.\n"
        "- Binding phrases name returned roles and order roles. For an order key, "
        "use direction/recency/tie-break wording, not the bare output noun; "
        "display-only wording is not enough. Multi-key order needs distinct "
        "request phrases; never bind one vague phrase to multiple concepts.",

        "# List Determinism Policy\n"
        "Lists need exact result: membership, order, limit, tie-breaks. Row-set controls "
        "must be in entity/request/contract; boundary "
        "words and direction agree (newest/latest vs oldest/earliest; asc/desc).\n"
        "- Rows must be distinguishable through requested output fields. If "
        "`query` reports duplicate projected answer rows, add natural visible "
        "field/aggregate. Do not shrink limits or add hidden handles.\n"
        "- If order leaves distinct-answer ties, ask for a natural visible "
        "tie-break before `query.order_by`, choose unique ordering, or return "
        "tied rows. Use sequence/rank only for named source record order, "
        "not generated rank. No artificial technical "
        "sequence/id/order wording. Max two order keys.",

        "# Scope Examples\n"
        "<example><draft_bad>{\"user_request\":\"show R\","
        "\"label\":\"s1_r\",\"query\":\"S1.R;S2.R\"}"
        "</draft_bad><commentary>Bad: hidden source choice"
        "</commentary></example>\n"
        "<example><draft_good>{\"user_request\":\"show S1 R\","
        "\"label\":\"s1_r\",\"query\":\"S1.R\"}</draft_good>"
        "<commentary>Good: source/label align"
        "</commentary></example>\n"
        "<example><draft_bad>entity={C.pk}; query C->P->siblings</draft_bad>"
        "<commentary>rewording as C-related is not enough"
        "</commentary></example>\n"
        "<example><draft_good>entity={P.pk}; query P->siblings</draft_good>"
        "<commentary>answer row set share parent scope"
        "</commentary></example>\n"
        "<example><draft_good>entity={R.pk}; query filters R.pk"
        "</draft_good><commentary>hidden current record "
        "and request scope match</commentary></example>",

        "# Feedback And Difficulty-Up Policy\n"
        "FeedbackError is not a new durable instruction source; pointer to an "
        "existing named policy. Preserve anchor/language; preserve target for "
        "repair/difficulty-up; switch target when policy says another "
        "label/scope. phrase repair: clean wording\n"
        "- Specificity: preserve anchor/target. Lists preserve "
        "row set/order/limit; scalar aggregates may add group/compare. Add "
        "one grounded related/derived dimension. smallest single structural "
        "strengthening; there is no fixed "
        "ladder; no difficulty jump.\n"
        "- Real strengthening changes answer work: aggregate/compare/group/"
        "order/related-row selection/row membership; not same-row fields, "
        "even via join. If just tried, switch work.\n"
        "- For lists, keep filters, order, limit, output "
        "fields/source meanings. Append one grounded dimension that changes "
        "lookup, comparison, order, or row reasoning; passive display fields are "
        "weak; ask for it in user_request/answer_contract.\n"
        "- Do not shrink the fixed list, remove/rename/replace fields, or "
        "combine a row-excluding filter with new outputs/order/cardinality.",

        "# Difficulty-Up Examples\n"
        "<example><draft_before>list R fields A,B</draft_before>"
        "<draft_after>R+C compare/order"
        "</draft_after><commentary>Good: changes what must be found"
        "</commentary></example>\n"
        "<example><draft_after>R+same-row C</draft_after>"
        "<commentary>Bad: passive width</commentary></example>",

        "# Task Shapes\n"
        "- Scalar: count/min/max/sum/avg over filters; avoid trivial 0/1.\n"
        "- Avoid single-record detail lookup.\n"
        "- List: ordered 3-5 homogeneous rows; if query count is outside range, "
        "switch target/scalar; natural order <=1 visible tie-break; "
        "avoid all matching when count >5.\n"
        "- Initial row/list labels: prefer 3-4 fields; max 5 before feedback; "
        "add one meaningful dimension; "
        "follow that tool's schema exactly.",
    ])

def _render_examples_pack(pack: ExamplePack) -> str | None:
    has_examples = pack.type_a_examples or pack.type_b_examples or pack.pitfalls
    if not has_examples:
        return None
    blocks: list[str] = [
        _render_context_value("label", pack.label),
        _render_context_block(
            "instruction",
            "These examples reflect the current DB's structure and observed "
            "cardinality. Use them as local templates only when they are "
            "consistent with system instructions, tool schemas, and tool "
            "descriptions. Generic structural patterns above are placeholders "
            "and may not match this DB's tables, fanout, or natural names.",
        ),
    ]
    if pack.type_a_examples:
        blocks.append(_render_context_json("type_a_examples", pack.type_a_examples))
    if pack.type_b_examples:
        blocks.append(_render_context_json("type_b_examples", pack.type_b_examples))
    if pack.pitfalls:
        blocks.append(_render_context_json("pitfalls", pack.pitfalls))
    return "\n\n".join(blocks)


def _render_affordance_map(affordance_map: dict[str, object]) -> str | None:
    blocks = [
        _render_context_block(
            "instruction",
            "Complete rule-based navigation map. Use it to choose what to "
            "inspect, then verify with live tools and canonical query.",
        ),
    ]
    table_cards = affordance_map.get("table_affordances")
    if isinstance(table_cards, list) and table_cards:
        blocks.append(_render_context_json("table_affordances", table_cards))
    path_cards = affordance_map.get("path_affordances")
    if isinstance(path_cards, list) and path_cards:
        blocks.append(_render_context_json("path_affordances", path_cards))
    return "\n\n".join(blocks) if len(blocks) > 1 else None


def build_synthesis_input(
    *,
    domain_name: str,
    scenario_description: str,
    requested_topic: str | None,
    task_language: str,
    schema_summary: dict[str, object],
    tool_surface_summary: dict[str, object],
    runtime_config: SynthesisRuntimeConfig,
    anchor_hint: dict[str, object] | None = None,
    data_profile: DataProfile | None = None,
    examples_pack: ExamplePack | None = None,
    affordance_map: dict[str, object] | None = None,
) -> str:
    # The callable tool surface is injected by the Agents SDK via the API
    # `tools` field, so it is intentionally not mirrored into user context.
    del tool_surface_summary
    sections: list[str] = []

    # ── Anchor (top for salience) ──
    if anchor_hint is not None:
        anchor_tag = (
            "candidate_starting_points"
            if isinstance(anchor_hint.get("candidate_entities"), list)
            else "starting_entity"
        )
        sections.append(
            _render_context_block(anchor_tag, _render_anchor_hint(anchor_hint))
        )

    # ── Session Context ──
    session_blocks: list[str] = []
    session_blocks.append(_render_context_value("domain_name", domain_name))
    session_blocks.append(
        _render_context_value("scenario_description", scenario_description)
    )
    if requested_topic:
        session_blocks.append(
            _render_context_value("topic_experiment_hint", requested_topic)
        )
        topic_semantics = _topic_hint_semantics_instruction(requested_topic)
        if topic_semantics is not None:
            session_blocks.append(
                _render_context_block("topic_semantics", topic_semantics)
            )
    language_name = LANGUAGE_NAMES.get(
        task_language, task_language
    )
    session_blocks.append(_render_context_value("task_language", task_language))
    session_blocks.append(_render_context_value("user_facing_language", language_name))
    session_blocks.append(
        _render_context_block(
            "language_instruction",
            f"Generate the question in {language_name}. Schema field names, "
            "JSON keys, and tool names must remain in English.",
        )
    )
    sections.append(
        _render_context_block("session_context", "\n\n".join(session_blocks))
    )

    # ── Environment ──
    environment_blocks: list[str] = []
    table_count = schema_summary.get("table_count")
    edge_count = schema_summary.get("edge_count")
    if isinstance(table_count, int):
        environment_blocks.append(_render_context_value("table_count", table_count))
    if isinstance(edge_count, int):
        environment_blocks.append(_render_context_value("fk_edge_count", edge_count))
    tables = schema_summary.get("tables")
    has_affordance_map = affordance_map is not None
    max_tables = runtime_config.prompt_schema_orientation_max_tables
    if not has_affordance_map and isinstance(tables, list):
        max_cols = runtime_config.prompt_schema_orientation_max_columns
        orientation_tables: list[dict[str, object]] = []
        for table in tables[:max_tables]:
            if not isinstance(table, dict):
                continue
            qualified_name = (
                table.get("qualified_name") or table.get("table_name")
            )
            columns = table.get("column_names") or []
            orientation_tables.append(
                {
                    "qualified_name": qualified_name,
                    "columns": list(columns)[:max_cols],
                }
            )
        if orientation_tables:
            environment_blocks.append(
                _render_context_json("schema_orientation_tables", orientation_tables)
            )
    if has_affordance_map:
        environment_blocks.append(
            _render_context_block(
                "instruction",
                "Complete table and relationship indexes are provided below. "
                "Verify readability in actual tool results before using a "
                "field in the label.",
            )
        )
    else:
        environment_blocks.append(
            _render_context_block(
                "instruction",
                "Schema orientation is for navigation only. Verify "
                "readability in actual tool results before using a field in "
                "the label.",
            )
        )
    sections.append(
        _render_context_block("database_context", "\n\n".join(environment_blocks))
    )

    # ── Topology ──
    topology_blocks: list[str] = []
    hub_tables = schema_summary.get("hub_tables")
    if isinstance(hub_tables, list) and hub_tables:
        topology_blocks.append(_render_context_json("hub_tables", hub_tables))
    bridge_tables = schema_summary.get("bridge_tables")
    if isinstance(bridge_tables, list) and bridge_tables:
        topology_blocks.append(_render_context_json("bridge_tables", bridge_tables))
        topology_blocks.append(
            _render_context_block(
                "bridge_table_instruction",
                "Bridge tables are id-only, 2+ FK tables. Traverse through "
                "them; do not use their fields.",
            )
        )
    if isinstance(tables, list):
        readable_tables = []
        id_only_tables = []
        for table in tables[:max_tables]:
            if not isinstance(table, dict):
                continue
            name = table.get("qualified_name") or ""
            surface = table.get("surface", "")
            readable_cols = table.get("readable_columns")
            if surface == "id-only":
                id_only_tables.append(str(name))
            elif isinstance(readable_cols, list) and readable_cols:
                readable_tables.append(
                    f"{name}({len(readable_cols)} fields)"
                )
        if readable_tables:
            topology_blocks.append(
                _render_context_json("readable_tables", readable_tables[:8])
            )
        if id_only_tables:
            topology_blocks.append(
                _render_context_json("id_only_tables", id_only_tables[:8])
            )
        fanout_hints: list[str] = []
        for table in tables[:max_tables]:
            if not isinstance(table, dict):
                continue
            fanout_in = table.get("fanout_in")
            if isinstance(fanout_in, list):
                for entry in fanout_in:
                    if isinstance(entry, dict):
                        src = entry.get("from", "")
                        fan = entry.get("fanout")
                        tgt = table.get("qualified_name", "")
                        if fan is not None:
                            fanout_hints.append(
                                f"{src}->{tgt} (~{fan}x)"
                            )
        if fanout_hints:
            topology_blocks.append(
                _render_context_json("fanout_relationships", fanout_hints[:6])
            )
    if topology_blocks and not has_affordance_map:
        sections.append(
            _render_context_block(
                "schema_topology",
                _render_context_block(
                    "instruction",
                    "Use this as a neutral relationship map; verify "
                    "candidates with live tools before using them.",
                )
                + "\n\n"
                + "\n\n".join(topology_blocks),
            )
        )

    # ── DB Affordance Map ──
    if affordance_map is not None:
        rendered_affordance_map = _render_affordance_map(affordance_map)
        if rendered_affordance_map is not None:
            sections.append(
                _render_context_block("db_affordance_map", rendered_affordance_map)
            )

    # ── Local Examples (per-DB pack) ──
    if examples_pack is not None:
        rendered_pack = _render_examples_pack(examples_pack)
        if rendered_pack is not None:
            sections.append(_render_context_block("local_examples", rendered_pack))

    # ── Data Distributions ──
    if data_profile is not None:
        rendered = data_profile.render()
        if rendered.strip():
            sections.append(
                _render_context_block(
                    "data_distributions",
                    _render_context_block(
                        "instruction",
                        "Use these to design realistic constraints "
                        "(budget thresholds, status filters, etc.).",
                    )
                    + "\n\n"
                    + _render_context_block("profile", rendered),
                )
            )

    context_body = "\n\n".join(
        section.strip()
        for section in sections
        if section.strip()
    )
    return _render_context_block("environment_context", context_body)
