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
        "Prepare grounded customer-facing task drafts; use "
        "`submit_draft`.",

        build_tool_call_budget_instruction(
            max_tool_calls=runtime_config.max_turns,
        ),

        "# Workflow\n"
        "1. Map: `schema_map`; choose a plausible root; the DB decides the domain\n"
        "2. Observe via `sample`/`profile`/`query`; use `neighborhood` for "
        "reachability/readable surfaces. Do not invent ids; no easy ids; "
        "hidden entity values must be grounded.\n"
        "3. Build a requestable label candidate: interesting, unique, "
        "verifiable, scoped, visible, deterministic, and natural to ask for.\n"
        "4. Check requestability: a realistic customer can ask for the exact "
        "fields, row set, order, and tie-breaks without technical or awkward "
        "control wording. If not, choose another label.\n"
        "5. Derive `user_request` and `topic` from that label; request "
        "exactly the label fields and row controls.\n"
        "6. Final `query(spec)` answers hidden entity/context, runs before "
        "`submit_draft`, supplies copied label JSON, and checks diagnostics; "
        "no global answer with a decorative entity.",

        "# Core Definitions\n"
        "- Verifiable label: final `query(spec)` exactly reproduces submitted "
        "JSON; every value is copied from that query.\n"
        "- Unique label: one correct structured answer for hidden entity/request. "
        "Visible ties must be returned, naturally distinguished, aggregated, or "
        "avoided; never rely on hidden ids/order/filters.\n"
        "- Source surface: user wording, label fields, and query path name the "
        "same role. If one phrase can map to several roles/surfaces, name "
        "chosen source in request/contract or choose another label. If no "
        "primary key, use a primary-key-backed "
        "path or aggregate; no hidden path guessing. When several reachable "
        "record surfaces share the same everyday noun, user_request must fix "
        "the chosen lifecycle/source role; do not depend on output_schema "
        "field names.",

        "# Request Contract\n"
        "Write only the customer's request body in the configured target "
        "language: ask, not answer.\n"
        "- Use first-person only when hidden context is requester/account/"
        "session/order/records; otherwise use neutral wording.\n"
        "- The customer does not know DB tables, ids, technical sequences/refs, "
        "or hidden handles. Hidden filter ids go in `entity`, never raw wording.\n"
        "- Match hidden scope: parent/list/history requests query that scope, "
        "not one child event/record unless asked. Do not attach `entity` to a "
        "global report.\n"
        "- If entity/filter has only a hidden handle and no observed "
        "customer-visible surface, do not expose it; choose another path.\n"
        "- Copy visible context values only from latest scoped query evidence.\n"
        "- Keep request realistic/compact. Field keys stay in JSON, not text; "
        "avoid schema-like aliases in parentheses. If controls need long "
        "tie-breaks/mechanical field lists, choose another label.\n"
        "- Bind modifiers to exact object/scope/source. If parent and child "
        "rows can match a phrase, name the surface; avoid child-row-filter "
        "wording unless the query uses it.",

        "# Label Contract\n"
        "Label is the structured result that answers `user_request`, not final prose.\n"
        "- `answer_contract.kind='scalar'` only for aggregate answers "
        "(`count/sum/avg/min/max`); `list` for selected rows or lookup fields. "
        "Scalar is aggregate.\n"
        "- Label Grounding Policy: copy label values from latest successful "
        "`query(spec)` result; Do not reformat strings, timestamps, casing, "
        "numbers, or types.\n"
        "- Scope label to hidden current subject/context or derived values; "
        "unrelated global answer is invalid.\n"
        "- Use semantic API-style field names, not raw DB aliases; values stay "
        "user-visible non-handles. Handle refs only when query marks them "
        "user-visible and request asks; never make a raw handle the main answer.\n"
        "- Bind answer representation exactly: no code/reference-to-display or "
        "source-field-to-related-field upgrade unless user_request names that "
        "role/representation; multiple answer surfaces are valid.\n"
        "- Source-sensitive fields (status/type/category/frequency/stage/route) "
        "need the same source role used by the query path; request wording "
        "must name it. Broad field words are invalid if another reachable "
        "surface can provide different values.\n"
        "- Keep output names faithful to source meaning; do not relabel one "
        "surface as another unless request names that role. Vague field words "
        "are invalid when several reachable sources could answer them; "
        "multiple columns are plausible.\n"
        "- `query.select` includes only returned label fields; every selected "
        "field becomes exact answer. Use `where`/`order_by` for context, "
        "filters, sorting. Do not select helper values unless user asks.\n"
        "- When one answer item combines facts from the same event/record, "
        "follow relationships through that event/record path. Avoid independent "
        "sibling joins that can pair unrelated child rows; one grounded state, "
        "not accidental pairing.\n"
        "- `answer_contract` only binds request phrases to label/order: kind, "
        "answer_phrase, constraints, limit, output/order "
        "bindings. No tables, columns, operators, or SQL; every phrase must be "
        "an exact substring of `user_request`; no duplicated evidence.\n"
        "- Binding phrases must name the returned field's role and each order "
        "key's role. If fields come from different roles/surfaces or "
        "multi-key order, user_request must distinguish roles before "
        "labeling; do not bind one vague phrase to multiple concepts.",

        "# List Determinism Policy\n"
        "Lists require one correct structured result: membership, order "
        "direction, limit, and tie-breaks. Row-set controls must be entity "
        "scope/request/contract. Match timestamp/date granularity; state "
        "direction explicitly: newest-first/oldest-first, asc/desc, or "
        "equivalent.\n"
        "- Rows must be distinguishable through requested output fields. If "
        "`query` reports duplicate projected answer rows, add a natural visible "
        "field, aggregate, or choose another task; never add hidden handles.\n"
        "- If order leaves distinct-answer ties, ask for a natural visible "
        "tie-break before `query.order_by`, choose unique ordering, or return "
        "tied rows. No artificial technical sequence/id/order wording.",

        "# Feedback And Difficulty-Up Policy\n"
        "FeedbackError is not a new durable instruction source; treat it as a "
        "pointer to an existing named policy plus failure evidence. Preserve "
        "anchored need/language; for recoverable feedback, change the smallest "
        "failing part. Do not reset to easier/global task or override policy. "
        "Keeping one source of policy prevents split guidance.\n"
        "- Specificity feedback: preserve kind, anchor, target, row set/query "
        "path. Make the smallest single structural strengthening DB supports; "
        "there is no fixed ladder. Keep the same task, not a new task or "
        "difficulty jump.\n"
        "- For lists, keep filters, order, limit, row set, and output "
        "fields/source meanings; append exactly one user-visible field from "
        "same row path/direct relationship, and ask for it in "
        "user_request/answer_contract.\n"
        "- Do not shrink the fixed list, remove/rename/replace fields, or "
        "combine a row-excluding filter with new outputs/order/cardinality.",

        "# Task Shapes\n"
        "Prefer data-service tasks: history lookup, shortlist, status summary, "
        "schedule, eligibility check, or plan-like list for arbitrary good DBs.\n"
        "- Scalar tasks: count, min, max, sum, or avg; filter large sets and "
        "avoid trivial 0/1 results.\n"
        "- List tasks: homogeneous ordered list with visible fields, filters, "
        "deterministic ordering, fixed limit. Prefer natural orders needing "
        "zero or one visible tie-break. For event/logs, use first/latest/top "
        "3-5 rows; avoid all matching when observed count exceeds 5. Include "
        "at least one non-handle visible field.\n"
        "- Keep initial row/list labels narrow: prefer 3-4 fields, max 5 before "
        "feedback. If too easy, use Difficulty-Up Policy to add one coherent "
        "field or relationship at a time for calibration.\n"
        "- Open-ended recommendations need deterministic filters, thresholds, "
        "ordering, limit, and tie-breaks; submit through `submit_draft` and "
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
