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
        rendered = _json.dumps(anchor_hint, ensure_ascii=False)
        return (
            f"Candidate starting points: {rendered}\n"
            "These are optional random observed rows for initial orientation, "
            "not answer hints or required topics. Use one if it helps avoid "
            "restarting from the first or smallest id; ignore them and inspect "
            "the DB if none fit. Choose a candidate only when it can serve as "
            "the hidden current subject/context of the customer's request; do "
            "not attach a candidate to an otherwise global task.\n"
            "If you choose one, first call `neighborhood` with that "
            "candidate's `table` and `row_id`, then use data tools and a final "
            "`query(spec)` to produce the exact label. Copy only the chosen "
            "candidate's `entity` object, encoded as JSON, into "
            "`submit_draft.entity_json`.\n"
            "`preview` and `relationship_summary` are orientation context, not "
            "final label evidence. The customer request may use visible "
            "preview values when natural, but must never expose raw primary-key "
            "or row_id values."
        )
    rendered = _json.dumps(anchor_hint, ensure_ascii=False)
    table = anchor_hint.get("table")
    pk_column = anchor_hint.get("pk_column")
    row_id = anchor_hint.get("row_id")
    if not isinstance(table, str) or not isinstance(pk_column, str):
        return f"Anchor: {rendered}"
    entity = anchor_hint.get("entity")
    if not isinstance(entity, dict):
        entity = {pk_column: row_id}
    row_id_json = _json.dumps(row_id, ensure_ascii=False)
    entity_json = _json.dumps(entity, ensure_ascii=False)
    return (
        f"Anchor: {rendered}\n"
        f"Anchor table: {table}\n"
        f"Anchor primary key: {pk_column} = {row_id_json}\n"
        f"submit_draft.entity_json: {entity_json}\n"
        f"When calling `neighborhood`, use `table` = {table!r} and "
        f"`row_id` = {row_id_json}; never pass `row_id: null`."
    )


def _topic_semantics_instruction(
    requested_topic: str,
) -> str | None:
    normalized = requested_topic.strip()
    if not normalized:
        return None
    return (
        f"Coverage hint: {normalized}. "
        "Treat as a soft hint. Start from a grounded "
        "label first; if the hint leads to a trivial or "
        "id-only label, ignore it and choose a better "
        "grounded topic."
    )


def build_synthesis_agent_instructions(
    runtime_config: SynthesisRuntimeConfig,
) -> str:
    return "\n\n".join([
        "You prepare grounded customer-facing task drafts from the current "
        "database. Use data tools to inspect the DB, then submit one candidate "
        "draft through `submit_draft`. Keep the draft small, deterministic, and "
        "reachable from tool evidence.",

        build_tool_call_budget_instruction(
            max_tool_calls=runtime_config.max_turns,
        ),

        "# Tools\n"
        "Use only the provided data tools. Why: the task must be reproducible "
        "from the same tool surface that later answers it.\n"
        "- `schema_map(root_table?, depth?)`: orient on tables, columns, and "
        "relationship paths.\n"
        "- `sample(target)`: observe real rows before "
        "choosing an entity or value.\n"
        "- `profile(target)`: inspect counts, ranges, and "
        "value distributions before choosing filters.\n"
        "- `neighborhood(table, row_id, max_per_edge?)`: inspect one observed "
        "entity's attributes and nearby relationships.\n"
        "- `query(spec)`: produce the exact canonical label. Use alias-qualified "
        "references like `{as, column}`; do not write SQL.",

        "# Workflow\n"
        "1. Start with `schema_map`, then choose a plausible root table from "
        "the current DB. Why: the DB decides the domain; no task shape is "
        "hard-coded.\n"
        "2. Use `sample`, `profile`, or `query` to observe a real entity and "
        "candidate values. Do not invent ids or choose a row only because its "
        "id is small/easy. Why: hidden entity values must be grounded.\n"
        "3. If the entity/path looks usable, inspect it with `neighborhood`; "
        "otherwise choose another observed entity/path. Why: some rows have no "
        "customer-facing task surface.\n"
        "4. Run one final `query(spec)` for the exact answer, scoped to the "
        "hidden entity or context derived from it, then submit. Why: the label "
        "must be copied from the latest query evidence and must not be a global "
        "answer with a decorative entity attached.\n"
        "5. On a specificity rejection, preserve the answer kind and target; "
        "add exactly one grounded filter, deterministic order/limit, repeated "
        "item, or item requirement, then resubmit quickly. Why: rejection asks "
        "for a narrower or richer version of the same task, not a new task.\n"
        "6. Stop on accept or overconstrained/terminal feedback.",

        "# User Request\n"
        "Write only the customer's request body in the configured target "
        "language. The request is the ask, not the organization's answer. Why: "
        "the final task should look like a real user asking an agent for "
        "information.\n"
        "- Use the customer's own voice: first-person (`my records`) or a direct "
        "request to the organization. Avoid service-side phrasing such as "
        "'we will help you'.\n"
        "- The customer does not know DB tables, rows, primary keys, foreign "
        "keys, or internal ids. Hidden ids may appear in `entity` and query "
        "filters, but not as raw user-facing wording.\n"
        "- Every draft must be about the hidden current subject/context or "
        "values derived from it. Do not attach `entity` to a global report; "
        "choose a different entity, path, or request instead.\n"
        "- If a filter/entity value has only a hidden structural handle and no "
        "observed customer-visible surface, do not expose that handle; choose "
        "a different entity, path, or constraint.\n"
        "- Use first-person or directly observed visible values only when the "
        "latest query evidence is scoped to them.",
        "- If the same answer concept can be reached through multiple "
        "relationship roles, make the role explicit in customer language or "
        "choose a different target. Why: ambiguous paths create multiple "
        "reasonable answers and weak RL signal.",

        "# Label And Contract\n"
        "The label is the structured result that answers `user_request`. It is "
        "not final prose. Why: exact structured labels make the task verifiable.\n"
        "- Use `kind='scalar'` only for aggregate answers "
        "(`count/sum/avg/min/max`). Use `kind='list'` with `fn='select'` for "
        "selected rows or lookup fields. Why: scalar contracts describe one "
        "aggregate result, while selected fields are row/list results.\n"
        "- Copy label values from the latest successful `query(spec)` result. "
        "Do not reformat strings, timestamps, casing, numbers, or types.\n"
        "- The label should be scoped to the hidden current subject/context, "
        "directly or through values derived from it. A draft with a hidden "
        "entity plus an unrelated global answer is invalid.\n"
        "- Use stable API-style field names. Prefer user-visible non-handle "
        "values for selected label outputs; do not expose hidden primary-key "
        "or foreign-key handles as answer values. A handle-like value is only "
        "appropriate when current query evidence marks it user-visible and the "
        "request naturally asks for that reference, not for a raw DB id.\n"
        "- Counts or aggregates over hidden handles are allowed because they do "
        "not expose handle values.\n"
        "- Every predicate, order, limit, and output in `answer_contract` must "
        "be present in the latest query evidence. Each `phrase` must be an "
        "exact substring of `user_request`. Why: the contract is how the draft "
        "meaning is checked without guessing.\n"
        "- The request must have exactly one correct structured result. For "
        "lists, fix membership, order, limit, and tie-breaks. For timestamps, "
        "ask for the exact timestamp when the label contains a timestamp; if "
        "the request asks only for a date, use a date-valued label.",

        "# Task Shape\n"
        "Prefer real data-service tasks: history lookup, shortlist, status "
        "summary, usage/billing summary, schedule, eligibility check, or a "
        "small plan-like list when the schema supports it. Why: the same code "
        "must adapt to arbitrary good DBs.\n"
        "- Scalar tasks: one aggregate such as count, min, max, sum, or avg over "
        "an FK-joined set. Include a natural filter when the unfiltered set is "
        "large, and avoid trivial 0/1 results.\n"
        "- List tasks: a homogeneous ordered list from one destination table, "
        "with visible fields, explicit filters, deterministic ordering, and a "
        "fixed small limit.\n"
        "- Open-ended recommendations are allowed only after you make them "
        "deterministic with visible filters, thresholds, ordering, limit, and "
        "tie-breaks. Submit the candidate through `submit_draft` and follow "
        "that tool's schema exactly.",
    ])

def _render_data_tool_lines(
    tool_surface_summary: dict[str, object],
    hint_limit: int,
) -> list[str]:
    lines: list[str] = []
    tool_count = tool_surface_summary.get("tool_count")
    if isinstance(tool_count, int):
        lines.append(f"- Data tools: {tool_count}")
    tools = tool_surface_summary.get("tools")
    if isinstance(tools, list):
        for tool in tools[:hint_limit]:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            description = tool.get("description")
            if not isinstance(name, str):
                continue
            if isinstance(description, str) and description.strip():
                lines.append(f"- {name} — {description}")
            else:
                lines.append(f"- {name}")
    return lines


def _render_examples_pack(pack: ExamplePack) -> str | None:
    has_examples = pack.type_a_examples or pack.type_b_examples or pack.pitfalls
    if not has_examples:
        return None
    lines: list[str] = [
        f"# Local Examples — {pack.label}",
        (
            "These examples reflect the current DB's structure and observed "
            "cardinality. Use them as local templates only when they are "
            "consistent with system instructions, tool schemas, and tool "
            "descriptions. Generic structural patterns above are placeholders "
            "and may not match this DB's tables, fanout, or natural names."
        ),
    ]
    if pack.type_a_examples:
        lines.append("\n**Type A (list of N child records) — local templates:**")
        lines.extend(f"- {entry}" for entry in pack.type_a_examples)
    if pack.type_b_examples:
        lines.append("\n**Type B (single scalar over an FK-joined set) — local templates:**")
        lines.extend(f"- {entry}" for entry in pack.type_b_examples)
    if pack.pitfalls:
        lines.append("\n**Local structural pitfalls to avoid:**")
        lines.extend(f"- {entry}" for entry in pack.pitfalls)
    return "\n".join(lines)


def _render_affordance_map(affordance_map: dict[str, object]) -> str | None:
    lines = [
        "# DB Affordance Map",
        (
            "Complete rule-based navigation map. Use it to choose what to "
            "inspect, then verify with live tools and canonical query."
        ),
    ]
    table_cards = affordance_map.get("table_affordances")
    if isinstance(table_cards, list) and table_cards:
        lines.append("## Complete table index")
        for card in table_cards:
            if not isinstance(card, dict):
                continue
            lines.append(
                "- "
                f"{card.get('table')}: structure={card.get('structure')}, "
                f"affordances={card.get('affordances') or []}, "
                f"readable={card.get('readable') or []}, "
                f"filters={card.get('categorical_filters') or []}, "
                f"metrics={card.get('numeric_metrics') or []}, "
                f"time={card.get('time_columns') or []}"
            )
    path_cards = affordance_map.get("path_affordances")
    if isinstance(path_cards, list) and path_cards:
        lines.append("## Complete relationship index")
        for card in path_cards:
            if not isinstance(card, dict):
                continue
            lines.append(
                "- "
                f"{card.get('path')}: fanout={card.get('fanout')}, "
                f"supports={card.get('supports') or []}, "
                f"readable={card.get('readable') or []}, "
                f"filters={card.get('filters') or []}, "
                f"metrics={card.get('metrics') or []}"
            )
    return "\n".join(lines) if len(lines) > 2 else None


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
    sections: list[str] = []

    # ── Anchor (top for salience) ──
    if anchor_hint is not None:
        anchor_title = (
            "Candidate Starting Points"
            if isinstance(anchor_hint.get("candidate_entities"), list)
            else "Starting Entity"
        )
        sections.append(
            f"# {anchor_title}\n"
            f"{_render_anchor_hint(anchor_hint)}"
        )

    # ── Session Context ──
    session_lines: list[str] = []
    session_lines.append(f"- Domain: {domain_name}")
    session_lines.append(f"- Scenario: {scenario_description}")
    if requested_topic:
        session_lines.append(
            f"- Requested topic hint: {requested_topic}"
        )
        topic_semantics = _topic_semantics_instruction(
            requested_topic,
        )
        if topic_semantics is not None:
            session_lines.append(
                f"- Topic semantics: {topic_semantics}"
            )
    language_name = LANGUAGE_NAMES.get(
        task_language, task_language
    )
    session_lines.append(
        f"- User-facing language: Generate the question in "
        f"{language_name}. Schema field names, JSON keys, and "
        f"tool names must remain in English."
    )
    sections.append(
        "# Session Context\n" + "\n".join(session_lines)
    )

    # ── Environment ──
    environment_lines: list[str] = []
    table_count = schema_summary.get("table_count")
    edge_count = schema_summary.get("edge_count")
    if isinstance(table_count, int):
        environment_lines.append(f"- Table count: {table_count}")
    if isinstance(edge_count, int):
        environment_lines.append(f"- FK edge count: {edge_count}")
    tables = schema_summary.get("tables")
    has_affordance_map = affordance_map is not None
    max_tables = runtime_config.prompt_schema_orientation_max_tables
    if not has_affordance_map and isinstance(tables, list):
        max_cols = runtime_config.prompt_schema_orientation_max_columns
        for table in tables[:max_tables]:
            if not isinstance(table, dict):
                continue
            qualified_name = (
                table.get("qualified_name") or table.get("table_name")
            )
            columns = table.get("column_names") or []
            environment_lines.append(
                f"- {qualified_name}: columns={list(columns)[:max_cols]}"
            )
    if has_affordance_map:
        environment_lines.append(
            "- Complete table and relationship indexes are provided "
            "below. Verify readability in actual tool results before "
            "using a field in the label."
        )
    else:
        environment_lines.append(
            "- Schema orientation is for navigation only. Verify "
            "readability in actual tool results before using a field "
            "in the label."
        )
    sections.append(
        "# Environment\n" + "\n".join(environment_lines)
    )

    # ── Tools Available ──
    hint_limit = runtime_config.prompt_tool_surface_hint_limit
    data_tool_lines = _render_data_tool_lines(
        tool_surface_summary, hint_limit
    )
    tool_section_lines: list[str] = []
    if data_tool_lines:
        tool_section_lines.append("## Callable data tools")
        tool_section_lines.extend(data_tool_lines)
    if tool_section_lines:
        sections.append(
            "# Tools Available\n" + "\n".join(tool_section_lines)
        )

    # ── Topology ──
    topology_lines: list[str] = []
    hub_tables = schema_summary.get("hub_tables")
    if isinstance(hub_tables, list) and hub_tables:
        topology_lines.append(
            f"- Hub tables (many inbound FKs): {hub_tables}."
        )
    bridge_tables = schema_summary.get("bridge_tables")
    if isinstance(bridge_tables, list) and bridge_tables:
        topology_lines.append(
            f"- Bridge tables (id-only, 2+ FKs): {bridge_tables}. "
            "Traverse through them, do not use their fields."
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
            topology_lines.append(
                f"- Readable: {', '.join(readable_tables[:8])}"
            )
        if id_only_tables:
            topology_lines.append(
                f"- Id-only: {', '.join(id_only_tables[:8])}"
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
            topology_lines.append(
                "- Fanout relationships: "
                + ", ".join(fanout_hints[:6])
            )
    if topology_lines and not has_affordance_map:
        sections.append(
            "# Schema Topology\n"
            + "Use this as a neutral relationship map; verify "
            "candidates with live tools before using them.\n"
            + "\n".join(topology_lines)
        )

    # ── DB Affordance Map ──
    if affordance_map is not None:
        rendered_affordance_map = _render_affordance_map(affordance_map)
        if rendered_affordance_map is not None:
            sections.append(rendered_affordance_map)

    # ── Local Examples (per-DB pack) ──
    if examples_pack is not None:
        rendered_pack = _render_examples_pack(examples_pack)
        if rendered_pack is not None:
            sections.append(rendered_pack)

    # ── Data Distributions ──
    if data_profile is not None:
        rendered = data_profile.render()
        if rendered.strip():
            sections.append(
                "# Data Distributions\n"
                "Use these to design realistic constraints "
                "(budget thresholds, status filters, etc.).\n"
                + rendered
            )

    return "\n\n".join(
        section.strip()
        for section in sections
        if section.strip()
    )
