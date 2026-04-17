"""Prompt builders for the single-agent synthesis loop."""

from __future__ import annotations

import json as _json

from rl_task_foundry.config.models import SynthesisRuntimeConfig
from rl_task_foundry.schema.profiler import DataProfile
from rl_task_foundry.synthesis.contracts import topic_phrase
from rl_task_foundry.synthesis.turn_budget import build_tool_call_budget_instruction

LANGUAGE_NAMES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
}


def _topic_semantics_instruction(
    requested_topic: str,
) -> str | None:
    normalized = requested_topic.strip()
    if not normalized:
        return None
    hinted_phrase = topic_phrase(normalized)
    if not hinted_phrase:
        return None
    return (
        f"Coverage hint: {hinted_phrase}. "
        "Treat as a soft hint. Start from a grounded "
        "label first; if the hint leads to a trivial or "
        "id-only label, ignore it and choose a better "
        "grounded topic."
    )


def build_synthesis_agent_instructions(
    runtime_config: SynthesisRuntimeConfig,
) -> str:
    return "\n\n".join([
        # ── Identity ──
        "You are a task-synthesis agent. Each tool call either "
        "inspects the database via one of the composer tools or "
        "commits a candidate task via submit_draft. Multiple "
        "independent solvers later attempt your task using a "
        "separate 9-primitive atomic calculus; their exact-match "
        "agreement rate decides acceptance.",

        # ── Commit Rule ──
        build_tool_call_budget_instruction(
            max_tool_calls=runtime_config.max_turns,
        ),

        # ── Never ──
        "# Never\n"
        "- Never write SQL. Use the composer DSL (`query`) instead.\n"
        "- Never weaken a label on rejection. Rejection is always "
        "caused by the request, not the label.\n"
        "- Never concatenate or reformat observed strings.\n"
        "- Never use internal identifiers (*_id) as answer fields.\n"
        "- Never apply more than one escalation per submit.",

        # ── Composer Tools ──
        "# Composer Tools\n"
        "You have five high-bandwidth tools. Each returns JSON you "
        "can keep reading directly.\n"
        "\n"
        "- `schema_map(root_table?, depth?)` — tables, columns, "
        "FK edges, plus hub/bridge tags. One call to orient.\n"
        "- `neighborhood(table, row_id, max_per_edge?)` — the "
        "anchor row's attributes plus per-edge sample IDs and "
        "counts. Use right after sampling an anchor.\n"
        "- `profile(table, column?, predicate?)` — row_count plus "
        "distinct/null counts; with `column` set, adds min/max and "
        "a top-k frequency list. Drives filter calibration.\n"
        "- `sample(table, n, seed?, predicate?)` — up to `n` "
        "representative rows. Seed makes the draw reproducible.\n"
        "- `query(spec)` — JSON DSL over filter + FK join chain + "
        "select or group_by+aggregate, plus sort/limit. One call "
        "computes any canonical answer an atomic chain could "
        "derive. Use it to produce the label.",

        # ── Solver Context ──
        "# Solver Context\n"
        "Solvers re-derive your canonical answer through nine "
        "atomic primitives on a schema-parameterized calculus:\n"
        "\n"
        "- Set-producing: `rows_where(table, column, op, value)` "
        "(ops: eq/in/lt/gt/lte/gte/like; no `any`), "
        "`rows_via(cursor, edge_label)`, "
        "`intersect(left, right)`.\n"
        "- Set-annotating: `order_by(cursor, column, direction)`.\n"
        "- Set-materializing: `take(cursor, n)` with `2 ≤ n ≤ 5`, "
        "`count(cursor)`, `aggregate(cursor, fn, column)` "
        "(sum/avg/min/max), `group_top(cursor, group_column, fn, "
        "n)` with `2 ≤ n ≤ 5`.\n"
        "- Row-reading: `read(table, row_id, columns)`.\n"
        "\n"
        "Design labels that force chaining. A grounded answer "
        "reachable by one primitive (e.g. a single `read` or a "
        "bare `take` on a base table) is solver-trivial and will "
        "lock pass_rate at 1.0. Aim for at least three primitives "
        "of required composition.",

        # ── Workflow ──
        "# Workflow\n"
        "Your ONLY immediate target is the minimum viable draft. "
        "Ignore later escalations until you receive a rejection.\n"
        "\n"
        "1. Call `schema_map` once if you are not already oriented; "
        "then `neighborhood(table, anchor_row_id)` to see the "
        "anchor's FK fan-out.\n"
        "2. Author the canonical answer with a single "
        "`query(spec)` call: a homogeneous list of 3 child records "
        "reached through a single foreign key from the anchor, "
        "every item sharing the same 1-2 keys "
        "(e.g. `[{rental_date, film_title}, …]`), sorted by one "
        "observed field in a fixed direction. Submit the result as "
        "the label and resurface the same constraint in the user-"
        "facing question.\n"
        "3. On too_easy, add exactly ONE dimension from the "
        "escalation axes below and resubmit within 2 composer "
        "calls.\n"
        "4. On too_hard, relax one clause (not the label) and "
        "resubmit. Never weaken the label itself.\n"
        "5. On accept, stop.\n"
        "\n"
        "Rewrite the user-facing request as a customer who knows "
        "nothing about databases. Every label constraint surfaces "
        "as a natural preference in the request.",

        "# Escalation Axes\n"
        "On each too_easy rejection, add ONE axis the current "
        "label does not yet have. Never remove prior structure; "
        "only add. Each axis has a different effect on the answer "
        "shape — shape-changing axes drop pass_rate faster.\n"
        "\n"
        "- **Cardinality** — change N (e.g. 3 → 5) or switch "
        "from fixed-N to \"all records matching the filter\". "
        "Changes answer size.\n"
        "- **Cross-item rule** — replace the sort key with a "
        "uniqueness, ordering, or conditional rule that relates "
        "list items. Changes which records appear and in what "
        "order.\n"
        "- **Composite** — two filters on different dimensions "
        "(categorical AND threshold). Narrows the row set.\n"
        "- **Filter** — a single categorical exclusion or "
        "threshold on an existing field. Narrows the row set, "
        "mildly.\n"
        "- **Width** — disallowed as an escalation. Adding more "
        "fields per record does not change the row set and does "
        "not drop pass_rate on this dataset. Never select Width "
        "after a too_easy rejection.\n"
        "\n"
        "Axes are structural. Pick the concrete field, category, "
        "or threshold from the current DB's schema and observed "
        "data distributions, never from a fixed template.",

        # ── After Rejection ──
        "# After Rejection\n"
        "A rejection is not a signal to explore more. Within 2 "
        "composer calls of rejection feedback, call submit_draft "
        "again. The anchor stays locked; only the label and the "
        "question change.",

        # ── Label Rules ──
        "# Label Rules\n"
        "- Copy strings verbatim from tool results. The runtime "
        "rejects unmatched strings.\n"
        "- One field per observed value. Keep first_name and "
        "last_name as separate slots.\n"
        "- Preserve types. Integers stay integers; do not "
        "stringify.\n"
        "- Use user-facing values, never internal IDs.\n"
        "- Every value referenced by the label — including filter "
        "thresholds, categorical filters, and cardinality targets "
        "— must already appear in a prior tool response. "
        "Ungrounded values are rejected as "
        "`no_new_grounded_observation`.",

        # ── Deterministic Answers ──
        "# Deterministic Answers\n"
        "The label must be the only correct answer. State the "
        "exact record count and the sort clause in the question "
        "so the list is unique (e.g. \"the first 3 rentals "
        "ordered by rental_date ascending\"). Never leave the "
        "count or order implicit. Solvers cannot `take` fewer "
        "than 2 or more than 5 rows in a single primitive call, "
        "so fixed-N targets within `[2, 5]` are directly "
        "reachable; anything outside forces a `count` + "
        "`aggregate` chain.",

        # ── submit_draft ──
        "# submit_draft\n"
        "```\n"
        "submit_draft(\n"
        '  topic = "short task description",\n'
        '  entity = \'{"pk_column": value}\',\n'
        "  label = {field: value, ...} or [{...}, {...}],\n"
        "  question = \"<entity>\\n{anchor}\\n</entity>\\n\\n"
        "<user-facing customer request>\"\n"
        ")\n"
        "```",
    ])


def _render_composer_tool_lines(
    tool_surface_summary: dict[str, object],
    hint_limit: int,
) -> list[str]:
    lines: list[str] = []
    tool_count = tool_surface_summary.get("tool_count")
    if isinstance(tool_count, int):
        lines.append(f"- Composer tools: {tool_count}")
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


def _render_solver_primitive_lines(
    tool_surface_summary: dict[str, object],
) -> list[str]:
    primitives = tool_surface_summary.get("solver_primitives")
    if not isinstance(primitives, dict):
        return []
    labels = {
        "set_producing": "set-producing",
        "set_annotating": "set-annotating",
        "set_materializing": "set-materializing",
        "row_reading": "row-reading",
    }
    lines: list[str] = []
    for key, label in labels.items():
        names = primitives.get(key)
        if not isinstance(names, list):
            continue
        filtered = [str(name) for name in names if isinstance(name, str)]
        if not filtered:
            continue
        lines.append(f"- {label}: {', '.join(filtered)}")
    return lines


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
) -> str:
    sections: list[str] = []

    # ── Anchor (top for salience) ──
    if anchor_hint is not None:
        sections.append(
            "# Starting Entity\n"
            f"Anchor: {_json.dumps(anchor_hint, ensure_ascii=False)}"
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
    max_tables = runtime_config.prompt_schema_orientation_max_tables
    if isinstance(tables, list):
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
    environment_lines.append(
        "- Schema orientation is for navigation only. "
        "Verify readability in actual tool results before "
        "using a field in the label."
    )
    sections.append(
        "# Environment\n" + "\n".join(environment_lines)
    )

    # ── Tools Available ──
    hint_limit = runtime_config.prompt_tool_surface_hint_limit
    composer_lines = _render_composer_tool_lines(
        tool_surface_summary, hint_limit
    )
    solver_lines = _render_solver_primitive_lines(tool_surface_summary)
    tool_section_lines: list[str] = []
    if composer_lines:
        tool_section_lines.append("## Composer toolset (for authoring)")
        tool_section_lines.extend(composer_lines)
    if solver_lines:
        tool_section_lines.append(
            "\n## Solver atomic calculus (what re-derives your answer)"
        )
        tool_section_lines.extend(solver_lines)
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
                "- High-fanout: "
                + ", ".join(fanout_hints[:6])
            )
    if topology_lines:
        sections.append(
            "# Schema Topology\n"
            + "\n".join(topology_lines)
        )

    # ── Data Distributions ──
    if data_profile is not None:
        rendered = data_profile.render()
        if rendered.strip():
            sections.append(
                "# Data Distributions\n"
                "Use these to design realistic constraints "
                "(budget thresholds, quality filters, etc.).\n"
                + rendered
            )

    return "\n\n".join(
        section.strip()
        for section in sections
        if section.strip()
    )
