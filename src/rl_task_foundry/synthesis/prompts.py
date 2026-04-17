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
        "inspects the database or commits a candidate task via "
        "submit_draft. Multiple independent solvers later attempt "
        "your task; their exact-match agreement rate decides "
        "acceptance.",

        # ── Commit Rule ──
        build_tool_call_budget_instruction(
            max_tool_calls=runtime_config.max_turns,
        ),

        # ── Never ──
        "# Never\n"
        "- Never write SQL.\n"
        "- Never weaken a label on rejection. Rejection is always "
        "caused by the request, not the label.\n"
        "- Never concatenate or reformat observed strings.\n"
        "- Never use internal identifiers (*_id) as answer fields.\n"
        "- Never apply more than one escalation per submit.",

        # ── Workflow ──
        "# Workflow\n"
        "Your ONLY immediate target is the minimum viable draft. "
        "Ignore later escalations until you receive a rejection.\n"
        "\n"
        "1. Inspect the anchor with up to 3 atomic calls to learn "
        "which user-facing fields exist along a multi-hop path.\n"
        "2. Submit immediately. The first draft is a multi-hop "
        "lookup returning one record with 1-2 user-facing fields. "
        "No filters, no lists, no constraints.\n"
        "3. On too_easy, add exactly ONE dimension from the "
        "escalation axes below and resubmit within 2 atomic "
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
        "only add. Axes are listed strongest to weakest — prefer "
        "the strongest axis still missing from the label.\n"
        "\n"
        "- **Cross-item rule** — uniqueness, ordering, or a "
        "conditional that relates list items (requires "
        "Cardinality already present).\n"
        "- **Cardinality** — return exactly N records (N ≥ 2) "
        "instead of one. Changes answer shape.\n"
        "- **Composite** — two filters on different dimensions "
        "(categorical AND threshold, for example).\n"
        "- **Filter** — a single categorical exclusion or "
        "threshold on an existing field.\n"
        "- **Width** — more fields per record, pulled from "
        "additional tables along the path.\n"
        "\n"
        "Width and a single Filter alone rarely shift pass_rate "
        "enough. The first escalation after a too_easy rejection "
        "should add Cardinality or a Composite filter unless the "
        "label already has one.\n"
        "\n"
        "Axes are structural. Pick the concrete field, category, "
        "or threshold from the current DB's schema and observed "
        "data distributions, never from a fixed template.",

        # ── After Rejection ──
        "# After Rejection\n"
        "A rejection is not a signal to explore more. Within 2 "
        "atomic calls of rejection feedback, call submit_draft "
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
        "The label must be the only correct answer. On any 1:N "
        "path, either narrow to one record with a constraint or "
        "return the full list. Never leave 'a customer' or 'one "
        "rental' when multiple rows satisfy the request.",

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
    family_counts = tool_surface_summary.get("family_counts")
    tool_count = tool_surface_summary.get("tool_count")
    if isinstance(tool_count, int):
        environment_lines.append(f"- Total atomic tools: {tool_count}")
    if isinstance(family_counts, dict):
        for family_name, meaning in [
            ("get", "retrieve one entry by ID"),
            ("find", "find entries matching a condition"),
            ("calc", "compute one statistic"),
            ("rank", "rank groups by a statistic"),
        ]:
            count = family_counts.get(family_name)
            if isinstance(count, int) and count > 0:
                environment_lines.append(
                    f"- {family_name}: {count} tools — {meaning}."
                )
    surfaces = tool_surface_summary.get("entity_surfaces")
    if isinstance(surfaces, list):
        hint_limit = runtime_config.prompt_tool_surface_hint_limit
        for item in surfaces[:hint_limit]:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name") or "")
            readable_fields = item.get("readable_fields")
            if not isinstance(readable_fields, list):
                continue
            readable = [str(f) for f in readable_fields]
            if readable:
                environment_lines.append(
                    f"- {tool_name}: readable fields={readable}"
                )
            else:
                environment_lines.append(
                    f"- {tool_name}: readable fields=[] (id-only)"
                )
    environment_lines.append(
        "- Schema orientation is for navigation only. "
        "Verify readability in actual tool results before "
        "using a field in the label."
    )
    sections.append(
        "# Environment\n" + "\n".join(environment_lines)
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
