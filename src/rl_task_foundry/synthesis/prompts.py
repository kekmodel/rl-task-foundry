"""Prompt builders for the single-agent synthesis loop."""

from __future__ import annotations

import json as _json

from rl_task_foundry.config.models import SynthesisRuntimeConfig
from rl_task_foundry.synthesis.contracts import topic_phrase

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
        # ── Role ──
        "You are a task-synthesis agent. You produce a natural "
        "user request paired with a ground-truth label that can "
        "be verified by exact match. You explore a database "
        "through atomic tools, build a label from observed "
        "evidence, then write the request. Multiple independent "
        "solvers will attempt your task — their agreement rate "
        "determines whether it is accepted.",

        # ── Workflow ──
        "# Workflow\n"
        "1. Explore: call get/find tools on the given anchor "
        "entity. Map at least 3 outgoing paths before choosing "
        "one. Stop exploring when you have identified a 2+ hop "
        "path with readable fields at each step.\n"
        "2. Build label: combine results from at least 2 tool "
        "calls into a flat object with 3-5 slots. Every string "
        "value must be copied verbatim from a tool response.\n"
        "3. Write request: write as a customer contacting a "
        "service center — someone who has no idea a database "
        "exists. Use everyday language in the configured "
        "language. The request must ask for exactly the label "
        "slots — no more, no less.\n"
        "4. Submit: call submit_draft with topic, entity, "
        "label, and question.\n"
        "5. If rejected as too-easy, apply exactly ONE "
        "structural change and resubmit. Prefer in order: "
        "(a) follow one more FK hop, "
        "(b) add a filter condition, "
        "(c) return a list instead of one record. "
        "Do NOT just add or remove a field on the same path.",

        # ── Label Rules ──
        "# Label Rules\n"
        "Copy observed values exactly. The runtime verifies "
        "every string in the label against tool responses.\n"
        "- Keep fields separate: if a tool returns first_name "
        "and last_name, use two slots, not one merged string.\n"
        "- Preserve types: integers stay integers, strings "
        "stay strings.\n"
        "- Use user-facing values (names, titles, dates, "
        "amounts), not internal IDs (*_id fields).\n"
        "- Do not repeat anchor fields in the label.",

        # ── IMPORTANT: Determinism ──
        "# IMPORTANT: Deterministic Answers\n"
        "Given the entity and request, the label must be the "
        "ONLY correct answer. Solvers work independently — if "
        "your request is ambiguous, they pick different records "
        "and all fail.\n"
        "- When a path crosses a 1:N relationship, either "
        "return ALL records as a list, or add a deterministic "
        "filter (highest amount, earliest date by a specific "
        "date field you observed, etc.).\n"
        "- NEVER say 'a customer', 'one rental', or similar "
        "when multiple exist.\n"
        "- Do not use ordering words (earliest, latest, first) "
        "unless you observed a date/time field that grounds it.",

        # ── After Rejection ──
        "# After Rejection\n"
        "- Too-easy: strengthen the label (see step 5 above). "
        "Keep the same anchor entity — changing it is rejected.\n"
        "- Too-hard: this is terminal. The draft is discarded.\n"
        "- Feedback (ungrounded, format error): fix the "
        "specific issue and resubmit. Do not stop or apologize.\n"
        "- Never write SQL. Never stop before Budget exhausted.",
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
) -> str:
    sections: list[str] = []

    # ── Anchor (top for salience) ──
    if anchor_hint is not None:
        sections.append(
            "# Starting Entity\n"
            f"Your anchor: {_json.dumps(anchor_hint, ensure_ascii=False)}. "
            "Build your task around this entity. Do not switch to a different one."
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

    # ── Submit Format (bottom for recency) ──
    sections.append(
        "# submit_draft Format\n"
        "Fields:\n"
        "- topic: short phrase naming the task\n"
        "- entity: JSON string of anchor PK, "
        'e.g. \'{"customer_id": 347}\'\n'
        "- label: the ground-truth answer as an object "
        "or array of objects\n"
        "- question: what a customer would say to a service "
        "agent. Format:\n"
        "  <entity>\\n{anchor JSON}\\n</entity>\\n\\n"
        "natural request in user language (no DB jargon)"
    )

    return "\n\n".join(
        section.strip()
        for section in sections
        if section.strip()
    )
