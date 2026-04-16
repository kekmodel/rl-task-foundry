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
        "customer request paired with a ground-truth label "
        "verified by exact match. Multiple independent solvers "
        "attempt your task — their agreement rate determines "
        "acceptance.",

        # ── Workflow ──
        "# Workflow\n"
        "**Start simple. Build complexity through rejections.**\n\n"
        "1. **Explore** the anchor entity with a few get/find "
        "calls. Check data distributions in the session.\n"
        "2. **First submit**: a simple multi-hop lookup "
        "(2+ tool calls, no constraints). Expect too-easy.\n"
        "3. **On each too-easy rejection**, keep your previous "
        "draft and apply **ONE** escalation — either add a "
        "constraint OR expand the structure (e.g. 1 record "
        "-> a list of 3 with cross-item rules):\n\n"
        "| Type | Example |\n"
        "| --- | --- |\n"
        "| Budget | total rental cost under $10 |\n"
        "| Preference | only PG-rated films |\n"
        "| Quality | rating above a threshold |\n"
        "| Uniqueness | no repeated categories in a list |\n"
        "| Conditional | if hotel is expensive, cheaper meals |\n"
        "| Cardinality | exactly 3 items |\n\n"
        "Use data distributions for realistic thresholds.\n"
        "4. **Rewrite** the request as a customer who knows "
        "nothing about databases. Constraints become natural "
        "preferences.\n"
        "5. **Submit** via submit_draft.\n\n"
        "Repeat 3-5 until accepted. NEVER write SQL.\n\n"
        "**Example escalation:**\n"
        "Round 1: simple lookup -> too-easy\n"
        "Round 2: + preference (filter by category) -> too-easy\n"
        "Round 3: + cardinality (find N items) -> too-easy\n"
        "Round 4: + uniqueness (no repeats across items) "
        "-> too-easy\n"
        "Round 5: + budget (total under threshold) -> accepted",

        # ── Label Rules ──
        "# Label Rules\n"
        "- Copy every string **verbatim** from tool results — "
        "the runtime rejects unmatched strings.\n"
        "- **One field per value**: do not merge first_name + "
        "last_name into one string.\n"
        "- Keep original types: integers stay integers.\n"
        "- Use user-facing values, not internal IDs (*_id).",

        # ── Deterministic Answers ──
        "# Deterministic Answers\n"
        "IMPORTANT: The label must be the **ONLY** correct "
        "answer. Solvers work independently — ambiguous "
        "answers cause disagreement and rejection.\n"
        "- On 1:N paths, add a constraint that narrows to "
        "exactly one record, or return ALL as a list.\n"
        "- NEVER say 'a customer' or 'one rental' when "
        "multiple exist.",
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
    data_profile: object | None = None,
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
    if data_profile is not None and hasattr(data_profile, "render"):
        rendered = data_profile.render()
        if rendered.strip():
            sections.append(
                "# Data Distributions\n"
                "Use these to design realistic constraints "
                "(budget thresholds, quality filters, etc.).\n"
                + rendered
            )

    # ── Submit Format (bottom for recency) ──
    sections.append(
        "# submit_draft\n"
        "```\n"
        "submit_draft(\n"
        '  topic = "short task description",\n'
        '  entity = \'{"pk_column": value}\',\n'
        "  label = {field: value, ...} or [{...}, {...}],\n"
        "  question = \"<entity>\\n{anchor}\\n</entity>\\n\\n"
        "customer request in user language\"\n"
        ")\n"
        "```"
    )

    return "\n\n".join(
        section.strip()
        for section in sections
        if section.strip()
    )
