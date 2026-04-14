"""Prompt builders for the single-agent synthesis loop."""

from __future__ import annotations

from rl_task_foundry.config.models import SynthesisRuntimeConfig
from rl_task_foundry.synthesis.contracts import DifficultyAxis, topic_phrase

LANGUAGE_NAMES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
}

TASK_LANGUAGE_INSTRUCTION = (
    "Generate the user-facing question in {language}. "
    "Schema field names, JSON keys, and tool names "
    "must remain in English."
)

_DIFFICULTY_AXIS_GUIDANCE_BY_AXIS: dict[DifficultyAxis, str] = {
    DifficultyAxis.SEARCH_COST: (
        "deepen the evidence path: add one more linked "
        "entity or require one more hop before the label "
        "is fixed. Preserve existing readable fields and "
        "extend the chain rather than replacing it."
    ),
    DifficultyAxis.SOLUTION_SPACE: (
        "enlarge the answer: add more fields, return a "
        "list instead of a scalar, or require choosing "
        "among candidates with a grounded tie-breaker. "
        "Preserve existing grounded slots and extend them."
    ),
    DifficultyAxis.CONSTRAINT_DENSITY: (
        "tighten the label: add a uniqueness rule, a "
        "stricter ordering, a subset condition, or combine "
        "filters so fewer answers remain valid. Keep the "
        "same path and add one extra grounded rule."
    ),
}


def difficulty_axis_guidance(axis: DifficultyAxis) -> str:
    return _DIFFICULTY_AXIS_GUIDANCE_BY_AXIS[axis]


def difficulty_axis_feedback(axis: DifficultyAxis) -> str:
    guidance = difficulty_axis_guidance(axis)
    return (
        f"Strengthen the label through {axis.value}. "
        f"{guidance[:1].upper()}{guidance[1:]}"
    )


DIFFICULTY_AXIS_GUIDANCE = (
    "Difficulty axes change the label itself, not just "
    "the question wording. "
    + " ".join(
        f"{axis.value}: {difficulty_axis_guidance(axis)}"
        for axis in DifficultyAxis
    )
    + " After a too-easy result, strengthen the label "
    "itself rather than only rewriting the question."
)


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
    sections = [
        (
            "Role",
            "You build grounded RLVR tasks from real "
            "database evidence. Discover a verifiable label, "
            "then render it as a natural user request. "
            "The end user sees only the <entity> block and "
            "the request — write as if for someone who knows "
            "nothing about the schema.",
        ),
        (
            "Principles",
            "1. Label before request — the request renders "
            "the label, not the other way around.\n"
            "2. Groundedness over style — a plain but exact "
            "task beats a polished but invented one.\n"
            "3. Topic hints are soft — ignore them if they "
            "lead to weak labels.\n"
            "4. Feedback is a signal — use rejections to "
            "repair, not to stop.",
        ),
        (
            "Workflow",
            "1. Research: inspect the anchor entry and map "
            "nearby paths (readable, id-only, countable, "
            "orderable, dead-end).\n"
            "2. Compare: evaluate multiple candidate paths "
            "before committing.\n"
            "3. Label: pick the strongest grounded path and "
            "build the canonical answer.\n"
            "4. Render: write a request that asks for every "
            "non-anchor slot — no more, no less.\n"
            "5. Retry: on feedback, keep the anchor, repair "
            "the smallest failing part, and change exactly "
            "one difficulty axis when asked.",
        ),
        (
            "Label Rules",
            # grounding
            "Every answer slot must come from a directly "
            "observed tool result — never guess, infer, or "
            "reformat values. "
            "Copy observed strings exactly; do not merge, "
            "shorten, or paraphrase them. "
            # anchor
            "anchor_entity must be a flat JSON object of "
            "primary-key fields to scalar values. "
            "The <entity> block identifies the subject, so "
            "do not duplicate anchor fields in the label "
            "unless the request explicitly asks for them. "
            # structure
            "question must follow: <entity>\\n{json}\\n"
            "</entity>\\n\\nuser request. "
            "The request must cover every non-anchor label "
            "slot and nothing extra. If a slot cannot be "
            "asked for naturally, remove it from the label. "
            # counts
            "Ground counts with an explicit anchor-scoped "
            "aggregate observation — not a global total. "
            # difficulty
            f"{DIFFICULTY_AXIS_GUIDANCE}",
        ),
        (
            "Prohibitions",
            "Do not submit while still exploring. "
            "Do not use raw column names (first_name, "
            "rental_date, etc.) in the request — rephrase "
            "as natural language the end user would say. "
            "Do not submit single-call labels (one tool "
            "call returns the full answer). "
            "Do not write or include SQL. "
            "Do not manufacture readable values from ids. "
            "Do not stop or apologize after rejection — "
            "continue until Accepted or Budget exhausted. "
            "Do not reset anchor or topic unless feedback "
            "proves the current path is ungroundable.",
        ),
    ]
    return "\n\n".join(f"# {title}\n{body}" for title, body in sections)


def build_synthesis_input(
    *,
    domain_name: str,
    scenario_description: str,
    requested_topic: str,
    task_language: str,
    schema_summary: dict[str, object],
    tool_surface_summary: dict[str, object],
    runtime_config: SynthesisRuntimeConfig,
) -> str:
    session_lines: list[str] = []
    environment_lines: list[str] = []

    session_lines.append(f"- Domain: {domain_name}")
    session_lines.append(f"- Scenario: {scenario_description}")
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
        "- User-facing language: "
        + TASK_LANGUAGE_INSTRUCTION.format(language=language_name)
    )

    table_count = schema_summary.get("table_count")
    edge_count = schema_summary.get("edge_count")
    if isinstance(table_count, int):
        environment_lines.append(
            f"- Table count: {table_count}"
        )
    if isinstance(edge_count, int):
        environment_lines.append(
            f"- Foreign-key edge count: {edge_count}"
        )
    tables = schema_summary.get("tables")
    if isinstance(tables, list):
        max_tables = (
            runtime_config.prompt_schema_orientation_max_tables
        )
        for table in tables[:max_tables]:
            if not isinstance(table, dict):
                continue
            qualified_name = table.get(
                "qualified_name"
            ) or table.get("table_name")
            columns = table.get("column_names") or []
            max_cols = (
                runtime_config.prompt_schema_orientation_max_columns
            )
            environment_lines.append(
                f"- {qualified_name}: "
                f"columns={list(columns)[:max_cols]}"
            )

    family_counts = tool_surface_summary.get("family_counts")
    tool_count = tool_surface_summary.get("tool_count")
    if isinstance(tool_count, int):
        environment_lines.append(
            f"- Total atomic tools: {tool_count}"
        )
    if isinstance(family_counts, dict):
        ordered_family_lines = [
            ("get", "retrieve one entry by ID"),
            ("find", "find entries matching a condition"),
            ("calc", "compute one statistic"),
            ("rank", "rank groups by a statistic"),
        ]
        for family_name, meaning in ordered_family_lines:
            count = family_counts.get(family_name)
            if isinstance(count, int) and count > 0:
                environment_lines.append(
                    f"- {family_name}: {count} tools — "
                    f"{meaning}."
                )
    # Schema topology hints — adaptive, derived from introspection
    topology_lines: list[str] = []
    hub_tables = schema_summary.get("hub_tables")
    if isinstance(hub_tables, list) and hub_tables:
        topology_lines.append(
            f"- Hub tables (many inbound FKs): "
            f"{hub_tables}. These are central entities — "
            "good anchors for multi-hop exploration."
        )
    bridge_tables = schema_summary.get("bridge_tables")
    if isinstance(bridge_tables, list) and bridge_tables:
        topology_lines.append(
            f"- Bridge tables (id-only, 2+ FKs): "
            f"{bridge_tables}. These connect entities but "
            "have no readable fields — traverse through "
            "them, do not use their fields in the label."
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
                f"- Readable entity tables: "
                f"{', '.join(readable_tables[:8])}"
            )
        if id_only_tables:
            topology_lines.append(
                f"- Id-only tables (no readable fields): "
                f"{', '.join(id_only_tables[:8])}"
            )
        # high-fanout edges
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
                "- High-fanout edges (one-to-many): "
                + ", ".join(fanout_hints[:6])
                + ". Paths through these expand the "
                "candidate set — useful for search_cost."
            )

    surfaces = tool_surface_summary.get("entity_surfaces")
    if isinstance(surfaces, list):
        hint_limit = (
            runtime_config.prompt_tool_surface_hint_limit
        )
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
                    f"- {tool_name}: "
                    f"readable fields={readable}"
                )
            else:
                environment_lines.append(
                    f"- {tool_name}: "
                    "readable fields=[] (id-only)"
                )
        environment_lines.append(
            "- Schema orientation is for navigation only. "
            "A listed column is not automatically "
            "answerable. Verify readability in actual tool "
            "results before using a field in the label."
        )

    sections: list[str] = [
        "# BOUNDARY\n"
        "Static rules end here. "
        "Everything below is specific to this session.",
        "# Session Context\n" + "\n".join(session_lines),
        "# Environment and State\n"
        + "\n".join(environment_lines),
    ]
    if topology_lines:
        sections.append(
            "# Schema Topology\n"
            + "\n".join(topology_lines)
        )
    return "\n\n".join(
        section.strip()
        for section in sections
        if section.strip()
    )
