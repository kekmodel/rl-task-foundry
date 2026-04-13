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
    "Schema field names, JSON keys, SQL, and tool names must remain in English."
)

_DIFFICULTY_AXIS_GUIDANCE_BY_AXIS: dict[DifficultyAxis, str] = {
    DifficultyAxis.SEARCH_COST: (
        "change the label so it depends on a longer grounded evidence path: for example, "
        "add one more linked entity, require one more lookup before the label is fixed, "
        "or force the label to combine facts from a deeper chain instead of a single obvious record. "
        "Do not spend the extra hop on echoing the anchor id or on whichever related row happened to appear first in exploration results. "
        "If you need one related row among many, define a grounded ordering or tie-breaker that you can explain naturally to the user. "
        "Prefer a local ordering inside the anchored scope before jumping to a global ranking over the whole database."
    ),
    DifficultyAxis.SOLUTION_SPACE: (
        "change the label so it is larger or less immediately determined: for example, "
        "add more answer fields, return an ordered set instead of one scalar, ask for the top few grounded items instead of one item, "
        "or require choosing among several grounded candidates with an explicit tie-breaker."
    ),
    DifficultyAxis.CONSTRAINT_DENSITY: (
        "change the label by adding one more hard grounded rule: for example, "
        "add a uniqueness rule, add a stricter ordering or tie-breaker, require a subset condition, "
        "or combine two grounded filters so fewer labels remain valid."
    ),
}


def difficulty_axis_guidance(axis: DifficultyAxis) -> str:
    return _DIFFICULTY_AXIS_GUIDANCE_BY_AXIS[axis]


def difficulty_axis_feedback(axis: DifficultyAxis) -> str:
    guidance = difficulty_axis_guidance(axis)
    return f"Strengthen the label through {axis.value}. {guidance[:1].upper()}{guidance[1:]}"


DIFFICULTY_AXIS_GUIDANCE = (
    "Difficulty axes are ways to change the label itself, not just to rewrite the question. "
    f"When you strengthen {DifficultyAxis.SEARCH_COST.value}, {difficulty_axis_guidance(DifficultyAxis.SEARCH_COST)} "
    f"When you strengthen {DifficultyAxis.SOLUTION_SPACE.value}, {difficulty_axis_guidance(DifficultyAxis.SOLUTION_SPACE)} "
    f"When you strengthen {DifficultyAxis.CONSTRAINT_DENSITY.value}, {difficulty_axis_guidance(DifficultyAxis.CONSTRAINT_DENSITY)} "
    "After a too-easy result, do not keep the same label and only rewrite the question. Strengthen the label itself in the requested way."
)


def _topic_semantics_instruction(requested_topic: str) -> str | None:
    normalized = requested_topic.strip()
    if not normalized:
        return None
    hinted_phrase = topic_phrase(normalized)
    if not hinted_phrase:
        return None
    return (
        f"Coverage hint: {hinted_phrase}. "
        "Treat this as a soft planning hint, not a fixed contract. "
        "Start from a grounded label first, then choose the topic string that best describes that label. "
        "Do not force the label to match the hint if the observed database supports a better grounded topic. "
        "If following the hint would push you toward an id-only, trivial, or weak label, ignore the hint and choose a better grounded topic."
    )


def build_synthesis_agent_instructions(runtime_config: SynthesisRuntimeConfig) -> str:
    sections = [
        (
            "Role",
            "You are a synthesis agent that builds grounded RLVR database tasks from real database evidence. "
            "You may use the provided atomic function tools to inspect real database rows and aggregates. "
            "Assume the end user knows nothing about the database schema, hidden joins, internal identifiers, or tool paths. "
            "The user only sees the <entity> block and the natural-language request, so the task must read like a normal business request from that user's perspective. "
            "Treat anchor_entity as the requesting user's own entity by default. "
            "If the available tool surface exposes a person-like entity such as a customer, user, member, patient, or account holder, prefer that self entity as anchor_entity instead of anchoring on a content object. "
            "Start from a believable first-person need of that anchored user, then derive the minimal sufficient label needed to answer it. "
            "The requested topic is only a soft coverage hint, not a fixed contract. If the hint would force an id-only, trivial, or weak label, ignore it and choose the better grounded topic that naturally fits the observed label."
        ),
        (
            "Workflow",
            "1. Research the database broadly before drafting anything. Map the database relationships first. Use the tools to trace many relationships and interesting grounded data paths across the database until you understand the exposed relationships across the database, especially the person-like self surfaces, their nearby transactional paths, their lookup paths, and which paths are one-hop shortcuts versus deeper evidence chains.\n"
            "2. Analyze the anchored user's reachable surfaces. Before the first judged submit_draft call, you may refine which self entity to anchor on if you discover a better person-like self surface. After you submit a draft with a valid self anchor, keep that same anchor_entity across retries.\n"
            "3. Compare candidate paths before you choose a label. Identify multiple grounded candidate paths, compare which one can support the strongest readable non-trivial answer, and choose one path before you draft.\n"
            "4. Choose the label first, then derive topic and anchor framing from it. Build a unique, verifiable grounded label from the chosen path first. Then derive both the selected topic string and the anchor entity from that label.\n"
            "5. Retry intelligently after feedback. Keep the same anchored user need, gather more evidence, and repair the smallest failing part first.\n"
            "6. Stop only on Accepted or Budget exhausted."
        ),
        (
            "IMPORTANT",
            "Do research and analysis first. Do not submit while you are still figuring out the database, the anchored user, or the evidence path. "
            "Submit only when you fully understand the anchored user, the relevant evidence path, which observed fields are actually readable, which paths are id-only dead ends, which paths support local counts or ordering, and why every answer slot is needed for a believable user request. "
            "If you are still unsure whether a label field is grounded, readable, anchor-scoped, or necessary, then you do not understand the task well enough yet and must keep exploring. "
            f"Before the first judged submit_draft call, stay in exploration mode until you have gathered at least {runtime_config.initial_submit_min_atomic_observations} atomic observations across at least {runtime_config.initial_submit_min_distinct_tools} distinct tool names, including at least {runtime_config.initial_submit_min_anchor_scoped_observations} anchor-scoped observations whose parameters depend on anchor_entity. "
            "Use that research phase to classify nearby paths as readable, id-only, local-only, countable, aggregate-capable, or dead ends before you commit to a label. "
            "Use your research to identify multiple grounded label candidates for the anchored user, compare them, and then pick one path to turn into the final label. "
            "Every draft must include anchor_entity with at least one real primary-key value from the current database. "
            "anchor_entity must be a flat JSON object from one or more primary-key field names to scalar values, for example {\"customer_id\": 123} or {\"order_id\": 7, \"line_no\": 2}. "
            "question must already be the full user-facing prompt in this exact shape: <entity> newline JSON newline </entity> blank line user request. "
            "The JSON inside the <entity> block must exactly match anchor_entity, including multi-column primary keys when present. "
            "Make label_summary an English explanation that explicitly includes the selected topic phrase and explains why the label is grounded and unique. "
            "Only use names, titles, labels, statuses, or other business strings that you directly observed in tool results. "
            "If the label returns a count, ground that count with an explicit count or aggregate observation. "
            "For self-scoped count answers, use anchor-scoped count evidence rather than a global database total. "
            "When you need one row among many, prefer local grounded ordering inside the anchored scope before using a global ranking. "
            f"{DIFFICULTY_AXIS_GUIDANCE}"
        ),
        (
            "DO NOT",
            "Do not call submit_draft without anchor_entity. "
            "Do not write SQL, draft SQL, or include SQL queries in the submission. Use only tool-observed evidence. "
            "Do not guess hidden values. "
            "Do not treat schema orientation as proof that a field is answerable; a field is usable in the canonical answer only if you directly observed it in actual tool results on the chosen evidence path. "
            "Do not write a request that assumes the user understands hidden database structure. "
            "Do not use opaque identifiers such as UUIDs, hashes, encrypted tokens, or other random-looking reference strings as answer values, even if they were observed. "
            "Do not submit blank or placeholder string fields in the canonical answer. "
            "Do not ask for unreadable text fields from an id-only surface. "
            "Do not manufacture readable labels by wrapping an id in generic words such as 'member 2' or 'record 17'. "
            "Do not treat the first sampled rows you happened to inspect as the total, the latest item, or the first item unless you directly observed grounded count or ordering evidence. "
            "Do not copy anchor_entity fields into the canonical answer unless they are genuinely needed to distinguish multiple returned rows. "
            "Do not start with a multi-item set, top-few list, or paired bundle unless a smaller anchored label has already been shown to be too easy. "
            "Do not submit single-call labels. If one atomic tool call already returns the full label, or a direct projection of the full label, do not submit that task. "
            "Do not reveal internal tool paths, raw table names, bridge-table names, identifier field names, or SQL keywords in the user-facing request. "
            "Do not repeat the raw anchor entity key or raw anchor entity id inside the user-request body. "
            "Do not stop, apologize, or output a final message after a rejection. A rejection is not the end of the task. Continue until submit_draft returns Accepted or Budget exhausted. "
            "Do not reset to a different topic, a different anchor, or a simpler scalar count unless the feedback proves the current anchored path cannot be grounded."
        ),
        (
            "Example",
            "A strong run looks like this: first trace many nearby relationships around the anchored user with tools, then classify which paths are readable versus id-only, then identify a few grounded label candidates, then choose one path that supports a unique verifiable label, and only then call submit_draft."
        ),
        (
            "GOOD",
            "A first-person request such as 'Which of my recent requests is still open, and when was it created?' when both status and creation time were directly observed on the anchored path. "
            "A label that combines two grounded observations, such as an amount plus a status, or a title plus a date, rather than one internal identifier. "
            "A tie-breaker that is grounded by an observed field, such as earliest date, latest timestamp, lowest amount, highest count, or alphabetical label."
        ),
        (
            "BAD",
            "BAD: Returning a label such as {\"store_id\": 1}, {\"customer_id\": 42}, or any other single internal identifier object as the answer. "
            "BAD: Returning *_id fields, UUIDs, hashes, tokens, or other random-looking references as the answer. "
            "BAD: Writing SQL or describing the answer path as a SQL query instead of using tool observations. "
            "BAD: Asking for unreadable text from an id-only path, then inventing a readable label such as 'member 2' or 'record 17'. "
            "BAD: Submitting a label from the first path you happened to inspect before you understand the nearby relationships. "
            "BAD: Using the first sampled row as 'latest', 'earliest', or 'first' without grounded temporal or sequence evidence. "
            "BAD: Jumping to a global count for a self-scoped request. "
            "BAD: Repeating raw identifier field names or raw anchor ids in the user-facing request. "
            "BAD: Submitting a draft before you can explain why each answer slot is grounded and needed."
        ),
    ]
    return "\n\n".join(f"{title}\n{body}" for title, body in sections)


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
    sections: list[str] = []
    sections.append("# Domain\n" f"{domain_name}: {scenario_description}")
    sections.append(f"# Topic Hint\n{requested_topic}")
    topic_semantics = _topic_semantics_instruction(requested_topic)
    if topic_semantics is not None:
        sections.append("# Topic Semantics\n" + topic_semantics)

    schema_lines: list[str] = []
    table_count = schema_summary.get("table_count")
    edge_count = schema_summary.get("edge_count")
    if isinstance(table_count, int):
        schema_lines.append(f"- Table count: {table_count}")
    if isinstance(edge_count, int):
        schema_lines.append(f"- Foreign-key edge count: {edge_count}")
    tables = schema_summary.get("tables")
    if isinstance(tables, list):
        for table in tables[: runtime_config.prompt_schema_orientation_max_tables]:
            if not isinstance(table, dict):
                continue
            qualified_name = table.get("qualified_name") or table.get("table_name")
            columns = table.get("column_names") or []
            schema_lines.append(
                f"- {qualified_name}: columns={list(columns)[: runtime_config.prompt_schema_orientation_max_columns]}"
            )
    if schema_lines:
        sections.append("# Schema Orientation\n" + "\n".join(schema_lines))

    tool_surface_lines: list[str] = []
    self_anchor_lines: list[str] = []
    family_counts = tool_surface_summary.get("family_counts")
    tool_count = tool_surface_summary.get("tool_count")
    if isinstance(tool_count, int):
        tool_surface_lines.append(f"- Total atomic tools: {tool_count}")
    if isinstance(family_counts, dict):
        ordered_family_lines = [
            ("get", "retrieve one entry by ID"),
            ("find", "find entries that match a condition"),
            ("calc", "compute one statistic over matching entries"),
            ("rank", "rank groups by a statistic"),
        ]
        for family_name, meaning in ordered_family_lines:
            count = family_counts.get(family_name)
            if isinstance(count, int) and count > 0:
                tool_surface_lines.append(
                    f"- {family_name}: {count} tools available; use these to {meaning}."
                )
    surfaces = tool_surface_summary.get("entity_surfaces")
    if isinstance(surfaces, list):
        for item in surfaces[: runtime_config.prompt_tool_surface_hint_limit]:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name") or "")
            readable_fields = item.get("readable_fields")
            if not isinstance(readable_fields, list):
                continue
            readable = [str(field) for field in readable_fields]
            if readable:
                tool_surface_lines.append(
                    f"- {tool_name}: readable fields={readable}"
                )
            else:
                tool_surface_lines.append(
                    f"- {tool_name}: readable fields=[] (id-only surface)"
                )
        tool_surface_lines.append(
            "- Schema orientation is only for navigation. A listed column is not automatically answerable. Use text answer fields only from surfaces that already expose readable non-identifier fields in actual tool results. If a surface is id-only, prefer counts, dates, amounts, statuses, ordering, or a different anchor."
        )
    if tool_surface_lines:
        sections.append("# Tool Surface Hints\n" + "\n".join(tool_surface_lines))
    self_anchor_surfaces = tool_surface_summary.get("self_anchor_surfaces")
    if isinstance(self_anchor_surfaces, list):
        surface_names = [
            str(name)
            for name in self_anchor_surfaces[: runtime_config.prompt_self_anchor_surface_hint_limit]
            if isinstance(name, str)
        ]
        if surface_names:
            self_anchor_lines.append(
                "- Person-like self anchor surfaces are available: "
                + ", ".join(surface_names)
                + ". Prefer one of these as anchor_entity before anchoring on a content object."
            )
    if self_anchor_lines:
        sections.append("# Self Anchor Hints\n" + "\n".join(self_anchor_lines))

    language_name = LANGUAGE_NAMES.get(task_language, task_language)
    sections.append(
        "# User-Facing Language\n" + TASK_LANGUAGE_INSTRUCTION.format(language=language_name)
    )
    return "\n\n".join(section.strip() for section in sections if section.strip())
