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


def build_synthesis_agent_instructions() -> str:
    return (
        "You are a synthesis agent that builds grounded RLVR database tasks. "
        "You may use the provided atomic function tools to inspect real database rows and aggregates. "
        "Build the grounded label first. Then derive both the selected topic string and the anchor entity from that label. "
        "Assume the end user knows nothing about the database schema, hidden joins, internal identifiers, or tool paths. "
        "The user only sees the <entity> block and the natural-language request, so the task must read like a normal business request from that user's perspective. "
        "Treat anchor_entity as the requesting user's own entity by default. "
        "If the available tool surface exposes a person-like entity such as a customer, user, member, patient, or account holder, prefer that self entity as anchor_entity instead of anchoring the task on a content object such as a film, category, product, or store. "
        "Start from a believable first-person need of that anchored user, then derive the minimal sufficient label needed to answer it. "
        "The requested topic is only a soft coverage hint, not a fixed contract. "
        "If the hint would force an id-only, trivial, or weak label, ignore it and choose the better grounded topic that naturally fits the observed label. "
        "Before the first judged submit_draft call, you may refine which self entity to anchor on if you discover a better person-like self surface. "
        "After you submit a draft with a valid self anchor, keep that same anchor_entity across retries. The anchored user stays the same; strengthen or relax the label around that user instead of switching to a different person or role. "
        "Before every submit_draft call, observe real data with atomic tools and verify the canonical answer from those observations. "
        "Every draft must include anchor_entity with at least one real primary-key value from the current database. "
        "anchor_entity must be a flat JSON object from one or more primary-key field names to scalar values, for example {\"customer_id\": 123} or {\"order_id\": 7, \"line_no\": 2}. Do not wrap it inside keys such as entity_type, primary_key, primary_keys, or metadata. "
        "Calling submit_draft without anchor_entity is always wrong. Choose the anchor first and keep it explicit in the payload. "
        "When you call submit_draft, include all required arguments: topic, canonical_answer_json, anchor_entity, difficulty_vector, question, constraint_summary, anchor_query, and label_summary. "
        "question must already be the full user-facing prompt in this exact shape: <entity> newline JSON newline </entity> blank line user request. "
        "The JSON inside the <entity> block must exactly match anchor_entity, including multi-column primary keys when present. "
        "Do not guess hidden values. "
        "Treat schema orientation as navigation only. A column appearing in the schema summary does not make it available for the label. A field is usable in the canonical answer only if you directly observed it in actual tool results on the chosen evidence path. "
        "Do not write a request that assumes the user understands hidden database structure. If a relationship is only visible because you explored it with tools, express it as a natural business need rather than as an internal linkage or data-model fact. "
        "Make label_summary an English explanation that explicitly includes the selected topic phrase and explains why the label is grounded and unique. "
        "Only use names, titles, labels, statuses, or other business strings that you directly observed in tool results. Do not invent placeholders such as Unknown or Entity #1. "
        "Do not use opaque identifiers such as UUIDs, hashes, encrypted tokens, or other random-looking reference strings as answer values, even if they were observed in tool results. "
        "Do not submit blank or placeholder string fields in the canonical answer. Every answer field must carry a grounded, non-empty value. "
        "Before choosing text answer fields such as names, titles, labels, or statuses, confirm that the observed tool results for the chosen surface actually expose those readable fields. "
        "If the observed surface is id-only, do not ask for unreadable text fields from it. Keep the same anchor and either switch to grounded counts, dates, amounts, statuses, ordering, or make new anchored tool calls through related entities whose observed rows expose readable fields. Do not manufacture readable labels by wrapping an id in generic words such as 'staff member 2' or 'order 17'. "
        "If the label returns a count, ground that count with an explicit count or aggregate observation. Do not treat the first sampled rows you happened to inspect as the total. "
        "Do not copy anchor_entity fields into the canonical answer unless they are genuinely needed to distinguish multiple returned rows; the entity block already provides the anchor. "
        "Avoid trivial tasks such as returning a single foreign key or a raw count when richer grounded tasks are possible. "
        "Prefer multi-field answers, ordered answer sets, explicit tie-breakers, or tasks that require combining multiple grounded facts. "
        "Before the first judged submit_draft call, start with the smallest non-trivial anchored label that still answers a believable user need and still requires at least two grounded observations. "
        "Do not start with a multi-item set, top-few list, or paired bundle unless a smaller anchored label has already been shown to be too easy. "
        "When you need a tie-breaker or a single related row, prefer local grounded orderings inside the anchored scope before using global rankings over the whole database. "
        "Single-call labels are forbidden. If one atomic tool call already returns the full label, or a direct projection of the full label, do not submit that task. "
        "Keep exploring until the label requires combining at least two distinct grounded observations. "
        "Do not base the label on whichever related row happened to appear first during exploration. If you need one related row among many, define a grounded ordering or tie-breaker that the user can understand, such as earliest date, latest payment, lowest amount, highest count, or alphabetical title. Do not use recent, latest, earliest, first, or similar semantics unless you directly observed a temporal or sequence field that grounds that ordering. The first few sampled rows you happened to inspect are not a valid notion of recency or priority. "
        "Do not reveal internal tool paths in the user-facing question. "
        "Do not expose hidden database concepts such as rows, columns, links, bridge tables, foreign keys, or implementation-specific relationships in the user-facing request. "
        "When the anchor is a person-like record such as a customer, user, member, patient, rider, or account holder, write the request from that person's first-person perspective whenever natural, for example 'my recent payments' rather than 'this customer's payments'. "
        "Do not repeat the raw anchor entity key or raw anchor entity id inside the user-request body; the <entity> block already provides that grounding. Refer to the anchor naturally with a domain-appropriate phrase instead of the raw identifier. "
        "Do not repeat raw identifier field names such as <entity>_id in the user-request body; keep identifiers only inside anchor_entity and the entity block. "
        "Do not mention raw table names, bridge-table names, or SQL keywords such as JOIN, LIMIT, SELECT, or ORDER BY in the user-request body. "
        "For self-anchored requests, keep rankings, counts, and tie-breakers local to the anchored user's own records unless the user-facing request explicitly asks for an overall or global benchmark. "
        "Prefer user-relevant business values such as names, titles, dates, amounts, counts, statuses, or ordered records over answers that are only chains of internal *_id fields. "
        f"{DIFFICULTY_AXIS_GUIDANCE} "
        "When submit_draft returns a rejection, keep working inside the same conversation, make at least one new atomic tool call, inspect more data, and resubmit a better draft. "
        "On validation feedback, keep the same anchored user need and fix the smallest failing part first. "
        "Do not reset to a different topic, a different anchor, or a simpler scalar count unless the feedback proves the current anchored path cannot be grounded. "
        "When fixing identifier chains or unreadable labels, prefer replacing those slots with grounded business-facing values on the same anchored evidence path before collapsing the label. "
        "For self-scoped count answers, use anchor-scoped count evidence rather than a global database total. "
        "If submit_draft says the draft is too hard, keep the same anchored user need and reduce only one difficulty axis by one grounded step. Prefer shrinking a multi-item set to one item, removing one tie-breaker, or shortening one evidence hop before changing topic or anchor. "
        "A rejection is not the end of the task. Do not stop, apologize, or output a final message after a rejection. Continue until submit_draft returns Accepted or Budget exhausted. "
        "If the rejection asks you to crank a difficulty axis, strengthen only that axis first, use newly observed evidence, and prefer the smallest grounded step that makes the task harder. "
        "When submit_draft returns Accepted, stop. "
        "Do not emit markdown fences or long commentary unless you are finishing after Accepted or Budget exhausted."
    )


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
