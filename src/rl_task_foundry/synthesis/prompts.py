"""Prompt builders for the single-agent synthesis loop."""

from __future__ import annotations

import re

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


def _topic_semantics_instruction(requested_topic: str) -> str | None:
    normalized = requested_topic.strip()
    if not normalized:
        return None
    topic_phrase = re.sub(r"[_\-\s]+", " ", normalized).strip()
    if not topic_phrase:
        return None
    return (
        f"Stay semantically tight to the requested topic: {topic_phrase}. "
        "Use the plain-language meaning of that topic as the semantic center of the task. "
        "Do not turn the task into generic lineage tracing across unrelated entities. "
        "Keep the final answer centered on readable business-facing values rather than opaque internal identifiers whenever the observed tool results allow it. "
        "If multiple grounded answers could fit, add an explicit tie-breaker or ask for an ordered answer set."
    )


def build_synthesis_agent_instructions() -> str:
    return (
        "You are a synthesis agent that builds grounded RLVR database tasks. "
        "You may use the provided atomic function tools to inspect real database rows and aggregates. "
        "The requested topic is fixed and must not be changed. "
        "You may change the anchor entity when you need a better grounded task. "
        "Before every submit_draft call, observe real data with atomic tools and verify the canonical answer from those observations. "
        "Every draft must include anchor_entity with at least one real primary-key value from the current database. "
        "anchor_entity must be a flat JSON object from primary-key field name to scalar value, for example {\"<pk_name>\": 123}. Do not wrap it inside keys such as entity_type, primary_key, primary_keys, or metadata. "
        "Calling submit_draft without anchor_entity is always wrong. Choose the anchor first and keep it explicit in the payload. "
        "When you call submit_draft, include all required arguments: canonical_answer_json, anchor_entity, difficulty_vector, question, constraint_summary, instance_space, and label_summary. "
        "Do not guess hidden values. "
        "Make label_summary an English explanation that explicitly includes the requested topic phrase and explains why the label is grounded and unique. "
        "Only use names, titles, labels, statuses, or other business strings that you directly observed in tool results. Do not invent placeholders such as Unknown or Entity #1. "
        "Do not submit blank or placeholder string fields in the canonical answer. Every answer field must carry a grounded, non-empty value. "
        "Before choosing text answer fields such as names, titles, labels, or statuses, confirm that the observed tool results for the chosen surface actually expose those readable fields. "
        "If the observed surface is id-only, do not ask for unreadable text fields from it. Switch to grounded counts, dates, amounts, statuses, ordering, or choose a different anchor or related entity whose observed rows expose readable fields. "
        "Avoid trivial tasks such as returning a single foreign key or a raw count when richer grounded tasks are possible. "
        "Prefer multi-field answers, ordered answer sets, explicit tie-breakers, or tasks that require combining multiple grounded facts. "
        "Single-call labels are forbidden. If one atomic tool call already returns the full label, or a direct projection of the full label, do not submit that task. "
        "Keep exploring until the label requires combining at least two distinct grounded observations. "
        "Do not reveal internal tool paths in the user-facing question. "
        "Do not literally include the token <entity> in the user-facing question. The runtime will render the entity block separately. "
        "Do not repeat the raw anchor entity key or raw anchor entity id inside the user-facing question; the <entity> block already provides that grounding. Refer to the anchor naturally with a domain-appropriate phrase instead of the raw identifier. "
        "Do not repeat raw identifier field names such as <entity>_id in the user-facing question; keep identifiers only inside anchor_entity. "
        "Do not mention raw table names, bridge-table names, or SQL keywords such as JOIN, LIMIT, SELECT, or ORDER BY in the user-facing question. "
        "Prefer user-relevant business values such as names, titles, dates, amounts, counts, statuses, or ordered records over answers that are only chains of internal *_id fields. "
        "When submit_draft returns a rejection, keep working inside the same conversation, make at least one new atomic tool call, inspect more data, and resubmit a better draft. "
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
) -> str:
    sections: list[str] = []
    sections.append("# Domain\n" f"{domain_name}: {scenario_description}")
    sections.append(f"# Requested Topic\n{requested_topic}")
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
        for table in tables[:8]:
            if not isinstance(table, dict):
                continue
            qualified_name = table.get("qualified_name") or table.get("table_name")
            columns = table.get("column_names") or []
            schema_lines.append(f"- {qualified_name}: columns={list(columns)[:8]}")
    if schema_lines:
        sections.append("# Schema Orientation\n" + "\n".join(schema_lines))

    tool_surface_lines: list[str] = []
    surfaces = tool_surface_summary.get("entity_surfaces")
    if not isinstance(surfaces, list):
        surfaces = tool_surface_summary.get("point_lookups")
    if isinstance(surfaces, list):
        for item in surfaces[:16]:
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
            "- Use text answer fields only from surfaces that already expose readable non-identifier fields. If a surface is id-only, prefer counts, dates, amounts, statuses, ordering, or a different anchor."
        )
    if tool_surface_lines:
        sections.append("# Tool Surface Hints\n" + "\n".join(tool_surface_lines))

    language_name = LANGUAGE_NAMES.get(task_language, task_language)
    sections.append(
        "# User-Facing Language\n" + TASK_LANGUAGE_INSTRUCTION.format(language=language_name)
    )
    return "\n\n".join(section.strip() for section in sections if section.strip())
