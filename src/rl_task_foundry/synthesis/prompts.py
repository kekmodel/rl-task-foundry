"""Prompt builders for the single-agent synthesis loop."""

from __future__ import annotations

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


def build_synthesis_agent_instructions() -> str:
    return (
        "You are a synthesis agent that builds grounded RLVR database tasks. "
        "You may use the provided atomic function tools to inspect real database rows and aggregates. "
        "The requested topic is fixed and must not be changed. "
        "You may change the anchor entity when you need a better grounded task. "
        "Before every submit_draft call, observe real data with atomic tools and verify the canonical answer from those observations. "
        "Every draft must include anchor_entity with at least one real primary-key value from the current database, for example {\"customer_id\": 148} or {\"store_id\": 1}. "
        "When you call submit_draft, include all required arguments: canonical_answer_json, anchor_entity, difficulty_vector, question, constraint_summary, instance_space, and label_summary. "
        "Do not guess hidden values. "
        "Avoid trivial tasks such as returning a single foreign key or a raw count when richer grounded tasks are possible. "
        "Prefer multi-field answers, ordered answer sets, explicit tie-breakers, or tasks that require combining multiple grounded facts. "
        "Single-call labels are forbidden. If one atomic tool call already returns the full label, or a direct projection of the full label, do not submit that task. "
        "Keep exploring until the label requires combining at least two distinct grounded observations. "
        "Do not reveal internal tool paths in the user-facing question. "
        "Do not repeat raw identifier field names such as customer_id, store_id, or address_id in the user-facing question; keep those only inside anchor_entity. "
        "Do not mention raw table names, bridge-table names, or SQL keywords such as JOIN, LIMIT, SELECT, or ORDER BY in the user-facing question. "
        "Prefer user-relevant business values such as names, titles, dates, amounts, counts, statuses, or ordered records over answers that are only chains of internal *_id fields. "
        "When submit_draft returns a rejection, keep working inside the same conversation, make at least one new atomic tool call, inspect more data, and resubmit a better draft. "
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
) -> str:
    sections: list[str] = []
    sections.append("# Domain\n" f"{domain_name}: {scenario_description}")
    sections.append(f"# Requested Topic\n{requested_topic}")

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

    language_name = LANGUAGE_NAMES.get(task_language, task_language)
    sections.append(
        "# User-Facing Language\n" + TASK_LANGUAGE_INSTRUCTION.format(language=language_name)
    )
    return "\n\n".join(section.strip() for section in sections if section.strip())
