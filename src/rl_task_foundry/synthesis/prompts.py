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
        "change the label by one grounded hop or fact while keeping the same anchored need. "
        "Prefer one extra connected lookup, one extra verified relation, or one extra local ordering step."
    ),
    DifficultyAxis.SOLUTION_SPACE: (
        "change the label by one grounded slot or one more connected ordered item while keeping the same anchored need."
    ),
    DifficultyAxis.CONSTRAINT_DENSITY: (
        "change the label by one grounded rule, filter, or tie-breaker while keeping the same anchored need."
    ),
}


def difficulty_axis_guidance(axis: DifficultyAxis) -> str:
    return _DIFFICULTY_AXIS_GUIDANCE_BY_AXIS[axis]


def difficulty_axis_feedback(axis: DifficultyAxis) -> str:
    guidance = difficulty_axis_guidance(axis)
    return f"Strengthen the label through {axis.value}. {guidance[:1].upper()}{guidance[1:]}"


DIFFICULTY_AXIS_GUIDANCE = (
    "Difficulty changes must change the label itself, not just the wording of the question. "
    "Start from the simplest grounded multi-step label, then raise difficulty only as needed. "
    f"For {DifficultyAxis.SEARCH_COST.value}, {difficulty_axis_guidance(DifficultyAxis.SEARCH_COST)} "
    f"For {DifficultyAxis.SOLUTION_SPACE.value}, {difficulty_axis_guidance(DifficultyAxis.SOLUTION_SPACE)} "
    f"For {DifficultyAxis.CONSTRAINT_DENSITY.value}, {difficulty_axis_guidance(DifficultyAxis.CONSTRAINT_DENSITY)} "
    "After difficulty feedback, keep the same anchored need, change exactly one axis, and stop once the draft is plausibly in an acceptable difficulty band."
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
        "Use it only if the observed label naturally fits; otherwise ignore it."
    )


def build_synthesis_agent_instructions(runtime_config: SynthesisRuntimeConfig) -> str:
    del runtime_config
    preamble = (
        "You are a synthesis agent for grounded database task generation. "
        "Choose the smallest non-trivial label that is fully supported by observed evidence and likely to pass validation. "
        "Prefer an initial draft that uses a connected multi-step evidence chain, usually at least two verified hops when the schema allows it. "
        "Then render that label as a natural user request."
    )
    sections = [
        (
            "Workflow",
            "1. Explore the anchor and nearby connected paths before drafting anything.\n"
            "2. Start with the simplest grounded multi-step label you can support. Avoid direct or one-hop drafts when a grounded two-hop path is available.\n"
            "3. If a row exposes references instead of readable values, resolve that chain step by step before using downstream business fields.\n"
            "4. Build one label from one verified evidence chain.\n"
            "5. Write a request that asks for every non-anchor slot in that label.\n"
            "6. After feedback, repair the smallest failing part. If the feedback is about difficulty, keep the same anchored need, change exactly one axis, and stop once the draft reaches a plausible acceptance band."
        ),
        (
            "Rules",
            "Treat the requested topic as a soft hint. Use it only when the observed label naturally fits.\n"
            "Use only tool-observed evidence. A field is answerable only after you observed it on the chosen path.\n"
            "Do not assume that a reference value from one row identifies a different entity until an observed hop confirms it.\n"
            "If a path stays reference-only or id-only, dereference it or choose a different path.\n"
            "Prefer user-relevant business values over opaque identifiers.\n"
            "Prefer a non-trivial connected label over a trivial direct lookup when both are grounded.\n"
            "Copy readable values exactly as observed. Do not merge separate fields into a new combined value unless that exact combined value was observed.\n"
            "Ground first/latest/earliest/count claims with local ordering or aggregate evidence.\n"
            "Prefer connected local paths over disconnected global lookups.\n"
            "Do not submit a label that can be read from one atomic tool call.\n"
            "Keep anchor_entity fixed across retries. The question must start with the exact literal tag pair <entity> and </entity>, then a blank line, then the request body. Never replace that tag with <customer>, <film>, or any entity-specific variant.\n"
            "The request must ask for every non-anchor answer slot and nothing else. If a slot is unnatural to ask for, remove it from the label.\n"
            "Do not add subject-name slots when the <entity> block already identifies the subject unless the request asks for them.\n"
            "Do not leak table names, SQL, hidden joins, or raw identifiers in the request.\n"
            "Continue until Accepted or Budget exhausted."
        ),
        (
            "Difficulty",
            DIFFICULTY_AXIS_GUIDANCE,
        ),
    ]
    rendered_sections = [f"{title}\n{body}" for title, body in sections]
    return "\n\n".join([preamble, *rendered_sections])


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
    # Keep tool-derived placeholders out of the prompt. Schema-derived map
    # information is allowed because it provides global structure rather than
    # leaking a preferred tool path.
    del tool_surface_summary
    session_lines: list[str] = []
    environment_lines: list[str] = []
    introspection_lines: list[str] = []

    session_lines.append(f"- Domain: {domain_name}")
    session_lines.append(f"- Scenario: {scenario_description}")
    session_lines.append(f"- Requested topic hint: {requested_topic}")
    topic_semantics = _topic_semantics_instruction(requested_topic)
    if topic_semantics is not None:
        session_lines.append(f"- Topic semantics: {topic_semantics}")
    language_name = LANGUAGE_NAMES.get(task_language, task_language)
    session_lines.append(
        "- User-facing language: "
        + TASK_LANGUAGE_INSTRUCTION.format(language=language_name)
    )

    table_count = schema_summary.get("table_count")
    edge_count = schema_summary.get("edge_count")
    if isinstance(table_count, int):
        environment_lines.append(f"- Table count: {table_count}")
    if isinstance(edge_count, int):
        environment_lines.append(f"- Foreign-key edge count: {edge_count}")

    tables = schema_summary.get("tables")
    if isinstance(tables, list):
        for table in tables[: runtime_config.prompt_schema_orientation_max_tables]:
            if not isinstance(table, dict):
                continue
            qualified_name = table.get("qualified_name") or table.get("table_name")
            columns = table.get("column_names") or []
            table_line = (
                f"- {qualified_name}: columns={list(columns)[: runtime_config.prompt_schema_orientation_max_columns]}"
            )
            primary_key = table.get("primary_key")
            if isinstance(primary_key, list) and primary_key:
                table_line += f"; primary_key={list(primary_key)}"
            relation_bits: list[str] = []
            outbound_edges = table.get("outbound_edges")
            inbound_edges = table.get("inbound_edges")
            if isinstance(outbound_edges, list) and outbound_edges:
                relation_bits.append(f"outbound={list(outbound_edges)[:3]}")
            if isinstance(inbound_edges, list) and inbound_edges:
                relation_bits.append(f"inbound={list(inbound_edges)[:3]}")
            if relation_bits:
                table_line += "; " + "; ".join(relation_bits)
            environment_lines.append(table_line)

    environment_lines.append(
        "- Schema map is for orientation only. Verify reachable evidence with tool calls before using downstream business fields."
    )

    introspection_rules = schema_summary.get("introspection_rules")
    if isinstance(introspection_rules, list):
        for rule in introspection_rules:
            if isinstance(rule, str) and rule.strip():
                introspection_lines.append(f"- {rule.strip()}")

    sections: list[str] = [
        "# Session Context\n" + "\n".join(session_lines),
        "# Environment and State\n" + "\n".join(environment_lines),
    ]
    if introspection_lines:
        sections.append("# Introspection Rules\n" + "\n".join(introspection_lines))
    return "\n\n".join(section.strip() for section in sections if section.strip())
