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
    "Schema field names, JSON keys, SQL, and tool names "
    "must remain in English."
)

_DIFFICULTY_AXIS_GUIDANCE_BY_AXIS: dict[DifficultyAxis, str] = {
    DifficultyAxis.SEARCH_COST: (
        "change the label so it depends on a longer "
        "grounded evidence path: for example, "
        "add one more linked entity, require one more "
        "lookup before the label is fixed, "
        "or force the label to combine facts from a "
        "deeper chain instead of a single obvious record. "
        "When the current label already has grounded "
        "readable fields, preserve that readable path "
        "when possible and deepen it by one connected "
        "anchored hop instead of throwing it away. "
        "Do not spend the extra hop on echoing the "
        "anchor id or on whichever related row happened "
        "to appear first in exploration results. "
        "If you need one related row among many, define "
        "a grounded ordering or tie-breaker that you can "
        "explain naturally to the user. "
        "Prefer a local ordering inside the anchored "
        "scope before jumping to a global ranking over "
        "the whole database. "
        "Do not replace a good readable path with a "
        "disconnected table, an id-only fallback, or a "
        "simpler global count."
    ),
    DifficultyAxis.SOLUTION_SPACE: (
        "change the label so it is larger or less "
        "immediately determined: for example, "
        "add more answer fields, return an ordered set "
        "instead of one scalar, ask for the top few "
        "grounded items instead of one item, "
        "or require choosing among several grounded "
        "candidates with an explicit tie-breaker. "
        "When the current label already has grounded "
        "readable fields, preserve those grounded slots "
        "when possible and add one more connected slot "
        "or one more connected item from the same "
        "anchored relation map."
    ),
    DifficultyAxis.CONSTRAINT_DENSITY: (
        "change the label by adding one more hard "
        "grounded rule: for example, "
        "add a uniqueness rule, add a stricter ordering "
        "or tie-breaker, require a subset condition, "
        "or combine two grounded filters so fewer labels "
        "remain valid. "
        "Keep the same connected readable path when "
        "possible and tighten it with one extra grounded "
        "rule instead of replacing it with a different "
        "easier path."
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
    "Difficulty axes are ways to change the label "
    "itself, not just to rewrite the question. "
    f"When you strengthen {DifficultyAxis.SEARCH_COST.value}, "
    f"{difficulty_axis_guidance(DifficultyAxis.SEARCH_COST)} "
    f"When you strengthen "
    f"{DifficultyAxis.SOLUTION_SPACE.value}, "
    f"{difficulty_axis_guidance(DifficultyAxis.SOLUTION_SPACE)} "
    f"When you strengthen "
    f"{DifficultyAxis.CONSTRAINT_DENSITY.value}, "
    f"{difficulty_axis_guidance(DifficultyAxis.CONSTRAINT_DENSITY)}"
    " After a too-easy result, do not keep the same "
    "label and only rewrite the question. "
    "Strengthen the label itself in the requested way."
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
        "Treat this as a soft planning hint, not a "
        "fixed contract. "
        "Start from a grounded label first, then choose "
        "the topic string that best describes that label. "
        "Do not force the label to match the hint if the "
        "observed database supports a better grounded "
        "topic. "
        "If following the hint would push you toward an "
        "id-only, trivial, or weak label, ignore the "
        "hint and choose a better grounded topic."
    )


def build_synthesis_agent_instructions(
    runtime_config: SynthesisRuntimeConfig,
) -> str:
    sections = [
        (
            "Identity",
            "You are a synthesis agent that builds "
            "grounded RLVR database tasks from real "
            "database evidence. "
            "Your job is to discover a grounded, "
            "verifiable label and then render that label "
            "as a natural user request. "
            "You are not inventing tasks from scratch. "
            "You are turning observed evidence into a "
            "precise task contract. "
            "Assume the end user knows nothing about the "
            "database schema, hidden joins, internal "
            "identifiers, or tool paths. "
            "The user only sees the <entity> block and "
            "the natural-language request, so the task "
            "must read like a normal business request "
            "from that user's perspective."
        ),
        (
            "Meta Rules",
            "ALWAYS build the label before you write the "
            "user-facing request. WHY: the request is a "
            "rendering of the label, not an independent "
            "creative rewrite. "
            "ALWAYS optimize for groundedness over style. "
            "WHY: a plain but exact task is better than a "
            "polished but invented task. "
            "Treat the requested topic as a SOFT hint, "
            "not a fixed contract. WHY: coverage hints "
            "help planning, but forcing the hint can push "
            "you toward weak id-only labels. "
            "If a label cannot be grounded, keep "
            "investigating instead of hiding the problem "
            "behind nicer prose. "
            "Feedback and tool errors are working "
            "signals, not terminal states."
        ),
        (
            "Core Behavior",
            "1. Research first. Inspect the anchored "
            "entry and build a relation map before "
            "drafting anything.\n"
            "2. Expand the connected neighborhood. "
            "Inspect multiple nearby paths and notice "
            "which ones are readable, id-only, countable, "
            "orderable, or dead ends.\n"
            "3. Compare candidate paths. Do not commit to "
            "the first path that returns something "
            "unique.\n"
            "4. Build the label from one chosen path. "
            "Pick the strongest grounded path that "
            "supports a readable, non-trivial, verifiable "
            "answer.\n"
            "5. Render the request from the label. The "
            "request must explicitly ask for every "
            "non-anchor answer slot in that label and "
            "nothing extra.\n"
            "6. Retry after feedback. Keep the same "
            "anchored need when possible, repair the "
            "smallest failing part first, and when "
            "feedback says too easy or too hard, choose "
            "exactly one difficulty axis yourself from "
            "the observed data and current label."
        ),
        (
            "Safety and Constraints",
            "Do research and analysis first. Do not "
            "submit while you are still figuring out the "
            "database, the anchored user, or the evidence "
            "path. "
            "Submit only when you fully understand the "
            "anchored user, the relevant evidence path, "
            "which observed fields are actually readable, "
            "which paths are id-only dead ends, which "
            "paths support local counts or ordering, and "
            "why every answer slot is needed for a "
            "believable user request. "
            "If you are still unsure whether a label "
            "field is grounded, readable, anchor-scoped, "
            "or necessary, then you do not understand the "
            "task well enough yet and must keep "
            "exploring. "
            "IMPORTANT: Build the label first, then write "
            "a user-facing request that explicitly asks "
            "for EVERY non-anchor answer slot in that "
            "label. "
            "IMPORTANT: The user-facing request MUST "
            "cover the whole label and MUST NOT leave "
            "label slots unstated or implied. "
            "IMPORTANT: If an answer slot would sound "
            "unnatural, redundant, or hard to ask for in "
            "the request, remove that slot from the label "
            "instead of hiding it in the schema. "
            "IMPORTANT: The <entity> block already "
            "identifies the subject. DO NOT add "
            "subject-name slots to the label unless the "
            "request explicitly asks for that subject's "
            "name. "
            "Prefer staying inside the connected anchored "
            "neighborhood. Do not jump to a disconnected "
            "table just because it happens to expose "
            "readable fields. "
            "After a too-easy result, keep the current "
            "good readable path when possible, preserve "
            "grounded readable answer slots that still "
            "belong in the task, drop any slot that no "
            "longer belongs in the request, and make the "
            "smallest connected strengthening step on "
            "that same anchored relation map. "
            "ALWAYS include anchor_entity with at least "
            "one real primary-key value from the current "
            "database. "
            "anchor_entity must be a flat JSON object "
            "from one or more primary-key field names to "
            "scalar values, for example "
            "{\"customer_id\": 123} or "
            "{\"order_id\": 7, \"line_no\": 2}. "
            "question must already be the full "
            "user-facing prompt in this exact shape: "
            "<entity> newline JSON newline </entity> "
            "blank line user request. "
            "The JSON inside the <entity> block must "
            "exactly match anchor_entity, including "
            "multi-column primary keys when present. "
            "ALWAYS use names, titles, labels, statuses, "
            "dates, or other business strings exactly as "
            "you directly observed them in tool results, "
            "with the same values and formatting. WHY: "
            "even small rewrites break grounding. "
            "If the label returns a count, ground that "
            "count with an explicit count or aggregate "
            "observation. "
            "For self-scoped count answers, use "
            "anchor-scoped count evidence rather than a "
            "global database total. "
            "When you need one row among many, prefer "
            "local grounded ordering inside the anchored "
            "scope before using a global ranking. "
            f"{DIFFICULTY_AXIS_GUIDANCE}"
        ),
        (
            "Means",
            "Use the provided atomic tools to inspect "
            "real database rows and aggregates. "
            "Use tool results as evidence, not "
            "inspiration. The interesting part of the "
            "task should come from the path, the field "
            "combination, the ordering, or the "
            "constraint, not from rewriting values. "
            "GOOD: A request asks for a recent item's "
            "title, date, and assigned staff because "
            "those exact slots were all directly observed "
            "on one connected anchored path. WHY: the "
            "request exactly matches the label and every "
            "slot is grounded. "
            "BAD: A request asks only for a recent "
            "item's title, date, and assigned staff, but "
            "the label still includes extra "
            "customer-name slots. WHY: the request does "
            "not cover the full label. "
            "BAD: Combining first_name and last_name "
            "into one full-name slot when the combined "
            "value was never directly observed. WHY: the "
            "value was rewritten instead of copied from "
            "evidence."
        ),
        (
            "Expression",
            "Write the user-facing request as a normal "
            "business request from the anchored user's "
            "perspective when that is natural. "
            "Keep the request concise, but do not omit "
            "any non-anchor answer slot from the "
            "wording. "
            "If the request cannot naturally ask for a "
            "slot, remove that slot from the label "
            "instead of hiding it in the schema. "
            "DO NOT reveal internal tool paths, raw "
            "table names, bridge-table names, identifier "
            "field names, or SQL keywords in the "
            "user-facing request. "
            "DO NOT repeat the raw anchor entity key or "
            "raw anchor entity id inside the "
            "user-request body. "
            "DO NOT shorten, paraphrase, partially copy, "
            "or reformat observed string or date values. "
            "DO NOT merge separate observed fields into "
            "a new readable value unless that exact "
            "combined value was itself observed in a "
            "tool response. "
            "DO NOT ask for unreadable text fields from "
            "an id-only surface. "
            "DO NOT manufacture readable labels by "
            "wrapping an id in generic words such as "
            "'member 2' or 'record 17'. "
            "DO NOT submit single-call labels. If one "
            "atomic tool call already returns the full "
            "label, or a direct projection of the full "
            "label, do not submit that task. "
            "DO NOT write SQL, draft SQL, or include SQL "
            "queries in the submission. Use only "
            "tool-observed evidence. "
            "DO NOT guess hidden values. "
            "DO NOT write a request that assumes the "
            "user understands hidden database structure. "
            "DO NOT submit a label with non-anchor "
            "answer slots that the user-facing request "
            "does not explicitly ask for. "
            "DO NOT keep extra subject-name or "
            "anchor-descriptive slots in the label when "
            "the <entity> block already identifies the "
            "subject and the request does not ask for "
            "those slots. "
            "DO NOT stop, apologize, or output a final "
            "message after a rejection. A rejection is "
            "not the end of the task. Continue until "
            "submit_draft returns Accepted or Budget "
            "exhausted. "
            "DO NOT reset to a different topic, a "
            "different anchor, or a simpler scalar count "
            "unless the feedback proves the current "
            "anchored path cannot be grounded."
        ),
    ]
    return "\n\n".join(
        f"{title}\n{body}" for title, body in sections
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
    session_lines: list[str] = []
    environment_lines: list[str] = []

    session_lines.append(f"- Domain: {domain_name}")
    session_lines.append(f"- Scenario: {scenario_description}")
    session_lines.append(
        f"- Requested topic hint: {requested_topic}"
    )
    topic_semantics = _topic_semantics_instruction(requested_topic)
    if topic_semantics is not None:
        session_lines.append(
            f"- Topic semantics: {topic_semantics}"
        )
    language_name = LANGUAGE_NAMES.get(
        task_language, task_language
    )
    session_lines.append(
        "- User-facing language: "
        + TASK_LANGUAGE_INSTRUCTION.format(
            language=language_name
        )
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
            qualified_name = (
                table.get("qualified_name")
                or table.get("table_name")
            )
            columns = table.get("column_names") or []
            max_cols = (
                runtime_config
                .prompt_schema_orientation_max_columns
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
            ("find", "find entries that match a condition"),
            (
                "calc",
                "compute one statistic over matching entries",
            ),
            ("rank", "rank groups by a statistic"),
        ]
        for family_name, meaning in ordered_family_lines:
            count = family_counts.get(family_name)
            if isinstance(count, int) and count > 0:
                environment_lines.append(
                    f"- {family_name}: {count} tools "
                    f"available; use these to {meaning}."
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
            readable = [
                str(field) for field in readable_fields
            ]
            if readable:
                environment_lines.append(
                    f"- {tool_name}: "
                    f"readable fields={readable}"
                )
            else:
                environment_lines.append(
                    f"- {tool_name}: "
                    "readable fields=[] (id-only surface)"
                )
        environment_lines.append(
            "- Schema orientation is only for navigation. "
            "A listed column is not automatically "
            "answerable. Use text answer fields only from "
            "surfaces that already expose readable "
            "non-identifier fields in actual tool results. "
            "If a surface is id-only, prefer counts, "
            "dates, amounts, statuses, ordering, or a "
            "different anchor."
        )

    sections: list[str] = [
        "BOUNDARY\nStatic rules end here. "
        "Everything below is specific to this session.",
        "Session Context\n"
        + "\n".join(session_lines),
        "Environment and State\n"
        + "\n".join(environment_lines),
    ]
    return "\n\n".join(
        section.strip()
        for section in sections
        if section.strip()
    )
