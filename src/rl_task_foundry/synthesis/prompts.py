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
        "inspects the database via one of the composer tools or "
        "commits a candidate task via submit_draft. Multiple "
        "independent solvers later attempt your task using a "
        "separate 9-primitive atomic calculus; their exact-match "
        "agreement rate decides acceptance.",

        # ── Commit Rule ──
        build_tool_call_budget_instruction(
            max_tool_calls=runtime_config.max_turns,
        ),

        # ── Never ──
        "# Never\n"
        "- Never write SQL. Use the composer DSL (`query`) instead.\n"
        "- Never weaken a label on rejection. Rejection is always "
        "caused by the request, not the label.\n"
        "- Never concatenate or reformat observed strings.\n"
        "- Never use internal identifiers (*_id) as answer fields.\n"
        "- Never apply more than one escalation per submit.",

        # ── Composer Tools ──
        "# Composer Tools\n"
        "You have five high-bandwidth tools. Each returns JSON you "
        "can keep reading directly.\n"
        "\n"
        "- `schema_map(root_table?, depth?)` — tables, columns, "
        "FK edges, plus hub/bridge tags. One call to orient.\n"
        "- `neighborhood(table, row_id, max_per_edge?)` — the "
        "anchor row's attributes plus per-edge sample IDs and "
        "counts. Use right after sampling an anchor.\n"
        "- `profile(table, column?, predicate?)` — row_count plus "
        "distinct/null counts; with `column` set, adds min/max and "
        "a top-k frequency list. Drives filter calibration.\n"
        "- `sample(table, n, seed?, predicate?)` — up to `n` "
        "representative rows. Seed makes the draw reproducible.\n"
        "- `query(spec)` — JSON DSL over filter + FK join chain + "
        "select or group_by+aggregate, plus sort/limit. One call "
        "computes any canonical answer an atomic chain could "
        "derive. Use it to produce the label.",

        # ── Solver Context ──
        "# Solver Context\n"
        "Solvers re-derive your canonical answer through nine "
        "atomic primitives on a schema-parameterized calculus:\n"
        "\n"
        "- Set-producing: `rows_where(table, column, op, value)` "
        "(ops: eq/in/lt/gt/lte/gte/like; no `any`), "
        "`rows_via(cursor, edge_label)`, "
        "`intersect(left, right)`.\n"
        "- Set-annotating: `order_by(cursor, column, direction)`.\n"
        "- Set-materializing: `take(cursor, n)` with `2 ≤ n ≤ 5`, "
        "`count(cursor)`, `aggregate(cursor, fn, column)` "
        "(sum/avg/min/max), `group_top(cursor, group_column, fn, "
        "n)` with `2 ≤ n ≤ 5`.\n"
        "- Row-reading: `read(table, row_id, columns)`.\n"
        "\n"
        "Design labels that force chaining. A grounded answer "
        "reachable by one primitive (e.g. a single `read` or a "
        "bare `take` on a base table) is solver-trivial and will "
        "lock pass_rate at 1.0. Aim for at least three primitives "
        "of required composition.",

        # ── Workflow ──
        "# Workflow\n"
        "Your ONLY immediate target is the minimum viable draft. "
        "Ignore later escalations until you receive a rejection.\n"
        "\n"
        "1. Call `schema_map` once if you are not already oriented; "
        "then `neighborhood(table, anchor_row_id)` to see the "
        "anchor's FK fan-out.\n"
        "2. Author the canonical answer with a single "
        "`query(spec)` call. Pick ONE of two task types — Type B "
        "and Type A are equal-weight options, not a default plus "
        "a fallback.\n"
        "\n"
        "**Selection hint.** Look at the anchor and ask what the "
        "most natural customer phrasing would be. If it is \"how "
        "many X do I have / does this have\", \"what was my "
        "earliest/latest Y\", or \"what is the largest/smallest "
        "Z\" — pick **Type B**. If it is \"show me my first 3 X\", "
        "\"list the records of Y\", \"give me the top-N by Z\" — "
        "pick **Type A**. Hub-like anchors (customer, film, "
        "category, country, staff) afford both phrasings; do not "
        "default to list every time. Across a batch of anchors, "
        "aim for roughly half Type B and half Type A.\n"
        "\n"
        "**Type B — single scalar over an FK-joined set.** One "
        "aggregate value computed from the anchor's child set. "
        "The label is a one-key dict holding that value; the "
        "question asks for the single number (or date) directly. "
        "Use `query(spec)` with `aggregate` and no `group_by` to "
        "produce a scalar row. Prefer `count`, `min`, `max` — "
        "they exact-match on integers and dates without float "
        "risk. `sum`/`avg` are valid but only on monetary or "
        "duration columns where both sides see the same Postgres "
        "numeric. Examples:\n"
        "  - anchor=customer → rental count: "
        "`{rental_count: 27}` (via `customer → rental`).\n"
        "  - anchor=film → actor count: `{actor_count: 5}` "
        "(via `film → film_actor → actor`, `count(*)`).\n"
        "  - anchor=category → film count: `{film_count: 64}` "
        "(via `category → film_category → film`).\n"
        "  - anchor=staff → earliest payment date: "
        "`{first_payment_date: '2005-05-24'}` (via `staff → "
        "payment`, `min(payment_date)`).\n"
        "  - anchor=customer → largest single payment: "
        "`{max_payment: 11.99}` (via `customer → payment`, "
        "`max(amount)`).\n"
        "  - anchor=country → customer count: "
        "`{customer_count: 5}` (via `country → city → address → "
        "customer`, `count(*)`).\n"
        "  - anchor=address → rental count at that address: "
        "`{rental_count: 12}` (via `address → customer → "
        "rental`).\n"
        "Type B forces the solver to chain `rows_where` → "
        "`rows_via` → `count`/`aggregate` (three primitives), and "
        "adding a child-side filter (`rows_where` on the joined "
        "set) gives four. It is NOT weaker than Type A — it is a "
        "different task type at step 2, not an escalation. The "
        "Escalation Axes below still apply to Type B if rejected "
        "as too_easy (add a Filter on the joined set, or a "
        "Composite pair of filters).\n"
        "\n"
        "**Type A — list of N child records (N ∈ {3, 5, 10, or "
        "'all records matching <filter>'}).** A homogeneous list "
        "from a single destination table, reached by the shortest "
        "FK chain from the anchor (single hop for leaf anchors "
        "like customer/rental/film; the natural chain for hub "
        "anchors like city/category). Every record's 1-2 keys "
        "must all live on that ONE destination table — never mix "
        "columns from two tables inside one record. Sort by one "
        "observed field on that same destination table in a fixed "
        "direction. Pick keys that match the destination's "
        "natural readable surface. Do NOT default to `{rental_"
        "date, return_date}` regardless of anchor; that is just "
        "one shape out of many. **N is a draft parameter — vary "
        "it**: N=3 is a common starting point but anchors with "
        "many children (customer→rental, film→actor, city→"
        "customer) often land in-band only at N=5, N=10, or '모든 "
        "…' (all matching, unbounded). Do not always default to "
        "N=3 — that choice alone has saturated the Cardinality "
        "axis at 0 across 15 prior accepts. Examples:\n"
        "  - anchor=customer → rental destination, **N=10**: "
        "`[{rental_date, return_date}, …]` × 10 (first 10 by "
        "rental_date) — '제 대여 기록 중 가장 빠른 10건'.\n"
        "  - anchor=film → actor destination (via film_actor), "
        "**N=all matching**: `[{first_name, last_name}, …]` for "
        "all actors on that film — '이 영화에 출연한 모든 배우'.\n"
        "  - anchor=city → customer destination (via address), "
        "**N=all matching**: `[{first_name, last_name}, …]` for "
        "all customers in that city — '해당 도시의 모든 고객'.\n"
        "  - anchor=category → film destination (via film_"
        "category), **N=5**: `[{title, release_year}, …]` × 5 "
        "or `[{title, rating}, …]` × 5 (top 5 by release_year "
        "desc) — '이 카테고리의 가장 최근 영화 5편'.\n"
        "  - anchor=staff → payment destination, **N=3**: "
        "`[{amount, payment_date}, …]` × 3 (a monetary+temporal "
        "pair, not name-only) — '해당 직원이 처리한 가장 최근 "
        "결제 3건'.\n"
        "Cross-table record shapes like `[{rental_date, film_"
        "title}, …]` — rental_date on rental and film_title on "
        "film in the same record — are NOT a valid Type A draft; "
        "the solver would have to join two tables per answer row "
        "and overshoot.\n"
        "3. On too_easy, add exactly ONE dimension from the "
        "escalation axes below and resubmit within 2 composer "
        "calls. **The task type picked on your first submit is "
        "locked for the whole anchor.** If attempt 1 was Type B "
        "scalar `{customer_count: 5}`, attempt 2 must still be a "
        "Type B scalar — add a filter on the joined set (e.g. "
        "`count` of customers in a specific city subset) or a "
        "Composite pair, never replace the scalar with a Type A "
        "list. If attempt 1 was Type A, stay Type A. Switching "
        "types is a weakening and will be rejected as "
        "crank_invalid.\n"
        "4. On too_hard, relax one clause (not the label) and "
        "resubmit. Never weaken the label itself. The same "
        "task-type lock applies.\n"
        "5. On accept, stop.\n"
        "\n"
        "Rewrite the user-facing request in the CUSTOMER'S OWN "
        "VOICE — first-person ask ('제 대여 기록을…') or a "
        "second-person request addressed to the organization "
        "('고객 54의 … 알려주세요'). The customer is seeking "
        "information; the solver is the organization answering. "
        "NEVER write the question from the organization's/staff's "
        "perspective — no '고객님', '귀하의', '확인해 드리겠습니다', "
        "'도와드리겠습니다', or any phrase where the speaker is "
        "serving the customer. The question is the ASK, not the "
        "response. Do not refer to the anchor row as '이 대여 "
        "기록' or similar schema-internal language; translate the "
        "anchor into a customer-natural reference (e.g. a date or "
        "the customer's own id said in their voice). "
        "**Numeric-ID filter values** (e.g. staff_id=1, film_id=473, "
        "category_id=6) must be resolved to the referent table's "
        "natural name or title via a lookup tool call (sample/read "
        "on that table) BEFORE phrasing the filter in the user "
        "voice — writing '직원 1 번', 'film 473번', '카테고리 6번' is "
        "schema-ese and fails voice check; write 'Mike Hillyer "
        "직원', 'Blade Runner', '액션 영화' instead. Every label "
        "constraint — count, sort, filter threshold — surfaces as "
        "a natural preference in the ask itself.",

        "# Escalation Axes\n"
        "On each too_easy rejection, add ONE axis the current "
        "label does not yet have. Never remove prior structure; "
        "only add. Each axis has a different effect on the answer "
        "shape — shape-changing axes drop pass_rate faster. "
        "**Cardinality is the least-observed axis (0 of 17 prior "
        "accepts) — prefer it over Cross-item when the initial "
        "draft already has ordering or when prior filter "
        "additions haven't dropped pass_rate enough.**\n"
        "\n"
        "- **Cardinality** — change N (e.g. 3 → 5, or 5 → '모든 "
        "matching records') or switch from fixed-N to \"all "
        "records matching the filter\". Changes answer size. "
        "Often unlocks band when pass_rate won't drop via "
        "filters alone.\n"
        "- **Composite** — two filters on different dimensions "
        "(categorical AND threshold). Narrows the row set.\n"
        "- **Filter** — a single categorical exclusion or "
        "threshold on an existing field. Narrows the row set, "
        "mildly.\n"
        "- **Cross-item rule** — replace the sort key with a "
        "uniqueness, ordering, or conditional rule that relates "
        "list items. Changes which records appear and in what "
        "order. Note: basic ordering (sort by date ASC, take N) "
        "is already part of the default Type A shape and no "
        "longer counts as Cross-item escalation — pick a true "
        "inter-item rule (uniqueness constraint, conditional "
        "between items).\n"
        "- **Width** — disallowed as an escalation. Adding more "
        "fields per record does not change the row set and does "
        "not drop pass_rate on this dataset. Never select Width "
        "after a too_easy rejection.\n"
        "\n"
        "Axes are structural. Pick the concrete field, category, "
        "or threshold from the current DB's schema and observed "
        "data distributions, never from a fixed template.",

        # ── After Rejection ──
        "# After Rejection\n"
        "A rejection is not a signal to explore more. Within 2 "
        "composer calls of rejection feedback, call submit_draft "
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
        "The label must be the only correct answer. For Type A, "
        "state the exact record count and the sort clause in the "
        "question so the list is unique (e.g. \"the first 3 "
        "rentals ordered by rental_date ascending\"); never leave "
        "the count or order implicit. For Type B, state the "
        "aggregate operation and the full filter window in the "
        "question so the single scalar is unambiguous (e.g. "
        "\"the total number of rentals I've made\", \"the date "
        "of my earliest payment\"); never leave which aggregate "
        "or which subset implicit. Solvers cannot `take` fewer "
        "than 2 or more than 5 rows in a single primitive call, "
        "so Type A fixed-N targets within `[2, 5]` are directly "
        "reachable; anything outside forces `count`/`aggregate`, "
        "which is also how every Type B answer resolves.",

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


def _render_composer_tool_lines(
    tool_surface_summary: dict[str, object],
    hint_limit: int,
) -> list[str]:
    lines: list[str] = []
    tool_count = tool_surface_summary.get("tool_count")
    if isinstance(tool_count, int):
        lines.append(f"- Composer tools: {tool_count}")
    tools = tool_surface_summary.get("tools")
    if isinstance(tools, list):
        for tool in tools[:hint_limit]:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            description = tool.get("description")
            if not isinstance(name, str):
                continue
            if isinstance(description, str) and description.strip():
                lines.append(f"- {name} — {description}")
            else:
                lines.append(f"- {name}")
    return lines


def _render_solver_primitive_lines(
    tool_surface_summary: dict[str, object],
) -> list[str]:
    primitives = tool_surface_summary.get("solver_primitives")
    if not isinstance(primitives, dict):
        return []
    labels = {
        "set_producing": "set-producing",
        "set_annotating": "set-annotating",
        "set_materializing": "set-materializing",
        "row_reading": "row-reading",
    }
    lines: list[str] = []
    for key, label in labels.items():
        names = primitives.get(key)
        if not isinstance(names, list):
            continue
        filtered = [str(name) for name in names if isinstance(name, str)]
        if not filtered:
            continue
        lines.append(f"- {label}: {', '.join(filtered)}")
    return lines


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
    environment_lines.append(
        "- Schema orientation is for navigation only. "
        "Verify readability in actual tool results before "
        "using a field in the label."
    )
    sections.append(
        "# Environment\n" + "\n".join(environment_lines)
    )

    # ── Tools Available ──
    hint_limit = runtime_config.prompt_tool_surface_hint_limit
    composer_lines = _render_composer_tool_lines(
        tool_surface_summary, hint_limit
    )
    solver_lines = _render_solver_primitive_lines(tool_surface_summary)
    tool_section_lines: list[str] = []
    if composer_lines:
        tool_section_lines.append("## Composer toolset (for authoring)")
        tool_section_lines.extend(composer_lines)
    if solver_lines:
        tool_section_lines.append(
            "\n## Solver atomic calculus (what re-derives your answer)"
        )
        tool_section_lines.extend(solver_lines)
    if tool_section_lines:
        sections.append(
            "# Tools Available\n" + "\n".join(tool_section_lines)
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
