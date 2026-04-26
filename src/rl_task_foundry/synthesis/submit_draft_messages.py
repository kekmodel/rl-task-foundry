"""Message-formatting helpers for submit_draft feedback.

Pure functions that compose the structured Rejected/Feedback/Accepted strings
the draft author reads back as tool output. None of these depend on the
controller — they accept the bits they need (diagnostics dict, attempt counts,
already-built error/important/next_step strings) and return strings.
"""

from __future__ import annotations


def _strip_message_status_prefix(text: str) -> str:
    for prefix in (
        "Rejected. ",
        "Feedback. ",
        "Accepted. ",
        "STATUS: REJECTED. PRIMARY ISSUE: ",
        "STATUS: FEEDBACK. PRIMARY ISSUE: ",
        "STATUS: ACCEPTED. RESULT: ",
        "RejectedError: ",
        "FeedbackError: ",
        "Accepted: ",
        "ToolError: ",
        "BudgetExhaustedError: ",
    ):
        if text.startswith(prefix):
            return text.removeprefix(prefix).strip()
    return text.strip()


def _render_structured_message(
    *,
    kind: str,
    primary: str | None = None,
    result: str | None = None,
    important: str | None = None,
    also_fix: list[str] | None = None,
    next_step: str | None = None,
    attempts_left: int | None = None,
) -> str:
    headline = _strip_message_status_prefix(result or primary or "")
    parts = [f"{kind}: {headline}"]
    if result:
        if primary:
            parts.append(f"Primary issue: {_strip_message_status_prefix(primary)}")
    elif primary:
        # headline already used the primary body
        pass
    if important:
        parts.append(f"Important: {important.strip()}")
    if also_fix:
        cleaned = [_strip_message_status_prefix(message) for message in also_fix if message.strip()]
        if cleaned:
            parts.append(f"Also fix: {' '.join(cleaned)}")
    if next_step:
        parts.append(f"Next step: {next_step.strip()}")
    if attempts_left is not None:
        parts.append(f"Attempts left: {attempts_left}.")
    return " ".join(parts)


def _format_ungrounded_value_guidance(diagnostics: dict[str, object] | None) -> str:
    if diagnostics is None:
        return ""
    raw_values = diagnostics.get("ungrounded_strings")
    if not isinstance(raw_values, list):
        return ""
    values = [str(value) for value in raw_values if isinstance(value, str)]
    if not values:
        return ""
    preview = ", ".join(repr(value) for value in values[:3])
    return f" Ungrounded values included: {preview}."


def _too_easy_retry_guidance(*, answer_kind: str | None = None) -> str:
    common = (
        " Keep the anchor entity locked; changing it is rejected. "
        "Use the current DB evidence to choose a feasible structural "
        "strengthening; there is no fixed ladder. Preserve existing "
        "grounded structure unless it was overconstrained. Replacing "
        "a field on the same path is not an escalation and will be "
        "rejected. Keep answer_contract.kind and the latest query output "
        "target fixed. "
    )
    if answer_kind == "scalar":
        return (
            common
            + "This is a scalar answer, so do not switch to a list, "
            "Cardinality, or Cross-item rule. Add a new grounded "
            "filter when the DB exposes a feasible visible dimension. "
            "Every added constraint phrase must appear verbatim in "
            "user_request and be represented in answer_contract."
        )
    if answer_kind == "list":
        return (
            common
            + "This is a list answer, so keep a selected-row query target. "
            "Pick a valid list strengthening supported by the current "
            "DB evidence: "
            "(a) Filter — add a categorical exclusion or threshold "
            "on the destination or join path, "
            "(b) Composite — a second filter on a different "
            "dimension than any existing filter, "
            "(c) Cardinality — grow the same repeated list "
            "structure one natural step (for example 1→2→3→5) "
            "or eventually switch to all matching items, "
            "(d) Item-complexity — keep the item count but add one "
            "grounded per-item requirement, related visible field, "
            "predicate, or deterministic tie-break from the same "
            "query path, "
            "(e) Order — add a secondary deterministic order_by "
            "without removing existing order clauses. Every added "
            "predicate, sort, cardinality, or item-complexity phrase "
            "must appear verbatim in user_request and be represented "
            "in answer_contract. Do not add passive display-only "
            "fields as difficulty."
        )
    return (
        common
        + "Pick a valid strengthening supported by current DB evidence: "
        "(a) Filter — add a categorical exclusion or threshold, "
        "preferably on a different table or business dimension for "
        "scalar counts, "
        "(b) Cardinality — return or grow a list of N items "
        "where N is declared in user_request and increases the same "
        "repeated structure gradually, "
        "(c) Item-complexity — make each item require one additional "
        "grounded condition, related visible field, or deterministic "
        "tie-break from the same query path, "
        "(d) Cross-item rule — uniqueness, ordering, or a "
        "conditional relating list items (requires an existing "
        "list), "
        "(e) Composite — a second filter on a different "
        "dimension than any existing filter. Every added predicate, "
        "sort, cardinality, or item-complexity phrase must appear "
        "verbatim in user_request and be represented in answer_contract."
    )
