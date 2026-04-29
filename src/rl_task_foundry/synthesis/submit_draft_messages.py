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


def _format_missing_request_phrase_guidance(
    diagnostics: dict[str, object] | None,
) -> str:
    if diagnostics is None:
        return ""
    binding_diagnostics = diagnostics.get("answer_contract_binding_diagnostics")
    if not isinstance(binding_diagnostics, dict):
        return ""
    raw_bindings = binding_diagnostics.get("missing_requested_by_phrase_bindings")
    if not isinstance(raw_bindings, list):
        return ""
    details: list[str] = []
    for item in raw_bindings[:3]:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        label_field = item.get("label_field")
        phrase = item.get("requested_by_phrase")
        if not isinstance(path, str) or not isinstance(phrase, str):
            continue
        if isinstance(label_field, str) and label_field:
            details.append(f"{path} label_field={label_field!r} phrase={phrase!r}")
        else:
            details.append(f"{path} phrase={phrase!r}")
    if not details:
        return ""
    return (
        " Missing request phrases: "
        + "; ".join(details)
        + ". Keep those label fields only if they still form a fluent user "
        "request; otherwise rerun with a cleaner field set or choose another "
        "label."
    )


def _format_missing_contract_phrase_guidance(
    diagnostics: dict[str, object] | None,
) -> str:
    if diagnostics is None:
        return ""
    raw_details = diagnostics.get("answer_contract_missing_phrase_details")
    details: list[str] = []
    if isinstance(raw_details, list):
        for item in raw_details[:3]:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            phrase = item.get("phrase")
            if isinstance(path, str) and isinstance(phrase, str):
                details.append(f"{path}={phrase!r}")
    if not details:
        raw_paths = diagnostics.get("answer_contract_missing_phrases")
        if isinstance(raw_paths, list):
            details = [str(path) for path in raw_paths[:3] if isinstance(path, str)]
    if not details:
        return ""
    return (
        " Missing contract phrases: "
        + "; ".join(details)
        + ". Label Contract reminder: each contract phrase must be copied as "
        "one contiguous substring from user_request; if a natural phrase was "
        "split by limit or order words, use a contiguous target substring or "
        "rewrite the request and contract together."
    )


def _format_missing_order_label_binding_guidance(
    diagnostics: dict[str, object] | None,
) -> str:
    if diagnostics is None:
        return ""
    binding_diagnostics = diagnostics.get("answer_contract_binding_diagnostics")
    if not isinstance(binding_diagnostics, dict):
        return ""
    raw_fields = binding_diagnostics.get("missing_order_label_bindings")
    if not isinstance(raw_fields, list):
        return ""
    fields = [repr(field) for field in raw_fields[:3] if isinstance(field, str)]
    if not fields:
        return ""
    return (
        " Missing returned order label bindings: "
        + ", ".join(fields)
        + ". Tool schema reminder: when a query.order_by key is returned in "
        "label_json, the matching answer_contract.order_bindings item must "
        "set label_field to that returned label field; use null only for an "
        "ordered key that is not returned."
    )


def _format_incremental_error_guidance(
    diagnostics: dict[str, object] | None,
) -> str:
    if diagnostics is None:
        return ""
    raw_errors = diagnostics.get("answer_contract_incremental_errors")
    if not isinstance(raw_errors, list):
        return ""
    errors = [str(error) for error in raw_errors[:5] if isinstance(error, str)]
    if not errors:
        return ""
    guidance = " Incremental diagnostics: " + ", ".join(errors) + "."
    if any(
        error in {"list_output_only", "no_new_structural_constraint"}
        for error in errors
    ):
        guidance += (
            " Difficulty-Up Policy reminder: same-row output additions only "
            "widen the displayed answer; they do not add lookup, comparison, "
            "group/aggregate, visible-order, or related-row reasoning."
        )
    if any(
        error in {"operation_changed", "list_row_filter_added", "cardinality_weakened"}
        for error in errors
    ):
        guidance += (
            " Restore the last solver-evaluated row set, order, limit, and "
            "output sources before adding a strengthening dimension; remove "
            "target switches, added row filters, or cardinality changes."
        )
    return guidance


def _format_duplicate_output_binding_guidance(
    diagnostics: dict[str, object] | None,
) -> str:
    if diagnostics is None:
        return ""
    binding_diagnostics = diagnostics.get("answer_contract_binding_diagnostics")
    if not isinstance(binding_diagnostics, dict):
        return ""
    raw_duplicates = binding_diagnostics.get("duplicate_output_binding_phrases")
    if not isinstance(raw_duplicates, list):
        return ""
    details: list[str] = []
    for item in raw_duplicates[:3]:
        if not isinstance(item, dict):
            continue
        phrase = item.get("requested_by_phrase")
        raw_fields = item.get("label_fields")
        if not isinstance(phrase, str) or not isinstance(raw_fields, list):
            continue
        fields = [str(field) for field in raw_fields if isinstance(field, str)]
        if fields:
            details.append(f"phrase={phrase!r} fields={fields!r}")
    if not details:
        return ""
    return (
        " Duplicate output binding phrases: "
        + "; ".join(details)
        + ". Label Contract reminder: one natural output slot should not be "
        "split across multiple label fields. Rerun the same target with one "
        "field per requested slot, or rewrite the request with distinct fluent "
        "phrases when the fields are genuinely separate."
    )


def _too_easy_retry_guidance(*, answer_kind: str | None = None) -> str:
    kind_note = ""
    if answer_kind in {"scalar", "list"}:
        kind_note = f" Current answer kind: {answer_kind}."
    return (
        " This draft failed specificity."
        f"{kind_note} Policy reminder: Difficulty-Up Policy is the repair "
        "source for specificity feedback on the current draft. Preserve the "
        "current anchor and target; for list labels preserve the evaluated row "
        "set, order, limit, source meanings, and existing output field request "
        "phrases. Do not add a narrowing row filter or lower the row count as a "
        "difficulty-up move. Add one grounded meaningful dimension with "
        "structural effect, supported by new evidence that changes lookup, "
        "comparison, group/aggregate, visible ordering, or related-row "
        "reasoning while keeping those rows. Do not only add display "
        "fields for the same selected row; same-row display/derived fields alone "
        "are still too direct. If that was just tried, switch answer work with "
        "aggregate, comparison, grouping, visible ordering, or related-row "
        "selection instead of adding another field. Related-row strengthening "
        "must be visible or aggregated; do not use hidden existence or "
        "primary-row filters as the added dimension. "
        "Request Contract reminder: keep user_request fluent and copy visible "
        "context/source values exactly; do not translate/transliterate them "
        "while strengthening. Do not switch topic or table family."
    )
