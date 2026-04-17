"""Message-formatting helpers for submit_draft feedback.

Pure functions that compose the structured Rejected/Feedback/Accepted strings
the synthesis agent reads back as tool output. None of these depend on the
controller — they accept the bits they need (diagnostics dict, attempt counts,
already-built error/important/next_step strings) and return strings.
"""

from __future__ import annotations

import re

_DATETIME_LITERAL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2}$")


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
    message = f" Ungrounded values included: {preview}."
    name_like_values = [
        value
        for value in values
        if " " in value
        and any(character.isalpha() for character in value)
        and not any(character.isdigit() for character in value)
    ]
    datetime_like_values = [value for value in values if _DATETIME_LITERAL_RE.fullmatch(value)]
    if name_like_values:
        message += " If the tool response exposed first_name and last_name separately, keep them as separate answer fields instead of merging them into one full-name string."  # noqa: E501
    if datetime_like_values:
        message += " If you use a date or timestamp field, copy the exact raw value from the chosen tool response row without changing its formatting."  # noqa: E501
    return message


def _too_easy_retry_guidance() -> str:
    return (
        " Keep the same entity — changing it is "
        "rejected because the requesting user has not "
        "changed. Extend the existing evidence path instead."
        " Pick ONE structural change (prefer earlier options): "
        "(a) follow one more FK hop to reach a new entity, "
        "(b) add a filter condition (date range, status, "
        "amount threshold), or "
        "(c) return a list of records instead of one. "
        "Do not just add or remove a single field on the "
        "same path — that rarely shifts difficulty enough."
    )
