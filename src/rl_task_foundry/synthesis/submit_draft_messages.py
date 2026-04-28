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
    kind_note = ""
    if answer_kind in {"scalar", "list"}:
        kind_note = f" Current answer kind: {answer_kind}."
    return (
        " This draft failed specificity."
        f"{kind_note} Apply the Difficulty-Up Policy from the system "
        "instructions before resubmitting. Preserve the prior query shape "
        "and answer fields; append, do not replace, any new answer field. "
        "Ask for it in user_request and answer_contract."
    )
