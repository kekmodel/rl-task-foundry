"""Pure validation helpers for submit_draft.

Stateless functions that walk canonical answers and atomic tool-call traces.
None of these depend on the controller or on ``SubmitDraftErrorCode`` — they
return primitives (``bool``, ``set[str]``, ``list[str]``) that the controller
turns into error codes and feedback messages.
"""

from __future__ import annotations

import re
from datetime import date, datetime


def _value_tree_contains_scalar(value: object, target: object) -> bool:
    if value == target:
        return True
    if isinstance(value, dict):
        return any(_value_tree_contains_scalar(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_value_tree_contains_scalar(item, target) for item in value)
    return False


def _tool_call_depends_on_anchor_entity(
    record: dict[str, object],
    *,
    anchor_entity: dict[str, object],
) -> bool:
    params = record.get("params")
    if not isinstance(params, dict):
        return False
    return any(_value_tree_contains_scalar(params, value) for value in anchor_entity.values())


def _collect_observed_strings(value: object, *, strings: set[str]) -> None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            strings.add(normalized)
        return
    if isinstance(value, datetime | date):
        normalized = value.isoformat().strip().lower()
        if normalized:
            strings.add(normalized)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_observed_strings(item, strings=strings)
        return
    if isinstance(value, list):
        for item in value:
            _collect_observed_strings(item, strings=strings)


def _collect_scalar_values(value: object, *, sink: set[object]) -> None:
    """Collect all scalar values from a tool result for anchor-chain tracking."""
    if isinstance(value, (str, int, float, bool)):
        sink.add(value)
        if isinstance(value, str):
            sink.add(value.strip().lower())
        return
    if isinstance(value, datetime | date):
        sink.add(value.isoformat())
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_scalar_values(item, sink=sink)
        return
    if isinstance(value, list):
        for item in value:
            _collect_scalar_values(item, sink=sink)


def _is_anchor_connected_call(
    *,
    tool_name: str,
    params: dict[str, object],
    result: object,
    anchor_entity: dict[str, object] | None,
    known_anchor_values: set[object],
) -> bool:
    """Check if a tool call is connected to the anchor entity chain.

    A call is anchor-connected if any of its parameter values appear in
    the anchor entity or in any previously observed anchor-connected result.
    GET calls with an ``id`` param matching a known anchor value count.
    FIND/CALC/RANK calls with a ``value`` param matching count.
    """
    del tool_name, result
    if anchor_entity is None:
        return True  # no anchor locked yet — assume connected
    anchor_values: set[object] = set()
    for v in anchor_entity.values():
        anchor_values.add(v)
        if isinstance(v, str):
            anchor_values.add(v.strip().lower())
    all_known = anchor_values | known_anchor_values
    for param_value in params.values():
        if isinstance(param_value, (dict, list)):
            continue
        if param_value in all_known:
            return True
        if isinstance(param_value, str) and param_value.strip().lower() in all_known:
            return True
    return False


def _collect_answer_strings(value: object, *, sink: list[str]) -> None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            sink.append(normalized)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_answer_strings(item, sink=sink)
        return
    if isinstance(value, list):
        for item in value:
            _collect_answer_strings(item, sink=sink)


def _disconnected_answer_strings(
    answer: object,
    *,
    observed_strings: set[str],
    anchor_connected_strings: set[str],
) -> list[str]:
    """Find answer strings observed only in anchor-disconnected tool calls."""
    answer_strings: list[str] = []
    _collect_answer_strings(answer, sink=answer_strings)
    disconnected: list[str] = []
    for value in answer_strings:
        if value not in observed_strings:
            continue  # ungrounded — caught by other check
        if value not in anchor_connected_strings:
            disconnected.append(value)
    return sorted(dict.fromkeys(disconnected))


def _rebuild_anchor_connected_strings(
    tool_calls: list[dict[str, object]],
    *,
    anchor_entity: dict[str, object],
) -> set[str]:
    """Replay tool calls and collect strings only from anchor-connected ones."""
    known_values: set[object] = set()
    connected_strings: set[str] = set()
    for call in tool_calls:
        name = str(call.get("tool_name", ""))
        params = call.get("params")
        if not isinstance(params, dict):
            params = {}
        result = call.get("result")
        if _is_anchor_connected_call(
            tool_name=name,
            params=params,
            result=result,
            anchor_entity=anchor_entity,
            known_anchor_values=known_values,
        ):
            _collect_observed_strings(result, strings=connected_strings)
            _collect_scalar_values(result, sink=known_values)
    return connected_strings


def _blank_string_paths(value: object, *, path: str = "$") -> list[str]:
    blank_paths: list[str] = []
    if isinstance(value, str):
        return [path] if not value.strip() else []
    if isinstance(value, dict):
        for key, item in value.items():
            blank_paths.extend(_blank_string_paths(item, path=f"{path}.{key}"))
        return blank_paths
    if isinstance(value, list):
        for index, item in enumerate(value):
            blank_paths.extend(_blank_string_paths(item, path=f"{path}[{index}]"))
        return blank_paths
    return []


def _ungrounded_answer_strings(
    answer: object,
    *,
    observed_strings: set[str],
) -> list[str]:
    answer_strings: list[str] = []
    _collect_answer_strings(answer, sink=answer_strings)
    ungrounded: list[str] = []
    for value in answer_strings:
        if value in observed_strings:
            continue
        ungrounded.append(value)
    return sorted(dict.fromkeys(ungrounded))


def _observed_anchor_readable_string_surface(
    tool_calls: list[dict[str, object]],
    *,
    anchor_entity: dict[str, object],
) -> bool:
    observed_strings: set[str] = set()
    for record in tool_calls:
        if not _tool_call_depends_on_anchor_entity(record, anchor_entity=anchor_entity):
            continue
        _collect_observed_strings(record.get("result"), strings=observed_strings)
    return bool(observed_strings)


_IDENTIFIER_FIELD_TOKEN_RE = re.compile(
    r"(?<![a-z0-9_])[a-z][a-z0-9_]*_ids?(?![a-z0-9_])",
    re.IGNORECASE,
)


def _contains_raw_identifier_token(text: str) -> bool:
    return _IDENTIFIER_FIELD_TOKEN_RE.search(text.lower()) is not None


def _contains_entity_placeholder_token(text: str) -> bool:
    lowered = text.lower()
    return "<entity>" in lowered or "&lt;entity&gt;" in lowered
