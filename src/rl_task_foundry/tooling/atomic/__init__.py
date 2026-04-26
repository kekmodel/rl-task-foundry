"""Solver atomic resource API for the RL target.

The package-level surface is intentionally the actor-visible v3 API only.
Internal cursor/calculus helpers live in their implementation modules, but are
not re-exported here.
"""

from rl_task_foundry.tooling.atomic.calculus import AtomicSession
from rl_task_foundry.tooling.atomic.cursor import CursorStore
from rl_task_foundry.tooling.atomic.tool_factory import (
    build_aggregate_records_tool,
    build_atomic_tools,
    build_count_records_tool,
    build_create_record_set_tool,
    build_filter_record_set_by_null_tool,
    build_filter_record_set_by_pattern_tool,
    build_filter_record_set_by_values_tool,
    build_filter_record_set_tool,
    build_follow_relation_tool,
    build_get_record_tool,
    build_intersect_record_sets_tool,
    build_list_record_refs_tool,
    build_list_records_tool,
    build_sort_record_set_tool,
)

__all__ = [
    "AtomicSession",
    "CursorStore",
    "build_aggregate_records_tool",
    "build_atomic_tools",
    "build_count_records_tool",
    "build_create_record_set_tool",
    "build_filter_record_set_by_pattern_tool",
    "build_filter_record_set_by_values_tool",
    "build_filter_record_set_by_null_tool",
    "build_filter_record_set_tool",
    "build_follow_relation_tool",
    "build_get_record_tool",
    "build_intersect_record_sets_tool",
    "build_list_record_refs_tool",
    "build_list_records_tool",
    "build_sort_record_set_tool",
]
