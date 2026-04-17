"""Composer analytic DSL — high-bandwidth recon tools for the synthesis agent.

See docs/spec/tooling-redesign.md. Philosophy is deliberately separate
from the atomic calculus: composer primitives are coarse one-call tools
designed for schema orientation, column profiling, and canonical-answer
authoring. Not an RL target.
"""

from rl_task_foundry.tooling.composer._session import ComposerSession
from rl_task_foundry.tooling.composer.neighborhood import neighborhood
from rl_task_foundry.tooling.composer.profile import profile
from rl_task_foundry.tooling.composer.query import query
from rl_task_foundry.tooling.composer.sample import sample
from rl_task_foundry.tooling.composer.schema_map import schema_map
from rl_task_foundry.tooling.composer.tool_factory import (
    build_composer_tools,
    build_neighborhood_tool,
    build_profile_tool,
    build_query_tool,
    build_sample_tool,
    build_schema_map_tool,
)

__all__ = [
    "ComposerSession",
    "build_composer_tools",
    "build_neighborhood_tool",
    "build_profile_tool",
    "build_query_tool",
    "build_sample_tool",
    "build_schema_map_tool",
    "neighborhood",
    "profile",
    "query",
    "sample",
    "schema_map",
]
