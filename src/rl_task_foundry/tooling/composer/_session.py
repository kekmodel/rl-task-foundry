"""Per-conversation state for the composer analytic toolset.

Composer primitives are stateless across calls — no cursor store, no
intermediate plans — so `ComposerSession` carries only the two pieces of
context every call needs: the schema snapshot and the asyncpg
connection. Kept separate from `tooling.atomic.calculus.AtomicSession`
per the redesign philosophy (atomic and composer evolve independently
and must not import from each other).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rl_task_foundry.tooling.common.schema import SchemaSnapshot


class _Row(Protocol):
    def __getitem__(self, key: str) -> object: ...


class _ConnectionLike(Protocol):
    async def fetch(self, sql: str, *args: object) -> Sequence[_Row]: ...
    async def fetchrow(self, sql: str, *args: object) -> _Row | None: ...
    async def fetchval(self, sql: str, *args: object) -> object: ...


@dataclass(slots=True)
class ComposerSession:
    """One synthesis conversation's composer state.

    Holds the schema snapshot (read-only) and an asyncpg connection. The
    caller owns the connection lifecycle.
    """

    snapshot: SchemaSnapshot
    connection: _ConnectionLike
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


__all__ = ["ComposerSession"]
