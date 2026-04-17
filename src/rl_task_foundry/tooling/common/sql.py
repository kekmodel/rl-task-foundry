"""SQL string helpers shared by atomic and composer toolsets.

Every SQL string produced by the tooling package must flow through these
helpers. Raw user input never reaches the SQL text; values go through
asyncpg parameters. Identifiers (table, column names) are validated against
the schema snapshot before quoting.
"""

from __future__ import annotations

import datetime as _dt


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def quote_table(schema_name: str, table_name: str) -> str:
    return f"{quote_ident(schema_name)}.{quote_ident(table_name)}"


def readonly_select(sql: str) -> str:
    """Collapse whitespace so traces stay short. Safe because we build SQL
    only from our own compile routines (no user input in the text).
    """
    return " ".join(sql.split())


_ALLOWED_SCALAR_TYPES = (int, float, str, bool, bytes, _dt.datetime, _dt.date, _dt.time)


def coerce_param(value: object) -> object:
    """Minimal coercion of Python values bound for asyncpg parameters.

    asyncpg accepts most Python scalars directly; this function rejects
    unsupported shapes early with a clear error. Lists pass through for
    `ANY($1)` predicates; dicts/tuples are rejected.
    """
    if value is None:
        return None
    if isinstance(value, _ALLOWED_SCALAR_TYPES):
        return value
    if isinstance(value, list):
        return [coerce_param(item) for item in value]
    raise TypeError(
        f"unsupported parameter type for tooling SQL: {type(value).__name__}"
    )
