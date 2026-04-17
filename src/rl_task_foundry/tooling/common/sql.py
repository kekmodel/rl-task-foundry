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


_INTEGER_TYPES = frozenset(
    {
        "integer",
        "bigint",
        "smallint",
        "serial",
        "bigserial",
        "smallserial",
        "int2",
        "int4",
        "int8",
    }
)
_FLOAT_TYPES = frozenset({"real", "double precision", "float4", "float8"})
_DECIMAL_TYPES = frozenset({"numeric", "decimal", "money"})
_BOOLEAN_TYPES = frozenset({"boolean", "bool"})
_TIMESTAMP_TYPES = frozenset(
    {
        "timestamp",
        "timestamp without time zone",
        "timestamp with time zone",
        "timestamptz",
    }
)
_DATE_TYPES = frozenset({"date"})
_TIME_TYPES = frozenset({"time", "time without time zone", "time with time zone"})


def coerce_scalar(value: object, data_type: str) -> object:
    """Coerce a JSON scalar to the Python type asyncpg expects for ``data_type``.

    LLM-sourced tool payloads arrive through JSON, so integer PKs can appear as
    strings (``"5244"``) and temporals as ISO strings. asyncpg's binary protocol
    rejects the wrong Python type, so every value flowing into a typed SQL
    parameter must pass through here. Lists recurse element-wise.
    """

    if isinstance(value, list):
        return [coerce_scalar(item, data_type) for item in value]
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if data_type in _INTEGER_TYPES:
        return int(stripped)
    if data_type in _FLOAT_TYPES:
        return float(stripped)
    if data_type in _DECIMAL_TYPES:
        return _decimal_from_str(stripped)
    if data_type in _BOOLEAN_TYPES:
        return _bool_from_str(stripped)
    if data_type in _TIMESTAMP_TYPES:
        return _dt.datetime.fromisoformat(stripped)
    if data_type in _DATE_TYPES:
        return _dt.date.fromisoformat(stripped)
    if data_type in _TIME_TYPES:
        return _dt.time.fromisoformat(stripped)
    return value


def _decimal_from_str(source: str) -> object:
    from decimal import Decimal

    return Decimal(source)


def _bool_from_str(source: str) -> bool:
    lowered = source.lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(f"cannot coerce {source!r} to boolean")


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
