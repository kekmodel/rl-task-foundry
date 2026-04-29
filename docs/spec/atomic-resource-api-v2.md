# Atomic Resource API v2

Status: implemented for the solver-facing atomic tool builders and hidden trace
AST emission.

## Goal

The solver-facing atomic tools are the action language that future actors learn
from. The surface should therefore look less like a database-internal API and
more like a normal resource-oriented endpoint workflow:

```text
call tool -> receive resource/data/error -> pass resource id to the next tool
```

The design keeps the current execution model (`CursorPlan`, `CursorStore`, SQL
compilation) but changes the actor-visible tool schema and responses.

## Principles

- Actor-visible observations must not contain repair hints or valid-action
  suggestions.
- Tool limits should model realistic API safety, not artificial anti-shortcut
  constraints.
- Long, high-quality traces should emerge from task distribution, reward, and
  curriculum, not from hiding natural actions such as `limit=1`.
- Internal execution metadata may be rich, but it must be separated from the
  actor-visible observation.
- The atomic tool set must remain expressive enough that targeted records,
  scalars, lists, and simple aggregates are reachable through composition.
- The generator must adapt to any introspectable PostgreSQL database schema,
  not to hand-written assumptions from Pagila, Postgres Air, or any other
  sample database.

## Database Adaptation Contract

The v2 surface is fixed, but its table enum, response metadata, and runtime
validators are generated from `SchemaSnapshot`.

For each database:

- table enums come from the introspected table set as canonical table handles
- record_set `columns`, `column_types`, and `primary_key` come from the current
  table
- record_set `relations` come from introspected foreign keys, including
  composite foreign keys, exposed in both forward and reverse directions
- record_set resources use set semantics: a resource contains unique target
  records by primary key, even after following a relation
- record id handling comes from each table's primary-key metadata
- scalar coercion comes from each column's PostgreSQL data type
- visibility filtering must be applied before the snapshot becomes a
  tool surface

The generator must not special-case domain names such as `customer`, `rental`,
`flight`, or `booking`. A new database should require configuration and
introspection only, not new Python tool code.

### Canonical Table Handles

Actor/composer-facing table identifiers are canonical handles, not raw
unqualified table names.

- If a table name is globally unique in the snapshot and cannot be confused
  with a qualified alias, the handle is the bare table name, for example
  `customer`.
- If the same table name appears in multiple schemas, the handle is
  `schema.table`, for example `public.customer` and `crm.customer`.
- If a raw table name contains `.`, it is qualified so it cannot shadow another
  table's `schema.table` alias.
- Tool outputs always return the canonical handle.
- Runtime lookup may accept a fully qualified `schema.table` alias for a unique
  table, but schemas expose only the canonical handle.
- Relation labels are opaque IDs built from canonical handles, for example
  `booking.customer_id->crm%2Ecustomer`. Simple identifiers stay readable;
  delimiter characters inside table or column names are percent-encoded so
  labels remain unique and copyable across arbitrary PostgreSQL identifiers.

This keeps common single-schema DBs readable while preserving correctness for
arbitrary multi-schema customer databases.

### Required Schema Features

The core surface can be generated for a database even if some tables are less
useful than others, but the runtime must handle the following cases explicitly.

| Schema shape | Expected behavior |
|---|---|
| Table has a primary key | `list_record_refs` emits record references that can be used by `get_record`. |
| Table has a composite primary key | record reference `id` is an array in primary-key column order. |
| Table has no primary key | record-set creation can name the table, but materialization, filtering, count, aggregate, record-reference listing, and `get_record` fail with `action_error` unless a stable synthetic record id policy is added later. |
| Table has no outgoing/incoming FK | `follow_relation` simply has no valid edge from that table; wrong-edge attempts fail with `action_error`. |
| Database has no FKs | record_set `relations` is empty; all single-table tools still generate. |
| FK is composite | exposed as a v2 relation when source/target column lists are non-empty and have the same length. |
| Table is a partition child | hidden by default when a partition parent is visible, unless partition testing is explicitly enabled. |
| Column type is unsupported for scalar coercion | reject values for that column with `request_error` or `action_error`; do not silently string-concatenate SQL. |

### Generation Invariants

For any generated DB surface:

- every table enum value resolves to exactly one `TableSpec`
- every table handle is unique, and ambiguous unqualified names are rejected
- every column listed in a record_set resource resolves on that resource's table
- every relation listed in a record_set resource resolves to one directed
  `TypedEdge`
- relation labels are unique per origin table after delimiter-safe encoding
- `follow_relation` validates that the edge originates at the source record-set
  table at runtime and deduplicates destination records by primary key
- `filter_record_set`, `sort_record_set`, `aggregate_records`, and `get_record` validate that
  referenced columns belong to the relevant table at runtime
- all SQL identifiers are quoted from validated schema metadata, never from
  untrusted raw strings
- all scalar values are passed as parameters, never interpolated into SQL

These invariants are what let the same action language apply across arbitrary
schemas while staying safe and deterministic.

## Resource Model

The old actor-visible `cursor_id` is replaced by a record-set resource handle.

```json
{
  "ok": true,
  "resource": {
    "id": "record_set_1",
    "type": "record_set",
    "table": "customer",
    "columns": ["customer_id", "store_id", "first_name"],
    "column_types": {
      "customer_id": "integer",
      "store_id": "integer",
      "first_name": "text"
    },
    "primary_key": ["customer_id"],
    "relations": ["customer<-rental.customer_id"]
  }
}
```

Internally this may still resolve to a content-hashed `CursorId` such as
`c_a81f...`. The hash is not shown to the actor. The session-local
`record_set_N` alias is the stable public handle for subsequent calls. The
resource also carries table-local columns and relation labels so actors can use
the current resource's metadata instead of reading a noisy database-wide column
enum.

## Response Envelope

All atomic tools return one of three visible shapes.

Resource-producing success:

```json
{
  "ok": true,
  "resource": {
    "id": "record_set_2",
    "type": "record_set",
    "table": "rental",
    "columns": ["rental_id", "customer_id", "rental_date"],
    "column_types": {
      "rental_id": "integer",
      "customer_id": "integer",
      "rental_date": "timestamp"
    },
    "primary_key": ["rental_id"],
    "relations": ["rental.customer_id->customer"]
  }
}
```

Data-producing success:

```json
{
  "ok": true,
  "data": {
    "count": 12
  }
}
```

Failure:

```json
{
  "ok": false,
  "error": {
    "type": "action_error",
    "code": "edge_wrong_origin"
  }
}
```

Error messages may be short, but they must not include a repair path, candidate
edge list, candidate column list, or next-action suggestion.

## Hidden Trace AST

Every v2 atomic tool appends a hidden semantic event to the per-run
`AtomicSession.trace_events` list. The actor never receives these events as tool
observations. After the solver runtime finishes, the orchestrator copies them
into `SolverResult.termination_metadata`:

```json
{
  "atomic_trace_version": "atomic-resource-api-v2.trace.v1",
  "atomic_trace_events": [
    {
      "action": "transform_resource",
      "operation": "filter_record_set",
      "visible_ok": true,
      "input_resource": {
        "id": "record_set_1",
        "type": "record_set",
        "table": "rental"
      },
      "predicate": {
        "column": "customer_id",
        "op": "eq",
        "value": 45
      },
      "output_resource": {
        "id": "record_set_2",
        "type": "record_set",
        "table": "rental"
      }
    }
  ]
}
```

The event stream is deliberately structural: it records resource transitions,
predicates, relation labels, pagination, aggregate descriptors, and result
shapes. Trace resources intentionally stay compact as `{id, type, table}` even
when the actor-visible resource includes table-local `columns`, `column_types`,
`primary_key`, and `relations`. It does not add repair hints to the
actor-visible response. Failed calls are recorded as `tool_error` events with
the visible error type/code and the submitted request payload.

## Error Taxonomy

Visible errors use three top-level types.

| Type | Meaning | Examples |
|---|---|---|
| `request_error` | The request does not match the endpoint contract. | invalid JSON, non-object input, missing field, wrong primitive JSON type, page size above hard API cap |
| `action_error` | The request shape is valid but impossible in the current environment state. | unknown record set, column not on record-set table, edge does not originate from record-set table, incompatible record-set intersection |
| `runtime_error` | The environment or runtime failed while executing a valid-looking action. | statement timeout, DB connection failure, response too large, internal bug |

## v4 Atomic Surface

The current solver-facing surface has twelve core tools. The main v4 change is
that value-shape-specific filters are split into endpoint-like tools, so schema
contracts can express scalar, list, pattern, and null predicates without relying
on runtime feedback.

```text
create_record_set(table) -> record_set resource
filter_record_set(record_set_id, column, op, value) -> record_set resource
filter_record_set_by_values(record_set_id, column, values) -> record_set resource
filter_record_set_by_pattern(record_set_id, column, pattern) -> record_set resource
filter_record_set_by_null(record_set_id, column, op) -> record_set resource
follow_relation(source_record_set_id, edge_label) -> record_set resource
intersect_record_sets(left_record_set_id, right_record_set_id) -> record_set resource
sort_record_set(record_set_id, column, direction) -> record_set resource
list_record_refs(record_set_id, limit, offset?) -> record_ref list
count_records(record_set_id) -> count
aggregate_records(record_set_id, fn, column) -> scalar
get_record(table, record_id, columns) -> object
```

### Mapping From Current Atomic Tools

| Current | v2 | Decision |
|---|---|---|
| `rows_where` | `create_record_set` + filter endpoints | split so unfiltered and incremental filtering are natural |
| `rows_via` | `follow_relation` | rename parameters, keep one FK hop |
| `intersect` | `intersect_record_sets` | rename parameters |
| `order_by` | `sort_record_set` | rename parameters |
| `take` | `list_record_refs` | allow natural `limit=1`; add optional pagination |
| `count` | `count_records` | envelope change |
| `aggregate` | `aggregate_records` | envelope change |
| `group_top` | removed from core v2 | too compressed for long-trace RL; later split if needed |
| `read` | `get_record` | envelope change |

## Tool Details

### `create_record_set`

Creates an unfiltered record-set resource for a table.

```json
{
  "table": "rental"
}
```

This replaces the need to invent dummy predicates when the task naturally starts
from a whole table.

### `filter_record_set`

Applies one scalar comparison predicate to an existing record set. Supported
operators are `eq`, `neq`, `lt`, `gt`, `lte`, and `gte`.

```json
{
  "record_set_id": "record_set_1",
  "column": "rental_date",
  "op": "gte",
  "value": "2005-01-01T00:00:00"
}
```

The column must belong to the record set's current table. Actors should choose
from the input resource's `columns` list. Runtime validation still enforces
membership and surfaces invalid choices as `action_error`.

### `filter_record_set_by_values`

Keeps records whose column value equals any value in a non-empty scalar list.

```json
{
  "record_set_id": "record_set_1",
  "column": "store_id",
  "values": [1, 2]
}
```

### `filter_record_set_by_pattern`

Applies one case-insensitive text pattern predicate. `%` may be used as a
wildcard for substring matching.

```json
{
  "record_set_id": "record_set_1",
  "column": "title",
  "pattern": "%airport%"
}
```

### `follow_relation`

Traverses exactly one directed FK edge.

```json
{
  "source_record_set_id": "record_set_1",
  "edge_label": "customer<-rental.customer_id"
}
```

The edge must originate at the source record set's table. Actors should choose
from the source resource's `relations` list. The actor-visible error does not
list valid edges. The label is an opaque token: actors copy it from the
resource instead of trying to parse or construct it.

The returned record set contains unique destination records by destination
primary key. Traversing from many source records to the same destination does
not create hidden duplicates.

### `list_record_refs`

Lists record references with ordinary API pagination.

```json
{
  "record_set_id": "record_set_1",
  "limit": 10,
  "offset": 0
}
```

`limit=1` is allowed. Very large requests are handled by realistic API limits:
hard page cap, max observation bytes, statement timeout, and reward failure for
unhelpful oversized traces.

Success returns record references, not full record objects.

```json
{
  "ok": true,
  "data": {
    "items": [
      {
        "type": "record_ref",
        "table": "rental",
        "id": 1520
      }
    ],
    "limit": 10,
    "offset": 0,
    "returned": 1
  }
}
```

### `get_record`

Reads selected columns for a single record reference or explicit table/record id.

```json
{
  "table": "rental",
  "record_id": 1520,
  "columns": ["rental_id", "customer_id", "rental_date"]
}
```

Composite primary keys continue to use array record ids in primary-key order.

## Hidden Trace AST

Actor-visible resources may include endpoint response metadata such as columns
and relations. Hidden trace resources stay compact, and the runtime may emit a
hidden trace AST for training-data analysis, filtering, reward shaping, and
curriculum.

Example for `follow_relation`:

```json
{
  "action": "transform_resource",
  "operation": "follow_relation",
  "input_resource": {
    "id": "record_set_1",
    "type": "record_set",
    "table": "customer"
  },
  "relation": {
    "edge_label": "customer<-rental.customer_id"
  },
  "output_resource": {
    "id": "record_set_2",
    "type": "record_set",
    "table": "rental"
  },
  "visible_ok": true
}
```

This trace AST is not part of the actor observation.

## Completeness Boundary

Within the intended SQL subset, actors should be able to reach values through
composition:

- whole table scans start with `create_record_set`
- predicates are chained with `filter_record_set`
- FK joins are chained with `follow_relation`
- multi-condition conjunction is represented by repeated filtering or
  `intersect_record_sets`
- deterministic selection is represented by `sort_record_set` + `list_record_refs`
- scalar facts are represented by `count_records`, `aggregate_records`, or
  `list_record_refs` + `get_record`
- pagination allows reaching later records when the task and reward justify it

The core v2 surface intentionally does not cover arbitrary SQL:

- OR / NOT / anti-join
- outer join
- window functions
- computed expressions
- date bucketing
- string transformation
- raw SQL
- multi-column FK traversal beyond existing execution support

Task generation should stay inside the reachable subset unless a later tool
extension explicitly broadens the actor interface.

## Implementation Plan

1. Add public record-set aliases to `CursorStore` while preserving internal
   content-hashed cursor ids.
2. Add atomic response helpers for `{ok, resource}`, `{ok, data}`, and
   `{ok, error}` envelopes.
3. Introduce request/action/runtime error classification for atomic tools.
4. Implement v2 tool builders in `tooling/atomic/tool_factory.py`.
5. Keep legacy cursor builders internal while tests and prompts migrate.
6. Update solver prompts and tests to use `record_set_id` terminology.
7. Remove `group_top` from the solver-facing v2 surface after composer and
   task generation no longer depend on it for solver traces.
8. Add hidden trace AST emission after the visible surface is stable.

## Acceptance Tests

- A chain can start with `create_record_set(table)` and list record references without
  a dummy predicate.
- `filter_record_set` on a record set rejects columns that do not belong to the record-set
  table with `action_error`.
- `follow_relation` rejects an edge from the wrong origin table with
  `action_error` and no candidate edge list.
- `list_record_refs(limit=1)` is accepted.
- Oversized fetch requests fail through the configured API cap or observation
  limit, not through an artificial `2 <= n <= 5` rule.
- Composite-PK record references still round-trip through `list_record_refs` and
  `get_record`.
- Hidden trace AST contains enough structure to reconstruct the resource
  transition graph without adding information to the actor-visible response.
