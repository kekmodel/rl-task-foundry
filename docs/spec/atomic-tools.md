# Atomic Tools

This page summarizes the current solver-facing atomic tool contract. The
historical v2 design details live in
[`atomic-resource-api-v2.md`](./atomic-resource-api-v2.md).

## Architecture

Atomic tools are generated per database from `SchemaSnapshot`. The action
language is fixed, while table enums and record_set metadata are generated from
the introspected schema.

The synthesis agent does not generate tool code. New databases should require
configuration and introspection, not hand-written Python tools.

## Current Solver Surface

The solver receives twelve resource-oriented tools:

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

The old internal `CursorPlan` engine still exists, but actor-visible responses
use record-set resource handles such as `record_set_1`.

`record_set` semantics are set semantics: a resource contains unique records by
the target table's primary key. `follow_relation` deduplicates destination
records, and `count_records` / `aggregate_records` operate over those unique
records. This keeps the actor-facing API consistent with ordinary resource
endpoints and prevents hidden join multiplicity from changing answers.

## Response Contract

All solver-visible atomic outputs use an API-style envelope.

```json
{
  "ok": true,
  "resource": {
    "id": "record_set_1",
    "type": "record_set",
    "table": "customer"
  }
}
```

```json
{
  "ok": true,
  "data": {
    "count": 12
  }
}
```

```json
{
  "ok": false,
  "error": {
    "type": "action_error",
    "code": "edge_wrong_origin"
  }
}
```

Visible errors must not include repair hints, valid-edge lists, candidate
columns, or next-action suggestions.

## Hidden Trace

The same calls append structural events to `AtomicSession.trace_events`.
Those events are copied to `SolverResult.termination_metadata` as
`atomic_trace_events` after the solver run, separate from actor-visible tool
observations.

## Atomicity Rules

Allowed:

- creating a record-set resource from one table
- applying one scalar, value-list, pattern, or null filter to an existing record set
- traversing one FK relation per call
- intersecting two record sets with the same target table
- annotating a record set with one sort column per call
- listing paginated record references
- counting or aggregating one record set
- getting selected columns from one record

Forbidden:

- raw SQL
- arbitrary multi-hop joins inside one tool
- high-level query DSL for the solver
- schema-map or valid-action hint tools for the solver
- grouped top-k shortcuts in the core solver surface
- window functions
- arbitrary helper tools that jump directly to the final answer

## Limit Policy

The actor may request natural page sizes, including `limit=1`. The environment
uses realistic API safety limits instead of artificial anti-shortcut constraints:

- read-only database access
- statement timeout
- configured maximum page size
- maximum observation size
- connection and concurrency limits

Oversized or unhelpful requests are handled by environment failure and reward,
not by hiding normal API actions from the actor.

## Completeness Boundary

Within the intended SQL subset, actors should be able to reach values by
composition:

- whole table scan: `create_record_set`
- conjunction: repeated filter endpoints or `intersect_record_sets`
- FK joins: chained `follow_relation`
- deterministic selection: `sort_record_set` + `list_record_refs`
- scalar facts: `count_records`, `aggregate_records`, or `list_record_refs` + `get_record`
- later records: `list_record_refs` with pagination

Out of scope:

- OR / NOT / anti-join
- outer join
- recursive CTE
- window functions
- computed expressions
- date bucketing
- string transformation
- raw SQL

Task generation must stay inside the reachable subset unless the actor
interface is deliberately extended.
