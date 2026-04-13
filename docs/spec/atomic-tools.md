# Atomic Tools

## Tool Architecture

Atomic tools are generated per database from schema structure. They are shared across all environments for the same `db_id`.

The synthesis agent does not generate tool code.

## Atomicity Rules

Allowed:

- single-table `SELECT`
- one-hop FK traversal
- single aggregate over a single table
- single grouped aggregate over one grouping column

Forbidden:

- multi-hop joins inside one tool
- subqueries
- window functions
- arbitrary helper tools that jump directly to the final answer

## Tool Families

- `T1` Point Lookup
- `T2` Bounded Enumeration
- `T3` Single-Column Filter
- `T4` FK Traversal
- `T5` Distinct Values
- `T6` Filtered Aggregate
- `T7` Sorted Top-K
- `T8` Grouped Aggregate Top-K

## Important Contract Details

- all multi-row tools accept a `limit` parameter
- runtime caps `limit` by `bounded_result_limit`
- `T6` includes `COUNT`
- `AVG` and floating-point `SUM` use DB-side `ROUND(..., float_precision)`
- `T7` supports both filtered and unfiltered top-k retrieval
- `T8` supports grouped `SUM/AVG/COUNT/MIN/MAX` with deterministic tie-break ordering
- filtered `T8` descriptions follow the same surface pattern as filtered `T7`:
  `Rank {group_column} groups for a specific {filter_column} by their {agg} in {table}, {direction}.`

## Compression Policy

When tool count must be reduced, the compression drop order is:

```text
aggregate -> grouped_aggregate -> sorted_top_k -> like -> distinct -> range -> in
```

The core lookup, traversal, and bounded enumeration surface is kept longest.

## SQL Contract

- database access is read-only
- tool output must stay within declared `returns_schema`
- deterministic ordering is required whenever multiple rows may be returned
- actor-visible tool semantics must remain stable across environments sharing the same database

## Seeded Row Ordering

- synthesis runs use one per-run shuffle seed for unordered multi-row tool results
- solver replicas use one per-replica shuffle seed for unordered multi-row tool results
- unordered `list_*`, `filter_*`, and reverse traversal tools use seeded row ordering
- `sorted_top_k` keeps its explicit sort key and uses the seed only as a deterministic tie-breaker
- scalar aggregate and count tools do not use shuffle ordering
