# Tooling Redesign: Asymmetric Composer/Solver Toolsets

> **Status:** design spec. All checklist items are complete. Landing chain: atomic scaffold `f63be79`, calculus `201f6f9`, atomic tool_factory `03e9690`, `Any` sweep `b3a5c44`, composer `schema_map` `7f92ae2`, `sample` `398d175`, `query` DSL `804f7da`, `profile` `92e54ce`, `neighborhood` `a9b6d78`, composer tool_factory `852d7c9`, synthesis rewire `0558b45`, solver rewire `52b3fdd`, prompt rewrite `4f0b6d5`, bundle export migration `fdee2f6`, atomic_tools codegen retirement `a51af0d`. Remaining open items are downstream of the redesign: iter13+ prompt tuning on the new surface, and an eventual composer rewrite of the proof environment task so it stops hand-coding a task draft.

## Motivation

The current `synthesis/atomic_tools.py` renders per-database Python source via string concatenation (~1300 lines) and produces one `find_<table>_by_<column>` tool per (table, column) pair. Two problems fell out of prompt-tuning observations (see `docs/prompt_tuning_log.md`, iter07–iter12):

1. **Shortcut modes weaken composition.** `find_<table>_by_<column>(op='any', sort_by=…, limit=N)` answers "top N rows of table X ordered by Y" in a single call. That turns a chaining task into a tool-selection task, contradicting the RL goal.
2. **Composer and solver share one tool surface.** Under same-model composer/solver pairing, the composer designs tasks at its own difficulty level; solvers trivially match. Observed as `pass_rate=1.0` lock-in across three trials (iter07 retry, iter10). The ceiling is *structural*, not a prompt defect.

The redesign splits toolsets by **role** and refounds each on its own philosophy.

## Roles and philosophies

### Solver atomic toolset — composition calculus

The solver is the RL training target. Its trace is the learning signal. Each tool must be an indivisible primitive such that:

- Any grounded fact in the database is reachable by a chain of primitives.
- No primitive's effect can be replicated in fewer calls by a chain of other primitives.

Out of scope for solver: bulk-list shortcuts, multi-parameter bundled queries, raw SQL, schema introspection tools.

### Composer analytic toolset — synthesis productivity

The composer is a generator, not an RL target. It needs to orient in schema, profile distributions, author tasks with calibrated difficulty, and verify canonical answers efficiently. It is free to use coarse, high-bandwidth tools.

In scope for composer: schema maps, column profiles, neighborhood sketches, one-call compound queries.

**Code isolation.** `tooling/atomic/` and `tooling/composer/` never import from each other. Both depend on `tooling/common/` for schema introspection, SQL helpers, and edge path resolution.

## Module layout

```
src/rl_task_foundry/tooling/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── schema.py       # SchemaSnapshot (immutable, JSON-serializable)
│   ├── sql.py          # quote_ident, quote_table, readonly guards, parameter coercion
│   └── edges.py        # TypedEdge (directed FK), path resolver
├── atomic/
│   ├── __init__.py
│   ├── cursor.py       # CursorPlan sum type + plan hashing + per-session store
│   ├── sql_compile.py  # CursorPlan → (sql, params)
│   ├── calculus.py     # 10 async primitives bound to AtomicSession
│   └── tool_factory.py # agents-SDK FunctionTool construction (post-vertical-slice)
└── composer/
    ├── __init__.py
    ├── query_dsl.py    # JSON spec → SQL (post-vertical-slice)
    ├── profile.py
    ├── schema_map.py
    └── tool_factory.py
```

## Solver atomic calculus

Ten primitives. The solver sees each as a separate tool with table/column/edge parameters typed as enums in the JSON schema.

### Set-producing

- `rows_where(table, column, op ∈ {eq,in,lt,gt,lte,gte,like}, value) → cursor`
  Create a filtered cursor. `op='any'` intentionally absent — no unfiltered entry point.
- `rows_via(cursor, edge) → cursor`
  Project through a typed FK edge. Multiplicity preserved (bag semantics); materialization steps decide whether to dedupe.
- `intersect(cursor_a, cursor_b) → cursor`
  Set intersection. Both operands must be cursors over the same target table.

### Set-annotating

- `order_by(cursor, column, direction) → cursor`
  Attaches an ordering. Doesn't execute SQL; consumed by `take`.

### Set-materializing

- `take(cursor, n) → [row_id, …]` with `2 ≤ n ≤ 5`.
  Dedup applied. `n=1` is intentionally disallowed to prevent `sort+limit=1` shortcuts that would replace `aggregate(max) + rows_where(eq, max)` chains.
- `count(cursor) → int`
  Bag count (multiplicity preserved).
- `aggregate(cursor, fn ∈ {sum,avg,min,max}, column) → value`
  Scalar aggregate over the cursor.
- `group_top(cursor, group_column, agg, n) → [(group_value, agg_value), …]` with `2 ≤ n ≤ 5`.
  Top-n groups by aggregate.

### Row reading

- `read(table, row_id, columns) → {column: value, …}`
  Reveal specific columns of one row. Separated from `rows_where` so "find by condition" and "read attributes" are distinct primitives.

### Total: 9 tools (not 10 — `order_by` is counted as set-annotating). Small fixed surface regardless of schema size.

### Cursor model

A cursor is an **opaque ID** backed by a content-hashed `CursorPlan`. Plans are composed immutably; `rows_via`, `intersect`, `order_by` return new plans wrapping their inputs. SQL is built and executed only at materialization (`take`/`count`/`aggregate`/`group_top`).

```python
CursorPlan = WhereNode | ViaNode | IntersectNode | OrderNode

@dataclass(frozen=True, slots=True)
class WhereNode:
    table: str
    column: str
    op: str
    value: Any
    # target table = self.table

@dataclass(frozen=True, slots=True)
class ViaNode:
    source: CursorPlan
    edge: TypedEdge
    # target table = edge.target_table

@dataclass(frozen=True, slots=True)
class IntersectNode:
    left: CursorPlan
    right: CursorPlan
    # target table = left.target_table (must equal right.target_table)

@dataclass(frozen=True, slots=True)
class OrderNode:
    source: CursorPlan
    column: str
    direction: Literal["asc", "desc"]
    # target table = source.target_table
```

A `CursorStore` maps cursor IDs to plans within an `AtomicSession`. The session scopes cursors to one conversation (solver run). On session close, the store is discarded.

### Multiplicity

- `rows_where` yields distinct rows (each row is itself).
- `rows_via` with a forward FK (many-to-one, e.g. `rental.customer_id → customer`) preserves multiplicity — each source rental contributes one customer occurrence to the target cursor.
- `rows_via` with a reverse FK (one-to-many, e.g. `customer ← rental.customer_id`) preserves multiplicity naturally.
- `intersect` takes intersection of distinct row IDs.
- `take` applies `DISTINCT` + `ORDER BY` + `LIMIT`. Tie-break by primary key (secondary order asc) for determinism.
- `count` and `aggregate` respect multiplicity by default.
- `group_top` always dedupes by `(group_column, aggregate)` before selecting top.

### Deterministic chains

- `take(cursor, n)` without a prior `order_by` uses the target table's primary key as the sole ORDER BY. Deterministic but uninformative — composer should design tasks that require explicit ordering.
- Tie-breaks always append primary key asc to the sort list, ensuring reproducibility across runs.

### Budget

The solver's call budget (30 tool calls) is preserved. Each atomic function call consumes one turn. The RL-facing trace records the function name, parameters, and output (plan summary or materialized values) — not raw SQL.

## Composer analytic DSL (preview — full spec next session)

Five primitives:

- `schema_map(root_table=None, depth=2)` — graph of tables/columns/edges with hub/bridge tags. One call to orient.
- `profile(table, column=None, predicate=None)` — column distributions (min/max/top-k modes/quartiles/distinct count). Designed for calibrating filter difficulty.
- `neighborhood(table, row_id, depth=2, max_per_edge=5)` — anchor-rooted entity graph with sample IDs per edge.
- `query(spec)` — JSON spec DSL over select/filter/join-via-edge/sort/limit/aggregate/group-by. One call expresses anything `atomic` can express as a chain. Used to author canonical answers.
- `sample(table, n=5, seed, predicate=None)` — representative rows.

The composer does not have `read`, `take`, `aggregate` primitives — it only sees the analytic DSL. It does not require atomic tools to compute canonical answers; its `query` output is the source of truth.

### Verification semantics

The canonical answer produced by composer `query(spec)` is stored verbatim in the task bundle. Calibration runs expose atomic tools to solver runtimes; solvers must re-derive the same value via atomic chains. Pass rate is measured by exact-match against the stored canonical answer. Composer's richer tool surface does not pollute the calibration signal because calibration is solver-measured.

## Integration points and migration

### Kept in place (unchanged this session)

- Current `synthesis/atomic_tools.py` codegen and its consumers (`synthesis_db.py`, `atomic_tool_materializer.py`, `bundle_exporter.py`, `solver/runtime.py`, `synthesis/runtime.py`).
- All existing tests.
- Per-DB `atomic_tools.py` materialization for bundle export.

### New (this session)

- `tooling/common/` — schema snapshot, SQL helpers, edge resolver.
- `tooling/atomic/cursor.py` — `CursorPlan` types + `CursorStore`.
- `tooling/atomic/sql_compile.py` — plan → SQL (Where/Via/Intersect id-stream + materializers).
- `tooling/atomic/calculus.py` — all 9 primitives executable end-to-end.
- `tooling/atomic/tool_factory.py` — 9 agents-SDK `FunctionTool` builders + `build_atomic_tools(session)` aggregator. Schema-baked enums for table/column/edge/op/fn/direction; ISO string → datetime coercion at the JSON boundary for temporal columns; errors surface as `{error, error_type}` JSON rather than raising.
- Integration tests at `tests/test_tooling_atomic_integration.py` and `tests/test_tooling_atomic_tool_factory.py` cover both direct calculus and tool-handler paths against sakila.

### Follow-ups (not part of the redesign proper)

- iter13+ prompt tuning against the new tool surface.
- Rewrite the proof environment task via the composer DSL so it stops hand-coding a `SynthesisTaskDraft` and instead flows through the same `SynthesisAgentRuntime` path as live traffic.

### Bundle export change (next session)

`bundle_exporter._export_database_bundle` currently copies `atomic_tools.py` + `atomic_tool_definitions.json`. After migration it copies `schema_snapshot.json` plus a small `tooling_version.json` so the env server loads `tooling.atomic.calculus` and instantiates tools from the snapshot. Task bundles stay unchanged in shape; only the database-layer artifact format changes.

## Open questions

- **`rows_via` over many-column FKs.** Sakila's FKs are single-column; IMDb has some composite FKs. Current `TypedEdge` represents single-column edges only. Document limitation; handle composite edges as a later extension.
- **`in` op list size.** Current bound is `MAX_BATCH_VALUES=128`. Keep the same bound; enforce in `rows_where` input validator.
- **Composer SQL escape hatch.** The composer design deliberately stays above raw SQL. If task diversity ceilings appear, revisit with a restricted `sql_readonly(sql, max_rows=N)` tool. Not in initial scope.
- **Cursor TTL in long synthesis conversations.** If qwen keeps cursor IDs in its prompt context across many turns, the `CursorStore` grows. Cap at ~100 cursors per session with LRU eviction; emit a warning if referenced cursor no longer exists.

## Why this is the right shape

1. **Tool count collapses.** Previously proliferated `find_<table>_by_<col>` per pair; now one `rows_where(table, column, …)` with enum params. Prompt/trace/test sizes shrink with schema size unchanged.
2. **Shortcuts unrepresentable.** No tool takes a table and returns top-N ordered rows in one call. The RL signal is pure composition by construction, not by prompt exhortation.
3. **Composer productivity separated from atomic discipline.** Composer's `query(spec)` can subsume a solver's 8-call chain in one call, breaking the same-model ceiling through tool asymmetry rather than model asymmetry.
4. **No codegen.** Runtime schema-parameterized tools replace per-DB Python file generation. New DB = no new files; just a schema snapshot.
5. **Bundle simplification.** Ship `schema_snapshot.json` instead of `atomic_tools.py`; the runtime tooling package owns all implementations.
