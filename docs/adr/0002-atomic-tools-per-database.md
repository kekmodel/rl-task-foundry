# ADR 0002: Atomic Tools per Database

- Status: Accepted
- Date: 2026-04-12
- Owners: rewrite lead, release owner

## Context

Milestone 1-2 of the older path-centric pipeline compiled narrow SQL tools from
schema-graph path traversal. That architecture lived primarily in the
`TierATaskFactory` and `tool_compiler` path and assumed that each environment
would carry a small task-shaped tool bundle.

The synthesis-agent rewrite moved task generation toward compositional
environments, but the tool side still inherited the environment-specific
`tool.py` pattern. The synthesis agent was expected to generate a narrow tool
artifact together with task, solution, and verifier code.

That inheritance created several structural problems:

- tool names leaked task intent directly, for example
  `get_customer_assignments_for_day`
- the solver could learn tool selection shortcuts instead of database query
  decomposition
- solver pass-rate became a biased predictor of true actor difficulty, so the
  25-75% quality band stopped reflecting what the actor would actually face
- RLVR VR principles (`hard to solve, easy to verify`) were only partially
  satisfied because the actor interface already encoded much of the solution
  structure in the tool layer

At the same time, the per-environment tool artifact kept a large amount of
legacy path-centric debt alive in Milestone 1-2 style code and multiplied code
surface area by environment count instead of database count.

## Decision

We replace the narrow per-environment tool architecture with a per-database
atomic tool architecture.

- tools are standardized at the database level
- the tool set is generated deterministically from the schema graph by a
  rule-based generator
- the synthesis agent no longer generates `tool.py`
- the synthesis agent now generates only:
  - `rendered_user_prompt`
  - `output_schema`
  - `solution.py`
  - `verifier.py`
  - `shadow_verifier.py`
  - `instance_space`

Atomic tools are defined as single SQL primitives.

- one tool call performs one primitive relational operation
- no multi-table join beyond one FK hop
- no subquery
- no `GROUP BY`
- no window function
- no single call may directly reveal the final answer

The v1 atomic tool families are:

1. T1 point lookup
2. T2 bounded enumeration
3. T3 single-column filter (`eq`, `range`, `like`, `in`)
4. T4 bidirectional one-hop FK traversal
5. T5 distinct values
6. T6 filtered single-column aggregates

The tool family is intended to be informally complete for the SQL subset this
project targets. Any answer reachable through ordinary `SELECT ... WHERE ...
JOIN via FK ... GROUP BY col ... ORDER BY ... LIMIT ...` patterns should be
reachable by chaining T1-T6 plus local deterministic post-processing in
`solution.py` or `verifier.py`.

All environments for the same `db_id` share the exact same atomic tool set. The
synthesis runtime materializes and caches the tool bundle the first time a
database is processed.

## Consequences

Positive:

- tool names no longer leak task intent, which structurally addresses parity
  leakage
- solver parity invariants become much stronger because the actor always sees a
  task-agnostic tool surface
- the actor must learn transferable DB query decomposition instead of
  environment-specific shortcuts
- registration policy becomes simpler because generated `tool.py` validation is
  removed
- server memory shifts from roughly `O(envs × tools × code)` to
  `O(dbs × tools × code)`
- the synthesis agent has less burden because it no longer owns tool generation

Trade-offs:

- `solution.py` and `verifier.py` become longer because they must chain atomic
  tools instead of leaning on narrow helpers
- actor context windows become larger because tool definitions are shared but
  broader
- early actor learning may be harder because atomic composition must be
  discovered through curriculum
- path-centric Milestone 1-2 code will require large-scale deletion, including
  legacy factories, compilers, and tests

## Alternatives Considered

1. Keep narrow per-environment tools.
   - rejected because tool-name leakage and legacy debt both remain
2. Use very general DB tools such as `list_tables` or `execute_sql`.
   - rejected because the chosen direction preserves narrow relational
     semantics while still removing task leakage
3. Hybrid architecture with atomic base tools plus task-specific narrow tools.
   - rejected because leakage remains and implementation complexity rises
     sharply
4. Prototype-first validation before committing to the transition.
   - considered early, but withdrawn after concluding that the VR signal is
     clean enough to justify a full commitment path

## Implementation Path

Detailed sequencing lives in `docs/plan.md`. The high-level phases are:

- Phase 1: atomic tool generator, canonicalization library, output-field
  metadata
- Phase 2: synthesis pipeline transition, database-level tool cache, prompt
  updates, subprocess transition
- Phase 3: solver backend transition to environment-contract-first execution
- Phase 4: full deletion of path-centric legacy code
- Phase 5: final documentation and operational cleanup

## Completeness Argument (Informal)

Given the six atomic tool families T1-T6 over a schema graph, any query
expressible as ordinary `SELECT ... FROM ... WHERE ... JOIN via FK ... GROUP BY
col ... ORDER BY ... LIMIT ...` can be reached by composition:

- point lookup maps to T1
- column filter maps to T3
- FK join across N hops maps to T4 chained N times
- `GROUP BY` plus aggregate maps to T5/T6 plus local grouping or reduction
- multi-condition `WHERE` maps to repeated T3 calls plus local intersection
- `ORDER BY` and `TOP-K` map to local deterministic sorting in `solution.py`

Queries outside the intended subset are explicitly excluded:

- recursive CTE / transitive closure
- window functions such as `lag`, `lead`, `rank`
- full-text search beyond `LIKE`

The synthesis prompt and acceptance policy must constrain generated tasks to the
expressible subset.

## Out of Scope

- the concrete environment API server implementation
- multi-database session routing or sharding strategy
- downstream actor curriculum design

## References

- `docs/spec.md`
- `docs/plan.md`
- `docs/adr/0001-custom-ast-preflight.md`
