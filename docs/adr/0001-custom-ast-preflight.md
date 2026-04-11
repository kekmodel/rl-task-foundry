# ADR 0001: Custom AST Preflight and Registration-Lane Subprocess Isolation

- Status: Accepted
- Date: 2026-04-11
- Owners: rewrite lead, release owner

## Context

The synthesis-agent rewrite needs to register generated `tools.py`, `solution.py`,
`verifier.py`, and `shadow_verifier.py` safely before they can participate in
self-consistency, shadow verification, or production solver execution.

We considered three broad approaches:

1. Trust generated code and run it in-process.
2. Use a general-purpose sandbox or policy engine such as RestrictedPython.
3. Use a narrow custom AST preflight plus subprocess isolation in the registration lane.

The v1 spec requires:

- explicit import allowlists
- rejection of reflection / dunder / dynamic access patterns
- function-signature validation per artifact kind
- subprocess isolation for untrusted generated code during registration
- main-process execution for production solver tool calls after registration succeeds

## Decision

We choose option 3:

- custom AST preflight implemented in project code
- persistent subprocess worker pool for registration-lane execution
- main-process solver lane for registered tools

## Why Not RestrictedPython

RestrictedPython is not the right fit for this project’s v1 boundary.

- We already have a deliberately narrow artifact DSL and can validate it directly.
- We need contract-aware checks that are specific to synthesis artifacts:
  `async def tool_name(conn, ...)`, `def solve(tools)`, and the staged verifier functions.
- We want clear, structured registration errors that map directly to artifact-generation
  feedback loops.
- The production design already splits lanes: registration is isolated, solver execution is
  not per-call sandboxed.

RestrictedPython could still be revisited later, but it would add a second policy model
without replacing the contract-specific checks we need anyway.

## Consequences

Positive:

- error messages stay project-specific and easy to feed back into synthesis prompts
- lane A / lane B split stays explicit
- we keep fast solver execution after registration succeeds

Tradeoffs:

- we own the AST denylist / allowlist logic
- subprocess worker lifecycle becomes part of the runtime surface
- security posture depends on keeping the preflight and subprocess contract narrow

## Implementation Notes

- Static policy lives in
  `src/rl_task_foundry/synthesis/registration_policy.py`
- Registration worker transport lives in
  `src/rl_task_foundry/synthesis/subprocess_pool.py`
  and
  `src/rl_task_foundry/synthesis/subprocess_worker.py`
- Lane planning lives in
  `src/rl_task_foundry/synthesis/runtime_policy.py`

## Superseded Sections (2026-04-12)

Sections covering R1~R7 validators for synthesis-generated `tool.py`
(async contract, parameterized query, read-only, bounded result set,
no cursor, no server-side state, deterministic SQL) are **partially
superseded by ADR 0002** (Atomic Tools per Database).

Under ADR 0002, synthesis no longer generates `tool.py`. Atomic tool
implementations are produced deterministically by the schema-driven
generator, which satisfies R1~R7 by construction. The AST preflight
for tool modules is therefore no longer needed.

Preflight validators for `solution.py`, `verifier.py`, and
`shadow_verifier.py` remain in effect and are extended in ADR 0002 to
check that these modules call only names present in the db-level
atomic tool set.

See ADR 0002 for the new validation contract.
