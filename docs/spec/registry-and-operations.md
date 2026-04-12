# Registry and Operations

## Quality Gate

The current acceptance path is intentionally simple.

```text
LABEL_CONSTRUCTION -> TASK_SYNTHESIS -> solver pass-rate -> registry
```

An environment is accepted only when solver pass-rate falls inside the configured band.

- pass rate below the lower bound: reject as too hard
- pass rate above the upper bound: request a harder retry
- pass rate inside the band: accept

The exact numeric band and rollout sizes are hyperparameters, not fixed semantics.

## Scheduler

The scheduler chooses `(db_id, topic)` pairs across the registry.

Goals:

- keep databases interleaved fairly
- respect per-topic backoff
- preserve coverage pressure across difficulty bands

## Provider Resilience

The synthesis pipeline should tolerate transient provider failures.

Diagnostics to keep:

- provider name
- model name
- synthesis phase
- backend failure taxonomy
- retry count

## Environment Registry

The registry is the durable source of accepted environments.

It maintains:

- filesystem environment records
- SQLite index tables
- difficulty-band coverage accounting
- semantic dedup candidates

## Manual Review

Manual review is for audit and debugging, not for replacing the quality gate.

Useful review targets:

- rendered prompt quality
- canonical answer plausibility
- semantic duplication
- topic distribution
- difficulty drift

## Proof Environment

A deterministic proof environment is kept as a vertical-slice fixture for end-to-end validation of:

- synthesis flow
- solver rollout
- registry commit
- bundle export

## Success Criteria

The system is successful when it can:

- ingest arbitrary read-only PostgreSQL databases
- generate grounded label-first environments
- reject too-easy and too-hard tasks automatically
- export self-contained serving bundles
- accumulate accepted environments durably in the registry

## Freeze Policy

The deleted path-centric stack stays in git history only. New work should extend the atomic-tool, label-first pipeline rather than reintroducing legacy surfaces.
