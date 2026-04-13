# Foundation

## Overview

RL Task Foundry turns a read-only PostgreSQL database into RLVR task bundles.

The system does not generate isolated lookup questions. Its unit of generation is a task bundle built on top of a database-level atomic tool bundle.

## Core Goals

- accept an arbitrary read-only PostgreSQL database
- inspect schema and live rows
- synthesize label-first task bundles
- keep only task bundles that pass the solver pass-rate quality gate
- accumulate accepted task bundles in a durable registry

## Operating Principles

1. Hard to solve, easy to verify.
2. Label correctness and uniqueness matter more than throughput.
3. Quality gates are authoritative; generation is disposable.
4. Each task bundle belongs to exactly one database.
5. Reward is binary exact match only.
6. New databases can be added without hand-written task logic.

## Clean Break Policy

The old path-centric stack is not authoritative.

- legacy `tools/`, `tasks/`, `truth/`, and `verification/` are deleted
- the authoritative runtime surface is `synthesis/`, `solver/`, and `pipeline/solver_orchestrator.py`
- the source of truth is DB-grounded label construction, not generated verifier code

## Authoritative Modules

- `config/`
- `infra/`
- `schema/introspect.py`, `schema/graph.py`, `schema/path_catalog.py`
- `synthesis/`
- `solver/backend_openai_agents.py`
- `pipeline/solver_orchestrator.py`
- `calibration/`
- `cli.py`

## System Architecture

```text
DB Registry
  -> Domain Scheduler
  -> Synthesis Orchestrator
  -> Quality Gate
  -> Task Registry
```

### Layer Breakdown

1. Schema and data discovery
2. Atomic tool generation and materialization
3. Label-first synthesis
4. Solver pass-rate quality gate
5. Registry persistence and bundle export
