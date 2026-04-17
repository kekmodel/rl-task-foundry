# Task Bundle Model

## Task Bundle as the Core Unit

The deployable unit is a task bundle layered on top of a shared database-level atomic tool bundle.

- all task bundles with the same `db_id` share the same atomic tools
- each task bundle contributes task metadata, one rendered prompt, and one canonical answer
- the task bundle is the durable RL dataset unit

## Bundle Layers

### Database Layer

Per database:

- `atomic_tools.py`
- `atomic_tool_definitions.json`

### Task Layer

Per task:

- `task.yaml`
- `instance.json`
- `canonical_answer.json`

The serving bundle contains only what the environment server needs at runtime.

## Registry vs Exported Bundle

The registry may keep extra bookkeeping artifacts for indexing and deduplication. The exported serving bundle is narrower and contains only runtime-serving data.

## Task Bundle Metadata

Each task bundle carries:

- `task_id`
- `db_id`
- `domain`
- `topic`
- `created_at`
- `generator_version`
- `tool_signature`
- `task_signature`
- `status`
- `quality_metrics`
- `rollout_constraints`

## Proposed vs Materialized Task Bundle

A proposed task bundle exists only during synthesis. A materialized task bundle has:

- one rendered prompt
- one canonical answer
- exact signatures
- registry metadata

Only accepted task bundles are materialized into the durable registry.
