# Environment Bundle Model

## Environment as the Core Unit

The deployable unit is an environment bundle layered on top of a shared database-level atomic tool bundle.

- all environments with the same `db_id` share the same atomic tools
- each environment contributes task metadata, rendered prompts, and canonical answers
- the environment bundle is the durable RL dataset unit

## Bundle Layers

### Database Layer

Per database:

- `atomic_tools.py`
- `atomic_tool_definitions.json`

### Environment Layer

Per environment:

- `environment.yaml`
- `instances.jsonl`
- `canonical_answers.jsonl`

The serving bundle contains only what the environment server needs at runtime.

## Registry vs Exported Bundle

The registry may keep extra bookkeeping artifacts for indexing and deduplication. The exported serving bundle is narrower and contains only runtime-serving data.

## Environment Metadata

Each environment carries:

- `env_id`
- `db_id`
- `domain`
- `topic`
- `difficulty_vector`
- `tool_signature`
- `task_signature`
- `status`
- `quality_metrics`
- `rollout_constraints`

## Proposed vs Materialized Environment

A proposed environment exists only during synthesis. A materialized environment has:

- rendered prompts
- canonical answers
- exact signatures
- registry metadata

Only accepted environments are materialized into the durable registry.
