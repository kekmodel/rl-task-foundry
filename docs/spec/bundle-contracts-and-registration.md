# Bundle Format, Contracts, and Registration

## Exported Serving Bundle

The serving bundle is the final runtime deliverable.

```text
bundle_root/
  databases/
    <db_id>/
      atomic_tools.py
      atomic_tool_definitions.json
  environments/
    <env_id>/
      environment.yaml
      instances.jsonl
      canonical_answers.jsonl
```

## Key Contracts

### `TaskContract`

The task contract captures the user-facing task surface.

Key fields:

- `question`
- `topic`
- `output_schema`
- `constraint_summary`
- `difficulty_vector`
- `instance_parameters`

### `EnvironmentContract`

The environment contract captures durable metadata.

Key fields:

- `env_id`
- `db_id`
- `domain`
- `topic`
- `atomic_tool_set_ref`
- `difficulty_vector`
- `tool_signature`
- `task_signature`
- `status`
- `quality_metrics`
- `rollout_constraints`
- `task`
- `instance_space`

### Materialized Instance Records

Per-instance materialized data contains:

- `instance_id`
- `rendered_user_prompt`
- `params`
- `anchor_values`

### Materialized Canonical Answer Records

Per-instance canonical answer data contains:

- `instance_id`
- `canonical_answer`
- `canonical_answer_json`
- `label_signature`

`label_signature` is derived from canonical answer JSON and is used for identity checks and diagnostics.

## Schema Extraction Rule

The runtime, not the synthesis agent, derives `output_schema` from the canonical answer.

This keeps the label authoritative and removes schema drift between generated schema text and actual label shape.

## Registry Commit Policy

A draft becomes durable only after:

1. synthesis succeeds
2. solver pass-rate falls inside the configured band
3. the registry accepts the exact signature or records a duplicate

Quality-rejected drafts are disposable and may be regenerated later.

## Deduplication Surface

Registry deduplication uses:

- exact signatures
- semantic dedup text
- difficulty band bucketing

The exact signature is authoritative for commit-time identity.
