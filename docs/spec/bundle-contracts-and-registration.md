# Bundle Format, Contracts, and Registration

## Exported Serving Bundle

The serving bundle is the final runtime deliverable.

```text
bundle_root/
  databases/
    <db_id>/
      atomic_tools.py
      atomic_tool_definitions.json
  tasks/
    <task_id>/
      task.yaml
      task.json
      instance.json
      canonical_answer.json
```

## Key Contracts

### `TaskContract`

The task contract captures the user-facing task surface.

Key fields:

- `question`
- `topic`
- `output_schema`
- `constraint_summary`
- `instance_parameters`

### `TaskBundleContract`

The task bundle contract captures durable metadata.

Key fields:

- `task_id`
- `db_id`
- `domain`
- `topic`
- `atomic_tool_set_ref`
- `created_at`
- `generator_version`
- `tool_signature`
- `task_signature`
- `status`
- `quality_metrics`
- `rollout_constraints`
- `task`

### Materialized Instance Record

Materialized runtime data contains:

- `rendered_user_prompt`
- `anchor_entity`

### Materialized Canonical Answer Record

Materialized answer data contains:

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
- semantic dedup text (MinHash LSH over question + topic + output shape)

The exact signature is authoritative for commit-time identity.
