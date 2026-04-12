# Synthesis Pipeline

## Authoritative Flow

The authoritative synthesis flow is label-first.

```text
SCHEMA_EXPLORATION
  -> CATEGORY_INFERENCE
  -> LABEL_CONSTRUCTION
  -> TASK_SYNTHESIS
```

## Stage 1: Schema and Data Exploration

The synthesis agent explores the real database using live atomic tools.

Goals:

- understand the schema at a practical level
- inspect sample rows
- identify grounded answer candidates
- understand what evidence paths are actually available

## Stage 2: Category Inference

The agent selects a `topic` string for the candidate environment.

This is a lightweight orientation step. The authoritative semantics are still set by the label.

## Stage 3: Label Construction

The agent produces the canonical answer first.

Current output includes:

- `canonical_answer_json`
- `anchor_entity`
- `difficulty_vector`
- `instance_parameters`
- `label_summary`
- `memory_summary`

The runtime derives the output schema from the canonical answer automatically.

## Stage 4: Task Synthesis

Given the fixed label, the agent writes the user-facing task request that makes that label the unique correct answer.

This stage does not generate verifier or shadow code.

## Prompt Policy

All system instructions, templates, and retry hints are written in English.

Only the generated user-facing task text follows `config.domain.language`.

## Prompt Constraints

Prompt inputs should stay LLM-friendly.

Keep:

- domain summary
- requested topic
- compact schema orientation
- previous phase outputs
- natural-language error feedback
- natural-language difficulty hint
- one few-shot example when helpful

Do not dump internal runtime state, raw debug payloads, or redundant tool-definition JSON into the prompt.

## Retry Model

Generation retries are allowed, but retries are not the source of truth.

- the label remains authoritative
- quality feedback can request a harder retry
- retries must change latent task semantics, not just wording
