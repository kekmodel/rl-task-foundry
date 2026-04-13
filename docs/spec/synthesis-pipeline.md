# Synthesis Pipeline

## Authoritative Flow

The authoritative synthesis flow is label-first and is executed by a single synthesis agent.

```text
explore real tools
  -> choose a grounded topic
  -> construct the label
  -> synthesize the user-facing request
  -> submit_draft
  -> feedback / retry in the same conversation
```

The agent keeps one conversation context throughout the loop. Retry feedback is returned as tool output and the same agent continues exploring before it submits again.

## Exploration

Goals:

- understand the schema at a practical level
- inspect sample rows
- identify grounded answer candidates
- understand what evidence paths are actually available

## Topic Selection

The agent selects a `topic` string for the candidate task bundle.

The topic is a soft orientation aid. The authoritative semantics are still set by the label.

## Label Construction

The agent produces the canonical answer first.

Current output includes:

- `canonical_answer_json`
- `anchor_entity`
- `difficulty_vector`
- `label_summary`

The runtime derives the output schema from the canonical answer automatically.

## Task Synthesis

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
- tool feedback
- natural-language difficulty hint

Do not dump internal runtime state, raw debug payloads, or redundant tool-definition JSON into the prompt.

## Retry Model

Generation retries are allowed, but retries are not the source of truth.

- the label remains authoritative
- `submit_draft` feedback can request a harder retry
- retries must change latent task semantics, not just wording
