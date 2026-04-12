# Difficulty and Actor Interface

## Topic Model

`topic` is a free-form string selected during synthesis.

The runtime may normalize or bucket topics for scheduling and reporting, but the contract is no longer a fixed enum taxonomy.

## Difficulty Model

Difficulty is a 3-axis scalar vector.

```yaml
difficulty:
  search_cost: 3.0
  solution_space: 3.0
  constraint_density: 4.0
```

### Axis Meaning

- `search_cost`: tool-call cost required to gather the needed evidence
- `solution_space`: size of the candidate answer space
- `constraint_density`: how sparse valid solutions are inside that space

### Crank Policy

- exactly one axis may increase per crank step
- difficulty must remain monotonic
- runtime chooses the next crank axis
- solver pass-rate feedback decides whether to crank harder, discard, or accept

## Actor-Facing Interface

The actor sees:

- a rendered user prompt
- the database-level atomic tool set
- a constant `submit_result` tool

The actor never sees the canonical answer.

### Parity Principle

The solver used during quality gating must face the same prompt surface and same tool surface that the training actor will face.

### `rendered_user_prompt`

The runtime materializes the final prompt from the label-first task contract.

Current shape:

```text
<entity>
{"customer_id": 148}
</entity>

{user_request}

# Submit Result Format
{auto_extracted_schema}
```

Rules:

- the `<entity>` block contains only anchor entity primary-key data
- the natural-language request is user-facing and follows `config.domain.language`
- the schema is derived from the canonical answer, not synthesized separately
- constraints must be expressed naturally inside the user request

### `submit_result`

The actor submits the final answer through `submit_result(answer_text: str)`.

`answer_text` must be a valid JSON string matching the rendered schema.
