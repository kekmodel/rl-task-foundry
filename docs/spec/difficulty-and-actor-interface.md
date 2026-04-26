# Difficulty and Actor Interface

## Topic Model

`topic` is a free-form string selected during synthesis.

The runtime may normalize or bucket topics for scheduling and reporting, but the contract is no longer a fixed enum taxonomy.

## Difficulty Model

Difficulty is measured empirically by solver pass rate. Axis names are
diagnostic language for logs and authoring feedback, not a fixed recipe for how
every database should be made harder.

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

- difficulty must remain monotonic
- the composer chooses a feasible strengthening direction from current DB
  evidence, tool observations, and feedback
- solver pass-rate feedback decides whether to crank harder, discard, or accept
- repeated-unit/list tasks should usually grow by curriculum (`1 -> 2 -> 3 ->
  5 -> 10` or all matching items) rather than jumping directly to the largest
  shape when that direction is supported by the current DB
- list difficulty can also grow by item complexity: each item may require an
  additional grounded condition, relationship, or tie-break. Passive display
  fields are not counted as meaningful difficulty.
- some directions will be impossible or unnatural for a given DB; the agent's
  job is to use the available tools and context to judge the feasible path, not
  to follow a global priority list

## Actor-Facing Interface

The actor sees:

- a rendered user prompt
- the database-level atomic tool set
- a constant `submit_result` tool

The actor never sees the canonical answer.

The actor is modeled as a customer-facing agent for the current database's
domain, not as a database engineer. The task text should ask for business
information in the user's language. The tools may expose generic API-like
resource operations over the database, but the natural-language task should not
require prior knowledge of table names, primary-key columns, join tables, or
internal schema design.

### Parity Principle

The solver used during quality gating must face the same prompt surface and same tool surface that the training actor will face.

The solver is a pure problem-solving role. It receives no system prompt that
explains synthesis, composer behavior, quality gates, pass rates, or training
purpose. It receives only the rendered customer problem, hidden entity block,
and generated tools that the eventual actor would see. The required structured
answer shape is exposed through the `submit_result` tool schema, like an API
endpoint contract, not through a prose/schema block in the user prompt.
In the OpenAI Agents backend this is an implementation invariant:
`Agent.instructions` stays `None`; all task-specific information enters through
`rendered_user_prompt` and tool schemas.

The actor learns through experience, not through authored strategy hints. Terms
such as curriculum, specificity feedback, Cardinality, Item-complexity, solver
pass rate, quality gate, or training purpose belong to composer/runtime logs and
authoring feedback only. They must not appear in the rendered actor prompt or
solver-facing system instructions.

### `rendered_user_prompt`

The runtime materializes the final prompt from the label-first task contract.

Current shape:

```text
<entity>
{"customer_id": 148}
</entity>

{user_request}
```

Rules:

- the `<entity>` block contains only anchor entity primary-key data
- the natural-language request is user-facing and follows `config.domain.language`
- the request is understandable without database knowledge
- visible task references come from domain data observed in the current database,
  not from generic phrases wrapped around internal IDs
- derived field names are `submit_result` keys, not final answer prose; concise
  API-style names are acceptable when they are stable and match the
  task-specific `submit_result` tool schema
- constraints must be expressed naturally inside the user request
- list order is part of the expected payload whenever the request/schema asks
  for a list; the actor should submit list items in the requested order

### `submit_result`

The actor submits the structured result through a task-specific
`submit_result(...)` tool. For object answers, the tool parameters are the
answer object fields directly. For non-object roots such as list answers, the
tool uses an `answer` wrapper because function tools require an object-shaped
parameter schema.

This tool call is the terminal, verifiable step of the learning trajectory.
The tool schema is derived from the canonical answer by runtime code, not
synthesized by the composer and not pasted into the user prompt.

`submit_result` is an exact payload endpoint. Its schema description tells the
actor to copy values from data-tool outputs without changing capitalization,
spelling, punctuation, whitespace, numeric precision, or date/time formatting
unless the user explicitly asked for that transformation. This is part of the
API contract, not a task-solving hint.

The final natural-language answer to the customer is downstream of this
terminal state and is not part of exact-match reward.
