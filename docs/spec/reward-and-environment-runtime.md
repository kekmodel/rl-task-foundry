# Reward and Task Runtime

## Label Truth

The canonical `submit_result` chosen during label construction is the source of
truth. It is the structured result object the actor must submit after using the
tools, not the final natural-language customer answer.

- the label is selected from real database evidence
- uniqueness and tie-breaks are resolved at label time
- runtime scoring does not depend on generated verifier code

Ambiguous or multi-answer requests are rejected at the authoring standard, not
rewarded by looser matching. Exact-match reward is only a high-quality signal
when the request has one correct structured result. That structured result may
contain a list, as long as the list's membership, order, limit, and tie-breaks
are part of the canonical answer.

Automatically inferred list output schemas are ordered by default. A solver that
returns the right list members in a different order is not treated as an exact
match unless a future task contract explicitly declares unordered semantics.

String values are exact too. If a data tool returns `ANGELINA`, submitting
`Angelina` is a mismatch. The submit endpoint should communicate this as schema
contract: copy values exactly from data-tool outputs unless the user requested a
transformation.

## Reward Function

Production reward is binary exact match.

Flow:

1. actor calls the task-specific `submit_result(...)` tool
2. runtime extracts the submitted structured result from the tool arguments
3. server canonicalizes against the output schema
4. server compares with the stored canonical answer
5. reward is `1` for exact match and `0` otherwise

## Solver Rollout Metrics

Calibration logs distinguish rollout volume from the pass-rate denominator.

- `planned_solver_runs`: solver calls intended for the quality gate
- `completed_solver_runs` / `total_solver_runs`: solver calls that returned a runtime result
- `evaluable_solver_runs`: solver calls counted in the pass-rate denominator
- `failed_solver_runs`: completed calls excluded because they are clearly infrastructural
- `matched_solver_runs`: exact-match successes among evaluable calls

Only high-confidence provider/runtime infrastructure failures, such as rate
limits, API timeouts, connection errors, authentication failures, and malformed
API requests, are excluded from `evaluable_solver_runs`. Unknown SDK failures,
`UserError`, max-turn termination, invalid submit calls, missing submit calls,
and wrong answers are actor/runtime outcomes and count as reward `0`.

## Environment Server

The environment server is a stateless Gym-style runtime consumer for synthesized task bundles.

### Infrastructure

Recommended Docker Compose components:

| Component | Role |
| --- | --- |
| PostgreSQL | serves the database and executes atomic tool SQL in read-only mode |
| Redis | holds episode state with TTL |
| `env-server` | stateless REST server, horizontally scalable |

### API Endpoints

- `GET /health`
- `POST /reset`
  - request: `{task_id}`
  - response: `{episode_id, observation, info}`
- `POST /step`
  - request: `{episode_id, tool_name, params}`
  - response: `{observation, reward, done, info}`

`submit_result` is a tool call, not a separate endpoint.

### Gym-Style Step Semantics

| Situation | Observation | Reward | Done |
| --- | --- | --- | --- |
| normal tool call success | tool execution result | `0` | `False` |
| `submit_result` | `-` | `0` or `1` | `True` |
| turn limit exceeded | `"Episode terminated: turn limit reached"` | `0` | `True` |
| `invalid_tool_name` | `"Error: unknown tool '{name}'"` | `0` | `False` |
| `invalid_params` | `"Error: invalid parameters"` | `0` | `False` |

### Server Errors

| Situation | HTTP | Action |
| --- | --- | --- |
| `sql_error` | `500` | discard the episode, mark the task unhealthy |
| Redis outage | `500` | discard the episode |
| `invalid_episode` | `404` | discard the episode |
| DB connection failure | `503` | discard the episode |

Malformed requests are handled as HTTP `4xx`.

Training infrastructure should terminate the session on any HTTP `4xx` or `5xx` and discard that trajectory.

### Episode State

Redis state shape:

```text
key:   episode:{episode_id}
value: {task_id, canonical_answer, output_schema, max_turns, current_turn}
ttl:   configurable
```

The canonical answer remains inside the server trust boundary.

### Step Flow

1. load episode state from Redis
2. increment `current_turn`
3. terminate with `reward=0` if the turn limit is exceeded
4. if the tool is `submit_result`, compute exact-match reward and terminate
5. if the tool name is unknown, return an error observation
6. otherwise execute SQL and return the observation with `reward=0`
7. if SQL execution fails, return server error and discard the episode

## Bundle Self-Containment

The serving bundle must be self-contained.

The environment server should not depend on synthesis-time memory, retry history, or generated verifier code.
