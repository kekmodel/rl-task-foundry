# Reward and Task Runtime

## Label Truth

The canonical answer chosen during label construction is the source of truth.

- the label is selected from real database evidence
- uniqueness and tie-breaks are resolved at label time
- runtime scoring does not depend on generated verifier code

## Reward Function

Production reward is binary exact match.

Flow:

1. actor calls `submit_result(answer_text)`
2. server parses JSON
3. server canonicalizes against the output schema
4. server compares with the stored canonical answer
5. reward is `1` for exact match and `0` otherwise

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
