# RL Task Foundry

RL Task Foundry is an experimental generator for database-grounded RL task
datasets. The goal is not to pass the current evaluator, but to produce
generalizable, verifiable, high-quality tasks from any sufficiently good DB.

## Start Here For AI Agents

Read these files before changing code:

1. `AGENTS.md`
2. `docs/experiments/first_principles.md`
3. `docs/experiments/rubric_dqs_v1.md`
4. `docs/experiments/registry.yaml`
5. `docs/experiments/tree.md`
6. `docs/spec.md`

The fixed constraints are the principles and evaluation metric. Everything
else, including specs, architecture, composer, solver, tools, prompts, pipeline,
and agent topology, is mutable.

## Current Improvement Protocol

- Treat Git commit history as the experiment graph.
- Start new experiments from the current baseline recorded in
  `docs/experiments/registry.yaml`.
- Use experiment branches named `exp/<node-id>-<slug>`.
- Record each experiment node in `docs/experiments/registry.yaml`.
- Do not tune to DB literals, known answers, row IDs, or observed bad-draft
  token strings.
- Only use precision-100 structural rejectors for hard runtime rejection.
- Keep low-quality accepted data as a hard failure.
- Preserve failed nodes instead of deleting them.

Production viability for v1:

- accepted sample count: 3
- productive loop budget: 900 seconds
- provider issue trials are excluded from the average and reported separately
- DB startup, pool creation, and schema/profile warm-up are excluded
- quality rejects count as productive loop cost

If accepted 3 are not produced within 900 productive seconds, the method is a
failed experiment node for production viability.

## Setup

Python 3.14 is required.

```bash
uv sync
cp .env.example .env
```

Fill only the provider keys and database DSNs you need in `.env`. Do not commit
real secrets. The default YAML configs expect local PostgreSQL databases.

Useful checks:

```bash
uv run pytest -q
uv run rl-task-foundry validate-config
```

Useful trial entrypoints:

```bash
uv run rl-task-foundry run-real-db-trial pagila artifacts/trials/pagila_001
uv run rl-task-foundry harvest pagila artifacts/harvest/pagila_001
```

`harvest` defaults to the v1 production check: accepted 3 within 15 productive
minutes.

## Artifact Policy

`artifacts/` is ignored by Git and is local runtime output.

Durable accepted data lives in the task registry under `artifacts/tasks`,
`artifacts/databases`, and `artifacts/task_registry.db`. Use `export-bundle` to
materialize serving bundles from that registry.

Per-trial debug output is for analysis only:

- `phase_monitors.jsonl`: phase-level state transitions
- `trial_events.jsonl`: unified composer/solver/runner event timeline
- `reasoning_content.jsonl`: optional provider-visible reasoning sidecar
- `traces/`: SDK/tool/session traces

Single `run-real-db-trial` runs export a self-contained trial bundle by
default, because they are often used for demos and inspection. `harvest` does
not export per-trial bundles by default; accepted tasks are already durable in
the registry, and repeated bundle copies make large experiments noisy. Pass
`--export-trial-bundles` only when debugging a specific harvest run.
