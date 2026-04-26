# Pipeline Lifecycle And State Boundaries

This document is the operational map for the end-to-end synthesis pipeline.
Read it before changing synthesis feedback, submit validation, solver rollout,
trial retry, harvest behavior, or registry acceptance.

The key invariant is simple: generation is disposable; accepted task bundles are
the only durable product. A composer conversation may be repaired only for
contract-level feedback. If solver rollout says a draft is not in the desired
pass-rate band, that trial is discarded unless the gate says the draft is too
easy and explicitly asks the same composer to make the same answer harder.

## Mental Model

```mermaid
flowchart TD
  DB["Read-only PostgreSQL DB"]
  SDB["SynthesisDb\nschema graph, snapshot, profile, anchor candidates"]
  RT["SynthesisAgentRuntime\none composer conversation"]
  COMP["Composer agent\nsystem policy + user run context"]
  TOOLS["Composer tools\nschema_map/sample/profile/neighborhood/query"]
  SUB["submit_draft\nschema + exact evidence checks"]
  DRAFT["SynthesisTaskDraft\nrendered user prompt + canonical label"]
  SOLVER["Solver rollout\nno system prompt, atomic tools, submit_result"]
  GATE["Quality gate\nexact reward pass-rate + CI band"]
  TRIAL["RealDbTrialRunner\ncommit/export or synthesis_failed"]
  HARVEST["HarvestRunner\nfresh independent trials until target/stall"]
  REG["Task registry + bundle export"]

  DB --> SDB
  SDB --> RT
  RT --> COMP
  COMP --> TOOLS
  TOOLS --> SUB
  SUB -- "feedback-only contract repair" --> COMP
  SUB -- "contract-valid draft" --> DRAFT
  DRAFT --> SOLVER
  SOLVER --> GATE
  GATE -- "accept" --> REG
  GATE -- "too_easy / high inconclusive" --> COMP
  GATE -- "too_hard / low inconclusive / terminal" --> TRIAL
  TRIAL -- "accepted or duplicate" --> REG
  TRIAL -- "failed trial summary" --> HARVEST
  HARVEST -- "new trial, new controller, new conversation" --> RT
```

## Code Ownership Map

- `src/rl_task_foundry/synthesis/synthesis_db.py`
  Owns per-DB cached context: schema graph, schema snapshot, data profile, DB
  pools, and optional random anchor candidates.
- `src/rl_task_foundry/synthesis/runtime.py`
  Owns one synthesis conversation at a time, builds composer tools, builds
  `SynthesisTaskDraft`, maps submit records to generation outcomes, and raises
  `SynthesisArtifactGenerationError` when no draft is accepted.
- `src/rl_task_foundry/synthesis/backend_openai_agents.py`
  Runs the composer agent. It finalizes only on `Accepted`, budget exhaustion,
  or controller terminal rejection.
- `src/rl_task_foundry/synthesis/submit_draft_tool.py`
  Owns `submit_draft` schema, exact evidence validation, retry feedback, and
  the bridge from contract-valid drafts into solver rollout.
- `src/rl_task_foundry/pipeline/solver_orchestrator.py`
  Owns solver runs, infra-failure exclusion, top-up attempts, exact reward
  counting, early stop, and pass-rate quality-gate classification.
- `src/rl_task_foundry/solver/backend_openai_agents.py`
  Runs the actor/solver with no system prompt, the generated atomic tools, and a
  task-specific strict `submit_result` schema.
- `src/rl_task_foundry/synthesis/real_db_trial.py`
  Owns one single-shot trial: create debug roots, run synthesis, commit/export
  accepted drafts, or return `synthesis_failed`.
- `src/rl_task_foundry/synthesis/harvest.py`
  Owns repeated independent trials until the target committed count is reached
  or the run stalls.

## Stage 1: Per-DB Context

`SynthesisDb` is long-lived per `db_id` and may be reused across many trials.
It is not the composer conversation.

It provides:

- introspected `SchemaGraph`
- materialized `SchemaSnapshot` for composer/solver tools and bundle export
- `DataProfile`
- optional random anchor candidate pool
- read-only DB connections through shared pools

Anchor candidates are environment randomization, not answer hints. They exist
to prevent stateless composer runs from repeatedly starting at the first or
smallest id. Composer may ignore them. Final label evidence still must come from
live tools and the latest canonical `query`.

## Stage 2: One Composer Conversation

`SynthesisAgentRuntime.synthesize_environment_draft()` creates exactly one
`SubmitDraftController` and one composer conversation for a trial. It gathers
the current DB context, builds the composer tools, instruments tool calls into
the controller, and calls the synthesis backend once.

Prompt layering:

- Composer system instructions are durable role-local policy.
- Composer user input is current-run context: language, domain/scenario,
  schema/profile/affordance map, tool surface, optional anchor candidates, and
  local examples.
- Composer should not be told actor, solver, pass-rate, training, registry, or
  dataset internals.

Composer tool calls are mirrored into `SubmitDraftController` by
`build_instrumented_composer_tools`. This telemetry is the evidence source for
grounding checks, latest-query checks, and trace logging.

## Stage 3: submit_draft State Machine

`submit_draft` has two different jobs:

1. Repair exact contract mistakes inside the same composer conversation.
2. When the draft is contract-valid, run solver rollout and classify quality.

Feedback-only validation keeps the same composer conversation alive. Examples:

- missing new grounded observation
- invalid or missing `entity`
- blank or ungrounded label strings
- answer-contract phrase absent from the user request
- no latest successful `query`
- label not exactly equal to the latest query result
- answer-contract predicates/order clauses absent from latest query evidence
- label directly exposing fields explicitly marked `internal` or `blocked`
- after too-easy feedback, no reward-visible label change

These are feedback-only because the same composer can make another tool call,
repair the contract, and resubmit within the same authoring episode.

Non-feedback rejection consumes a submission attempt. If the submit budget is
exhausted, the composer backend finalizes and the trial fails.

Solver-backed quality outcomes have special state semantics:

| Quality outcome | Same composer continues? | Meaning |
| --- | --- | --- |
| `accept` | No | Draft is stored on the controller and becomes a task bundle. |
| `reject_too_easy` | Yes | Same answer target must be strengthened; composer gets feedback. |
| `calibration_inconclusive` with point estimate above band | Yes | Treated as "still too direct"; same target should be strengthened. |
| `reject_too_hard` | No | Terminal discard. The current conversation is over. |
| `calibration_inconclusive` with point estimate not above band | No | Terminal discard as not clearly reachable enough. |

The name `reject_too_hard` is a quality-gate bucket, not a proof of the cause.
It can mean overconstrained, actor-unreachable, tool-trace-unfriendly,
ambiguous/non-unique for exact match, or otherwise low-quality. That is
acceptable: all of those are reasons to discard the trial.

## Stage 4: Draft Materialization

When `submit_draft` has a contract-valid payload, runtime builds a
`SynthesisTaskDraft`:

- `TaskContract.question` is the customer-facing request body.
- `instance_parameters` are the hidden `entity` handle.
- `canonical_answer_json` is the exact label JSON.
- `output_schema` is inferred directly from the canonical label.
- list labels infer ordered list schemas with exact length.
- `rendered_user_prompt` prepends the hidden `<entity>` block to the request.

The label is the actor-visible `submit_result` object, not final customer prose.
The final natural-language answer is downstream of the verifier boundary.

## Stage 5: Solver Rollout

The solver/actor receives:

- no system prompt
- the rendered user prompt with hidden entity block
- generated atomic DB tools
- a task-specific strict `submit_result` tool

The solver must use tools and terminate by calling `submit_result`. The
`submit_result` schema is inferred from the canonical label and carries exact
copy-value descriptions. Missing or invalid submit calls are actor/runtime
outcomes and count as failures unless they are clearly infrastructural.

`SolverOrchestrator` runs batches until it reaches the target number of
evaluable solver runs or an exact early-stop decision. Current development
configuration is:

- pass-rate band `[0.5, 0.9]`
- `max_solver_runs = 20`
- `solver_batch_size = 4`
- `ci_alpha = 0.1`

Infrastructure failures are excluded only when they are clearly provider/runtime
infrastructure failures, such as rate limits, timeouts, API connection errors,
auth errors, and similar transport/service failures. The orchestrator schedules
replacement attempts up to a finite budget.

These count in the denominator:

- wrong exact answer
- schema mismatch
- invalid submit
- missing submit
- max-turn termination
- unknown SDK `UserError`
- model/tool protocol failures that returned a solver result

## Stage 6: Quality Gate

Reward is binary exact match after schema canonicalization. The quality gate
uses exact Clopper-Pearson binomial bounds for early-stop decisions and stores
the two-sided interval as quality metadata.

The gate is intentionally allowed to reject both difficult and low-quality
drafts. Solver pass-rate is the statistical detector for gray-zone failures that
hard validation should not guess: poor naturalness, under-specified ordering,
multi-answer requests, awkward trace surfaces, or tasks that actors do not
reliably solve with the given tools.

Do not replace that statistical role with heuristic validators. Hard validation
is only for 100%-precision contract violations, and even then the state
transition must be correct.

## Stage 7: Trial Boundary

`RealDbTrialRunner` is a single-shot trial wrapper.

On accepted draft:

1. commit the draft to the task registry
2. classify duplicate vs committed
3. export the bundle
4. return an accepted or duplicate trial summary

On `SynthesisArtifactGenerationError`, provider failure, provider unavailable,
or runtime failure:

1. log phase-monitor diagnostics
2. return `RealDbTrialStatus.SYNTHESIS_FAILED`
3. do not continue the same composer conversation

A failed `run-real-db-trial` command is therefore one failed trial, not an
internal retry loop.

## Stage 8: Harvest Boundary

`HarvestRunner` is the retry loop that turns disposable generation into a
target number of accepted bundles.

It runs many independent single-shot trials, optionally in parallel, until:

- `target_committed` accepted tasks are committed, or
- no new commit lands before `stall_timeout_seconds`.

Each trial gets a fresh `RealDbTrialRunner`, fresh `SubmitDraftController`, and
fresh composer conversation. Shared DB pools, solver orchestrator, registry,
exporter, and `SynthesisDb` may be reused for efficiency, but conversation state
does not cross trial boundaries.

Failures such as `too_hard`, validation exhaustion, provider errors, or
duplicates are discarded attempts. Harvest simply starts another trial.

## Change Checklist

Before changing validation, feedback, prompts, tool descriptions, solver
rollout, or harvest behavior, answer these questions in the PR or tuning log:

1. Which state owns this problem: schema, tool description, prompt, feedback,
   terminal trial discard, solver statistics, registry, or offline review?
2. If this rule fires, should the same composer conversation continue?
3. If not, do not implement it as feedback-only `submit_draft` repair.
4. Can the runtime prove the violation with 100% precision from schema,
   explicit config, latest query evidence, or exact reward-visible values?
5. If precision is below 100%, leave it to prompts/tool schemas/examples,
   qualitative trace review, or solver pass-rate.
6. Does the rule rely on table-name, column-name, tool-name, or generated-text
   tokens? If yes, it violates the no-semantic-token-heuristics rule.
7. Could this rule hide a valid task shape in an arbitrary good DB? If yes, it
   must not be a hard validator.
8. Does the change leak composer, solver, actor, pass-rate, training, or
   registry internals across role boundaries?

## Practical Decision Table

| Situation | Correct layer |
| --- | --- |
| Tool argument shape is wrong | JSON schema / strict schema / concise field description |
| Label is not copied from latest query | `submit_draft` feedback-only validation |
| Label exposes explicit `internal` or `blocked` source field | `submit_draft` feedback-only validation |
| Composer produced a too-easy draft | Same conversation feedback; preserve answer kind/target and strengthen |
| Actor pass-rate is too low | Terminal trial discard; harvest starts fresh |
| Actor pass-rate is too high | Same conversation feedback only when gate classifies too-easy/high inconclusive |
| Task may be ambiguous, awkward, or low-quality but not exactly provable | Solver pass-rate, trace review, prompt/schema tuning |
| Provider rate limit or transport failure | Exclude as infra and top up, within attempt budget |
| Unknown model/tool protocol failure | Count as evaluable failure unless explicitly classified as infra |
| Duplicate accepted task | Registry duplicate; not a composer retry |

## Anti-Patterns

- Adding a feedback validator just because a condition is detectable, without
  checking whether that condition should terminate the trial.
- Treating `reject_too_hard` as only "too difficult". It also carries other
  low-pass quality failures.
- Letting hard validation steal the statistical job of solver rollout.
- Repairing terminal low-quality drafts inside the same composer conversation.
- Adding semantic token heuristics to "help" DB adaptation.
- Moving current-run DB facts into composer system instructions.
- Giving the solver a system prompt or composer/quality-gate internals.
