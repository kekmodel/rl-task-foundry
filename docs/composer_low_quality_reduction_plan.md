# Composer Low-Quality Reduction Plan

This document is an experiment plan, not a normative spec. Normative boundaries
remain in `docs/spec/foundation.md` and `docs/spec/pipeline-lifecycle.md`.

## Goal

Reduce low-quality composer drafts without weakening the solver signal.

Low-quality means the submitted task cannot be answered from the rendered user
request, hidden entity, database data, and solver tool surface because the row
set, ordering, source path, output mapping, or evidence path is under-specified,
unstable, or unavailable.

The goal is not to maximize raw accept rate. A difficult but tool-solvable task
should survive qualitative review even when current solver pass rate is low.
The main risk is low-quality accepted data.

## Non-Negotiable Constraints

- No DB literal, generated-text, table-name, column-name, or token-containment
  heuristics may be used as validator evidence.
- Hard validators are only for facts the runtime can prove with 100% precision
  from schema metadata, explicit config, latest query evidence, or exact
  reward-visible values.
- Durable policy lives in the composer system prompt first. Feedback only points
  back to an existing policy and current failure evidence.
- Tool schemas/descriptions own local callable contracts. They must not become
  a second broad policy source.
- Normal generation does not receive an external topic target. The composer
  submits a topic that summarizes the final grounded draft it actually built.
- A subagent may advise or produce a structured intermediate artifact, but an
  LLM subagent must not become the authoritative accept/reject judge.

## Current Failure Classes

1. **Output drift**
   The final request asks for one surface, while selected label fields include
   extra fields or a different representation. Recent examples include
   medication tasks returning route/status/stop time when the request no longer
   asks for those fields.

2. **Order and tie-break drift**
   The query uses a tie-break that determines row membership or order, but the
   request and answer contract do not clearly expose that tie-break. Some cases
   are caught by existing order validators; remaining cases are often semantic
   wording drift rather than a 100%-precision structural violation.

3. **Too-easy recovery jump**
   After `reject_too_easy`, the composer sometimes changes topic, path, row set,
   or label meaning instead of making one incremental strengthening to the same
   task.

4. **Solver-inaccessible source surfaces**
   The composer can author a canonical `query`, but the solver cannot revisit
   the same row surface with atomic tools. No-primary-key row labels are now
   structurally guarded; other reachability gaps may remain.

5. **Role overload**
   One composer conversation currently owns DB exploration, entity choice,
   surface choice, topic, user request, query, label, answer contract, feedback
   recovery, and difficulty adjustment. Most low-quality failures come from
   losing consistency across those responsibilities, not from lack of raw DB
   access.

## Layering Strategy

The system should improve in this order:

1. **Structured contract inside the existing flow**
   Add machine-readable bindings that make the composer state why each output
   field and ordering key belongs to the user-facing request.

2. **Better source-surface metadata**
   Enrich composer tool outputs with structural facts about solver reachability
   and row stability. Use hard validation only when the violation is exact.

3. **Small DB-neutral examples**
   Add examples only for repeatedly observed cross-DB failure patterns. Examples
   must be abstract and must not mention real table names, DB literals, or
   domain-specific values.

4. **Planning tool**
   If the existing composer still chooses unstable surfaces, add a tool that
   returns solver-compatible task surface candidates. This tool should narrow
   search space, not generate final tasks.

5. **Subagent decomposition**
   Split responsibilities only after the intermediate artifacts are structured
   enough to pass safely between roles. Subagents are producers and auditors,
   not quality gates.

## Candidate Architecture

### Existing Flow With Stronger Contracts

Keep one composer conversation and `submit_draft`, but add explicit bindings.

Proposed `answer_contract` extension:

```json
{
  "kind": "list",
  "answer_phrase": "<exact user_request phrase>",
  "constraint_phrases": ["<exact user_request phrase>"],
  "limit_phrase": "<exact user_request phrase or null>",
  "output_bindings": [
    {
      "field": "result_field",
      "requested_by_phrase": "<exact user_request phrase>"
    }
  ],
  "order_bindings": [
    {
      "output_or_ref": "ordered_field",
      "direction": "desc",
      "requested_by_phrase": "<exact user_request phrase>"
    }
  ]
}
```

Validator scope:

- precise: every submitted label field has an `output_binding`
- precise: each `requested_by_phrase` is an exact substring of `user_request`
- precise: each binding field exists in label/latest query output
- not precise: whether the phrase semantically means the field

This does not solve semantics by itself. It forces the composer to expose its
claim in structured form, making drift easier to catch by prompt feedback,
qualitative audit, and future exact checks.

### Planning Tool

If binding contracts do not reduce surface drift enough, add a
`plan_task_surface` composer tool.

The tool should return structural candidates, not final prose:

```json
{
  "anchor": {"table": "T_anchor", "entity": {"pk": "..."}},
  "surface_path": ["T_anchor", "T_child", "T_display"],
  "record_surface": {
    "table": "T_child",
    "has_primary_key": true,
    "solver_revisitable": true
  },
  "candidate_outputs": [
    {"field": "event_time", "visible": true, "source_role": "row field"},
    {"field": "display_name", "visible": true, "source_role": "related display"}
  ],
  "candidate_orders": [
    {
      "field": "event_time",
      "direction": "desc",
      "requires_user_phrase": true
    }
  ],
  "structural_risks": ["same-order ties need an exposed tie-break"]
}
```

Correct use:

- orient the composer toward primary-key-backed, solver-revisitable surfaces
- identify candidate visible outputs and order fields
- expose structural risks early

Incorrect use:

- writing the final task for the composer
- deciding quality with an LLM-like judgment
- hiding policy or examples inside tool output

### Subagent Decomposition

Subagents can help only if each role has a narrow output contract.

Recommended roles:

- **Surface Scout**
  Produces solver-compatible surface candidates from schema/profile/neighborhood
  evidence. It does not write the user request, topic, label, or answer
  contract.

- **Draft Writer**
  Converts one chosen surface candidate into topic, user request, query, label,
  and answer contract. This role remains responsible for final coherence.

- **Contract Auditor**
  Advisory only. It flags likely output drift, order drift, source-surface
  mismatch, and topic drift. It cannot accept or reject. Deterministic
  validators and solver rollout remain authoritative.

- **Difficulty Adjuster**
  Used only after too-easy feedback. It receives the prior accepted-like draft
  state and proposes one incremental strengthening that preserves answer kind,
  entity, row set, query path, and output source meanings.

Bad subagent splits:

- a Quality Judge that accepts/rejects drafts
- independently exploring agents that each produce unrelated drafts
- subagents with their own durable policy wording
- a solver simulator embedded in composer authoring

## Experiment Sequence

Each experiment must use the mandatory qualitative audit from
`docs/runbook.md`: accepted tasks are classified as clean, borderline,
low-quality accepted, or inconclusive; rejected tasks are classified as
hard-good, low-quality, infra/provider failure, or inconclusive.

### Experiment 1: Binding Contract Design, No Enforcement

Change:

- Add optional `output_bindings` and `order_bindings` to `answer_contract`.
- Update prompt/tool schema descriptions to ask for them when possible.
- Do not reject missing bindings yet.
- Log binding coverage in diagnostics or phase monitor.

Purpose:

- Measure whether the composer can produce useful structured bindings without
  destabilizing generation.
- Learn exact binding shapes before enforcing them.

Success criteria:

- no role leak or duplicated policy
- no DB-specific examples
- no low-quality accepted increase versus recent MIMIC baseline
- at least 60% of submitted list drafts include plausible bindings

Stop criteria:

- bindings are mostly blank, copied mechanically, or unrelated to query outputs
- composer spends too many turns repairing binding format instead of grounding
  the task

### Experiment 2: Binding Feedback, Still Not Terminal

Change:

- Make missing binding evidence feedback-only when the runtime can prove the
  field/order key exists but no structured binding was supplied.
- Keep semantic phrase-to-field judgment out of validation.

Purpose:

- Reduce output/order drift while preserving same-conversation repair.

Success criteria:

- low-quality accepted remains zero in five-trial Kimi batch
- fewer final drafts contain unrequested visible output fields
- hard-good rejected tasks are not converted into low-quality feedback loops

Stop criteria:

- feedback causes repeated topic/path resets
- too many MaxTurnsExceeded failures from binding repair

### Experiment 3: Difficulty-Up State Lock Diagnostics

Change:

- Enrich too-easy feedback diagnostics with previous topic, entity, query path,
  row set signature, output source meanings, and allowed delta.
- Do not add a new policy; point back to Difficulty-Up Policy.

Purpose:

- Reduce non-incremental jumps after `reject_too_easy`.

Success criteria:

- too-easy retries preserve topic/path/entity more often in smoke batches
- no new low-quality accepted tasks

Stop criteria:

- diagnostics are too verbose for model behavior or token budget
- composer still jumps despite clear structured state

### Experiment 4: Source Reachability Metadata

Change:

- Add structural reachability metadata to composer results where exact:
  primary-key-backed row source, solver-revisitable record surface, and
  unsupported relationship reasons.
- Validate only exact no-access cases.

Purpose:

- Prevent solver-inaccessible source surfaces beyond the current no-PK guard.

Success criteria:

- no false positive on derived aggregates or PK-backed related displays
- fewer low-quality rejected drafts caused by unreachable source paths

Stop criteria:

- reachability status depends on semantic guesses rather than tool capability

### Experiment 5: `plan_task_surface` Tool Prototype

Change:

- Add a composer-only planning tool that returns solver-compatible candidate
  surfaces and structural risks.
- The final canonical answer still comes from `query`.

Purpose:

- Reduce role overload before adding subagents.

Success criteria:

- improved clean accepted count or reduced low-quality rejected count in
  MIMIC demo without topic hints
- no collapse into repetitive templates
- accepted tasks remain natural and DB-swappable

Stop criteria:

- tool becomes a hidden task generator
- trace diversity collapses
- composer ignores the tool or treats candidates as mandatory topics

### Experiment 6: Surface Scout Subagent

Change:

- Split only the surface planning step into a subagent or internal planning
  backend. It produces the same structured candidate format as
  `plan_task_surface`.
- Draft Writer remains the final composer conversation.

Purpose:

- Reduce composer overload while avoiding policy split.

Success criteria:

- lower low-quality rejected count caused by bad surface choice
- no increase in topic/request/query drift
- clear audit trail from scout candidate to final draft

Stop criteria:

- subagent introduces conflicting policy
- handoff context becomes larger than the current single-agent prompt savings
- final composer treats scout output as an external topic target

### Experiment 7: Contract Auditor Subagent

Change:

- Add an advisory audit pass before `submit_draft`, or as a retry assistant
  after feedback. The auditor writes findings only; runtime validators remain
  authoritative.

Purpose:

- Catch likely drift cases that are not precise enough for hard validation.

Success criteria:

- fewer low-quality accepted/borderline tasks
- findings are concise and actionable
- no duplicate durable policy text

Stop criteria:

- auditor overrules valid hard-good tasks
- auditor output becomes the main instruction source
- cost/latency is not justified by quality improvement

## Measurement Protocol

For every experiment:

- run targeted unit tests for the changed contract/tool/prompt surface
- run `ruff` for touched files
- run a five-trial no-topic MIMIC demo Kimi batch unless the change is purely
  documentation-only
- record raw accept count, clean accepted count, borderline accepted count,
  low-quality accepted count, hard-good rejected count, low-quality rejected
  count, infra failures, and inconclusive cases
- compare against the most relevant recent baseline, not against an arbitrary
  historical best

Current baseline to compare against:

- `artifacts/trial_20260428_mimiciv_demo_kimi_no_topic_batch5_01`
- `artifacts/trial_20260428_mimiciv_demo_no_pk_guard_kimi_no_topic_batch5_01`

## Decision Rules

- If an experiment reduces low-quality accepted but lowers raw accept rate, keep
  investigating; raw accept is secondary.
- If an experiment raises clean accepted count while also raising low-quality
  accepted count, treat it as a regression.
- If an experiment mainly increases low-quality rejected count, decide whether
  it is exposing existing noise or making the composer worse by qualitative
  audit.
- If a proposed validator cannot guarantee precision 100%, move it to prompt,
  schema shape, advisory audit, or qualitative review.
- If a proposed subagent needs its own durable policy, do not add it until the
  policy is promoted to the main composer system prompt or a tool-local schema
  contract.

## First Implementation Target

Start with Experiment 1.

Reason:

- It targets the currently repeated output/order drift directly.
- It does not enforce semantics prematurely.
- It creates structured evidence needed before deciding whether stricter
  feedback, a planning tool, or subagents are justified.
- It is reversible if the composer cannot use the binding shape reliably.

Initial implementation should keep `output_bindings` and `order_bindings`
optional, add diagnostics only, and preserve the existing accepted/rejected
state machine.
