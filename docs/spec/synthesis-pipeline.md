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

Exploration must be adaptive to the current database. The synthesis agent may
use the current schema graph, relationship fanout, profiles, samples, and local
example packs as evidence. It must not rely on hard-coded assumptions from a
particular sample database.

The schema map is the composer's exploration map. It is used to identify
DB-native topics, reachable relationship paths, hub/bridge structure, readable
surfaces, and plausible strengthening directions. The map is not proof by
itself: profiles, samples, neighborhoods, and final queries decide whether a
candidate path is grounded, readable, nontrivial, and suitable for a unique
label.

### Non-Negotiable Adaptation Boundary

DB adaptation must be structural and evidence-based. It must never interpret
names as semantic evidence.

The synthesis pipeline must not use token, substring, prefix/suffix, regex,
case, word-splitting, or name-list logic over table names, column names, tool
names, generated user text, or DB literal values rendered into text to decide:

- topic quality or topic family
- anchor quality
- table role
- visible/readable/user-facing status
- privacy or sensitivity
- metric/filter/time/business affordances beyond DB type metadata
- naturalness or customer-facing validity
- hard accept/reject decisions
- retry feedback branch selection
- diversity

The only allowed signals for these decisions are explicit configuration,
database structure, declared type metadata, profiler statistics, sampled rows,
relationship counts, exact query evidence, and exact schema/contract checks.
Literal occurrence or absence of an observed DB value in `user_request` or label
text is not exact schema/contract evidence, even when that value came from the
composer's own latest query rather than from a hand-written list. Query evidence
may prove exact structured facts such as label equality, handle equality,
column visibility, row order, or answer-distinguishable ties; it must not be
converted into natural-language value-containment validation.
This boundary does not ban registry near-duplicate detection over generated task
text; MinHash tokenization/shingling is valid there because deduplication is
string-surface similarity, not schema or task-validity inference.

This boundary exists because the same code must run across arbitrary databases.
Names are local conventions, often multilingual, abbreviated, misleading,
legacy, or absent. Treating name tokens as domain facts makes the system
sample-DB-biased instead of DB-adaptive, destroys topic diversity, introduces
false rejects, and weakens RLVR by replacing solver pass-rate evidence with
hand-written priors.

Rule-based DB adaptation is allowed and expected at the context layer. The
runtime may derive affordance summaries from introspection and live profiling,
including structural table classes, readable surfaces, path fanout, filterable
columns, and candidate list/aggregate surfaces. These summaries should help the
composer see what the current DB makes possible. They must not be treated as a
task oracle, verifier, or fixed difficulty ladder; every accepted label still
needs live evidence and a canonical query.

The rule-based layer exists to make composer authoring faster and better. It may
organize context, highlight affordances, and reduce avoidable exploration, but it
does not own the authoring decision. Composer remains responsible for using the
context and tools, selecting a feasible task shape, proving the label, and
adapting to feedback.

The first implemented context artifact is `DB Affordance Map`. It contains
rule-derived table cards and path cards. Table cards summarize structural
classes, readable fields, filters, metrics, time columns, and whether a table is
anchorable by graph structure. Path cards summarize anchor-to-child relationship
opportunities, fanout, readable surfaces, filters, metrics, and supported task
shapes such as ordered lists, cardinality/counts, numeric aggregates, timelines,
or categorical filters.

The second context artifact is the random candidate anchor pool. It exists to
break the stateless composer's natural tendency to restart every episode from
the first or smallest primary-key value. The pool is rule-based and
database-agnostic: it scores all eligible primary-key tables from schema
metadata, samples actual rows, attaches visible preview fields and lightweight
relationship counts, and gives the composer optional starting entities. It is
not a task oracle or a forced topic. Composer may use one candidate, ignore all
of them, or inspect different rows through tools; accepted labels still require
live canonical query evidence. The runtime exposes this as an explicit
experiment knob: `synthesis.runtime.anchor_candidates_enabled` with
`anchor_candidate_limit`. Production trial configs keep this enabled by default
so stateless composer runs do not collapse back to the first visible primary-key
row.

Anchor sampling is evaluated by RLVR-start metrics, not by downstream composer
success alone. Candidate-level metrics score anti-degeneracy, visible preview
surface, positive relationship surface, and dead-anchor risk. Pool-level metrics
track mean and p10 candidate score, preferred-rate, preview-rate, positive
relation-rate, dead-anchor-rate, id-one-rate, table coverage, and normalized
table entropy. These metrics are diagnostics for improving the sampler
distribution; they are not verifier rules and must not reject generated task
drafts.

The sampler uses a single quality-diversity algorithm. Each table's schema-level
quality score comes from readable surfaces, incoming and outgoing relationships,
time/numeric affordances, and structural penalties. For each candidate slot, the
sampler draws from all eligible tables with weight:

```text
table_quality_score * structure_novelty_bonus * repeated_table_penalty
```

This balances high-quality starting points with cross-table diversity without
forcing low-quality structural tables into every episode. After a table is
chosen, the sampler tries a small number of random rows from that table and
keeps the row with the higher RLVR-start candidate score. This is part of the
sampling algorithm, not a post-hoc candidate filter.

In the composer input, the pool is rendered as `Candidate Starting Points`, not
`Starting Entity`. The wording must preserve three contracts:

- candidates are optional orientation context, not answer hints or required
  topics
- preview and relationship summaries are not final label evidence; final labels
  still come from data tools and a canonical `query(spec)`
- hidden primary-key and `row_id` values may be used in tool calls and
  `submit_draft.entity_json`, but must not appear in the customer-facing request

## Topic Selection

The agent selects a `topic` string for the candidate task bundle.

The topic is a soft orientation aid. The authoritative semantics are still set by the label.

## Label Construction

The agent produces the canonical `submit_result` first. This label is inside
the training/evaluation trust boundary: it is the structured result object that
the actor must submit after composing tool/API responses. It is not the final
natural-language answer shown to the customer.

The label must be unique for the generated request. If the request admits
multiple valid structured results, multiple tie-breaks, or partial answers, the
draft is not a good RLVR task even when one candidate label is grounded.
List-valued labels are valid when the entire list is canonical: membership,
order, limit, and tie-breaks are all fixed by the request and evidence. Composer
should narrow the subset, specify ordering/limits, or choose an exact aggregate
scope until only one structured result object is correct.

The automatically inferred output schema preserves list order. If the label is a
list, the order in the latest canonical query is part of the expected payload.

Current output includes:

- `canonical_answer_json`
- `anchor_entity`
- `label_signature` (sha256 of canonical answer JSON)

The runtime derives the output schema from the canonical answer automatically. Solver pass rate is the sole difficulty signal — there is no agent-supplied difficulty vector.

Solver pass rate is computed over evaluable solver attempts, not over every
planned API call. The monitor must still report planned, completed, evaluable,
failed, and matched counts separately. A solver run is excluded only when the
failure is clearly infrastructural. Unknown `UserError` failures count as
evaluable failed attempts, because otherwise a task can be falsely rejected as
too easy when only one of several solver calls actually submits.

Calibration decisions use exact one-sided Clopper-Pearson binomial bounds for
statistically decisive too-easy / too-hard rejection and early termination; the
default confidence level is 90% (`ci_alpha = 0.1`). The two-sided
Clopper-Pearson interval is still recorded as quality metadata. When the
observed point estimate falls outside the target band but the exact directional
bound is not decisive, the result is `calibration_inconclusive`, not a hard
difficulty rejection. Point-in-band drafts may still be accepted with the CI
recorded as quality metadata.

After a too-easy rejection, the next draft must change the reward-visible
canonical label, not only the wording or output field name. For single-field
scalar labels, if the answer operation is unchanged and the scalar value is
canonically identical to the previous submission, the draft is rejected as not
strengthened. This is an exact verifier-safety rule, not a semantic difficulty
heuristic.

## Task Synthesis

Given the fixed label, the agent writes the user-facing task request that makes
that label the unique correct structured result.

This stage does not generate verifier or shadow code.

The request must be written for a customer or end user who does not know the
database schema. User-facing text should be grounded in visible business data:
names, titles, dates/times, amounts, categories, statuses, locations, and the
user's own visible account or reference. It must not expose table names, primary
keys, foreign keys, row/record jargon, internal IDs, or other schema-internal
language.

The generated task should feel like a real end-user data-service request, not a
database exercise. Good shapes include domain-adapted history lookups,
shortlists, status summaries, billing/usage summaries, schedules, eligibility
checks, or plan-like lists when those shapes are supported by the current
schema. These are structural shapes, not domain templates: the composer chooses
the concrete shape from observed tables, relationships, readable fields, and
data distributions.

Difficulty should grow from the current database's affordances, not from a
global ladder. A task may start from a minimum viable repeated unit and, after
specificity feedback, grow the same structure one natural step at a time
(`1 -> 2 -> 3 -> 5 -> 10` or eventually all matching items) when the current
database supports that shape. In other databases, item complexity, filters,
ordering, or another grounded dimension may be the feasible path.

Repeated payloads have two separate curriculum axes: the number of items and the
complexity of each item. Item complexity is not passive width. It is a new
grounded per-item requirement, relationship, visible related field, predicate,
or deterministic tie-break that changes what the actor must find for every
item. Display-only field additions are not enough by themselves.

Composer strengthening remains autonomous. Runtime feedback and context should
help the agent judge the next feasible direction; they should not force a
database-independent priority list.

Open-ended recommendation wording is only valid after it has been made
deterministic. The request must state the grounded filters, thresholds,
ordering, limit, and tie-breaks that make exactly one structured result object
correct.

The canonical answer field names do not need to be customer prose. They are
agent-visible `submit_result` keys, similar to an API response schema. Query
output aliases should be stable, deterministic result keys that match the
rendered submit schema. The final customer-facing answer is downstream of
`submit_result` and is not the verifier boundary.

## Prompt Policy

All system instructions, templates, and retry hints are written in English.

Only the generated user-facing task text follows `config.domain.language`.

The composer prompt is scoped to draft authoring only. It should not mention
synthetic datasets, RLVR, actors, solvers, pass rates, training, registry
operations, or downstream evaluation internals. The composer only needs to know
how to inspect the current database evidence, construct a canonical label,
write a customer-facing request, express the request-binding answer contract,
and call `submit_draft`.

### Absolute Composer System/User Boundary

This boundary is non-negotiable. It is the composer API contract, not a prompt
style guideline. A change that moves current-run DB context into `system`, or
uses `user` text to override durable role policy, is a design regression.

The synthesis backend has two LLM-facing prompt layers:

- `system` / agent instructions: built by `build_synthesis_agent_instructions`
- `user` / run input: built by `build_synthesis_input`

Use the system layer for durable, role-local policy that remains true across
every database and every generation run:

- composer role and stopping condition
- tool-use workflow invariants
- customer-facing request and exact-label principles
- validation/guidance hierarchy
- compact, DB-neutral good/bad patterns when they teach a universal behavior
- bans on cross-role leaks, sample-DB priors, and semantic token heuristics

Do not put run-specific facts in the system layer: current domain facts,
requested topic hints, schema maps, table/column lists, data profiles, anchor
candidates, local examples, provider-specific observations, or previous trial
outcomes.

Use the user layer for the current assignment packet:

- target language for generated user-facing text
- domain/scenario description from config
- requested topic or coverage hint, if any
- schema summary, DB Affordance Map, and data profile
- current tool-surface summary
- optional candidate starting points
- local example pack for the current database

User-layer instructions are allowed because the runtime is explicitly asking the
composer to draft one task for this DB. They should be treated as current-run
context and soft tasking, not as policy that overrides the system role contract.
If a rule belongs beside a callable argument, prefer tool schema or parameter
description over either prompt layer.

Forbidden boundary crossings:

- putting schema maps, affordance maps, data profiles, anchor candidates, local
  examples, requested topic hints, provider quirks, or trial observations in
  `system`
- putting role-isolation exceptions, RLVR/solver/pass-rate/training internals,
  sample-DB priors, semantic token heuristics, or hard-validator policy in
  `user`
- using either prompt layer when JSON Schema or parameter descriptions can state
  the callable contract directly

Common synthesis prompts and feedback are part of the shared product surface.
They must work across arbitrary introspectable PostgreSQL databases. They may
name structural operations (`query`, relationship traversal, filter, aggregate,
order, limit, answer contract) but must not depend on domain-specific table names
or business facts from pagila, postgres_air, or any other sample database.
`answer_contract` itself should remain minimal: answer shape plus exact
user-request phrases. Query table/column/operator evidence is derived from the
latest successful `query` result rather than retyped by the composer.
Composer query guidance must also preserve label meaning without hard-coding DB
names: selected fields are submitted label fields, not helper context; and when
one answer item combines facts from the same event or record, joins should
follow that event/record path instead of independently joining sibling child
sets from the same root.
This does not make the composer DB-blind: during a run, the composer should be
DB-aware through the current schema map, generated tool schemas, live samples,
profiles, neighborhoods, and query results.

Control priority:

1. Put exact, machine-checkable requirements into the tool schema.
2. Put short contract semantics into tool descriptions.
3. Use minimal prompts and generalized good/bad examples for cross-domain
   authoring patterns that schemas cannot express.
4. Use feedback to repair contract mistakes after a failed submission.
5. Use hard validation only for 100%-precision violations.
6. Let solver pass-rate statistically judge the remaining uncertainty.

This order is intentional. A validator that is not 100%-precision contaminates
the dataset by hiding valid task shapes; a prompt or schema hint may guide the
composer without pretending to prove semantic invalidity.

Minimal good/bad examples are allowed when they teach a cross-domain pattern
that prose rules alone do not reliably induce. They should show abstract
anti-patterns and desired patterns, such as avoiding raw internal ids in
customer-facing wording, not sample-database facts. Each example block should
stay short and be tied to the reason for the rule.

Per-database adaptation is allowed only through runtime-provided context:
schema summaries, live tool observations, data profiles, local examples, and
configured domain metadata. Local example packs may guide DB adaptation only
when they remain consistent with durable system policy, tool schemas, tool
descriptions, and 100%-precision validation boundaries.

## Prompt Constraints

Prompt inputs should stay LLM-friendly.

Keep:

- domain summary
- requested topic (optional — agent may infer from schema)
- compact schema orientation (hub/bridge tables, fanout edges, readable vs id-only)
- DB Affordance Map cards
- tool feedback

Do not dump internal runtime state, raw debug payloads, or redundant tool-definition JSON into the prompt.

DB Affordance Map cards are generated from the current introspected database,
not from per-database Python branches. They should be cross-checked on multiple
schemas before trial changes are trusted. Current invariants:

- table cards exclude partition children when the visible parent table exists
- table/card scoring and anchor sampling do not infer business roles from table
  or column-name tokens
- visible affordance columns are chosen structurally: user-visible, non-primary
  key, non-foreign key
- visibility labels come from explicit config/default metadata, not table or
  column-name tokens
- hard submit validation does not reject drafts by token patterns in generated
  user text
- hard submit validation does not reject drafts by DB literal occurrence or
  absence in generated user text or label text
- time affordances come from DB type metadata, not column-name tokens
- numeric metrics come from DB type/profile metadata, not semantic name filters
- path cards include a relation label with source and target join columns, so
  duplicate table-pair edges such as departure vs arrival remain distinct

## Retry Model

Generation retries are allowed, but retries are not the source of truth.

- See [Pipeline Lifecycle And State Boundaries](./pipeline-lifecycle.md) for
  the authoritative state-transition map. In particular, feedback-only
  `submit_draft` repair and terminal trial discard are different states; do not
  add retry feedback for failures that should discard the trial and let harvest
  start fresh.
- the label remains authoritative
- `submit_draft` feedback can request a harder retry
- retries must change latent task semantics, not just wording
- retry feedback must stay structural and database-neutral; it should point to
  evidence-visible axes such as preserving the answer kind and query output
  target, adding a grounded filter, tightening a filter pair, changing list
  cardinality, or preserving order, not to domain-specific fixes
- retry feedback must not reveal how the draft is evaluated. It should say what
  to change in the draft surface and contract, not mention solvers, actors,
  pass rates, training, or dataset construction.
