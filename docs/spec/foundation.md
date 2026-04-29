# Foundation

## Overview

RL Task Foundry turns a read-only PostgreSQL database into RLVR task bundles.

The system does not generate isolated lookup questions. Its unit of generation is a task bundle built on top of a database-level atomic tool bundle.

The product goal is database-swappable customer-agent RLVR data generation:
given a different read-only PostgreSQL database, the same project code should
introspect that database, adapt its tool surface and synthesis behavior to the
observed schema and rows, and produce customer-facing RLVR task bundles for that
domain. Adding a new database should not require domain-specific prompt forks,
hand-written task templates, custom verifier code, or per-database Python tool
implementations.

## Core Goals

- accept an arbitrary read-only PostgreSQL database
- inspect schema and live rows
- adapt prompts, feedback, and generated tool schemas from the current database
- synthesize label-first task bundles
- phrase tasks as customer-facing requests for users who do not know the database
- keep only task bundles that pass the solver pass-rate quality gate
- accumulate accepted task bundles in a durable registry

## Operating Principles

1. Hard to solve, easy to verify.
2. Label correctness and uniqueness matter more than throughput.
3. Quality gates are authoritative; generation is disposable.
4. Each task bundle belongs to exactly one database.
5. Reward is binary exact match only.
6. New databases can be added without hand-written task logic.
7. Common prompts and feedback must be database-neutral, while each composer
   run must be database-aware through the current schema, tool schemas, and
   live tool observations. Shared text may refer to structural concepts such as
   anchors, relationships, filters, ordering, aggregates, and answer shapes, but
   must not encode sample-database-specific tables, columns, or business
   assumptions.
8. The generated task surface is for a customer or end user of the domain, not
   a database operator. Internal tables, rows, primary keys, foreign keys,
   internal IDs, and schema jargon stay in hidden metadata and tools, not in the
   natural-language request.
9. Pipeline roles are isolated. Each agent receives only the information needed
   for its own job. Composer drafts grounded tasks; solver answers the rendered
   customer problem; runtime handles orchestration, quality measurement,
   registry, and export. No agent should be told another role's internals.
10. Runtime database lanes are read-only by default. Solver, composer/control,
    introspection, profiling, and smoke connections apply DB-level read-only
    session settings; fixture/setup DDL must use an explicit mutating lane.
11. The learning trajectory terminates at `submit_result`. The actor is a
    customer-center agent that receives a user request about its hidden entity,
    uses generated data tools to gather and compose the needed information, and
    submits a structured result object. The label is this agent-visible
    `submit_result` object, similar to an API response assembled from tool
    outputs; the final customer-facing natural-language answer is downstream of
    the exact-match verifier boundary.
    The hidden entity is intentional actor context, not something to remove:
    it represents the current customer/session/object that lets the agent look
    up the user's information. It is also not a decorative anchor. Composer
    should draft tasks whose label is scoped to that hidden entity or values
    derived from it; unrelated global reports with a hidden entity attached are
    invalid authoring targets. Because indirect scope can be legitimate, hard
    validation must not reject entity-scope issues unless the violation is
    provable with 100% precision.
12. Ambiguous multi-answer tasks are bad training signals. A user request must
    identify one unique structured result; list-valued labels are fine when the
    whole list is canonical, including membership, order, limit, and tie-breaks.
    If several result objects could reasonably satisfy the request, exact-match
    reward becomes less informative even when the stored label is grounded.
13. Hard validators must be conservative and only enforce 100%-precision
    contracts such as JSON shape, type compatibility, latest-query exact match,
    required provenance metadata, direct label exposure of fields explicitly
    marked `internal`/`blocked`, and reward-visible label changes. Structural
    signals such as PK/FK handle status may guide prompts and tool schemas, but
    must not become hard rejection rules unless they are part of an explicit
    100%-precision policy contract. Heuristic signals such as inferred entity
    connectivity, naturalness, or likely user intent may be logged or used to
    improve prompts/tool schemas, but must not become hard rejection rules.
14. Absolute rule: no semantic token heuristics. Runtime code must never infer
    business meaning, safety, naturalness, difficulty, topic quality, visibility,
    or validity from substrings/tokens in table names, column names, tool names,
    or generated user text. This includes regexes, suffix/prefix checks, token
    lists, and name-category maps such as "customer-like", "payment-like",
    "identifier-like", or "maintenance timestamp". Such logic is banned even
    when it appears helpful for privacy, naturalness, or weak-model guidance.
    Runtime code also must not infer predicate expression, membership validity,
    or user intent by checking whether a DB literal value appears as a substring
    in generated user text or submitted label text.

## Absolute Rule: No Semantic Token Heuristics

This rule is non-negotiable for the project.

Forbidden:

- table-name, column-name, tool-name, or generated-text token lists
- substring, prefix, suffix, regex, casing, or word-splitting rules used to infer
  domain meaning
- DB-value literal containment checks against generated user text or label text
  used to infer that a filter, membership rule, visibility rule, or task intent
  is expressed
- token-based role assignment, topic scoring, anchor scoring, affordance
  suppression, visibility/sensitivity classification, privacy classification,
  naturalness checks, hard validation, or retry feedback branches
- sample-DB-shaped priors hidden behind generic names

Allowed:

- exact lookup of explicit user configuration, such as visibility overrides
- structural database metadata: PK/FK flags, uniqueness, nullability, declared
  data types, FK graph degree, row estimates, and relationship fanout
- measured live facts: profile statistics, sampled row non-emptiness,
  relationship counts, query evidence, and exact label equality
- mechanical string handling that does not infer semantics, such as SQL
  identifier quoting, env-var interpolation, stable relation labels, rendering
  observed names back to the model, and exact substring checks for structured
  answer-contract phrases supplied by the composer as pointers into
  `user_request`, not as DB-value semantic evidence
- registry near-duplicate detection over generated task text, including MinHash
  tokenization/shingling, because its purpose is string-surface similarity, not
  DB semantic inference

### DB Literal Containment Boundary

A DB literal is any concrete value observed from the current database or tool
evidence: strings, enum/status/code values, numbers, booleans, dates,
timestamps, UUIDs, handles, and redaction placeholders. The ban applies even
when the value was discovered dynamically during the current run. A runtime rule
is still heuristic if it asks whether an observed DB value occurs in generated
natural-language text.

Forbidden as validation or feedback evidence:

- `value in user_request`, `value in label_json`, or `value in
  answer_contract_phrase` checks used to infer that a predicate, filter,
  membership rule, visibility rule, or task intent is expressed
- case-folded, tokenized, regex, prefix/suffix, stemming, translation,
  synonym, or paraphrase variants of that containment check
- sample-derived allow/deny vocabularies that turn observed DB values into
  natural-language accept/reject rules
- rejecting a draft because a query predicate value such as a status code is
  not literally or semantically visible in the request or label text

Allowed exact uses of DB literals:

- executing typed predicates inside the database as part of an observed query
- exact structured equality checks: canonical label versus latest query result,
  reward-visible label changes, submitted result versus label, or entity handle
  values versus explicit anchor values
- exact lookup through explicit configuration or an explicit structured mapping
  supplied outside the model's generated prose
- rendering observed values back to the model, logs, traces, or task artifacts
  without treating their textual occurrence as proof of semantics

Future structured predicate bindings may be validated for shape: a predicate can
reference a concrete `constraint_phrase` id, the phrase can be required to exist
in `user_request`, and the predicate can be required to appear in the latest
query evidence. The runtime still must not decide that the phrase semantically
means the DB value by substring, token, translation, or paraphrase matching.
Semantic value-to-phrase equivalence is valid only when it comes from explicit
configuration or another 100%-precision structured source.

### Visibility Metadata

`visibility` is column-level policy metadata, not a semantic guess from a name
or value. The current value set is `blocked`, `internal`, and `user_visible`
(`Visibility = Literal["blocked", "internal", "user_visible"]` in
`infra/visibility.py`; config uses the same literal set).

- `blocked`: not user-visible. Direct label exposure is rejected. PK/FK blocked
  columns may still be available as opaque handles for entity scope and
  navigation when the schema marks them as handles.
- `internal`: available to internal tooling where exposed by the surface, but
  not a customer-facing answer value. Direct label exposure is rejected.
- `user_visible`: eligible to appear in user-facing answer values when the task
  asks for it and the value is grounded in observed tool/query evidence.

Visibility comes from explicit config/default metadata and snapshot
materialization. Runtime code must not infer visibility from column-name tokens.

Reason:

The product goal is DB-swappable RLVR data generation. Arbitrary good databases
do not share English naming conventions, suffixes, demo-schema nouns, or privacy
vocabulary. Token heuristics therefore create false priors that are not learned
from the current DB: they hide valid surfaces, over-promote familiar demo
schemas, collapse diversity, reject potentially valid drafts, and contaminate
the solver pass-rate signal that should measure task difficulty. They also
smuggle author assumptions into a pipeline whose job is to adapt from schema
structure, live rows, and exact tool evidence. When a concept matters, it must
come from explicit config, DB metadata, observed data, or a 100%-precision
contract, not from name tokens.

## Decision Ledger

Normative project decisions should be recorded here when they affect multiple
spec areas.

1. The project is DB-swappable by design. Common code, prompts, feedback, and
   generated tool schemas adapt from introspection and live observations; they
   must not encode one sample database's schema or business assumptions.
2. Composer is autonomous inside its authoring role. The system gives it schema
   maps, profiles, samples, neighborhoods, queries, and feedback so it can judge
   feasible DB-native topics and strengthening directions. The system must not
   hard-code a global difficulty ladder.
3. `schema_map` is an exploration map, not proof. It helps identify topics,
   relationship paths, hub/bridge structure, readable surfaces, and possible
   strengthening directions; final label evidence still comes from live tool
   observations and canonical `query` results.
4. Actor and solver learn or solve through experience with the rendered customer
   request and tools. Authoring concepts such as curriculum, specificity
   feedback, Item-complexity, Cardinality, pass rate, quality gates, and training
   purpose must not leak into the rendered actor prompt or solver system prompt.
5. List-valued labels are first-class exact-match targets. A list is valid when
   the whole payload is canonical: membership, order, limit, and tie-breaks are
   fixed by the request and evidence.
6. Difficulty can grow by more items, harder items, filters, ordering, or other
   grounded dimensions when the current DB supports them. Passive display-width
   additions are not meaningful difficulty unless they change what must be found.
7. Random anchor candidates are environment randomization, not authoring hints.
   They should be selected by a DB-agnostic rule-based sampler to prevent a
   stateless composer from repeatedly starting at `id=1` or the first sampled
   row. Candidate anchors may include hidden primary-key metadata, visible row
   previews, and relationship counts for composer tooling, but raw ids must not
   be exposed in the customer-facing request.
8. Anchor sampling should be optimized as a quality-diversity distribution.
   Metrics quantify the distribution's value as RLVR starting context, including
   visible surface, relationship surface, anti-degeneracy, table coverage, and
   entropy. These metrics are used to improve the sampler itself, not to reject
   individual task drafts.
9. Rule-based adaptation is responsible for DB affordance context, not for
   choosing the final task. Rules may summarize tables, paths, readable
   surfaces, fanout, distribution, and task-shape/strengthening affordances
   from introspection and live profiling. The composer still chooses and proves
   the concrete label with tools.
10. Rule-based context is an acceleration layer for composer judgment. Its job is
   to make the current DB easier and faster to understand, not to constrain the
   composer to one path or replace authoring judgment. Outside that context
   preparation, composer is responsible for using tools well, selecting the
   task shape, grounding the label, and adapting after feedback.
11. Runtime logic must not use semantic token heuristics over table names,
    column names, or generated user text for topic scoring, anchor scoring, role
    assignment, affordance suppression, visibility/sensitivity classification,
    or hard rejection. It may use explicit configuration, structural metadata,
    and measured facts: PK/FK flags, FK graph degree, declared visibility
    labels, DB type metadata, row/profile statistics, sampled preview
    non-emptiness, and relationship counts. Relationship labels may include
    column names to identify the actual join, but names must not be interpreted
    as business meaning. Runtime logic also must not treat occurrence or
    absence of DB literal values in generated user text or label text as proof
    that a predicate is expressed, visible, hidden, or valid.
12. DB-adaptive context should describe logical data surfaces, not storage
    implementation details. Partition children are hidden when their parent
    table is visible; primary keys and foreign keys are treated as tool handles
    by structure; and relationship labels include the joining columns so
    multiple foreign keys between the same table pair remain distinct.
13. Shared prompts should be minimal, general, and reasoned. A short good/bad
    example block is acceptable when it teaches a cross-domain output pattern,
    but examples must stay abstract and must not smuggle in one database's
    schema, values, or business story. Trial-specific observations may be
    recorded in experiment logs, but they must be translated into
    schema-neutral structural patterns before entering shared prompts.
14. Schema descriptions are the first line of control for LLM-facing contracts.
    Prefer precise tool/input schema descriptions over retry feedback, and
    prefer retry feedback over hard validation unless the runtime can prove a
    violation exactly.
15. Validation and guidance have different jobs. Hard validation is reserved for
    conservative, 100%-precision violations only. Everything else should be
    shaped through minimal prompts, generalized good/bad examples, generated tool
    schemas, and concise tool descriptions; remaining uncertainty is judged
    statistically by solver pass-rate rather than by heuristic rejection.
16. Absolute rule: prompt-layer boundaries are part of the pipeline contract.
    System prompts may contain only durable, role-local policy; user prompts carry
    current-run context and tasking. Do not put database facts, schema maps,
    anchor candidates, local examples, optional edge-case topic hints, provider
    quirks, or trial observations into system prompts. Do not use user prompts
    to override role isolation, validation precision, no-token-heuristic rules,
    or tool schema contracts.
17. Pipeline state transitions are part of the correctness contract. Before
    changing synthesis feedback, submit validation, solver rollout, trial retry,
    harvest behavior, or registry acceptance, consult
    [Pipeline Lifecycle And State Boundaries](./pipeline-lifecycle.md). In
    particular, feedback-only `submit_draft` repair and terminal trial discard
    are different states. A rule being detectable is not enough reason to make
    it same-conversation feedback.
18. Too-hard quality failures are terminal by design. Empirically, asking the
    same composer conversation to weaken a hard or low-quality draft tends to
    burn turns, drift semantics, and produce another weak candidate. Too-easy
    drafts may be strengthened in place because the same answer target can
    usually be narrowed with one grounded constraint; too-hard drafts should be
    discarded so harvest can start from a fresh conversation and anchor context.
19. Trial analysis must separate gate status from human quality judgment. The
    configured pass-rate band is a calibration policy, not the definition of
    good data. When a trial is too hard or inconclusive, inspect the canonical
    query, label, sampled rows, solver traces, and tool errors to decide whether
    the draft is low-quality, ambiguous, or data-broken, or whether it is a
    sound but difficult problem for the current solver/tool surface. The
    decisive distinction is tool-solvability: if the submitted task can be
    answered from the database using only the solver's provided tools, but the
    sampled solvers fail, it is a difficult but good problem. If the submitted
    task cannot be answered from the prompt, data, and solver tool surface
    because the row set, ordering, entity, output mapping, or evidence path is
    under-specified or unavailable, it is low-quality. This review must stay
    structural and trace-based; do not justify the judgment with DB-literal or
    token-containment heuristics.

## Validation And Guidance Boundary

Hard validators are not quality critics. They are accept/reject gates for facts
the runtime can prove exactly: schema/type shape, exact latest-query evidence,
required provenance, explicitly configured exposure policy, source table
primary-key availability for row-valued labels, and exact reward-visible label
changes. If a rule could reject a valid task in an arbitrary well-designed
database, it is not precise enough for hard validation.
DB literal containment in natural-language requests or labels is not validator
evidence; value-to-language alignment requires an explicit structured binding or
belongs in prompt/tool guidance and solver pass-rate review.

For the larger set of desirable behaviors, use LLM-facing contract design:

- prompts should be minimal, general, and explain why the rule exists
- examples should be abstract good/bad patterns that transfer across domains
- tool schemas should encode required structure whenever JSON Schema can express
  it
- tool descriptions should be concise and contract-shaped, not long policy
  essays
- tool feedback should repair contract mistakes, but should not become the main
  control mechanism

The remaining gray zone is intentional. Difficulty, naturalness, and task
usefulness are evaluated by solver pass-rate and qualitative trace review. A low
pass rate is evidence to investigate, not proof that the data is bad. This
preserves the solver signal instead of replacing it with brittle heuristics.

## Absolute Rule: Prompt Layer Boundary

Prompt placement is not a style preference; it is an interface invariant.

- `system` is for durable role-local policy: role, stopping condition,
  tool-use invariants, customer-facing request principles, exact-label
  principles, validation/guidance hierarchy, and cross-role leak bans.
- `user` is for the current assignment packet: domain/scenario text, target
  language, optional edge-case topic hints, schema/affordance/profile context,
  tool surface summary, anchor candidates, local examples, and retry/tasking
  context.
- tool schema and parameter descriptions outrank prompt prose for callable
  argument contracts. If the requirement belongs beside one field, put it there.
- hard validation remains reserved for 100%-precision violations. Prompt
  placement must never be used to smuggle heuristic rejection rules into the
  runtime.

This boundary protects DB-swappability and role isolation. System prompts remain
portable across all databases; user prompts adapt to the current database without
becoming policy; tool schemas express machine-checkable contracts; solver
pass-rate handles the remaining uncertainty.

Topic ownership follows the same boundary. Normal generation does not receive a
topic target from outside the composer. The composer submits the topic that best
describes the grounded draft it actually built. A topic hint may appear only as
edge-case experiment context, never as the default generation target.

## Clean Break Policy

The old path-centric stack is not authoritative.

- legacy `tools/`, `tasks/`, `truth/`, and `verification/` are deleted
- the authoritative runtime surface is `synthesis/`, `solver/`, and `pipeline/solver_orchestrator.py`
- the source of truth is DB-grounded label construction, not generated verifier code

## Authoritative Modules

- `config/`
- `infra/`
- `schema/introspect.py`, `schema/graph.py`, `schema/path_catalog.py`
- `synthesis/`
- `solver/backend_openai_agents.py`
- `pipeline/solver_orchestrator.py`
- `calibration/`
- `cli.py`

## System Architecture

```text
DB Registry
  -> Domain Scheduler
  -> Synthesis Orchestrator
  -> Quality Gate
  -> Task Registry
```

### Layer Breakdown

1. Schema and data discovery
2. Atomic tool generation and materialization
3. Label-first synthesis
4. Solver pass-rate quality gate
5. Registry persistence and bundle export
