# RL Task Foundry Spec Index

This file is the entrypoint to the current RL Task Foundry product/runtime
contract snapshot. These specs are mutable experiment artifacts, not immutable
axioms. They may be rewritten, replaced, or deleted when an experiment changes
the architecture, as long as the change obeys `AGENTS.md`,
`docs/experiments/first_principles.md`, and `docs/experiments/rubric_dqs_v1.md`.

When a spec changes, record the reason and evaluation impact in the experiment
node metadata.

## Reading Order

1. [Foundation](./spec/foundation.md)
2. [Task Bundle Model](./spec/environment-bundle-model.md)
3. [Difficulty and Actor Interface](./spec/difficulty-and-actor-interface.md)
4. [Atomic Tools](./spec/atomic-tools.md)
5. [Synthesis Pipeline](./spec/synthesis-pipeline.md)
6. [Pipeline Lifecycle and State Boundaries](./spec/pipeline-lifecycle.md)
7. [Reward and Task Runtime](./spec/reward-and-environment-runtime.md)
8. [Bundle Format, Contracts, and Registration](./spec/bundle-contracts-and-registration.md)
9. [Registry and Operations](./spec/registry-and-operations.md)

## Quick Map

- [Foundation](./spec/foundation.md): goals, principles, clean-break rationale, architecture
- [Task Bundle Model](./spec/environment-bundle-model.md): task bundle as the core unit, metadata, materialization model
- [Difficulty and Actor Interface](./spec/difficulty-and-actor-interface.md): topic model, difficulty model, rendered prompt, `submit_result`
- [Atomic Tools](./spec/atomic-tools.md): current solver-facing tool surface, response contract, SQL boundary
- [Synthesis Pipeline](./spec/synthesis-pipeline.md): label-first phases, prompt constraints, retry behavior
- [Pipeline Lifecycle and State Boundaries](./spec/pipeline-lifecycle.md): end-to-end runtime state transitions, feedback vs terminal discard boundaries, and change checklist
- [Reward and Task Runtime](./spec/reward-and-environment-runtime.md): canonicalization, reward, environment server, episode semantics
- [Bundle Format, Contracts, and Registration](./spec/bundle-contracts-and-registration.md): bundle filesystem, schemas, static registration policy
- [Registry and Operations](./spec/registry-and-operations.md): quality gate, scheduler, registry, review, proof environment, success criteria

## Current Contract Scope

When a rule appears in only one linked spec document, that linked document
describes the current contract for that topic. This index is intentionally
short and should not duplicate details.

Spec documents do not create new immutable principles. If a spec conflicts with
the experiment first principles or DQS-v1 hard gates, the first-principles and
DQS documents win. The no-semantic-token-heuristics language in
[Foundation](./spec/foundation.md) is a spec-level restatement of that higher
rule, not a separate fixed spec constraint.
