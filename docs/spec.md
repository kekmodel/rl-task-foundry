# RL Task Foundry Spec Index

This file is the entrypoint to the RL Task Foundry specification. Normative content is split into focused documents under [`docs/spec/`](./spec/).

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

## Authoritative Rule

When a rule appears in only one linked document, that linked document is authoritative for that topic. This index is intentionally short and should not duplicate normative details.

The absolute no-semantic-token-heuristics rule in [Foundation](./spec/foundation.md) applies across the whole project and overrides local convenience arguments in synthesis, tooling, visibility, validation, feedback, and registry logic.
