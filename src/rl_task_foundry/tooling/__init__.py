"""Asymmetric tool surfaces for synthesis composer and solver roles.

See docs/spec/atomic-tools.md and docs/spec/synthesis-pipeline.md for the
current contract.

- `tooling.atomic`: composition calculus exposed to the RL solver.
- `tooling.composer`: analytic DSL exposed to the synthesis composer.
- `tooling.common`: shared helpers (schema snapshot, SQL, FK edges).

The atomic and composer subpackages never import from each other.
"""
