"""Asymmetric tool surfaces for synthesis composer and solver roles.

See docs/spec/tooling-redesign.md for the design philosophy.

- `tooling.atomic`: composition calculus exposed to the RL solver.
- `tooling.composer`: analytic DSL exposed to the synthesis composer (next session).
- `tooling.common`: shared helpers (schema snapshot, SQL, FK edges).

The atomic and composer subpackages never import from each other.
"""
