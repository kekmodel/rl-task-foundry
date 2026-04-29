# Docs

- Development baseline: Python 3.14 via `uv`
- Recommended setup:
  - `uv venv --python 3.14`
  - `uv sync --dev`

- [../AGENTS.md](../AGENTS.md)
  - 현재 실험/개선 작업의 최상위 원칙과 필수 읽기 순서
- [spec.md](./spec.md)
  - 현재 제품/런타임/도구 계약 snapshot의 index. 원칙/DQS를 지키는 한 변경 가능
- [experiments/first_principles.md](./experiments/first_principles.md)
  - 평가 해킹 방지 원칙과 실험 그래프의 불변 규칙
- [experiments/rubric_dqs_v1.md](./experiments/rubric_dqs_v1.md)
  - DQS-v1 accept/reject 정성평가 루브릭과 promotion 기준
- [experiments/evaluator_subagent.md](./experiments/evaluator_subagent.md)
  - Codex subagent 기반 독립 DQS 평가자 생성/리포트 프로토콜
- [experiments/tree.md](./experiments/tree.md)
  - git branch/worktree 기반 실험 탐색 그래프 운영 규칙
- [experiments/registry.yaml](./experiments/registry.yaml)
  - 현재 baseline, evaluation policy, 실험 노드 metadata source of truth
- [sample_databases.md](./sample_databases.md)
  - Pagila/Postgres Air 로컬 PostgreSQL bootstrap 경로

Legacy plans, tuning logs, stale ADRs, and generated schema reports are kept in
git history only. Do not treat old experiment notes as current policy.
