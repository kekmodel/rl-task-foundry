# Docs

- Development baseline: Python 3.14 via `uv`
- Recommended setup:
  - `uv venv --python 3.14`
  - `uv sync --dev`

- [spec.md](./spec.md)
  - 제품 목표, 데이터 계약, runtime 경계, throughput/verification 원칙
- [plan.md](./plan.md)
  - 구현 순서, 모듈 책임, 완료 기준, 테스트 전략
- [composer_low_quality_reduction_plan.md](./composer_low_quality_reduction_plan.md)
  - composer low-quality 감소를 위한 계약/툴/subagent 실험 순서
- [experiments/first_principles.md](./experiments/first_principles.md)
  - 평가 해킹 방지 원칙과 실험 그래프의 불변 규칙
- [experiments/rubric_dqs_v1.md](./experiments/rubric_dqs_v1.md)
  - DQS-v1 accept/reject 정성평가 루브릭과 promotion 기준
- [experiments/tree.md](./experiments/tree.md)
  - git branch/worktree 기반 실험 탐색 그래프 운영 규칙
- [sample_databases.md](./sample_databases.md)
  - Pagila/Postgres Air 로컬 PostgreSQL bootstrap 경로
