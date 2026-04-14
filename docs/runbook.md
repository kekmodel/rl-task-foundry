# Rewrite Runbook

## Production Training Freeze

status: `ACTIVE`

owner:

- release owner / rewrite lead가 freeze 상태 유지와 해제 판단을 맡는다

effective_from:

- `2026-04-11`

scope:

- synthesis-agent hybrid rewrite가 proof environment와 quality gate를 통과하기 전까지 유지

## Prohibited Actions

- 기존 path-centric baseline artifact를 production RL training source로 사용하는 것
- rewrite 이전 generator에서 나온 dataset을 accepted production corpus로 승격하는 것
- proof task 이전에 production task registry를 운영 시작하는 것
- review pack 같은 별도 qualitative snapshot surface를 다시 도입하는 것

## Allowed Actions

- spec / plan / core contract 작업
- synthetic proof task 구축
- registration policy / runtime isolation 구현
- artifact 기반 정성 평가
- infra regression test 유지

## Exit Criteria

freeze는 아래가 모두 충족될 때만 해제 후보가 된다.

- proof task vertical slice 완료
- hybrid verifier A/B/C/D mandatory gate 구현
- self-scoped task consistency와 solver pass-rate quality filter 구현
- artifact 기반 정성 평가에서 반복적으로 품질 기준을 충족

## Operator Checklist

rewrite 진행 중에는 아래를 확인한다.

- baseline regression이 깨지지 않았는가
- 새 artifact가 training source로 오인되어 누적되지 않았는가
- qualitative review가 registry bundle과 debug traces 기준으로 수행되고 있는가
- production acceptance / registry commit은 milestone 달성 전까지 막혀 있는가

## Artifact Inspection

정성 평가는 review pack이 아니라 아래 artifact를 본다.

현재 qualitative baseline snapshot:

- [`artifacts/evals/real_db_assignment_entity_literal_20260414`](/Users/jd/Documents/workspace/rl-data-harness/.dev/worktree/quick-harbor/artifacts/evals/real_db_assignment_entity_literal_20260414)
- baseline phase monitor: [phase_monitors.jsonl](/Users/jd/Documents/workspace/rl-data-harness/.dev/worktree/quick-harbor/artifacts/evals/real_db_assignment_entity_literal_20260414/debug/phase_monitors.jsonl:1)
- latest status: `accepted`
- latest accepted task: `task_assignment_a032aeb7b99d95a8`
- latest notable behavior:
  - strict `submit_draft` schema no longer fails at SDK setup time
  - literal `<entity>` prompt guidance removed `question_entity_block_required` as the dominant failure mode
  - the accepted run passed through one `reject_too_easy` turn before succeeding on a one-axis strengthening step

- synthesis loop / rejection 흐름:
  - [`artifacts/.../debug/phase_monitors.jsonl`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
- synthesis agent 대화 / tool 사용:
  - [`artifacts/.../debug/traces/synthesis/transcripts/*.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
  - [`artifacts/.../debug/traces/synthesis/tool_traces/*.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
- solver run 행동:
  - [`artifacts/.../debug/traces/transcripts/*.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
  - [`artifacts/.../debug/traces/tool_traces/*.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
- accepted task 내용:
  - [`artifacts/.../bundle/tasks/*/task.yaml`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
  - [`artifacts/.../bundle/tasks/*/instance.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
  - [`artifacts/.../bundle/tasks/*/canonical_answer.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)

눈으로 데이터 생성 과정을 볼 때는 보통 `phase_monitors.jsonl`부터 보고, 이상한 draft가 있으면 대응하는 `synthesis/transcripts/*.json`와 solver transcript를 같이 본다.
