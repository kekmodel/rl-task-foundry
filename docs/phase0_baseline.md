# Phase 0 Baseline Snapshot

이 문서는 Plan 4의 `Phase 0: Freeze and Baseline` 산출물이다.

## Snapshot Metadata

- recorded_at: `2026-04-11 13:31:34 KST`
- release_owner: `current rewrite lead`
- branch: `codex/review-pack-archive-temp1`
- commit: `52b6c97`
- purpose: `rewrite 회귀 기준과 freeze 상태를 고정한다`

## Infra Regression Baseline

rewrite 중 계속 보호할 baseline green suite는 아래다.

- command: `uv run pytest -q`
- result: `124 passed in 14.06s`
- verified_at: `2026-04-11 13:31:34 KST`

이 숫자는 `infra / solver / calibration / orchestration skeleton` 회귀 기준으로 사용한다.

## Phase 0 Outcome

현재 기준:

- green suite baseline이 재확인됐다
- production training freeze는 [`runbook.md`](/Users/jd/Documents/workspace/rl-data-harness/docs/runbook.md)에 기록됐다
- qualitative inspection은 runtime artifact 기준으로 수행한다

## Notes

- 이 baseline은 training source가 아니다
- qualitative review는 registry bundle과 debug traces를 기준으로 한다

## Latest Baseline Refresh

이 섹션은 rewrite 중 비교 기준으로 다시 사용할 최신 baseline snapshot을 기록한다.

### Snapshot Metadata

- refreshed_at: `2026-04-14 15:36:12 KST`
- branch: `quick-harbor`
- commit: `112aed7`
- purpose: `strict submit_draft schema 및 literal <entity> prompt patch 이후 accepted baseline을 재고정한다`

### Current Infra Regression Baseline

- command: `uv run pytest -q`
- result: `178 passed in 2.43s`
- verified_at: `2026-04-14 15:36:12 KST`

### Current Qualitative Baseline Snapshot

- artifact_root: [`artifacts/evals/real_db_assignment_entity_literal_20260414`](/Users/jd/Documents/workspace/rl-data-harness/.dev/worktree/quick-harbor/artifacts/evals/real_db_assignment_entity_literal_20260414)
- phase_monitor: [phase_monitors.jsonl](/Users/jd/Documents/workspace/rl-data-harness/.dev/worktree/quick-harbor/artifacts/evals/real_db_assignment_entity_literal_20260414/debug/phase_monitors.jsonl:1)
- status: `accepted`
- task_id: `task_assignment_a032aeb7b99d95a8`
- solver summary:
  - submission 2 calibration: `matched_solver_runs=6`, `pass_rate=1.0`
  - final accepted draft: `solver_pass_rate=4/6`
- accepted path summary:
  - anchor: `customer_id=1`
  - answer slots: `title`, `rental_date`, `return_date`, `staff_first_name`, `staff_last_name`, `payment_date`
  - acceptance change: `reject_too_easy` 이후 `solution_space` 한 축 강화로 통과

이 snapshot은 다음 feature 실험에서 비교할 기준점으로 사용한다.
