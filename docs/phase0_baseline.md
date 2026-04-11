# Phase 0 Baseline Snapshot

이 문서는 Plan 4의 `Phase 0: Freeze and Baseline` 산출물이다.

## Snapshot Metadata

- recorded_at: `2026-04-11 13:31:34 KST`
- release_owner: `current rewrite lead`
- branch: `codex/review-pack-archive-temp1`
- commit: `52b6c97`
- purpose: `path-centric baseline을 archive로 고정하고, rewrite 회귀 기준을 남긴다`

## Baseline Review Pack

clean-break rewrite 이전 baseline review pack snapshot은 아래 artifact로 고정한다.

- markdown: [`review_pack.md`](/Users/jd/Documents/workspace/rl-data-harness/review_packs/20260411-114636-aggregate-focus-v2/review_pack.md)
- jsonl: [`review_pack.jsonl`](/Users/jd/Documents/workspace/rl-data-harness/review_packs/20260411-114636-aggregate-focus-v2/review_pack.jsonl)

선정 기준:

- latest archived review pack
- path-centric generator의 마지막 qualitative baseline
- 이후 synthesis-agent rewrite와 직접 비교 가능한 artifact

portable reference:

- `review_packs/20260411-114636-aggregate-focus-v2/review_pack.md`
- `review_packs/20260411-114636-aggregate-focus-v2/review_pack.jsonl`

## Infra Regression Baseline

rewrite 중 계속 보호할 baseline green suite는 아래다.

- command: `uv run pytest -q`
- result: `124 passed in 14.06s`
- verified_at: `2026-04-11 13:31:34 KST`

이 숫자는 `infra / solver / calibration / orchestration skeleton` 회귀 기준으로 사용한다.

## Phase 0 Outcome

현재 기준:

- baseline review pack snapshot이 명시됐다
- green suite baseline이 재확인됐다
- production training freeze는 [`runbook.md`](/Users/jd/Documents/workspace/rl-data-harness/docs/runbook.md)에 기록됐다

## Notes

- 이 baseline은 training source가 아니다
- review pack은 qualitative reference artifact이며 production acceptance gate가 아니다
- 이후 rewrite 진행 중 새 review pack은 별도로 누적하되, 이 snapshot은 비교 기준으로 유지한다
