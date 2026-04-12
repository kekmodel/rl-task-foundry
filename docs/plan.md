# Plan 4: Synthesis-Agent Hybrid Rewrite Plan

## Objective

이 계획의 목적은 RL Task Foundry를 `path-centric task generator`에서  
`synthesis-agent driven environment generator`로 전면 재설계하는 것이다.

새 시스템은 read-only PostgreSQL DB를 등록하면, compositional task environment를 자동 생성하고, canonical answer + exact-match reward 기반 quality gate를 통과한 environment만 registry에 누적한다.

## Rewrite Ground Rules

1. task generation 관련 코드의 incremental polishing은 중단한다
2. 먼저 spec과 core contracts를 고정한다
3. 그 다음 proof environment를 하나 끝까지 증명한다
4. generalized environment expansion은 proof environment 이후로 미룬다
5. infra / solver / orchestration skeleton은 최대한 유지한다

## Deliverable

Plan 4가 끝나면 아래가 가능해야 한다.

1. DB registry에 새 DB를 등록한다
2. scheduler가 DB를 선택해 synthesis loop를 돈다
3. synthesis agent가 db-level atomic tool bundle을 참조하는 environment bundle을 생성한다
4. code registration policy가 generated code를 검사한다
5. cross-instance와 solver pass-rate quality gate를 통과한 environment만 registry에 커밋한다
6. exported bundle로 환경 품질을 사람이 spot-check할 수 있다

## Explicit Non-Deliverable

이번 계획은 아래를 하지 않는다.

- path-centric baseline을 계속 보강하는 것
- baseline accepted dataset을 training source로 쓰는 것
- multi-DB joint task generation
- fuzzy / subjective verifier

## Authoritative Modules After Transition

2026-04-12 기준 authoritative module surface는 아래다.

- `config/`
- `infra/`
- `schema/introspect.py`, `schema/graph.py`, `schema/path_catalog.py`
- `synthesis/`
  - atomic tool generator
  - canonicalize + `compute_reward`
  - label-first synthesis runtime
  - registry / scheduler / bundle exporter
- `solver/backend_openai_agents.py`
- `solver/runtime.py`, `solver/models.py`
- `calibration/`
- `pipeline/environment_orchestrator.py`
- `cli.py`

삭제된 path-centric legacy stack은 git history에만 남고 authoritative runtime surface에는 포함되지 않는다.

## Deleted Legacy Modules

아래 legacy surface는 `C11`에서 삭제 완료됐다.

- `tools/`
- `tasks/`
- `truth/`
- `verification/`
- `pipeline/orchestrator.py`
- `pipeline/review_pack.py`
- `pipeline/export.py`
- `pipeline/manifest.py`

## Execution Order

이 계획은 아래 순서로 실행한다.

### Phase 0: Freeze and Baseline

작업:

- release owner가 latest qualitative artifact snapshot을 baseline으로 기록
- current green suite를 infra regression baseline으로 명시
- production training 금지 상태를 문서와 runbook에 기록
- phase 0 산출물을 `docs/phase0_baseline.md`, `docs/runbook.md`에 남긴다

Acceptance:

- baseline snapshot path가 명시돼 있다
- baseline suite가 회귀 기준으로 남아 있다
- freeze 상태가 runbook에 명시돼 있다

### Milestone 1: Core Contracts (Deprecated)

이 milestone은 per-environment narrow tool 가정을 포함한 초기 synthesis core contract를 고정했던 historical 단계다.

남는 가치:

- trust field materialization
- draft vs materialized environment 경계
- difficulty vector
- registry-friendly typed contract discipline

deprecated 이유:

- `ToolContract`와 per-env generated tool surface를 중심으로 contract가 조직돼 있다
- verifier contract가 staged fact checker 기준이다
- actor-facing parity contract와 canonical answer contract가 아직 중심이 아니다

이 milestone의 산출물은 Milestone M-Atomic-Transition에서 전면 삭제되거나 재정의된다.

### Milestone 2: Code Registration Policy and Runtime Isolation (Deprecated)

이 milestone은 synthesis-generated `tool.py`를 포함하는 registration lane을 기준으로 설계된 historical 단계다.

남는 가치:

- custom AST preflight
- subprocess isolation
- worker pool
- registration error schema

deprecated 이유:

- generated `tool.py`
- `tool_self_test`
- tool-specific validator 경로

가 atomic-tool-per-database 결정 이후 authoritative path가 아니게 되었다.

향후 authoritative registration 대상은 `solution.py`, `verifier.py`, `shadow_verifier.py`다.

이 milestone의 산출물은 Milestone M-Atomic-Transition에서 전면 삭제되거나 재정의된다.

### Milestone 3: Synthesis Agent Runtime Skeleton

목표:

- OpenAI Agents SDK 기반 synthesis meta-agent runtime skeleton을 유지한다

작업:

- schema exploration + sample row inspection
- category inference phase
- label-first phase split
  - label construction
  - task synthesis
  - code generation
- structured output contract
- explicit memory / tool trace contract
- provider resilience reuse
- runtime trust-field materialization
- single-db runtime binding
- async-safe shared cache / registration pool initialization

Acceptance:

- 단일 DB에서 단일 category에 대해 environment draft 하나를 생성할 수 있다
- provider circuit breaker / cooldown / quota rebalance가 synthesis runtime에도 적용된다
- draft는 structured phase output을 통해 생성된다
- agent는 trust field를 제안하지 않고 runtime이 authoritative metadata를 채운다

주의:

- 이 milestone이 초기에 가정했던 per-env narrow tool generation 경로는 아래 M-Atomic-Transition에서 대체된다

### Milestone M-Atomic-Transition: Atomic Tools per Database (Completed 2026-04-12)

목표:

- narrow per-env tool architecture를 atomic-tool-per-database architecture로 전면 전환한다

상태:

- acceptance criteria 충족
- Phase 1~4 구현 완료
- C13은 구현 후 spec/plan을 코드 기준으로 동기화하는 문서 마감 단계다
- authoritative atomic tool family surface는 `T1~T8`이다
- exact tool count는 schema snapshot의 함수이며 stale fixed count를 source of truth로 두지 않는다

Acceptance criteria:

- atomic tool set은 schema graph로부터 deterministic하게 생성된다
- 같은 `db_id`의 모든 environment가 동일 atomic tool set을 공유한다
- synthesis agent는 `tool.py`를 생성하지 않고, `solution.py`는 atomic tool만 호출한다
- solver backend가 `EnvironmentContract` 기반으로 동작하고 legacy `TaskSpec` 경로가 제거된다
- path-centric legacy (`tool_compiler`, `TaskSpec`, `TierATaskFactory` 등)가 완전히 삭제된다
- parity invariants 7개가 test로 강제된다
- environment bundle이 self-contained 구조로 export된다

Commit sequence (C1-C13):

Phase 1 — Foundation

- C1: `feat: add atomic tool generator from schema graph` ✅
- C2: `feat: add schema-driven canonicalize library` ✅
- C3: `feat: add output field canonicalization metadata` ✅

Phase 2 — Synthesis transition

- C4: `refactor: remove tool_source from generated artifact bundle` ✅
- C5: `feat: materialize db-level atomic tool bundle during synthesis` ✅
- C6: `refactor: update synthesis prompt to consume atomic tool set` ✅
- C7: `refactor: switch self-consistency subprocess to atomic tool reference` ✅
- C8: `feat: materialize per-instance canonical answers with triple oracle` ✅

Phase 3 — Solver backend transition

- C9: `refactor: switch solver backend to environment contract` ✅
- C10: `refactor: pipeline orchestrator to environment-contract-first` ✅

Phase 4 — Legacy deletion and bundle export (Completed)

- C11: `chore: delete legacy path-centric tool pipeline`
  - 삭제 대상 test 범위도 함께 고정한다
    - `tests/test_task_factory.py`
    - `tests/test_task_package_composer.py`
    - `tests/test_tool_compiler.py`
    - `tests/test_tool_model_naming.py`
    - `tests/test_tool_naming_eval.py`
    - `tests/test_truth_canonicalize.py`
    - `tests/test_ground_truth_generator.py`의 legacy path-centric 부분
  - C11을 C12보다 먼저 두는 이유는 exporter 구현 시 legacy 분기를 고려하지 않도록 authoritative path를 먼저 단일화하기 위해서다
- C12: `feat: environment bundle exporter` ✅
  - `export-bundle` CLI가 registry snapshot을 environment API server layout으로 내보낸다

Phase 5 — Documentation finalization

- C13: `docs: final spec/plan sync after implementation` ✅
  - atomic tool count 실측
  - staged verifier contract와 original `compute_canonical_answer(...)` target의 차이 명시
  - synthesis prompt 최종 wording
  - exporter가 만든 실제 bundle shape
  를 spec/plan에 backfill한다

Dependencies:

- blocked by: none
- blocks: Milestone 4, Milestone 10-11의 tool-related authoritative implementation

Risks:

- synthesis agent가 더 긴 `solution.py` / `verifier.py`를 안정적으로 생성하는지 실측 필요
- tool definition context window가 10K~20K token까지 증가할 수 있음
- legacy code / test 삭제 중 일시적 coverage 하락 가능

### Milestone 4: Verification and Reward Enforcement

목표:

- exact-match reward path를 authoritative source로 고정하고, synthesis-time verifier contract의 다음 수렴점을 정리한다

현재 상태:

- schema-driven canonicalization과 pure `compute_reward(...)`는 이미 구현됐다
- environment-contract-first solver / orchestrator path도 구현됐다
- 현재 authoritative triple oracle은 `solve + staged verifier + staged shadow verifier` 구조다
- 즉 원래 계획의 `compute_canonical_answer(tools) -> dict` 단일-entrypoint verifier는 아직 future simplification target이다

작업:

- schema-driven canonicalization
- staged verifier contract 유지 여부 또는 `compute_canonical_answer(tools) -> dict` 단일-entrypoint 수렴 여부 결정
- `solution.py`, `verifier.py`, `shadow_verifier.py` triple oracle 운영 계약 고정
- pure `compute_reward(submitted_text, canonical_answer, output_schema)` contract
- `json_decode_failed`, `schema_mismatch`, `em_mismatch` failure taxonomy
- actor-facing parity invariant test
- cross-instance verification이 canonical answer materialization과 직교함을 명시
- pass-rate band 재calibration 계획 수립
  - Milestone 5 proof environment에서 solver pass rate를 실측한다
  - 기존 `25-75%` band가 여전히 유효한지 또는 조정이 필요한지 결정한다
  - 결정 결과를 config와 `docs/spec.md`의 Quality Gate section에 반영한다

Acceptance:

- verification이 EM + canonicalize로 단일화된다
- triple oracle이 self-consistency의 공식 형태가 된다
- current staged verifier contract와 future simplification target 사이의 경계가 문서와 code path에서 명시된다
- cross-instance verification은 instance-level multiplexing으로 유지된다
- pass-rate band는 atomic tool 전환 이후 다시 calibration 대상임이 문서와 code path에 반영된다

### Milestone 5: Proof Environment Vertical Slice

목표:

- atomic tool bundle 위에서 compositional proof environment 하나를 end-to-end로 통과시킨다

운영 결정:

- first proof DB는 synthetic fixture DB를 쓴다
- Sakila는 proof DB로 쓰지 않는다

권장 proof task:

- itinerary 또는 assignment
- 3개 이상 slot
- uniqueness
- conditional threshold
- locality constraint
- `list[object]` answer

작업:

- grounded sample-row exploration
- label construction first, task synthesis second
- proof task의 constraint 구조 먼저 고정
- 그 task를 지원하는 최소 synthetic schema 설계
- synthetic fixture DB 구축
- rendered prompt / solution synthesis
- label-first canonical answer materialization

Acceptance:

- unique canonical answer가 materialize된다
- grounded label과 canonical answer가 일치한다
- exported bundle review에서 사람이 봐도 compositional task다

### Milestone 6: Generation Retry Policy and Difficulty Escalation

목표:

- synthesis loop의 종료 조건과 난이도 crank를 atomic tool architecture 위에 명시적 policy로 구현한다

작업:

- `max_generation_attempts`
- schema/category phase 재사용 + artifact generation phase retry
- artifact diagnostics를 다음 attempt input에 주입
- quality gate feedback을 다음 harder retry input에 주입
- infeasible discard path
- discard consumes budget 정책
- `db_id x category` failure counter와 backoff queue
- `max_consecutive_category_discards` / `category_backoff_duration_s` config로 local runtime backoff enforce
- scheduler가 읽을 수 있도록 runtime category status snapshot 노출
- tool set change -> solution invalidation policy
- label weakening 금지 규칙
- one-axis-per-step difficulty crank
- crank termination policy

Acceptance:

- artifact diagnostics는 regeneration retry로 이어진다
- canonical answer를 느슨하게 바꾸는 label weakening이 reject된다
- difficulty vector가 monotonic하게 증가한다
- 연속 discard threshold를 넘긴 `db_id x category`는 backoff error로 즉시 차단된다
- 반복 discard가 scheduler-level backoff로 이어진다

### Milestone 7: Cross-Instance Consistency

목표:

- cross-instance consistency를 production mandatory로 붙인다

작업:

- instance_space contract
- deterministic instance sampler
- instance별 서로 다른 valid canonical answer 확인

Acceptance:

- instance별 다른 valid canonical answer가 correctly materialize된다

### Milestone 8: Quality Gate 5-Stage Pipeline

목표:

- environment acceptance를 atomic tool / canonical answer 기준 orchestrated gate로 만든다

단계:

1. label construction and canonical materialization
2. cross-instance consistency
3. solver pass-rate band
4. registry dedup
5. coverage policy

작업:

- stage별 rejection reason schema
- retry / discard policy
- CI-based early termination
- 공식 reward source는 exact-match reward function으로 고정

Acceptance:

- environment는 5단계를 모두 통과해야 accepted 된다
- rejection reason이 structured하게 남는다

### Milestone 9: Real DB Single-Environment Validation

목표:

- synthetic proof 뒤에 real DB 적응성 공백을 없앤다

현재 메모:

- pre-label-first runtime flow에서는 real DB trial이 12회 연속 실패했고, 2026-04-12 기준 trial taxonomy 수집 결과가 prompt/runtime redesign 입력이 되었다
- authoritative follow-up은 grounded exploration + label-first pipeline으로 real DB single-environment acceptance를 재시도하는 것이다

작업:

- Sakila에서 single environment 생성
- 가능한 한 두 번째 real DB에서도 single environment 생성
- domain inference / atomic tool selection / canonical answer materialization failure taxonomy 수집
- prompt / synthesis policy 보강

Acceptance:

- synthetic fixture가 아닌 real DB에서 environment 하나를 end-to-end로 생성할 수 있다
- real DB failure mode가 taxonomy로 정리되어 다음 단계 입력으로 남는다

### Milestone 10: Domain Scheduler and Multi-DB Coordination

목표:

- 여러 DB를 registry에서 순차 처리하는 production loop를 만든다

작업:

- DB add/remove contract
- round-robin / priority queue
- per-DB progress tracking
- retry / backoff / starvation protection
- checkpoint / resume
- environment API server 연결 지점 명시
- stateless + `session_id` + async request model을 orchestration boundary로 문서화
- `SynthesisDomainScheduler` helper로 `category_status()` snapshot 소비
- `SynthesisOrchestrator`로 `registry -> snapshot build -> scheduler decision -> single-db runtime.run_next` 흐름 연결
- `SynthesisRegistryRunner`로 `registry -> checkpoint filter -> orchestrator.run_next -> successful pair checkpoint mark` loop 연결

Acceptance:

- DB가 동적으로 추가되어도 scheduler가 처리할 수 있다
- DB 간 task가 섞이지 않는다
- orchestrator가 db별 runtime cache를 유지하고 선택된 db에만 draft 생성을 위임한다
- runner가 성공한 `db_id x category` pair를 checkpoint에 기록하고 resume 시 다시 생성하지 않는다
- runner summary가 typed run outcome과 lightweight step summary를 제공한다

### Milestone 11: Environment Registry, Dedup, Coverage

목표:

- accepted environment와 db-level atomic tool bundle을 durable registry로 관리한다

작업:

- filesystem registry
- sqlite index
- WAL + single-writer append queue 정책
- `databases/{db_id}/` atomic tool bundle 저장
- `environments/{env_id}/` bundle 저장
- exact dedup
- MinHash semantic dedup
- registry read/query surface
- category taxonomy mapping
- difficulty-band coverage counters
- generator version compatibility policy
- selective regeneration queue scheduling policy
- bundle exporter와 cross-reference

Acceptance:

- near-duplicate environment를 exact + semantic 두 단계로 잡을 수 있다
- v1 semantic dedup은 MinHash threshold로 동작한다
- v1 exact dedup은 instance_space를 제외한 environment contract identity를 기준으로 삼는다
- db-level atomic tool bundle과 env-level audit bundle이 함께 durable commit된다
- category/difficulty coverage를 계산하고 skew를 볼 수 있다
- registry snapshot과 semantic dedup candidate를 조회할 수 있다

### Milestone 12: Coverage Planner

목표:

- registry inventory 대비 부족한 `db x category x difficulty band` cell을 계획적으로 채운다

작업:

- coverage target config source-of-truth
- zero-count cell을 포함한 coverage deficit planner
- pair-level aggregate deficit ranking
- coverage plan CLI surface

Acceptance:

- planner가 registry file의 db/category inventory를 기준으로 zero-count cell까지 포함한 deficit plan을 계산한다
- planner가 `db x category` pair별 total deficit을 정렬해서 다음 scheduler priority 입력으로 넘길 수 있다
- tracked difficulty band와 per-band target이 config source-of-truth로 고정된다

### Cross-Cutting Workstream: Manual Bundle Review and Observability

이건 마지막 milestone이 아니라 전 과정에 걸친 cross-cutting requirement다.

작업:

- exported bundle에 rendered prompt / instance / canonical answer가 모두 포함되도록 유지
- atomic tool set summary와 actor-facing definitions를 review surface로 유지
- verifier / shadow audit source를 함께 보관
- environment-level metrics 노출
- quality taxonomy 기록 포맷 고정

Acceptance:

- 각 milestone마다 small exported bundle batch를 생성해 정성 평가할 수 있다
- qualitative rubric 기준으로 pass/fail을 기록할 수 있다

## Execution Strategy

이 plan은 prototype-first가 아닌 full commitment 접근을 따른다  
([ADR 0002](adr/0002-atomic-tools-per-database.md) 참조).

- parallel dual-path 유지 없음
- 각 commit은 green state를 유지한다
- legacy 경로는 M-Atomic-Transition Phase 4에서 일괄 삭제한다
- rollback은 foundation을 남기고 transition phase를 보류하는 방식으로만 수행한다

Rollback trigger:

1. C5~C7 진행 중 생성된 `solution.py`의 50% 이상이 parse 실패 또는 triple oracle disagreement를 유발한다
2. 작은 proof DB에서 triple oracle agreement rate가 `80%` 미만으로 지속된다
3. atomic tool definition context가 synthesis model context window의 `30%` 이상을 차지해 prompt truncation 또는 구조적 overflow가 반복된다

관측 메모:

- C1~C12 구현 동안 위 rollback trigger는 발동하지 않았다

## Quality Filter Defaults

초기 제안값:

```yaml
quality_filter:
  attempts_per_env: 10
  lower_pass_rate: 0.25
  upper_pass_rate: 0.75
  ci_alpha: 0.1
  safe_early_termination: true
  shadow_disagreement_threshold: 0.05
  min_cross_instances: 3
  max_generation_attempts: 5
  max_difficulty_cranks: 6
```

품질 우선이므로 비용 상한은 낮은 우선순위다.

## Test Strategy

### Layer 1: Infra Regression

현재 green suite는 아래를 계속 보호한다.

- db pool
- checkpoint
- budget
- solver runtime
- calibration skeleton
- provider resilience

### Layer 2: New Paradigm Proof Tests

신규 테스트는 아래 중심으로 시작한다.

- registration policy AST rejection
- subprocess isolation contract
- canonicalize contract
- triple oracle consistency
- self-consistency loop
- difficulty crank monotonicity
- cross-instance checker stability
- proof environment binary verification

### Layer 3: Parity and Bundle Invariants

atomic tool architecture에서는 actor-facing parity와 bundle self-containment이 별도 레이어로 보호돼야 한다.

운영 결정:

- solver prompt constant
- tool definitions identical per `db_id`
- `Submit Result Format` 밖 field-name leakage 금지
- `submit_result(answer_text: string)` constant contract
- environment bundle self-containment

## Review Strategy

이 rewrite에서는 exported bundle 정성 평가가 필수다.

반드시 반복한다.

1. small environment batch 생성
2. exported bundle의 `environment.yaml`, `instances.jsonl`, `canonical_answers.jsonl`, atomic tool definitions, audit source를 직접 읽는다
3. 품질 문제를 taxonomy로 기록한다
4. prompt / policy / category inference를 수정한다

즉 green tests만으로 진행 결정을 내리지 않는다.

## Success Criteria

이 계획의 성공은 아래로 판단한다.

- generated environment가 lookup task처럼 보이지 않는다
- solver가 실제 composition reasoning을 해야 한다
- canonical answer + exact-match reward path가 deterministic하게 동작한다
- shadow / cross-instance / pass-rate 품질 필터가 모두 동작한다
- arbitrary DB를 registry에 추가해도 pipeline이 돌아간다
- 최근 exported bundle 10개 중 최소 7개가 아래 rubric을 만족한다
  - `compositional_structure`
  - `constraint_density`
  - `branching_or_threshold`
  - `grounded_verification`
  - `natural_user_request`
  - `non_lookup_shape`
  - 앞의 네 항목은 mandatory, 총 6개 중 최소 5개 통과

## Stop Conditions

아래 중 하나면 구현을 멈추고 문서로 돌아간다.

- verifier가 다시 self-consistent but ungrounded Python checker로 흐를 때
- actor-facing parity invariant가 흐려질 때
- proof environment 없이 generalized environment expansion부터 만들려 할 때
- exported bundle 품질이 낮은데도 green tests만으로 진행하려 할 때
