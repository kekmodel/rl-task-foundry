# Plan 4: Synthesis-Agent Hybrid Rewrite Plan

## Objective

이 계획의 목적은 RL Task Foundry를 `path-centric task generator`에서  
`synthesis-agent driven environment generator`로 전면 재설계하는 것이다.

새 시스템은 read-only PostgreSQL DB를 등록하면, compositional task environment를 자동 생성하고, hybrid verifier 기반 quality gate를 통과한 environment만 registry에 누적한다.

## Rewrite Ground Rules

1. task generation 관련 코드의 incremental polishing은 중단한다
2. 먼저 spec과 core contracts를 고정한다
3. 그 다음 proof environment를 하나 끝까지 증명한다
4. generalized factory는 proof environment 이후로 미룬다
5. infra / solver / orchestration skeleton은 최대한 유지한다

## Deliverable

Plan 4가 끝나면 아래가 가능해야 한다.

1. DB registry에 새 DB를 등록한다
2. scheduler가 DB를 선택해 synthesis loop를 돈다
3. synthesis agent가 environment 4-tuple을 생성한다
4. code registration policy가 generated code를 검사한다
5. self-consistency, shadow verifier, cross-instance, solver pass-rate quality gate를 통과한 environment만 registry에 커밋한다
6. review pack으로 환경 품질을 사람이 spot-check할 수 있다

## Explicit Non-Deliverable

이번 계획은 아래를 하지 않는다.

- path-centric baseline을 계속 보강하는 것
- baseline accepted dataset을 training source로 쓰는 것
- multi-DB joint task generation
- fuzzy / subjective verifier

## Preserved Modules

다음은 유지 또는 최소 수정 재사용을 목표로 한다.

- `config/`
- `infra/`
- `solver/backend_openai_agents.py`
- `calibration/`
- `pipeline/orchestrator.py` skeleton
- `schema/introspect.py`, `schema/graph.py`, `schema/path_catalog.py`
- `verification/shadow.py` 개념
- `infra/json_chat_client.py`
- `cli.py`
- `pipeline/review_pack.py` 구조

## Rewrite Modules

전면 재작성 대상으로 본다.

- `tasks/factory.py`
- `tasks/composer.py`
- `tasks/question_generation.py`
- `tasks/package_validation.py`
- `tasks/provenance.py`
- `truth/generator.py`
- `truth/canonicalize.py` contract 부분
- `tools/compiler.py`
- `tools/sql_templates.py`
- `tools/model_naming.py`
- `tools/naming_eval.py`
- `verification/compare.py`

## Execution Order

이 계획은 아래 순서로 실행한다.

### Phase 0: Freeze and Baseline

작업:

- release owner가 latest review pack을 baseline snapshot으로 기록
- current green suite를 infra regression baseline으로 명시
- production training 금지 상태를 문서와 runbook에 기록
- phase 0 산출물을 `docs/phase0_baseline.md`, `docs/runbook.md`에 남긴다

Acceptance:

- baseline snapshot path가 명시돼 있다
- baseline suite가 회귀 기준으로 남아 있다
- freeze 상태가 runbook에 명시돼 있다

### Milestone 1: Core Contracts

목표:

- synthesis pipeline의 핵심 타입을 고정한다

작업:

- `EnvironmentContract`
- `ToolContract`
- `TaskContract`
- `SolutionContract`
- `VerifierContract`
- `ShadowVerifierContract`
- `InstanceSpaceContract`
- `CrossInstanceSet`
- `MaterializedFactsSchema`
- `DifficultyAxes`

핵심 결정:

- facts dict schema는 verifier contract의 일부다
- solution은 oracle/reference 전용이다
- difficulty는 vector다

Acceptance:

- one environment를 코드 없이 JSON/YAML spec으로 기술할 수 있다
- fact stage / constraint stage 경계를 타입으로 표현할 수 있다

### Milestone 2: Code Registration Policy and Runtime Isolation

목표:

- generated code를 등록/실행할 안전한 경계를 만든다

작업:

- AST preflight rules
- dunder / reflection / dynamic attribute 금지
- import allowlist
- signature validation
- registration error schema
- subprocess execution contract
- subprocess worker DB access strategy 고정
- solver runtime tool execution lane 고정
- timeout / process-memory limit / call-count guard
- custom AST preflight implementation 선택과 RestrictedPython 비채택 사유를 [ADR 0001](adr/0001-custom-ast-preflight.md)로 문서화

Acceptance:

- 허용되지 않은 import / file I/O / reflection / raw DB connect가 등록 단계에서 reject된다
- registration lane은 persistent subprocess worker pool을 사용한다
- subprocess worker는 bounded async DB pool을 자체 보유한다
- solver runtime은 등록 완료된 tools를 main process에서 직접 실행한다

### Milestone 3: Synthesis Agent Runtime Skeleton

목표:

- OpenAI Agents SDK 기반 synthesis meta-agent runtime을 만든다

작업:

- schema exploration phase
- category inference phase
- tool/task/solution/verifier generation phase
- explicit memory / tool trace contract
- provider resilience reuse

Acceptance:

- 단일 DB에서 단일 category에 대해 environment draft 하나를 생성할 수 있다
- provider circuit breaker / cooldown / quota rebalance가 synthesis runtime에도 적용된다

### Milestone 4: Hybrid A + Hybrid B Enforcement

목표:

- verifier 신뢰도의 기본 제약을 enforced contract로 만든다

작업:

- `fetch_facts(answer, tools) -> facts`
- `facts_match_answer_claims(answer, facts)`
- `check_constraints(answer, facts)`
- aggregate는 `check_constraints()`에서만 계산한다는 규칙 고정
- Stage 2/3에서 tool call 금지 enforcement
- Runtime instrumentation으로 factual claim vs tool usage 로그 수집

Acceptance:

- pure-hardcoded verifier는 reject된다
- Stage 2/3에서 tool 재호출 verifier는 reject된다
- materialized facts schema가 verifier artifact에 포함된다
- aggregate constraint가 raw facts만으로 재계산된다

### Milestone 5: Proof Environment Vertical Slice

목표:

- compositional proof environment 하나를 end-to-end로 통과시킨다

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

- synthetic fixture DB 구축
- tool synthesis
- task synthesis
- solution synthesis
- verifier synthesis
- self-consistency loop

Acceptance:

- solution이 verifier를 통과한다
- verifier는 hybrid A/B를 만족한다
- review pack에서 사람이 봐도 compositional task다

### Milestone 6: Self-Consistency Policy and Difficulty Escalation

목표:

- synthesis loop의 종료 조건과 난이도 crank를 명시적 policy로 구현한다

작업:

- `max_self_consistency_iterations`
- infeasible discard path
- discard consumes budget 정책
- `db_id x category` failure counter와 backoff queue
- tool change -> solution/verifier invalidation policy
- verifier weakening 금지 규칙
- one-axis-per-step difficulty crank
- crank termination policy

Acceptance:

- verifier relaxation으로만 통과하는 gaming이 reject된다
- difficulty vector가 monotonic하게 증가한다
- 반복 discard가 scheduler-level backoff로 이어진다

### Milestone 7: Hybrid C + Hybrid D

목표:

- shadow verifier와 cross-instance consistency를 production mandatory로 붙인다

작업:

- shadow prompt strategy 2종 구현
- session separation
- temperature separation
- instance_space contract
- deterministic instance sampler
- instance별 서로 다른 valid solution 확인
- shadow verifier도 same Hybrid A/B contract와 registration policy를 따르도록 enforcement

Acceptance:

- shadow disagreement rate가 측정된다
- instance별 다른 valid solution이 verifier에 의해 correctly accepted된다
- shadow verifier가 weaker contract로 빠지지 않는다

주의:

- model family diversity는 proof stage optional이다
- second family backend가 준비되면 production accepted env에서 mandatory로 승격한다

### Milestone 8: Quality Gate 5-Stage Pipeline

목표:

- environment acceptance를 orchestrated gate로 만든다

단계:

1. code registration policy
2. self-consistency
3. shadow verifier agreement
4. cross-instance consistency
5. solver pass-rate band

작업:

- stage별 rejection reason schema
- retry / discard policy
- CI-based early termination
- 공식 pass/fail은 primary verifier, shadow는 diagnostics only로 명시

Acceptance:

- environment는 5단계를 모두 통과해야 accepted 된다
- rejection reason이 structured하게 남는다

### Milestone 9: Real DB Single-Environment Validation

목표:

- synthetic proof 뒤에 real DB 적응성 공백을 없앤다

작업:

- Sakila에서 single environment 생성
- 가능한 한 두 번째 real DB에서도 single environment 생성
- domain inference / tool synthesis / verifier grounding failure taxonomy 수집
- prompt / synthesis policy 보강

Acceptance:

- synthetic fixture가 아닌 real DB에서 environment 하나를 end-to-end로 생성할 수 있다
- real DB failure mode가 taxonomy로 정리되어 다음 단계 입력으로 남는다

### Milestone 10: Domain Scheduler

목표:

- 여러 DB를 registry에서 순차 처리하는 production loop를 만든다

작업:

- DB add/remove contract
- round-robin / priority queue
- per-DB progress tracking
- retry / backoff / starvation protection
- checkpoint / resume

Acceptance:

- DB가 동적으로 추가되어도 scheduler가 처리할 수 있다
- DB 간 task가 섞이지 않는다

### Milestone 11: Environment Registry, Dedup, Coverage

목표:

- accepted environment를 durable registry로 관리한다

작업:

- filesystem registry
- sqlite index
- WAL + single-writer append queue 정책
- exact dedup
- semantic dedup
- category taxonomy mapping
- difficulty-band coverage counters
- generator version compatibility policy
- selective regeneration queue scheduling policy

Acceptance:

- near-duplicate environment를 exact + semantic 두 단계로 잡을 수 있다
- category/difficulty coverage를 계산하고 skew를 볼 수 있다

### Cross-Cutting Workstream: Review Pack and Observability

이건 마지막 milestone이 아니라 전 과정에 걸친 cross-cutting requirement다.

작업:

- review pack에 constraint summary 추가
- verifier / shadow summary 추가
- instance summary 추가
- environment-level metrics 노출
- quality taxonomy 기록 포맷 고정

Acceptance:

- 각 milestone마다 small review batch를 생성해 정성 평가할 수 있다
- qualitative rubric 기준으로 pass/fail을 기록할 수 있다

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
  require_all_instances_pass_verifier_consistency: true
  max_self_consistency_iterations: 5
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
- verifier stage separation
- self-consistency loop
- difficulty crank monotonicity
- cross-instance checker stability
- proof environment binary verification

### Layer 3: Tool Self-Test

agent-generated tools는 self-consistency 전에 self-test를 통과해야 한다.

운영 결정:

- tool self-test도 synthesis agent가 tool synthesis 단계의 artifact로 함께 작성한다
- runtime은 등록 전에 이 self-test를 실행하고 실패 시 tool synthesis 단계로 되돌린다

- happy-path lookup
- empty result behavior
- timeout behavior
- deterministic ordering

## Review Strategy

이 rewrite에서는 review pack 정성 평가가 필수다.

반드시 반복한다.

1. small environment batch 생성
2. question / tool set / verifier summary / constraint summary를 직접 읽는다
3. 품질 문제를 taxonomy로 기록한다
4. prompt / policy / category inference를 수정한다

즉 green tests만으로 진행 결정을 내리지 않는다.

## Success Criteria

이 계획의 성공은 아래로 판단한다.

- generated environment가 lookup task처럼 보이지 않는다
- solver가 실제 composition reasoning을 해야 한다
- verifier는 DB-grounded deterministic binary reward를 준다
- shadow / cross-instance / pass-rate 품질 필터가 모두 동작한다
- arbitrary DB를 registry에 추가해도 pipeline이 돌아간다
- 최근 review pack 10개 중 최소 7개가 아래 rubric을 만족한다
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
- fact/constraint stage 경계가 흐려질 때
- proof environment 없이 generalized factory부터 만들려 할 때
- review pack 품질이 낮은데도 green tests만으로 진행하려 할 때
