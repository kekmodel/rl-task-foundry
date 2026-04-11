# Plan 2: Composition-Centric Rewrite Plan

## Objective

이 계획의 목적은 RL Task Foundry를 `path-centric single-value lookup generator`에서  
`composition-centric constraint satisfaction task generator`로 전면 재설계하는 것이다.

이번 계획은 clean break를 전제로 한다.

- 기존 task generation layer는 더 이상 incremental improvement하지 않는다
- infra / solver / orchestration 자산은 최대한 보존한다
- rewrite는 `spec -> core contract -> proof task -> generalized factory` 순서로 진행한다

## Deliverable

Plan 2가 끝나면 아래가 가능해야 한다.

1. DB schema에서 entity / relation primitive를 읽는다
2. path catalog 위에 plan catalog를 생성한다
3. `CompositeTaskPlan + ConstraintDSL + AnswerSchema v2` 기반 task contract를 만든다
4. deterministic candidate enumeration 또는 compiled checker로 valid solution set을 정의한다
5. solver answer를 execution-based constraint checker로 binary verification한다
6. entity-centric tools로 composition task를 풀게 한다
7. pass-rate band를 보고 slot/constraint 중심으로 난이도를 조절한다
8. review pack으로 생성 품질을 지속적으로 정성 평가한다

## Explicit Non-Deliverable

이번 계획은 아래를 하지 않는다.

- path-centric Tier A와 composition-centric Tier C를 병행 운영
- 현재 path-centric accepted dataset을 production training에 사용
- 기존 task factory를 장기간 유지한 채 우회 보강

## Rewrite Constraints

### Constraint 1: Code Freeze Boundary

task generation 관련 코드는 더 이상 점진 개선하지 않는다.

문서 합의 전에는 아래만 허용된다.

- spec / plan 문서 수정
- baseline snapshot 보존
- review artifact 확인

### Constraint 2: Infra Preservation

아래는 가능한 한 유지한다.

- `infra/`
- `config/`
- `solver/backend_openai_agents.py`
- `calibration/`
- `pipeline/orchestrator.py`의 runtime skeleton
- `schema/introspect.py`, `schema/graph.py`
- `infra/json_chat_client.py`

### Constraint 3: Binary Reward 유지

새 verifier도 RLVR reward는 binary만 사용한다.

- all hard constraints satisfied -> `1`
- otherwise -> `0`

diagnostic per-constraint scores는 기록만 한다.

### Constraint 4: Determinism 우선

composition task여도 verifier noise는 허용하지 않는다.

- deterministic candidate enumeration
- stable canonical solution selection
- compiled checker replay
- tie-break / float / date / NULL 규칙 명시

## Milestones

### Milestone 0: Freeze and Baseline

목표:

- 현재 path-centric baseline을 archive하고 rewrite 입력으로 삼는다

작업:

- 최신 review pack archive 유지
- current green test suite를 baseline으로 기록
- production run 금지 상태 명시

Acceptance:

- baseline review pack 경로가 문서에 기록돼 있다
- rewrite 기간 동안 accepted dataset export는 training source로 사용되지 않는다

### Milestone 1: Core Contract Redesign

목표:

- composition-centric core contract를 타입 수준에서 정의한다

작업:

- `CompositeTaskPlan`
- `EntitySlot`
- `ConstraintDSL`
- `AnswerSchema v2`
- `GroundTruth v2`
- `VerifyResult` diagnostic extension

Files:

- `tasks/models.py`
- `truth/schemas.py`
- `verification/policies.py`

Acceptance:

- one-file prototype로 trip-planning 수준 task contract를 표현할 수 있다
- nested / array answer schema가 타입으로 표현된다

### Milestone 2: Constraint Checker and Solution Methodology

목표:

- `exact field match`가 아니라 `constraint execution`을 verifier source of truth로 만든다

작업:

- ConstraintDSL evaluator 설계
- candidate enumeration contract 설계
- canonical solution selection contract 설계
- checker payload serialization 설계
- deterministic rules 공통 모듈 설계

Files:

- `truth/generator.py`
- `verification/compare.py`
- `truth/canonicalize.py`

Acceptance:

- valid solution이 여러 개인 task도 verifier가 binary pass/fail을 결정할 수 있다
- per-constraint diagnostics를 기록할 수 있다

### Milestone 3: Entity-Centric Tool Surface

목표:

- path-bound tool bundle을 entity-centric tool bundle로 대체한다

작업:

- entity retrieval tool classes 정의
- attribute selection contract 정의
- deterministic bounded query rule 정의
- `L1/L2` naming layer 유지 전략 정의

Files:

- `tools/models.py`
- `tools/compiler.py`
- `tools/sql_templates.py`

Acceptance:

- 하나의 composition proof task를 solver가 entity-centric tools만으로 풀 수 있다
- tool surface가 schema chain 설명 없이도 answerable하다

### Milestone 4: Plan Catalog

목표:

- path catalog 위에 composition template abstraction을 올린다

작업:

- plan template families 정의
- slot graph abstraction 정의
- difficulty feature 정의
- template viability checks 정의

Files:

- `schema/path_catalog.py`
- `tasks/factory.py`
- 신규 `plan catalog` 모듈

Acceptance:

- one path가 아니라 multiple relation primitive를 합쳐 하나의 plan template를 만들 수 있다
- difficulty가 slot/constraint 기준으로 계산된다

### Milestone 5: Proof Task Vertical Slice

목표:

- trip-planning 수준의 compositional proof task 하나를 end-to-end로 통과시킨다

권장 proof task:

- itinerary / assignment / bundle construction 중 하나
- 3개 이상 slot
- uniqueness constraint 포함
- conditional budget constraint 포함
- answer schema는 array of objects

작업:

- fixture DB 또는 synthetic fixture schema 준비
- plan 생성
- canonical solution / checker 생성
- solver execution
- binary verification
- review pack 생성

Acceptance:

- review pack에서 사람이 봐도 compositional task로 인정된다
- solver가 실제로 constraint reasoning을 해야 한다
- verifier noise가 없다

### Milestone 6: Question Composer and Judge Rewrite

목표:

- question / tool presentation / judge를 composition contract 기준으로 다시 짠다

작업:

- question composer agent prompt 전면 수정
- task package judge rubric 전면 수정
- tool presentation judge와의 관계 재정의
- seed fallback 정책 재정의

Files:

- `tasks/composer.py`
- `tasks/package_validation.py`
- `tasks/question_generation.py`

Acceptance:

- question이 single-value lookup phrasing을 벗어난다
- review pack 정성평가에서 “실제 사용자 요청”처럼 보인다

### Milestone 7: Generalized Composite Factory

목표:

- proof task 하나에서 끝내지 않고 arbitrary DB primitive 위에 composition task를 생성한다

작업:

- generic template matching
- deterministic anchor/reference sampling
- constraint instantiation
- negative outcome policy 재정의

Files:

- `tasks/factory.py`

Acceptance:

- 특정 fixture DB에 특화되지 않고 다른 relational schema에도 적용 가능하다
- review pack에서 family 다양성이 실제로 나타난다

### Milestone 8: Orchestrator Integration

목표:

- 새 task contract를 기존 orchestration skeleton에 연결한다

작업:

- composed task -> solver -> checker -> export
- calibration decision을 slot/constraint difficulty 기준으로 갱신
- checkpoint / budget / circuit breaker 경로 유지

Files:

- `pipeline/orchestrator.py`
- `calibration/*`

Acceptance:

- rolling orchestration, provider resilience, checkpoint, budget가 새 paradigm에서도 유지된다

### Milestone 9: Review and Export Readiness

목표:

- rewrite 결과를 사람이 계속 볼 수 있는 artifact loop를 고정한다

작업:

- review pack schema 확장
- constraint summary 포함
- canonical solution / valid solution semantics 포함
- accepted export schema 재정의

Files:

- `pipeline/review_pack.py`
- `pipeline/export.py`

Acceptance:

- review pack만 봐도 task quality와 verifier contract를 이해할 수 있다

## First Implementation Sequence

코드 재작성 순서는 아래를 따른다.

1. `spec.md` 확정
2. `plan.md` 확정
3. core contract 타입 정의
4. proof task vertical slice
5. verifier / checker proof
6. tool surface proof
7. composer / judge proof
8. generalized factory
9. orchestrator reintegration

즉 generalized rewrite 전에 반드시 `proof task`를 먼저 끝까지 증명한다.

## Recommended Proof Task Shape

proof task는 아래 특성을 가져야 한다.

- slot 3개 이상
- cross-slot uniqueness
- one conditional constraint
- one numeric threshold
- one locality / compatibility constraint
- answer format은 list of objects

예:

- 3-day itinerary
- multi-shift roster
- constrained bundle assembly

## Difficulty Redefinition

adaptive difficulty는 아래 순서로 올린다.

1. slot count
2. candidate width
3. constraint count
4. conditional depth
5. uniqueness scope
6. temporal window complexity
7. tool presentation level

더 이상 `required_hops`가 중심 난이도 축이 아니다.

## Test Strategy

rewrite 중 테스트는 두 층으로 나눈다.

### Layer A: Infra Regression

현재 green suite는 계속 유지한다.

목적:

- db pool
- checkpoint
- budget
- solver runtime
- provider resilience
- calibration skeleton

### Layer B: New Paradigm Proof

새 테스트는 proof task 중심으로 시작한다.

포함:

- contract validation
- checker determinism
- multiple valid solution handling
- binary verification
- review pack quality smoke

## Review Strategy

이 rewrite에서는 정성 평가가 필수다.

반드시 반복한다.

1. small review pack 생성
2. 질문 / tool set / answer format / constraint summary를 사람 눈으로 확인
3. 문제 유형을 기록
4. prompt / contract / template를 수정

즉 테스트 green만으로 진행 판단을 하지 않는다.

## Success Criteria

이 계획의 성공은 아래로 판단한다.

- generated task가 single-value lookup처럼 보이지 않는다
- solver가 실제 composition reasoning을 해야 한다
- verifier가 deterministic binary reward를 준다
- arbitrary relational DB에도 적용 가능한 template abstraction이 있다
- review pack 정성 평가에서 예시 수준 이상의 quality가 반복적으로 나온다

## Stop Conditions

아래 중 하나면 다시 설계를 멈추고 조정한다.

- verifier가 canonical exact-match에 다시 의존하게 될 때
- question만 자연스러워지고 core contract는 lookup에 머물 때
- template가 특정 fixture schema에만 맞을 때
- review pack 품질이 높지 않은데도 test green만으로 진행하려 할 때

