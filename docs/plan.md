# Plan 3: Synthesis-Agent Hybrid Rewrite Plan

## Objective

이 계획의 목적은 RL Task Foundry를 `path-centric task generator`에서  
`synthesis-agent driven environment generator`로 전면 재설계하는 것이다.

새 시스템은 임의의 read-only PostgreSQL DB를 등록하면, 해당 DB에 대해 compositional task environment를 자동 생성하고, hybrid verifier 기반 quality gate를 통과한 환경만 registry에 누적한다.

## Rewrite Ground Rules

1. task generation 관련 코드의 incremental polishing은 중단한다
2. 먼저 문서와 core contract를 고정한다
3. 그 다음 proof environment 하나를 끝까지 만든다
4. generalized factory는 proof environment 이후로 미룬다
5. infra / solver / orchestration skeleton은 최대한 유지한다

## Deliverable

Plan 3가 끝나면 아래가 가능해야 한다.

1. DB registry에 새 DB를 등록한다
2. scheduler가 DB를 하나 선택해 synthesis loop를 돈다
3. synthesis agent가 environment 4-tuple을 생성한다
4. code registration policy가 generated code를 검사한다
5. self-consistency, shadow verifier, cross-instance, solver pass-rate quality gate를 통과한 environment만 registry에 커밋한다
6. review pack에서 compositional task quality를 사람이 지속적으로 확인할 수 있다

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

## Milestones

### Milestone A: Freeze, Baseline, and Registry Foundations

목표:

- rewrite 기간의 baseline과 registry contract를 고정한다

작업:

- latest review pack을 baseline snapshot으로 기록
- current green suite를 infra regression baseline으로 명시
- environment registry directory layout 설계
- sqlite registry index schema 설계
- DB registry metadata schema 설계

산출물:

- `docs/spec.md`
- `docs/plan.md`
- registry schema draft

Acceptance:

- baseline snapshot path가 문서에 기록돼 있다
- registry schema가 env/db/category/difficulty/status/pass-rate를 표현할 수 있다

### Milestone B: Core Contracts

목표:

- environment generation의 핵심 타입을 고정한다

작업:

- `EnvironmentContract`
- `ToolContract`
- `TaskContract`
- `VerifierContract`
- `ShadowVerifierContract`
- `CrossInstanceSet`

필수 요구:

- environment directory 레이아웃과 1:1 대응
- function signature contract 포함
- metadata hash / provenance fields 포함

Acceptance:

- environment 하나를 코드 없이 JSON/YAML만으로 기술할 수 있다
- verifier stage separation을 타입으로 표현할 수 있다

### Milestone C: Code Registration Policy

목표:

- generated Python code를 안전하게 등록 / 실행할 수 있게 한다

작업:

- AST import allowlist
- forbidden builtins / attributes
- forbidden syntax rules
- function signature validation
- runtime timeout contract
- call-count / memory guard contract
- readonly DB mediation contract

파일 후보:

- 신규 `synthesis/registration_policy.py`
- `infra/` runtime wrappers

Acceptance:

- 허용되지 않은 import / file I/O / subprocess / eval이 등록 단계에서 차단된다
- tool / solution / verifier signature mismatch는 즉시 reject된다

### Milestone D: Synthesis Agent Runtime

목표:

- OpenAI Agents SDK 기반 synthesis meta-agent runtime을 만든다

작업:

- schema exploration phase
- category inference phase
- tool/task/solution/verifier generation phase
- repair / retry loop
- explicit memory / tool trace contract

재사용:

- `solver/backend_openai_agents.py` 구조
- `submit_result()` 개념 대신 environment draft finalization event

Acceptance:

- 단일 DB에서 단일 category에 대해 environment draft 하나를 생성할 수 있다

### Milestone E: Hybrid A + Hybrid B Enforcement

목표:

- verifier 신뢰도의 핵심 제약을 enforced contract로 만든다

작업:

- fact check must use tools
- fact stage / constraint stage separation
- static analyzer로 verifier body 검사
- runtime instrumentation으로 tool usage trace 수집

Acceptance:

- pure-hardcoded verifier는 등록 단계에서 reject된다
- fact-free constraint-only verifier는 reject된다

### Milestone F: Proof Environment Vertical Slice

목표:

- compositional proof environment 하나를 end-to-end로 통과시킨다

권장 proof task:

- trip planning 비슷한 fixture
- 3개 이상 slot
- uniqueness
- conditional threshold
- locality constraint
- answer schema는 `list[object]`

작업:

- fixture DB 또는 synthetic DB 준비
- tools synthesis
- task synthesis
- solution synthesis
- verifier synthesis
- self-consistency loop

Acceptance:

- solution이 verifier를 통과한다
- verifier는 hybrid A/B 제약을 만족한다
- review pack에서 사람이 봐도 compositional task로 인정된다

### Milestone G: Hybrid C + Hybrid D

목표:

- shadow verifier와 cross-instance consistency를 production mandatory로 붙인다

작업:

- independent shadow verifier generation
- disagreement logging
- instance parameterization
- cross-instance validation loop

Acceptance:

- shadow disagreement rate가 측정된다
- instance별 서로 다른 valid solution을 verifier가 correctly accept한다

### Milestone H: Quality Gate 5-Stage Pipeline

목표:

- 전체 quality filter를 orchestrated pipeline으로 만든다

단계:

1. code registration policy
2. self-consistency
3. shadow verifier agreement
4. cross-instance consistency
5. solver pass-rate band

작업:

- stage별 rejection reason schema
- stage별 retry / discard policy
- CI-based early termination

Acceptance:

- environment는 5단계를 모두 통과해야 accepted 된다
- rejection reason이 structured하게 남는다

### Milestone I: Domain Scheduler and Multi-DB Production Loop

목표:

- 여러 DB를 registry에서 순차 처리하는 production loop를 만든다

작업:

- DB add/remove contract
- round-robin / priority scheduler
- per-DB progress tracking
- retry / backoff / starvation protection
- checkpoint / resume

Acceptance:

- DB가 동적으로 추가되어도 scheduler가 처리할 수 있다
- DB 간 task가 섞이지 않는다

### Milestone J: Environment Registry, Dedup, Coverage

목표:

- accepted environment를 durable registry로 관리한다

작업:

- filesystem registry
- sqlite index
- dedup signatures
- coverage counters
- generator versioning

Acceptance:

- 같은 DB 안에서 near-duplicate environment를 감지할 수 있다
- category / difficulty coverage를 계산할 수 있다

### Milestone K: Review Pack and Observability

목표:

- 사람이 계속 품질을 볼 수 있는 artifact loop를 고정한다

작업:

- review pack에 constraint summary 추가
- verifier summary / shadow summary 추가
- instance summary 추가
- environment-level metrics 노출

Acceptance:

- review pack만 봐도 environment 품질과 verifier contract를 이해할 수 있다

## First Proof Sequence

실제 구현 순서는 아래가 좋다.

1. core contracts
2. registration policy
3. synthesis runtime skeleton
4. hybrid A/B enforcement
5. proof environment vertical slice
6. hybrid C/D
7. quality gate pipeline
8. scheduler
9. registry / dedup / coverage

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
- verifier stage separation
- self-consistency loop
- cross-instance checker stability
- proof environment binary verification

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

## Stop Conditions

아래 중 하나면 구현을 멈추고 문서로 돌아간다.

- verifier가 다시 self-consistent but ungrounded Python checker로 흐를 때
- question만 자연스럽고 contract는 여전히 lookup일 때
- proof environment 없이 generalized factory부터 만들려 할 때
- review pack 품질이 낮은데도 green tests만으로 진행하려 할 때

