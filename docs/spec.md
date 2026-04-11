# RL Task Foundry — Synthesis-Agent Hybrid Design Spec

## Overview

이 프로젝트의 목표는 사람이 read-only PostgreSQL DB를 등록하면, 그 DB를 바탕으로 RLVR용 compositional task environment를 자동 생성하고, 엄격한 quality gate를 통과한 environment만 registry에 누적하는 상시 파이프라인을 만드는 것이다.

핵심 paradigm은 더 이상 `path-centric single-value lookup`이 아니다.

새 기준은 아래와 같다.

- unit of generation은 `task`가 아니라 `<environment, tools, task, verifier>` 4-tuple이다
- synthesis agent가 tool, task, solution, verifier를 함께 합성한다
- verifier는 Python으로 표현되더라도 DB-grounded hybrid rules를 강하게 따라야 한다
- solver는 compositional reasoning을 풀고, verifier는 deterministic binary reward를 준다

대표 목표 task는 아래 수준이다.

- trip planning
- constrained roster / assignment
- bundle assembly with thresholds
- no-repeat recommendation
- conditional budget / locality / uniqueness planning

즉 이 시스템은 “DB에서 값을 하나 꺼내는 질문 생성기”가 아니라,  
“임의 DB에 대해 compositional reasoning environment를 자동 생성하고 품질 필터링하는 production pipeline”이다.

## Why A Clean Break Is Required

기존 path-centric generator는 다음 상한을 가진다.

- answer shape가 대부분 scalar / count / bool / small record에 머문다
- question이 자연스러워져도 core contract는 lookup이다
- uniqueness, threshold, conditional branching, multi-slot coupling을 자연스럽게 담기 어렵다
- reverse aggregate, multi-field record, sampling 보강을 해도 complexity ceiling 자체는 깨지지 않는다

따라서 필요한 것은 점진 개선이 아니라 paradigm 전환이다.

## Operating Principles

1. `hard to solve, easy to verify`
2. verifier 신뢰도는 RLVR의 핵심 자산이며 절대 희생하지 않는다
3. quality가 throughput보다 우선이다
4. arbitrary relational DB에 대응해야 하며 schema/domain 하드코딩을 금지한다
5. DB 간 task를 섞지 않는다. 한 environment는 정확히 한 DB에 속한다
6. DB는 언제든 registry에 추가될 수 있어야 한다
7. reward는 binary만 사용하고 diagnostics는 별도 채널로 기록한다

## System Goal

시스템의 장기 목표는 아래다.

- 사용자는 read-only PostgreSQL DB만 등록한다
- system이 schema와 sample row를 탐색한다
- system이 domain category를 추론한다
- synthesis agent가 environment 4-tuple을 생성한다
- self-consistency, shadow verifier, cross-instance, solver pass-rate quality gate를 통과한 environment만 registry에 커밋한다
- registry는 계속 커지고 scheduler는 DB별로 round-robin 또는 priority queue로 작업한다

## Non-Goals

- 기존 path-centric generator와 새 synthesis generator를 병행 유지하는 것
- baseline accepted dataset을 production RL training source로 사용하는 것
- fuzzy / embedding similarity verifier
- verifier correctness를 사람 수동 리뷰에 의존하는 운영
- DB 직접 write가 필요한 environment generation

## Clean Break Policy

이 rewrite는 clean break다.

- 기존 path-centric baseline은 freeze한다
- baseline review pack은 archive로 보존한다
- production run은 새 hybrid pipeline이 proof되기 전까지 금지한다
- task generation code의 incremental polishing은 중단한다

## Assets To Preserve

다음 자산은 최대한 유지하거나 최소 수정 재사용한다.

- `config/` 구조 및 load 메커니즘
- `infra/`
  - db pool
  - events
  - budget
  - checkpoint
  - storage
  - privacy
- `solver/backend_openai_agents.py`
  - `submit_result()`
  - structured output
  - explicit terminal action
- `calibration/`
  - pass-rate band
  - Clopper-Pearson CI
  - safe early stop
- `pipeline/orchestrator.py`
  - rolling orchestration
  - provider semaphore
  - provider circuit breaker
  - checkpoint / budget / export skeleton
- `schema/introspect.py`, `schema/graph.py`, `schema/path_catalog.py`
- `verification/shadow.py`의 independent-verifier 개념
- `infra/json_chat_client.py`
- `cli.py`
- `pipeline/review_pack.py`의 shell 구조

## Rewrite Targets

전면 재설계 대상으로 본다.

- `tasks/factory.py`
- `tasks/composer.py`
- `tasks/question_generation.py`
- `tasks/package_validation.py`
- `tasks/provenance.py`
- `truth/generator.py`
- `truth/canonicalize.py`의 contract 부분
- `tools/compiler.py`
- `tools/sql_templates.py`
- `tools/model_naming.py`
- `tools/naming_eval.py`
- `verification/compare.py`

## System Architecture

```text
DB Registry
  -> Domain Scheduler
  -> Synthesis Orchestrator
  -> Quality Gate
  -> Environment Registry
```

### Layer Breakdown

1. `Schema + Domain Discovery`
   - schema introspection
   - FK graph
   - sample row inspection
   - domain/category inference

2. `Synthesis Layer`
   - tool synthesis
   - task synthesis
   - solution synthesis
   - verifier synthesis
   - difficulty crank

3. `Registration and Runtime Safety Layer`
   - code registration policy
   - function contract validation
   - subprocess isolation
   - DB access mediation

4. `Quality Gate Layer`
   - self-consistency
   - shadow verifier
   - cross-instance consistency
   - solver pass-rate filter

5. `Registry + Orchestration Layer`
   - filesystem environment registry
   - sqlite index
   - dedup
   - coverage
   - scheduler

## Environment as the Core Unit

### Environment 4-Tuple

각 environment는 아래 4-tuple이다.

- `environment`
- `tools`
- `task`
- `verifier`

보조 아티팩트:

- `solution`
- `shadow_verifier`
- `instance_space`
- `instances`

### Filesystem Layout

```text
environments/
  <env_id>/
    environment.yaml
    tools.py
    task.json
    solution.py
    verifier.py
    shadow_verifier.py
    instance_space.yaml
    instances/
      instance_1.json
      instance_2.json
      instance_3.json
```

### Environment Metadata

`environment.yaml`은 최소 아래를 포함한다.

```yaml
env_id: str
db_id: str
domain: str
category: str
difficulty_vector: dict[str, int | float]
created_at: str
generator_version: str
tool_signature: str
task_signature: str
verifier_signature: str
status: draft | accepted | rejected | archived
quality_metrics:
  self_consistency_pass: bool
  shadow_disagreement_rate: float
  solver_pass_rate: float
  solver_ci_low: float
  solver_ci_high: float
```

## Task Category Taxonomy

category는 agent가 완전히 자유 텍스트를 만들지 않는다. runtime taxonomy에 매핑한다.

초기 taxonomy:

- `itinerary`
- `assignment`
- `bundle_selection`
- `eligibility_filter`
- `no_repeat_recommendation`
- `threshold_routing`
- `temporal_planning`
- `other`

agent는 free-form proposal을 만들 수 있지만 registry에는 taxonomy-mapped value로 저장한다.

## Difficulty Model

difficulty는 단일 level이 아니라 vector다.

```python
DifficultyAxes:
  SLOT_COUNT
  CONSTRAINT_COUNT
  CONDITIONAL_DEPTH
  THRESHOLD_TIGHTNESS
  UNIQUENESS_SCOPE
  TEMPORAL_SPAN
  CANDIDATE_WIDTH
```

운영 원칙:

- 한 escalation step에서는 정확히 한 축만 증가시킨다
- difficulty crank는 monotonic vector를 가져야 한다
- 종료 조건은 pass-rate band 수렴 또는 iteration budget 소진이다

## Tool Contract

generated tool은 아래 signature를 가져야 한다.

```python
async def tool_name(conn, **kwargs) -> Any
```

tool은 read-only DB mediation layer만 통해 동작한다.

### Tool Reuse Policy

v1에서는 tool registry를 environment 간 공유하지 않는다.

- tools는 per-environment 독립 artifact다
- dedup은 tool text보다 `task + verifier semantics`를 더 강하게 본다
- 같은 의미의 tool이 구현 세부만 달라도 semantic dedup에서 잡을 수 있어야 한다

## Task Contract

`task.json`은 최소 아래를 표현해야 한다.

- natural-language question
- structured output schema
- constraint summary
- difficulty vector
- category
- instance parameters

answer는 flat field에 제한되지 않는다.

- object
- list[object]
- nested structures

## Solution Contract

`solution.py`는 oracle/reference 역할이다.

### Role

- self-consistency 단계에서만 사용
- cross-instance 생성 검증에서 사용
- production solver runtime에는 절대 노출되지 않는다

### Signature

```python
def solve(tools) -> dict
```

solution은 multiple valid solutions 중 하나를 안정적으로 반환하면 된다. canonicality는 verifier가 아니라 solution의 책임이 아니다.

## Verifier Contract

verifier는 hybrid DB-grounded contract를 따라야 한다.

### Signature

```python
def verify(answer, tools) -> bool
```

### Official Judgment

- primary verifier가 공식 pass/fail을 결정한다
- shadow verifier는 동일 attempt에 대해 같이 실행되지만 disagreement 통계 수집용이다
- shadow는 production acceptance gate에는 쓰이지만 공식 reward source는 아니다

## Hybrid Verifier Rules

### Hybrid A — Tool Call Presence Is Only A Weak Baseline

verifier는 answer의 factual claim을 검사할 때 tool call을 통해 DB 사실을 다시 조회해야 한다.

허용:

- `hotel_name -> tools.get_infos_by_hotel(["price"], hotel_name)`
- `restaurant_name -> tools.get_city_by_restaurant(...)`

금지:

- answer literal을 하드코딩된 문자열/숫자와만 비교
- tool 호출 없이 pure Python if/else만으로 fact correctness 판정

정적 baseline check:

- verifier 본문에 `tools.` 호출이 최소 1회 이상 있어야 한다

하지만 이것만으로는 충분하지 않다.  
실질적 방어는 `Hybrid C + Hybrid D + runtime instrumentation`이 담당한다.

### Hybrid B — Stage Boundary Must Be Explicit

verifier는 아래 세 함수를 가진 명시적 계약을 따라야 한다.

```python
def fetch_facts(answer, tools) -> dict:
    ...

def facts_match_answer_claims(answer, facts) -> bool:
    ...

def check_constraints(answer, facts) -> bool:
    ...
```

#### Stage 1: Fact Materialization

`fetch_facts(answer, tools)`는 answer에 등장하는 entity와 attribute를 DB-grounded 사실로 materialize한다.

- tool call 사용 필수
- constraint 판단 금지
- 반환값은 `facts` dict

#### Stage 2: Claim Consistency

`facts_match_answer_claims(answer, facts)`는 answer가 facts와 일치하는지만 본다.

- 추가 tool call 금지
- pure function only

#### Stage 3: Constraint Checking

`check_constraints(answer, facts)`는 facts dict만 입력으로 받아 hard constraints를 평가한다.

- 추가 tool call 금지
- pure function only

즉 facts dict가 Stage 1과 Stage 2/3 사이의 유일한 경계 계약이다.

### MaterializedFacts Contract

environment는 `task.json`에 facts schema를 선언해야 한다.

```python
FactSpec:
  key: str
  entity_ref: str
  attribute: str
  value_type: str
  nullable: bool
```

`fetch_facts()`는 반드시 이 schema에 맞는 dict를 반환해야 한다.

이 계약 덕분에:

- Stage 2/3에서 tool 재호출 우회를 막을 수 있다
- verifier review와 diagnostics가 쉬워진다

### Hybrid C — Shadow Verifier Independence Strategy

shadow verifier는 아래 독립성 요건을 가진다.

mandatory:

- 다른 synthesis session
- 다른 prompt template
- 다른 temperature

preferred:

- 다른 model family

운영 결정:

- proof-stage에서는 model family diversity는 optional이다
- production-grade accepted environment에서는 second family backend가 준비되면 model-family-diverse shadow를 mandatory로 승격한다

#### Prompt Strategy

primary verifier prompt:

- top-down
- task statement와 constraints를 직접 verifier로 옮기게 함

shadow verifier prompt:

- bottom-up
- tool surface와 task output을 보고 필요한 invariants를 재구성하게 함

즉 둘은 같은 내용을 같은 방식으로 paraphrase하지 않는다.

### Hybrid D — Cross-Instance Generation Is Runtime-Owned

cross-instance variation의 주체는 runtime이다.

agent는 `instance_space.yaml`만 정의한다.

```yaml
anchor_query: ...
parameter_spaces:
  budget_bucket: [low, mid, high]
  day_count: [2, 3]
sampling_strategy: deterministic_hash
```

runtime는 이 parameter space에서 deterministic sampler로 N instances를 만든다.

요구:

- instance별 anchor / threshold / context가 달라야 한다
- valid solution도 instance별로 달라져야 한다

즉 agent가 instance JSON 전체를 고정으로 박는 구조가 아니라, runtime-owned variation을 강제한다.

## Code Registration Policy

generated Python code는 in-process로 실행하지 않는다.  
`AST preflight + subprocess isolation`이 mandatory다.

운영 결정:

- v1은 `RestrictedPython`을 전면 채택하지 않는다
- 대신 narrow DSL 전제를 둔 custom AST preflight + subprocess isolation을 사용한다
- production promotion 전에 denylist 우회 사례가 발견되면 RestrictedPython 또는 동등한 safe-subset compiler 도입을 재평가한다

### Static Policy

AST 검사에서 아래를 강제한다.

- import allowlist
- 금지 builtins / attributes
- 금지 syntax
- signature contract
- dunder access ban

#### Forbidden Symbols

- `open`
- `eval`
- `exec`
- `compile`
- `__import__`
- `getattr`
- `setattr`
- `globals`
- `locals`
- `vars`

#### Forbidden Attribute Patterns

- `__*__` dunder attribute 전부 금지
- dynamic attribute assembly 금지
- string literal subscript를 통한 reflective access 금지

#### Import Policy

허용 예:

- `datetime`
- `math`
- `json`
- `re`
- `decimal`
- `typing`
- `collections.abc`
- runtime-provided tool facade

금지 예:

- `os`
- `sys`
- `subprocess`
- `socket`
- raw DB client
- arbitrary file IO helpers

### Registration Error Schema

policy 위반 시 agent에 아래 구조로 피드백한다.

```python
RegistrationError:
  code: str
  line: int | None
  col: int | None
  node_type: str | None
  detail: str
  suggestion: str | None
```

### Runtime Isolation Strategy

v1에서는 generated code를 subprocess로 분리 실행한다.

- `solution.py`, `verifier.py`, `shadow_verifier.py`, generated `tools.py`는 subprocess worker에서 실행
- timeout은 subprocess 단위로 강제한다
- memory limit도 프로세스 단위로 강제한다
- container는 v1 필수는 아니지만, subprocess는 mandatory다

즉 과거의 “sandbox 불필요” 가정은 수정된다.  
container는 optional이지만 subprocess isolation은 필수다.

## Synthesis Agent Loop

### Stage 1: Schema and Category Discovery

- schema introspection
- sample row inspection
- domain/category proposal

### Stage 2: 4-Tuple Synthesis

agent는 아래를 함께 만든다.

- `tools.py`
- `task.json`
- `solution.py`
- `verifier.py`

### Stage 3: Self-Consistency Iterate

기본 loop:

1. solution 실행
2. primary verifier 실행
3. 실패 시 수정 제안
4. iteration budget이 남아 있으면 다시 synthesize

운영 규칙:

- `max_self_consistency_iterations`를 가진다
- tool이 수정되면 solution/verifier를 모두 invalidate하고 재생성한다
- task를 “불가능”으로 분류하고 discard하는 경로가 있어야 한다

#### Gaming Guard

verifier 수정은 임의 완화가 아니다.

- difficulty vector는 self-consistency 수정 과정에서 감소하면 안 된다
- constraint count / conditional depth / threshold strictness를 줄이는 verifier 수정은 disallowed
- verifier를 약화시키는 수정이 필요하면 environment를 discard하고 category synthesis로 되돌린다

즉 self-consistency는 “solution을 맞추는 수리”는 허용하지만, “검사를 느슨하게 하는 수리”는 허용하지 않는다.

### Stage 4: Difficulty Escalation

difficulty crank는 아래 enum 중 하나만 한 iteration에 증가시킨다.

- `SLOT_COUNT`
- `CONSTRAINT_COUNT`
- `CONDITIONAL_DEPTH`
- `THRESHOLD_TIGHTNESS`
- `UNIQUENESS_SCOPE`
- `TEMPORAL_SPAN`
- `CANDIDATE_WIDTH`

종료 조건:

- pass-rate CI가 band 안으로 들어옴
- `max_difficulty_cranks` 소진
- `N`회 연속 infeasible 판정

agent는 임의로 난이도를 올리지 않고 runtime policy가 선택한 axis만 올린다.

### Stage 5: Shadow and Cross-Instance

- independent shadow verifier 생성
- deterministic cross-instance generation
- disagreement / consistency 측정

### Stage 6: Solver Quality Filter

- 여러 solver attempts 실행
- primary verifier로 공식 판정
- shadow verifier로 disagreement 측정
- pass-rate band와 CI 판정

## Quality Gate

Environment는 아래 다섯 단계를 모두 통과해야 accepted 된다.

1. code registration policy
2. self-consistency
3. shadow verifier agreement
4. cross-instance consistency
5. solver pass-rate band

quality filter 예시:

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

## Domain Scheduler and Provider Resilience

### Domain Scheduler

- DB add/remove 지원
- round-robin / priority queue
- DB 단위 독립 처리
- per-DB progress 추적

### Synthesis Provider Resilience

synthesis agent도 solver와 동일하게 provider resilience를 사용한다.

- provider semaphore
- circuit breaker
- cooldown
- quota rebalance

즉 synthesis orchestrator는 기존 `pipeline/orchestrator.py`의 resilience skeleton을 재사용한다.

## Environment Registry

registry는 filesystem + sqlite index의 이중 구조를 가진다.

### SQLite Index

최소 기록 항목:

- env_id
- db_id
- domain
- category
- difficulty_vector
- pass_rate
- CI interval
- shadow disagreement
- created_at
- status
- tool_signature
- task_signature
- verifier_signature
- generator_version

운영 메모:

- sqlite index는 WAL mode + single-writer append queue를 기본으로 한다
- parallel synthesis worker는 직접 index에 동시 write하지 않고 writer lane을 통해 반영한다

### Dedup

dedup은 두 단계다.

1. exact dedup
   - normalized tool signature
   - normalized task signature
   - normalized verifier AST signature
2. semantic dedup
   - task embedding
   - constraint summary embedding

minor naming variation으로 near-duplicate가 통과하면 안 된다.

### Generator Version Policy

- `generator_version`이 바뀌어도 기존 environment를 즉시 삭제하지 않는다
- registry는 versioned coexistence를 허용한다
- 다만 current production target version과 mismatch인 environment는 `legacy` 상태로 마킹할 수 있어야 한다
- major contract change가 발생하면 selective regeneration queue를 만든다

### Coverage

coverage는 아래 grid로 본다.

- `db_id x category x difficulty band`

difficulty band는 difficulty vector를 bucketized한 결과다.

## Review Pack

review pack은 mandatory production gate가 아니다.  
quality gate와 독립적인 qualitative spot-check artifact다.

포함:

- final question
- tool set
- output schema
- constraint summary
- verifier summary
- shadow verifier status
- instance summary
- canonical solution reference

운영 역할:

- 품질 taxonomy 작성
- prompt / policy 회귀 감지
- proof environment 검토

즉 verifier correctness의 source of truth는 아니고, 운영 guardrail이다.

## Proof Environment Decision

첫 proof environment는 Sakila 위에서 억지로 만들지 않는다.

운영 결정:

- first proof는 synthetic fixture DB를 사용한다
- Sakila는 schema introspection / baseline regression 용도로만 남긴다

이유:

- trip-planning 수준 compositional task를 자연스럽게 표현하기 어렵다
- proof task를 schema에 억지로 맞추면 설계가 다시 lookup 쪽으로 끌려간다

## Success Criteria

성공은 아래로 판단한다.

- generated environment가 lookup task처럼 보이지 않는다
- solver가 실제 composition reasoning을 해야 한다
- verifier는 DB-grounded deterministic binary reward를 준다
- shadow / cross-instance / pass-rate 품질 필터가 모두 동작한다
- arbitrary DB를 registry에 추가해도 pipeline이 돌아간다
- 최근 생성한 review pack 10개 중 최소 7개가 “예시 수준 이상” 품질로 정성 평가된다

## Freeze Policy

이 spec 기준으로:

- 기존 task generation code는 incremental polishing을 중단한다
- rewrite는 `spec -> plan -> core contracts -> proof environment -> generalization` 순으로 간다
- proof environment가 나오기 전 production run은 금지한다
