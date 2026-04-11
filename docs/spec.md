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

### Proposed vs Materialized Environment

artifact generation phase에서 agent가 직접 반환하는 객체는 완전한 `EnvironmentContract`가 아니다.

- agent는 `proposed_environment`만 제안한다
- `proposed_environment`에는 agent-authored field만 들어간다
  - `tools`
  - `task`
  - `solution`
  - `tool_self_test`
  - `verifier`
  - `shadow_verifier`
  - `instance_space`
  - `cross_instance_set`
- runtime은 registration gate가 통과된 뒤에만 `EnvironmentContract`를 materialize한다

runtime-owned trust field는 agent가 제안하지 않는다.

- `env_id`
- `db_id`
- `domain`
- `category`
- `created_at`
- `generator_version`
- `tool_signature`
- `task_signature`
- `verifier_signature`
- `status`
- `quality_metrics`

materialization 시점에는 runtime이 위 필드를 다시 계산하고, 최종 payload를 `EnvironmentContract.model_validate(...)`로 재검증한다.

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

축 의미:

- `SLOT_COUNT`
  - solver가 채워야 하는 slot 또는 entity selection 수
- `CONSTRAINT_COUNT`
  - verifier가 강제하는 hard constraint 수
- `CONDITIONAL_DEPTH`
  - 조건 분기의 최대 중첩 깊이
- `THRESHOLD_TIGHTNESS`
  - 0.0~1.0 normalized strictness
- `UNIQUENESS_SCOPE`
  - uniqueness가 동시에 걸리는 slot/entity 범위 수
- `TEMPORAL_SPAN`
  - day-equivalent horizon
- `CANDIDATE_WIDTH`
  - 탐색해야 하는 후보 폭

## Tool Contract

generated tool은 아래 signature를 가져야 한다.

```python
async def tool_name(conn, **kwargs) -> Any
```

tool은 read-only DB mediation layer만 통해 동작한다.

### Error Behavior

tool contract는 success schema만이 아니라 empty/timeout 동작도 명시해야 한다.

- empty result behavior
  - `return_empty`
  - `return_null`
  - `raise_not_found`
- timeout behavior
  - `raise_timeout`
  - `return_error`

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

`constraint_summary`는 review pack / observability용 human-readable summary다.  
실제 authoritative constraint semantics는 verifier code가 책임진다.

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
- v1 registration policy는 이를 `fetch_facts()` 내부의 최소 1회 `tools.*` 호출로 구체화한다

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
- 반환값은 slot/entity 단위의 raw facts여야 한다
- aggregate 값은 이 stage에서 미리 계산하지 않는다

#### Stage 2: Claim Consistency

`facts_match_answer_claims(answer, facts)`는 answer가 facts와 일치하는지만 본다.

- 추가 tool call 금지
- pure function only

#### Stage 3: Constraint Checking

`check_constraints(answer, facts)`는 facts dict만 입력으로 받아 hard constraints를 평가한다.

- 추가 tool call 금지
- pure function only
- 합계, 평균, 개수, min/max 같은 aggregate 계산은 이 stage에서 수행한다

즉 facts dict가 Stage 1과 Stage 2/3 사이의 유일한 경계 계약이다.

v1 registration policy는 아래를 정적으로 enforce한다.

- `verify()`는 `fetch_facts() -> facts_match_answer_claims() -> check_constraints()` 3-stage pipeline을 호출해야 한다
- `verify()`는 `tools.*`를 직접 호출할 수 없다
- `fetch_facts()`는 최소 1회 `tools.*`를 호출해야 한다
- `fetch_facts()`는 `sum/min/max/len/any/all/sorted` 같은 aggregate helper로 pre-aggregated metric을 만들 수 없다
- `facts_match_answer_claims()`와 `check_constraints()`는 `tools`를 참조하거나 호출할 수 없다
- `facts_match_answer_claims()`와 `check_constraints()`는 다른 verifier stage를 호출할 수 없다
- `fetch_facts()`는 `answer`를 읽어 materialization 대상을 정해야 한다
- `facts_match_answer_claims()`는 `answer`와 `facts`를 모두 읽어야 한다
- `check_constraints()`는 `facts`를 읽어야 한다
- `facts_match_answer_claims()`와 `check_constraints()`는 constant `True` / `False` return으로 축약될 수 없다

registration report는 verifier/shadow verifier마다 stage analysis를 기록해,
tool call count와 pure-stage 위반 여부를 diagnostics로 남긴다.

runtime은 이 registration report를 그대로 버리지 않고 draft-level diagnostics로 승격한다.

- 성공한 draft는 `registration_diagnostics`에 static/probe 분석 요약을 가진다
- registration 실패는 `SynthesisRegistrationError(report, diagnostics)`로 승격되어 self-consistency loop가 실패 원인을 바로 읽을 수 있다
- 즉 registration gate는 bool barrier이면서 동시에 다음 iteration을 위한 structured feedback channel이다

v1 dynamic probe는 registration lane subprocess worker에서 아래를 추가 확인한다.

- synthetic answer sample로 `fetch_facts()`, `facts_match_answer_claims()`, `check_constraints()`, `verify()`를 실제 실행한다
- `fetch_facts()` 반환값이 dict인지 확인한다
- `fetch_facts()` 반환 key set이 declared facts schema key set과 일치하는지 확인한다
- `facts_match_answer_claims()`, `check_constraints()`, `verify()`가 bool을 반환하는지 확인한다
- `verify()`의 최종 결과가 staged outcome (`facts_match` 실패면 `False`, 아니면 `check_constraints`)와 일치하는지 확인한다

이 probe는 full semantic correctness 증명이 아니라, weak verifier와 schema drift를 registration 시점에 조기 차단하는 defense-in-depth layer다.

### MaterializedFacts Contract

environment는 `task.json`에 facts schema를 선언해야 한다.

```python
FactSpec:
  key: str
  entity_ref: str
  attribute: str
  value_type: str  # enum constrained
  nullable: bool
  cardinality: one | many
```

허용 value type:

- `str`
- `int`
- `float`
- `bool`
- `date`
- `datetime`
- `list[str]`
- `list[int]`
- `list[float]`

`fetch_facts()`는 반드시 이 schema에 맞는 dict를 반환해야 한다.

운영 결정:

- `fetch_facts()`는 per-entity raw attribute bag만 materialize한다
- facts는 slot 또는 entity collection을 표현할 수 있지만 pre-aggregated metric은 포함하지 않는다
- `"3일 호텔 가격 총합"`, `"식당 평점 평균"` 같은 aggregate는 항상 `check_constraints()`가 raw facts에서 계산한다

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

#### Contract Symmetry

shadow verifier도 weaker contract를 허용하지 않는다.

- 동일한 `verify(answer, tools) -> bool` signature를 따른다
- 동일한 `fetch_facts / facts_match_answer_claims / check_constraints` 3-stage 경계를 따른다
- 동일한 code registration policy와 subprocess isolation을 적용받는다
- 동일하게 tool-grounded fact materialization이 필수다

즉 “다른 각도”는 contract를 느슨하게 하는 것이 아니라, 어떤 facts를 materialize하고 어떤 invariant를 더 강하게 보느냐의 차이로 구현한다.

### Hybrid D — Cross-Instance Generation Is Runtime-Owned

cross-instance variation의 주체는 runtime이다.

agent는 `instance_space.yaml`만 정의한다.

```yaml
anchor_query:
  sql: SELECT ...
  outputs: [anchor_id, city_id]
parameters:
  budget_bucket:
    kind: enum
    values: [low, mid, high]
  day_count:
    kind: int_range
    min: 2
    max: 3
sampling:
  strategy: deterministic_hash
  seed: 17
instance_count: 3
```

runtime는 이 parameter space에서 deterministic sampler로 N instances를 만든다.

### InstanceSpace Contract

`instance_space.yaml`은 아래를 명시해야 한다.

- `anchor_query.sql`
  - read-only SQL
  - deterministic ordering이 가능해야 한다
- `anchor_query.outputs`
  - downstream instance builder가 사용할 named columns
- `parameters`
  - `enum`
  - `int_range`
  - `float_range`
  - `date_range`
  - `derived_bucket`
- `sampling.strategy`
  - `deterministic_hash`
  - `grid`
  - `stratified_hash`
- `sampling.seed`
  - deterministic source of truth
- `instance_count`
  - 생략 시 quality filter의 `min_cross_instances`를 따른다
  - 지정 시 runtime minimum보다 작을 수 없다

요구:

- instance별 anchor / threshold / context가 달라야 한다
- valid solution도 instance별로 달라져야 한다

즉 agent가 instance JSON 전체를 고정으로 박는 구조가 아니라, runtime-owned variation을 강제한다.

## Code Registration Policy

unregistered generated Python code는 in-process로 실행하지 않는다.  
`AST preflight + subprocess isolation`이 mandatory다.

운영 결정:

- v1은 `RestrictedPython`을 전면 채택하지 않는다
- 대신 narrow DSL 전제를 둔 custom AST preflight + subprocess isolation을 사용한다
- production promotion 전에 denylist 우회 사례가 발견되면 RestrictedPython 또는 동등한 safe-subset compiler 도입을 재평가한다
- 이 선택의 사유는 spec 본문과 [ADR 0001](adr/0001-custom-ast-preflight.md)에 함께 기록한다

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

v1에서는 runtime facade를 import로 주입하지 않는다.

- generated code는 runtime-provided function arguments (`tools`, `facts`)를 직접 사용한다

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

v1에서는 generated code를 두 개의 runtime lane으로 나눈다.

#### Lane A: Registration and Verification Lane

아래 코드는 persistent subprocess worker pool에서만 실행한다.

- `solution.py`
- `verifier.py`
- `shadow_verifier.py`
- tool self-test during registration
- generated `tools.py` during self-consistency / verifier / shadow verification

운영 방식:

- subprocess worker는 persistent pool이다
- 각 worker는 자기 프로세스 안에서 작은 read-only async DB pool을 가진다
- worker당 connection 수는 bounded다
- worker protocol은 최소한 `validate_module`, `execute_module_entrypoint`, `run_tool_self_test` 요청을 지원한다
- timeout은 subprocess 단위로 강제한다
- memory limit도 프로세스 단위로 강제한다
  - Linux에서는 enforced guard로 취급한다
  - macOS 개발 환경에서는 best-effort / near no-op일 수 있다
- main process의 asyncpg pool을 subprocess와 공유하지 않는다
- registration runner는 `tool + tool_self_test + solution + verifier + shadow_verifier` 번들을 한 번에 평가하는 orchestration helper를 가진다
  - Milestone 2에서는 `tool_self_test`만 runtime execution 대상이다
  - `solution`, `verifier`, `shadow_verifier`는 아직 static registration policy만 통과한다
  - tool self-test는 lightweight facade를 통해 실행되며, 이 단계에서는 live DB connection 대신 `conn=None`이 주입된다

#### Lane B: Production Solver Runtime Lane

solver runtime은 per-tool-call subprocess roundtrip을 하지 않는다.

- registration policy를 통과한 registered tool만 main process에서 import하여 실행한다
- solver는 기존 `DatabasePools` 기반 main-process tool execution path를 재사용한다
- subprocess isolation은 “신뢰되지 않은 generated code의 등록/검증”에 mandatory이며, production solver loop의 per-call isolation을 의미하지 않는다

즉 과거의 “sandbox 불필요” 가정은 수정되지만, 동시에 “모든 tool call을 subprocess로 우회”하는 구조도 채택하지 않는다.  
container는 v1 필수는 아니지만 subprocess isolation은 registration lane에서 필수다.

## Synthesis Agent Loop

### Stage 1: Schema and Category Discovery

- schema introspection
- sample row inspection
- domain/category proposal

phase output contract:

- `SchemaExplorationOutput`
  - `domain_hypothesis`
  - `candidate_categories`
  - `memory_summary`

운영 규칙:

- `candidate_categories`는 비어 있으면 안 된다
- category inference는 반드시 이 structured output을 입력으로 사용한다

### Stage 2: 4-Tuple Synthesis

agent는 아래를 함께 만든다.

- `tools.py`
- `task.json`
- `solution.py`
- `verifier.py`

artifact generation phase output contract:

- `ArtifactGenerationOutput`
  - `proposed_environment`
  - `artifacts`
  - `memory_summary`

운영 규칙:

- `proposed_environment`는 trust field를 포함하지 않는다
- registration bundle runner가 먼저 통과해야 한다
- registration 실패 시 draft는 materialize되지 않는다
- registration이 통과된 뒤 runtime이 trust field를 채워 `EnvironmentContract`를 만든다

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
- discard는 iteration budget을 소비한 것으로 본다
- discard는 `db_id x category` failure counter를 증가시킨다
- 연속 discard가 임계치를 넘으면 해당 `db_id x category`를 backoff queue로 보낸다

#### Gaming Guard

verifier 수정은 임의 완화가 아니다.

- difficulty vector는 self-consistency 수정 과정에서 감소하면 안 된다
- constraint count / conditional depth / threshold strictness를 줄이는 verifier 수정은 disallowed
- verifier를 약화시키는 수정이 필요하면 environment를 discard하고 category synthesis로 되돌린다

category synthesis로 되돌린다는 뜻은:

- 같은 category 안에서 새 environment draft를 다시 제안할 수 있다
- failure counter가 누적되면 scheduler가 해당 category를 일시 backoff할 수 있다
- backoff가 해제되기 전까지는 다른 category 또는 다른 DB를 우선 진행한다

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

Milestone 3 skeleton의 기본 구현은 `models.composer`를 synthesis backend model로 재사용하며,
phase별 backend list를 주입해 circuit breaker 이후 healthy provider로 fallback할 수 있게 한다.
또 각 phase는 explicit memory entry와 tool trace entry를 남긴다.
`previous_outputs`는 phase별 authoritative structured output이고, `memory`는 retry/fallback을 포함한 compressed execution summary다.
artifact generation 응답은 `proposed_environment + artifacts` 구조를 사용하고,
registration bundle runner를 통과한 뒤에만 draft로 승격된다.
`env_id`, signatures, `status`, `quality_metrics`, `generator_version` 같은 trust field는 runtime이 재생성한다.
runtime 인스턴스는 v1에서 single-db다. 즉 하나의 `SynthesisAgentRuntime`은 첫 호출의 `db_id`에 bind되며, 다른 `db_id`를 처리하려면 새 runtime 인스턴스를 생성한다.
shared graph cache와 registration pool 초기화는 내부 async lock으로 보호한다.

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

### Review Rubric

review pack의 “예시 수준 이상” 평가는 아래 rubric으로 기록한다.

- `compositional_structure`
  - multi-slot 또는 multi-entity coupling이 필요하다
- `constraint_density`
  - non-trivial hard constraint가 5개 이상이다
- `branching_or_threshold`
  - conditional branching 또는 numeric threshold가 존재한다
- `grounded_verification`
  - verifier summary상 DB-grounded fact stage가 분명하다
- `natural_user_request`
  - 실제 사용자가 할 법한 자연어 요청이다
- `non_lookup_shape`
  - 단순 single-value lookup으로 축약되지 않는다

운영 기준:

- 앞의 네 항목은 mandatory다
- 전체 6개 중 최소 5개를 만족하면 “예시 수준 이상”으로 기록한다

## Proof Environment Decision

첫 proof environment는 Sakila 위에서 억지로 만들지 않는다.

운영 결정:

- first proof는 synthetic fixture DB를 사용한다
- Sakila는 schema introspection / baseline regression 용도로만 남긴다
- synthetic proof 이후에는 Sakila와 두 번째 real DB에서 single-environment validation을 반드시 거친다

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
