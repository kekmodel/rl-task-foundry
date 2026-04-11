# RL Task Foundry — Composition-Centric Design Spec

## Overview

이 프로젝트의 목표는 PostgreSQL 스키마를 읽어 RLVR용 synthetic dataset을 대량 생산하는 것이다.  
다만 핵심 task paradigm은 더 이상 `path-centric single-value lookup`이 아니다.

새 기준은 아래와 같다.

- solver가 실제로 조합적 계획과 제약 만족을 풀어야 한다
- verifier는 여전히 noise 없는 deterministic binary reward를 줘야 한다
- task는 하나의 값 조회가 아니라 `여러 entity slot + cross-slot constraints`를 가진 composition problem이어야 한다

대표 예시는 trip planning, roster assignment, bundle construction, multi-day itinerary, eligibility-constrained matching 같은 과제다.

이 문서는 기존 path-centric baseline을 대체하는 새 기준 문서다. 현재 코드베이스의 infra / orchestration / solver substrate 자산은 최대한 유지하고, task generation / truth generation / verification contract를 composition paradigm으로 재설계한다.

## Why This Rewrite

현재 single-value lookup 중심 생산물은 다음 한계가 있다.

- RLVR 학습 신호가 약하다
- 실제 사용자 요청 수준의 compositional reasoning으로 일반화되기 어렵다
- question quality를 올려도 core contract가 여전히 단순 조회에 머문다
- path/hop 난이도만으로는 진짜 계획형 문제의 complexity를 표현할 수 없다

따라서 설계 중심을 아래처럼 이동한다.

- `path`는 더 이상 task 그 자체가 아니다
- `path catalog`는 `plan catalog`를 만들기 위한 하위 primitive다
- verifier는 `exact field match`보다 `constraint satisfaction`을 평가한다
- tool surface는 path-bound lookup 집합보다 entity-centric retrieval 집합으로 이동한다

## Core Principles

1. `hard to solve, easy to verify`
2. task는 `composition-first`, path는 그 구성 primitive다
3. solver runtime과 RL rollout runtime은 같은 상태 기계를 사용한다
4. reward는 끝까지 binary이며, diagnostic은 별도 채널로 남긴다
5. correctness는 SDK가 아니라 domain contract와 constraint checker가 책임진다
6. dataset quality는 테스트 숫자보다 generated review pack의 정성 평가를 통해 계속 검증한다

## Goals

- PostgreSQL schema에서 entity, relation, attribute 구조를 읽는다
- composition template를 자동 생성할 수 있는 `plan catalog`를 만든다
- entity slot과 constraint set을 가진 task contract를 생성한다
- deterministic solver-visible tools를 entity-centric surface로 제공한다
- valid solution set을 enumerate하거나 constraint checker를 compile한다
- solver 답을 constraint execution으로 검증한다
- 여러 solver replica를 병렬 실행해 pass-rate band에 맞춰 난이도를 조절한다
- review pack과 accepted dataset을 함께 관리한다

## Non-Goals

- 자연어 품질만 개선한 single-value lookup dataset 생산
- fuzzy grading이나 semantic similarity reward
- human-in-the-loop labeling이 필수인 verifier
- production training run을 현재 rewrite 중간 산출물에 바로 연결하는 것

## Clean Break Policy

이 rewrite는 `Tier A path-centric`와 `Tier C composition-centric`를 병행 운영하지 않는다.

- 기존 path-centric baseline은 freeze한다
- 기존 review pack은 baseline snapshot으로 보존한다
- production run은 composition contract가 proof되기 전까지 금지한다
- 새 구현은 기존 task generation layer를 대체하는 clean break를 목표로 한다

## Preserved Assets

그대로 유지하거나 최대한 재사용한다.

- `infra/`
  - DB pool
  - events
  - budget
  - checkpoint
  - storage
  - privacy
- `config/` 구조와 load 메커니즘
- `solver/backend_openai_agents.py`
  - `submit_result()`
  - explicit terminal action
  - structured output handling
- `verification/shadow.py`와 verification framework 골격
- `calibration/`
  - pass-rate band
  - Clopper-Pearson CI
  - safe early stop
- `pipeline/orchestrator.py`
  - rolling semaphore
  - provider circuit breaker
  - checkpoint / budget / export 골격
- `cli.py`, `review_pack.py` 구조
- `schema/graph.py`, `schema/introspect.py`
- `infra/json_chat_client.py`

## Rewrite Targets

전면 재설계 대상은 아래다.

- `tasks/factory.py`
- `tasks/composer.py`
- `tasks/package_validation.py`
- `truth/generator.py`
- `tools/compiler.py`
- `tools/sql_templates.py`
- `verification/compare.py`

부분 보강 대상으로는 아래가 있다.

- `truth/canonicalize.py`
  - nested / array / multi-slot answer 지원 확장
- `schema/path_catalog.py`
  - 유지하되 상위의 `plan catalog`가 필요

## New Architecture

시스템은 다섯 층으로 나뉜다.

1. `Schema + Entity Layer`
   - schema introspection
   - PK/FK graph
   - entity profile
   - attribute visibility

2. `Plan Catalog Layer`
   - composition template library
   - slot graph
   - constraint family
   - difficulty features

3. `Task Synthesis Layer`
   - contract-first task generation
   - question composition
   - tool presentation composition
   - task package judge

4. `Solution + Verification Layer`
   - deterministic solution enumeration
   - canonical solution selection
   - execution-based constraint checker

5. `Runtime + Orchestration Layer`
   - solver swarm
   - calibration
   - checkpoint / budget / export

### High-Level Flow

```text
Postgres
  -> schema introspection
  -> entity graph + relation graph
  -> plan catalog
  -> composite task contract
  -> question + presented tool bundle composition
  -> deterministic solution enumerator / constraint checker
  -> solver swarm execution
  -> execution-based verification
  -> adaptive calibration
  -> review pack / accepted dataset export
```

## Agent Runtime Contract

solver와 RL rollout agent는 같은 step contract를 따른다.

```text
Observation -> Policy Call -> Action -> Tool Result -> State Update
```

여기서 action은 다음을 포함한다.

- entity-centric retrieval tool 호출
- optional summarization event
- `submit_result()` terminal action

메모리, summary, tool trace는 모두 explicit event다.  
최종 제출은 반드시 `submit_result()` tool 호출로만 처리한다.

## Core Task Paradigm

### Old Paradigm

- 하나의 anchor row
- 하나의 path
- 하나의 scalar/list/count answer
- verifier는 mostly field equality

### New Paradigm

- 하나 이상의 anchor / reference entity
- 여러 `slot`
- slot 간 제약
- answer는 nested rows / arrays / records일 수 있음
- verifier는 constraint execution

즉 task는 `무슨 경로를 따라 무엇을 읽느냐`보다  
`어떤 entities를 골라 어떤 global constraints를 만족시키느냐`가 핵심이다.

## Core Data Model

### Entity Slot

```python
EntitySlot:
  slot_id: str
  entity_type: str
  cardinality: "one" | "many"
  source_plan_node: str
  visible_attributes: list[str]
  required_output_fields: list[str]
```

### Constraint DSL

constraint는 solver 답이 만족해야 하는 predicate다.

```python
Constraint:
  constraint_id: str
  kind:
    - equality
    - inequality
    - range
    - membership
    - uniqueness_across
    - cardinality
    - conditional
    - implication
    - ordering
    - budget_sum
    - threshold_by_bucket
    - same_location_as
    - no_repeat
  args: dict[str, object]
  severity: "hard"
  description: str
```

### CompositeTaskPlan

```python
CompositeTaskPlan:
  plan_id: str
  template_family: str
  slots: list[EntitySlot]
  constraints: list[Constraint]
  objective:
    "find_any_valid_solution" | "find_canonical_solution"
  candidate_sources: list[str]
  difficulty_features: dict[str, float | int | bool | str]
```

### AnswerSchema v2

```python
AnswerField:
  name: str
  type:
    - "string"
    - "int"
    - "float"
    - "bool"
    - "date"
    - "datetime"
    - "enum"
    - "object"
    - "list[string]"
    - "list[int]"
    - "list[object]"
  nullable: bool
  ordered: bool
  canonicalizer: str
  description: str
  visibility: "user_visible" | "internal" | "blocked"
  source_columns: list[str]

AnswerSchema:
  version: "v2"
  fields: list[AnswerField]
  primary_output_format: "json_object" | "json_array"
```

### TaskSpec v2

```python
TaskSpec:
  task_id: str
  domain: str
  language: str
  label_tier: "A" | "B"
  question_family: str
  question: str
  outcome_type: "answer" | "no_result" | "clarify" | "deny"
  answer_schema: AnswerSchema
  composite_plan_id: str
  selected_tool_level: 1 | 2
  tool_bundle_id: str
  presented_tool_bundle_id: str | None
  provenance_requirements: list[str]
  difficulty_features: dict[str, float | int | str | bool]
  contract_metadata: dict[str, object]
```

`anchor_table`, `selected_path_id`, `required_hops`는 더 이상 중심 contract가 아니다.  
필요하면 metadata에 남길 수 있지만 source of truth는 `CompositeTaskPlan`이다.

### GroundTruth v2

```python
GroundTruth:
  task_id: str
  expected_outcome_type: "answer" | "no_result" | "clarify" | "deny"
  canonical_solution: dict[str, object] | list[dict[str, object]] | None
  valid_solution_signature: str
  checker_kind: "compiled_constraint_checker"
  checker_payload: dict[str, object]
  row_context: list[dict[str, object]]
  answer_schema_version: str
```

중요한 점:

- ground truth는 단일 값이 아닐 수 있다
- verifier는 canonical solution exact match 대신 `constraint checker`를 실행한다
- canonical solution은 review / debugging / seeding 용도다

## Execution-Based Verification

### Verification Contract

solver 답을 받으면 verifier는 아래를 수행한다.

1. strict schema validation
2. canonicalization
3. constraint checker execution
4. binary pass/fail 결정
5. diagnostic per-constraint result 기록

### Binary Reward Principle

RLVR reward는 여전히 binary다.

- 모든 hard constraint를 만족하면 `1`
- 하나라도 깨면 `0`

하지만 diagnostic은 richer하게 남긴다.

```python
VerifyResult:
  pass_exact: bool
  failure_reason: str | None
  constraint_diagnostics: list[{
    constraint_id: str,
    passed: bool,
    reason: str | None
  }]
  provenance_pass: bool
  shadow_verifier_status: str | None
```

즉 partial reward는 주지 않되, 어떤 constraint가 깨졌는지는 남긴다.

### Determinism Rules

constraint checker는 아래를 공통으로 강제한다.

- float는 configured precision으로 round
- ordering은 total order를 가져야 함
- top-k는 tie-break를 명시해야 함
- list answer는 deterministic order를 가져야 함
- date / datetime은 timezone과 cast rule을 명시해야 함
- NULL semantics를 명시해야 함
- candidate enumeration은 deterministic ordering을 사용해야 함

## Ground Truth Methodology

### Canonical Approach

기본 방법은 `constraint set -> deterministic candidate enumeration -> checker payload`다.

흐름:

1. candidate entities를 SQL로 읽는다
2. slot assignment 후보를 deterministic order로 나열한다
3. constraint DSL evaluator가 valid assignment를 찾는다
4. objective가 `find_canonical_solution`이면 stable ordering으로 첫 valid solution을 canonical로 선택한다
5. solver 답 검증 시에는 canonical exact match가 아니라 checker를 다시 실행한다

### Multiple Valid Solutions

valid solution이 여러 개일 수 있다.

이 경우 verifier 전략은:

- 기본: `any valid solution`이면 pass
- canonical solution은 review/debug용 reference로만 사용
- dedup signature는 canonical solution이 아니라 `plan structure + checker payload hash` 기준으로 만든다

## Tool Surface Redesign

### Principle

tool은 더 이상 `path lookup compiler` 중심이 아니다.  
solver가 필요한 entities와 attributes를 조합할 수 있게 해야 한다.

### Entity-Centric Tool Classes

예시:

- `get_all_{entity}_by_{filter}`
- `get_{entity}_options_for_{context}`
- `get_infos_by_{entity}(entity_ref, keys)`
- `get_related_{entity}(entity_ref, relation)`
- `get_temporal_events_by_{entity}(entity_ref, keys)`
- `submit_result(answer_json)`

tool은 가능한 한 다음 성질을 가진다.

- read-only
- bounded
- deterministic ordering
- attribute selection 가능

### L1 / L2

`L1/L2` tool level은 유지한다.

- `L1`
  - direct
  - canonical
  - rule-based
- `L2`
  - model-generated presentation
  - same semantics, harder surface

하지만 difficulty의 핵심은 naming이 아니라 composition complexity다.

## Plan Catalog

`path catalog` 위에 `plan catalog`를 둔다.

### Role

plan catalog는 여러 relation/path primitive를 조합해 reusable composition template를 만든다.

### Example Template Families

- itinerary / schedule
- assignment / roster
- bundle selection
- top-k with constraints
- no-repeat recommendation
- threshold-conditioned selection
- temporal window planning

### Difficulty Features

기존 hop/fanout 외에 아래가 핵심이다.

- slot_count
- candidate_width
- constraint_count
- conditional_depth
- uniqueness_scope_count
- numerical_threshold_count
- temporal_window_count
- cross-slot coupling
- distractor_density

## Question Composition

질문은 반드시 contract-first로 생성한다.

순서:

1. `CompositeTaskPlan`
2. `AnswerSchema v2`
3. `GroundTruth / Checker payload`
4. sanitized question context
5. question composer agent
6. task package judge agent

질문은 다음 성질을 가져야 한다.

- 이 DB를 소유한 기업의 AI agent에게 실제 user가 할 법한 요청
- schema / DB 구조 노출 금지
- answer leak 금지
- plan semantics와 coherent해야 함

## Task Package Judge Agent

judge agent는 semantic validation을 맡는다.

입력:

- question
- answer schema
- tool bundle
- ground truth context
- plan summary

루브릭:

- 자연스러운 domain language인가
- answer leak가 없는가
- schema exposure가 없는가
- answer schema와 semantic coherence가 맞는가
- solver가 주어진 tool set으로 풀 수 있는가

## Calibration

adaptive calibration의 중심도 바뀐다.

이전:

- hop
- fanout
- tool level

이후:

- slot count
- constraint count
- conditional depth
- candidate width
- uniqueness scope
- temporal windows
- tool level

즉 solver pass-rate를 보고 다음 layer에서 question만 바꾸는 것이 아니라  
plan 자체를 강화한다.

## Review Pack

review pack은 이제 필수 artifact다.

포함 항목:

- final question
- submit format
- presented tool bundle
- plan summary
- constraint summary
- canonical solution
- question strategy
- judge summary

review는 테스트보다 먼저 품질을 드러내는 장치다.  
새 paradigm에서는 정성 평가가 development loop의 1급 입력이다.

## Migration Notes

현재 path-centric baseline은 아래 용도로만 남긴다.

- infra / solver / orchestration regression test
- schema introspection fixture
- rewrite 전 baseline snapshot

하지만 future production dataset은 composition-centric pipeline만 사용한다.

## Freeze Policy

이 문서 기준으로 다음 원칙을 적용한다.

- task generation 관련 코드는 spec/plan 합의 전까지 incremental polishing을 중단한다
- rewrite는 `spec -> plan -> one vertical proof task -> generalization` 순으로 진행한다
- first proof task는 trip-planning 수준의 composition fixture를 택한다

