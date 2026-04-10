# RL Task Foundry — Design Spec

## Overview

이 프로젝트의 목표는 PostgreSQL 스키마를 읽어 멀티홉 추론이 필요한 고객센터형 질의를 자동 생성하고, 여러 solver agent로 풀게 한 뒤, deterministic SQL verification으로 정답을 판정해 RLVR용 synthetic dataset을 대량 생산하는 것이다.

핵심 원칙:

1. `hard to solve, easy to verify`
2. solver runtime과 judge runtime을 분리한다
3. solver와 RL rollout agent는 같은 상태 기계를 사용한다
4. memory와 summarization은 숨은 프레임워크 기능이 아니라 명시적 상태 전이로 다룬다
5. dataset correctness는 SDK가 아니라 우리 도메인 레이어가 책임진다
6. batch throughput은 언어보다 orchestration, provider saturation, checkpoint, verification 설계로 확보한다

이 문서는 `OpenAI Agents SDK`를 solver substrate로 사용하되, 예전 설계의 핵심 생산 메커니즘인 `adaptive calibration`, `tool level`, `outcome diversity`, `shadow verifier`, `multi-provider diversity`, `provider resilience`를 다시 포함한 기준 문서다.

## Goals

- PostgreSQL 스키마를 읽어 PK/FK 기반 멀티홉 question family를 생성한다
- read-only DB tools를 자동 생성한다
- anchor row별 task를 만든다
- deterministic SQL로 ground truth와 canonical answer를 생성한다
- 답이 있는 case뿐 아니라 `no_result`, `clarify`, `deny` case도 생성한다
- 여러 solver replica를 병렬 실행한다
- solver output을 typed canonicalization 후 exact / provenance 기준으로 verification한다
- shadow verifier로 judge disagreement rate를 감시한다
- adaptive difficulty loop로 target pass-rate band에 수렴하도록 난이도를 조절한다
- dedup, diversity, coverage를 관리한다
- run state를 durable하게 기록하고 resume 가능하게 만든다
- solver 구조를 RL rollout agent에도 재사용할 수 있게 만든다

## Non-Goals

- SDK가 verification, calibration, provenance를 대신해주길 기대하지 않는다
- 코딩 에이전트용 범용 파일/쉘/웹 환경을 solver에 기본 제공하지 않는다
- hidden summarization이나 opaque memory compaction을 canonical runtime state로 사용하지 않는다
- multi-tenant SaaS control plane을 v1의 전제로 두지 않는다

## Architecture

시스템은 네 개의 층으로 나뉜다.

1. `Schema + Tool Compiler`
   - DB 구조 이해
   - sensitivity classification
   - path catalog
   - canonical semantic tool generation
2. `Task Factory`
   - task composition
   - task-aware tool presentation
   - ground truth
   - validation
   - dedup / coverage
3. `Agent Runtime`
   - OpenAI Agents SDK 기반 solver 실행
   - explicit transcript / memory state
   - tool tracing
4. `Judge + Orchestrator`
   - verification
   - calibration
   - budget / checkpoint / export

### High-Level Flow

```text
Postgres
  -> schema introspection
  -> path catalog + sensitivity policy
  -> canonical tool compilation (L1 semantic source)
  -> task composition + presented tool bundle generation
  -> ground truth SQL
  -> task validation
  -> solver swarm execution
  -> verification
  -> adaptive calibration
  -> dedup / coverage checks
  -> accepted dataset export
```

### Module Responsibilities

- `schema`
  - tables, columns, PK/FK graph, fanout, uniqueness, nullable, shortcut path 분석
- `tools`
  - read-only canonical tool spec 정의, SQL compiler, tool registry
- `tasks`
  - question family 생성, anchor row materialization, outcome type selection, task-aware tool presentation, task validation
- `truth`
  - answer schema, verification SQL, canonical answer 생성
- `solver`
  - OpenAI Agents SDK 기반 multi-backend solver runtime
- `verification`
  - typed comparison, outcome-aware verification, exact / shadow scoring
- `calibration`
  - pass-rate banding, adaptive difficulty loop, canary, safe early termination
- `infra`
  - DB pool, checkpoint, budget, events, storage, privacy
- `pipeline`
  - rolling orchestration, export, run manifest

## OpenAI Agents SDK Positioning

OpenAI Agents SDK는 `solver substrate`로만 사용한다.

SDK가 맡는 역할:

- model invocation
- tool calling loop
- sessions
- tracing
- handoff 같은 선택적 agent primitive
- function tool schema transport

SDK가 맡지 않는 역할:

- answer schema definition
- task generation correctness
- ground truth generation
- provenance rules
- verification policy
- calibration policy
- run durability
- dataset export

기본 구현은 `openai` / `openai_compatible`를 먼저 지원하되, solver runtime은 provider-specific model adapter를 통해 `anthropic`, `google` 계열도 추가 가능한 구조를 유지한다. calibration label은 최소 두 개 이상의 provider family에서 측정 가능해야 한다.

### Agent Runtime Contract

solver와 RL rollout agent는 동일한 step contract를 따른다.

```text
Observation -> Policy Call -> Action -> Tool Result -> State Update
```

state는 아래를 포함한다.

- transcript
- explicit memory entries
- summaries
- tool trace
- per-turn metadata
- reward / verification attachment

메모리 요약은 반드시 명시적 이벤트로 기록된다. 요약이 발생하면 다음 세 항목이 남아야 한다.

1. 요약 이전 원본 범위
2. 요약 결과
3. 요약을 생성한 정책 혹은 도구 정보

### Terminal Submission Contract

최종 답 제출은 SDK `output_type`에 위임하지 않고, 명시적 `submit_result()` tool 호출로 처리한다.

원칙:

- `submit_result()`는 terminal action이다
- 일반 tool의 argument/schema 오류는 recoverable tool error일 수 있다
- 하지만 `submit_result()`는 schema를 이미 tool definition으로 알고 호출하는 것이므로, schema mismatch도 terminal fail로 처리한다
- terminal fail은 reward를 바꾸기 위한 정보가 아니라, monitoring / analysis를 위한 종료 사유로만 기록한다
- verifier는 structured output이 없을 때 solver 종료 사유를 failure reason으로 승계할 수 있어야 한다

즉 solver가 학습해야 하는 종료 판단은 아래 세 가지다.

1. 더 조사할지
2. 지금 제출할지
3. 어떤 구조화된 답을 제출할지

## Core Data Model

### Answer Schema

`answer_schema`는 task/ground truth/solver output/verification 전부를 묶는 1급 계약이다.

```python
AnswerField:
  name: str
  type: "string" | "int" | "float" | "bool" | "date" | "datetime" | "enum" | "list[string]" | "list[int]"
  nullable: bool
  ordered: bool
  canonicalizer: str
  description: str
  visibility: "user_visible" | "internal" | "blocked"
  source_columns: list[str]

AnswerSchema:
  version: str
  fields: list[AnswerField]
  primary_output_format: "json_object"
```

### TaskSpec

```python
TaskSpec:
  task_id: str
  anchor_table: str
  anchor_pk_column: str
  anchor_pk_value: str
  domain: str
  language: str
  label_tier: "A" | "B"
  question_family: str
  question: str
  outcome_type: "answer" | "no_result" | "clarify" | "deny"
  answer_schema: AnswerSchema
  selected_path_id: str
  required_hops: int
  tool_level: 1 | 2
  tool_bundle_id: str
  presented_tool_bundle_id: str | None
  provenance_requirements: list[str]
  difficulty_features: dict[str, float | int | str | bool]
  sensitivity_policy: str
```

### TaskPackage

```python
PresentedToolSpec:
  name: str
  description: str
  semantic_key: str
  kind: str
  parameter_names: list[str]
  output_fields: list[str]
  name_source: "rule_based" | "model_generated" | "fallback_alias"
  presentation_role: "core" | "distractor"

PresentedToolBundle:
  bundle_id: str
  canonical_bundle_id: str
  path_id: str
  tool_level: 1 | 2
  question_family: str
  outcome_type: "answer" | "no_result" | "clarify" | "deny"
  tools: list[PresentedToolSpec]
  generation_metadata: dict[str, object]

TaskPackage:
  task: TaskSpec
  presented_tool_bundle: PresentedToolBundle
  presentation_options: list[PresentedToolBundle]
  available_tool_levels: list[int]
```

`TaskPackage`는 stable `L1` base option과 optional `L2` option을 함께 가질 수 있다. 다만 실제 solver 실행에 노출하는 presented bundle은 execution config의 `task_composer.selected_tool_level`로 run 시작 시 고정된다.
현재 task factory는 `TaskContractDraft(answer_schema, outcome_type, selected_path, proof obligation)`을 먼저 만든다. factory가 만드는 `question`은 provisional seed일 수 있으며, production run에서는 task composer가 path context와 sanitized row context를 받아 user-facing final question을 다시 합성한다.

### GroundTruth

```python
GroundTruth:
  task_id: str
  expected_outcome_type: "answer" | "no_result" | "clarify" | "deny"
  verification_sql: str
  sql_params: dict[str, object]
  canonical_answer: dict[str, object] | None
  row_context: list[dict[str, object]]
  answer_schema_version: str
  provenance_path: list[str]
```

### SolverResult

```python
SolverResult:
  task_id: str
  solver_id: str
  provider: str
  model: str
  replica_index: int
  response_type: "answer" | "no_result" | "clarify" | "deny" | None
  transcript_ref: str
  tool_trace_ref: str
  raw_output_text: str
  structured_output: dict[str, object] | None
  explicit_memory_events: list[dict[str, object]]
  token_usage: dict[str, int]
  latency_ms: int
  turn_count: int
  status: "completed" | "invalid_submit" | "tool_error" | "model_error" | "timeout"
  termination_reason: str | None
  termination_metadata: dict[str, object]
```

### VerifyResult

```python
VerifyResult:
  task_id: str
  solver_id: str
  pass_exact: bool
  field_scores: dict[str, bool]
  provenance_pass: bool
  canonical_prediction: dict[str, object] | None
  shadow_verifier_status: "not_run" | "match" | "disagree"
  shadow_pass_exact: bool | None
  shadow_failure_reason: str | None
  failure_reason: str | None
```

### AcceptedExample

```python
AcceptedExample:
  task: TaskSpec
  ground_truth: GroundTruth
  solver_results: list[SolverResult]
  verification_results: list[VerifyResult]
  pass_rate: float
  calibration_band: tuple[float, float]
  rollout_record: dict[str, object]
  mean_correct_solver_turns_rounded: int | None
  training_metadata: dict[str, object]
  export_payload: dict[str, object]
```

### Run State

운영 상태의 source of truth는 `run.db`다. JSONL은 최종 export와 append-only event mirror일 뿐이다.

`run.db`는 최소 아래를 저장한다.

- runs (`run_id`, `config_hash`, `created_at`)
- processed anchors
- composed tasks (`run_id` scoped)
- accepted / rejected tasks (`run_id` scoped)
- solver runs
- verification results (`run_id` scoped)
- budget ledger
- dedup signatures
- coverage counters
- run manifest

현재 구현 slice에서는 두 경로가 열려 있다.

- `generate-task-specs -> TaskSpec JSONL`
- `generate-review-pack -> review_pack.jsonl + review_pack.md`
- `TaskSpec JSON/JSONL -> orchestrator -> solver -> verifier -> run.db`

즉 완전한 adaptive task factory 이전에도, anchor/path 기반 Tier A task spec을 headless하게 생성할 수 있고, 이미 정의된 task spec 묶음을 실행한 뒤 `run-summary`를 `run.db`만으로 재구성할 수 있어야 한다. 이 실행 경로는 task 간에도 bounded in-flight set으로 rolling 처리되며, provider saturation은 provider semaphore가 따로 제어한다.

정성 품질 평가는 선택이 아니라 필수다. 따라서 generated task는 rule-based unit test만으로 충분하다고 간주하지 않고, review pack을 통해 질문/답 계약/tool surface를 사람 눈으로 샘플링 검토할 수 있어야 한다.

## Pipeline Flow

### 1. Schema Introspection

- PostgreSQL catalog를 읽어 table/column/PK/FK 정보를 수집한다
- uniqueness, fanout, nullable, cardinality estimate를 계산한다
- sensitivity classification을 위한 column metadata를 수집한다

### 2. Sensitivity Policy

컬럼은 세 클래스로 분류한다.

- `blocked`
  - tool output에도 노출하지 않는다
- `internal`
  - 내부 reasoning / trace에는 존재할 수 있으나 export answer에는 넣지 않는다
- `user_visible`
  - task answer에 사용할 수 있다

이 정책은 blanket PII exclusion이 아니라 visibility policy다. 따라서 "고객 본인이 볼 수 있는 배송기사 연락처" 같은 값은 `user_visible`로 허용 가능하다.

### 3. Path Catalog Build

- anchor table에서 시작해 reachable FK path를 탐색한다
- path별 hop 수, fanout, shortcut 후보를 기록한다
- question family별 admissible path를 등록한다

### 4. Tool Compilation

- path catalog와 sensitivity policy를 읽어 solver용 tool set을 만든다
- 모든 tool은 read-only SQL template에서 생성된다
- tool surface는 `entity lookup`, `relation traversal`, `count`, `exists`, `aggregate`, `timeline` 범주로 제한한다
- tool set은 난이도 variant를 가진다: `L1`, `L2`

### 5. Task Composition

- anchor row와 selected path를 바탕으로 question family를 고른다
- question text는 domain-configured user/agent roles에 맞는 natural language로 생성한다
- `outcome_type`을 `answer | no_result | clarify | deny` 중에서 선택한다
- 질문은 tool use를 유도하되, lexical leak가 없어야 한다
- required hops와 answer_schema는 composer 이전에 이미 정해진다
- final question synthesis는 composer model이 담당한다
- composer는 selected path, question family, outcome type, answer contract, sanitized row context를 보고 질문을 쓴다
- 질문은 "이 DB를 소유한 조직의 AI 에이전트에게 실제 end user가 할 요청"처럼 들려야 하며, schema/tool/database terminology를 드러내면 안 된다
- composer는 질문 생성 후 answer_schema와의 semantic coherence를 별도 검증해야 한다
- seed fallback이 발생하면 `question_source`와 generation metadata를 export에 남겨서 학습 데이터에서 구분 가능해야 한다
- qualitative review pack은 final synthesized question, submit format, tool set, hidden answer key를 함께 저장해서 사람이 하나씩 품질을 점검할 수 있어야 한다
- current automatic factory slice는 `status_lookup`, `causal_chain`, `aggregate_verification`를 우선 지원한다
- `timeline_resolution`은 temporal scalar가 Tier A proof obligation을 만족할 때만 자동 생성한다

### 6. Ground Truth Generation

- selected path와 answer_schema를 이용해 deterministic verification SQL을 생성한다
- canonical answer를 answer_schema 타입에 맞게 정규화한다
- `outcome_type`이 `answer`가 아니면 canonical answer 대신 expected outcome branch를 생성한다
- provenance path도 함께 기록한다

### 7. Task Validation

task는 solver를 돌리기 전에 아래를 통과해야 한다.

- answer_schema complete
- lexical leak 없음
- ambiguity 없음
- blocked/internal column policy 위반 없음
- shortcut answer 가능성 없음
- provenance path valid
- answer cardinality valid
- outcome_type과 ground truth branch 일치
- 형식적 체크는 코드로 수행하되, 의미적 품질 검사는 task package judge agent가 수행한다
  - 질문이 자연스러운 도메인 언어인가
  - 질문에 답이 leak되지 않았는가
  - 질문에 schema/DB 구조가 노출되지 않았는가
  - 질문과 answer_schema가 의미적으로 정합하는가
  - solver가 주어진 tool set으로 답할 수 있는가

### 8. Solver Swarm Execution

- 여러 solver replica를 병렬 실행한다
- solver는 OpenAI Agents SDK 기반 `AgentRuntime`으로 실행한다
- solver가 사용할 수 있는 도구는 compiled DB tools만이다
- optional memory / summarization / handoff 기능은 runtime 옵션으로 켠다
- solver replica는 모델 family별 고정 quota로 배분한다
- solver는 최종 답을 `submit_result()` tool로 제출한다
- 일반 tool argument 오류는 recoverable tool error로 취급할 수 있지만, `submit_result()` schema mismatch는 terminal fail이다

### 9. Verification

- solver structured output을 answer_schema 기준으로 canonicalize한다
- `fail_on_internal_field_leak=true`이면 `answer_schema`에서 `internal`로 표시된 필드를 최종 structured output에 제출하는 순간 verifier는 fail한다
- deterministic ground truth와 outcome-aware exact / fieldwise 비교한다
- provenance violation은 fail로 처리한다
- 일정 샘플에는 shadow verifier를 추가 실행해 disagreement rate를 측정한다
- structured output이 없으면 solver 종료 사유를 failure reason에 연결해 분석 가능해야 한다

### 10. Adaptive Calibration

- pass-rate가 target band 안에 들어오지 않으면 동일 anchor에서 난이도를 조절하며 재시도한다
- too easy / too hard는 confidence interval 기준으로만 확정한다
- 난이도는 한 번에 한 축만 조절한다: `required_hops -> condition_complexity -> fanout_ambiguity`
- canary phase에서 초반 추정치를 만들고, 필요 시 full solve까지 확장한다
- max iteration 도달 시 best-so-far fallback을 허용한다

### 11. Dedup / Coverage / Export

- semantic near-dup 제거
- table/path/question family coverage를 균형 있게 유지
- accepted task만 JSONL export
- accepted example에는 학습용 메타데이터로 `mean_correct_solver_turns_rounded`를 포함할 수 있다
- run manifest에 전체 통계를 남긴다

## Tool Set Design

### Problem Definition

solver는 자유 SQL을 실행하지 않는다. solver가 보는 world는 compiled tool set이다. 따라서 task difficulty는 `DB 자체`보다 `tool set under-specifiedness`, `tool 탐색 난이도`, `required provenance depth`에서 나온다.

### Difficulty Axes

난이도는 아래 축으로 측정한다.

1. required hop count
2. condition complexity
3. fanout ambiguity
4. distractor density
5. aggregation complexity
6. temporal reasoning complexity
7. tool level (`L1` / `L2`, run partition)
8. answer cardinality

`tool naming opacity`는 추상 메타데이터가 아니라 실제 `tool_level` 메커니즘으로 구현한다.

### Tool Classes

- `lookup_*`
  - 특정 entity를 식별자나 좁은 filter로 조회
- `list_related_*`
  - anchor entity에 연결된 row 집합 조회
- `count_*`
  - 제한된 count
- `exists_*`
  - boolean condition 확인
- `aggregate_*`
  - sum, avg, min, max 같은 deterministic aggregate
- `timeline_*`
  - ordered events 조회

### Tool Set Levels

- `L1`
  - direct naming
  - 테이블/관계가 직관적으로 드러난다
  - rule-based compiler가 canonical tool 이름을 생성한다
  - column split이 적고 탐색 난이도가 가장 낮다
- `L2`
  - semi-indirect naming
  - production에서는 model-generated naming이어야 한다
  - 일부 schema cue는 남을 수 있지만, 전체 path를 그대로 이름에 풀어쓰면 안 된다
  - 관계명과 컬럼군이 일부 분산된다
  - solver가 더 많은 후보 중에서 올바른 chain을 찾아야 한다
  - 목표는 "찾을 수는 있지만 바로 보이지 않는" 수준이다
`tool_level`은 같은 run 안에서 바꾸는 calibration 축이 아니라 run partition이다. `L1`은 answerability baseline이고, `L2`는 실제 탐색 난이도를 높이는 production tool set이다. pass-rate history와 difficulty warm-start는 `tool_level`별로 따로 유지한다.

중요:

- `L1`은 rule-based canonical compiler가 생성한다
- `L2`는 task composer가 question context와 함께 composer model을 호출해 presented tool bundle을 생성한다
- model은 질문과 visible tool surface를 함께 설계해야 하며, naming을 path metadata만 보고 따로 생성하면 안 된다
- execution config는 `selected_tool_level`로 실제 run에서 어떤 level을 노출할지 결정한다
- rule-based alias는 bootstrapping/dev fallback일 수는 있지만 production difficulty label의 source of truth가 되어서는 안 된다
- `L2`가 full-path literal naming이면 실패다
- fallback alias는 개발용 안전망일 뿐이며, production run에서 `tool_level=L2`의 최종 label source가 되어서는 안 된다

### Tool Validation

각 tool은 아래를 만족해야 한다.

- read-only SQL template 기반
- parameterized query only
- blocked column 미노출
- declared provenance path 보유
- stable output schema 보유
- predictable cardinality bound 보유
- 같은 semantic capability가 level별 naming/layout variant를 가져도 결과 의미는 동일
- `L2` variant는 task-aware presented bundle이어야 하며, validation 실패 시 fallback alias를 임시로 쓸 수 있어도 run metadata에 fallback 사실을 기록해야 한다
- task package는 stable `L1` base presentation과 optional `L2` presentation option을 함께 가질 수 있어야 한다
- naming evaluator는 최소한 duplicate/invalid name, raw schema overlap, level-policy violation을 측정해야 한다
- `L2` acceptance:
  - L1보다 raw identifier overlap이 낮아야 한다
  - 전체 path chain을 이름에 직접 노출하면 안 된다
  - 지나치게 opaque해서 schema cue가 0이 되는 것도 피한다
  - quality gate를 통과하지 못하면 fallback alias로 downgrade하고 metadata에 이유를 남긴다

## Label Verifiability — Tier System

### Tier A

v1의 기본 모드다.

허용:

- raw field extraction
- count
- exists / boolean
- bounded list

규칙:

- `lookup`, `list_related`, `count`, `exists`만 tool surface에 포함한다
- aggregate / timeline / grouped statistic capability는 생성하지 않는다
- reward noise를 최소화하는 low-noise baseline tier다

### Tier B

실험 모드다.

허용:

- deterministic aggregate (`sum`, `avg`, `min`, `max`)
- ordered timeline lookup
- multiple aggregate composition
- temporal window reasoning
- ranking with deterministic tie-break
- multi-field normalized answers

규칙:

- `Tier A` 규칙을 유지한 상태에서 aggregate / timeline capability를 추가로 연다
- aggregate SQL은 `NULL` 값을 제외하도록 `WHERE aggregate_column IS NOT NULL`를 강제한다
- `sum` / `avg` on numeric-like columns는 `verification.float_precision`으로 scale을 고정한다
- ordered timeline은 temporal column 정렬 + PK tie-break를 반드시 포함한다
- grouped aggregate, ranking, temporal window는 determinism rule set이 구현된 뒤에만 enable한다

### Tier C

금지한다.

- subjective summarization
- fuzzy semantic judgement
- unverifiable free-form explanation
- policy/legal reasoning without deterministic label

## Detailed Module Design

### Schema Explorer

입력:

- DB connection
- include/exclude schema 설정

출력:

- table catalog
- column stats
- PK/FK graph
- path candidates
- sensitivity hints

핵심 규칙:

- schema 탐색은 control-plane lane에서 수행
- statistics sampling은 bounded query budget 안에서만 수행

### Tool Compiler

입력:

- path catalog
- sensitivity policy
- tool design rules

출력:

- `ToolSpec`
- `ToolBundle`
- runtime registration payload
- canonical semantic bundle (`L1`)

핵심 규칙:

- `safe_columns`는 visibility policy 기반으로 계산
- answerable question family와 reachable tool surface가 서로 모순되지 않아야 한다
- compiler output은 semantic source of truth다
- `L2` presented bundle은 compiler가 직접 만드는 것이 아니라 task composer가 canonical bundle을 재표현한 결과다
- level 간 semantic equivalence는 canonical bundle 기준으로 검증한다
- aggregate tool은 identifier-like numeric column (`id`, `*_id`, PK/FK`)에 대해 생성하지 않는다
- aggregate tool SQL은 `NULL` 제외와 numeric rounding 규칙을 공통 helper로 적용한다

### Task Composer

입력:

- anchor row
- selected path
- question family template
- answer_schema
- canonical tool bundle (`L1`)

출력:

- `TaskPackage`
- `PresentedToolBundle`

핵심 규칙:

- question text는 answer literal을 leak하지 않는다
- task composition 실패는 compose failure로 기록하지만 solver budget을 소비하지 않는다
- required_hops와 answer fields는 composer가 임의로 바꾸지 못한다
- `outcome_type`은 config의 negative outcome ratio를 따르되, ground truth로 검증 가능해야 한다
- calibration 중 난이도 조절은 한 번에 한 축만 바꾼다
- `L2` 도구 이름과 설명은 question context와 함께 생성되어야 한다
- production `L2`는 context-free naming pass로 만들지 않는다

### Ground Truth Generator

입력:

- TaskSpec
- selected path
- answer_schema

출력:

- GroundTruth

핵심 규칙:

- SQL은 deterministic해야 한다
- answer_schema 없는 task는 ground truth 생성 금지
- canonical answer는 verifier가 그대로 재사용한다
- `no_result`, `clarify`, `deny`는 positive answer branch와 별도 verifier branch를 가진다

### Task Validator

입력:

- TaskSpec
- GroundTruth

검사 항목:

- answer_schema completeness
- blocked/internal leakage
- ambiguity
- unsupported cardinality
- lexical leak
- shortcut
- provenance mismatch
- language quality minimum bar
- outcome branch validity

### Provenance Validator

solver가 제출한 답은 correct value만 맞아도 충분하지 않다. solver가 접근 가능한 tool graph 상에서 answer가 reachable해야 한다.

검사 항목:

- selected path와 tool trace 일치 여부
- required entity traversal 존재 여부
- unsupported hidden path 사용 여부

v1에서는 strict path proof 대신 아래 pragmatic rule을 사용한다.

- solver가 사용한 tool trace에 required path edges가 모두 나타나면 provenance pass
- ground truth answer만 맞고 trace가 부족하면 fail 혹은 flagged fail

### Solver Runtime

solver runtime은 `OpenAI Agents SDK` 위에 얇게 만든 `AgentRuntime`이다.

구성:

- prompt template
- tool registry adapter
- explicit runtime state
- provider-specific model adapter
- optional session backend
- tracing integration
- explicit `submit_result()` tool

규칙:

- canonical transcript는 우리 runtime이 보관한다
- SDK session은 convenience layer일 뿐 source of truth가 아니다
- summarization은 explicit memory event로만 허용
- reward / verification attachment를 위해 every step metadata를 기록한다
- final answer는 SDK `output_type`이 아니라 `submit_result()` tool call trace로 canonicalize한다
- `turn_count`는 solver run metadata로 기록한다

### Solver Swarm

swarm은 framework-native multi-agent보다 orchestration layer 개념이다.

기능:

- solver replica launch
- fixed quota by solver/model family
- provider-aware concurrency control
- timeout / retry / backoff
- circuit breaker / cooldown / quota rebalancing
- task queueing
- result collection

solver 수의 source of truth는 `models.solvers[*].replicas` 합계 하나뿐이다.

### Verification Engine

입력:

- GroundTruth
- SolverResult
- AnswerSchema

출력:

- VerifyResult

규칙:

- typed canonicalization first
- outcome_type branch first
- exact match만 정식 reward / accept 기준으로 사용
- fieldwise diagnostics는 디버깅과 error analysis용으로만 제공한다
- shadow verifier disagreement를 측정한다
- provenance fail은 correctness fail과 분리 기록

### Calibration Loop

입력:

- verified solver outcomes
- target pass-rate band

출력:

- accepted / rejected decision
- rejection reason

adaptive calibration rule:

- 초기값은 `initial_required_hops`, `initial_condition_complexity`
- `selected_tool_level`은 run 시작 시 고정하고, 동일 run 안에서는 바꾸지 않는다
- solver replica는 `canary_replica_count`만 먼저 실행하고, 판단이 불확실할 때만 `full_replica_limit`까지 점진적으로 확장한다
- canary 이후 확장은 `post_canary_batch_size` 단위 micro-batch로 진행하고, 각 batch 뒤에 decision을 다시 계산한다
- band를 벗어나면 confidence interval을 보고 방향을 결정한다
- 한 번에 한 축만 조절한다: `required_hops -> condition_complexity -> fanout_ambiguity`
- 조절 후에도 불확실하면 동일 difficulty로 한 번 더 샘플링한다
- `max_iterations_per_anchor` 도달 시 best-so-far fallback을 허용한다

현재 vertical slice에서는 adaptive loop를 보수적으로 적용한다.

- 같은 anchor 안에서 반복 실행은 이미 동작한다
- 현재 path를 바꾸는 조절은 `status_lookup`, `causal_chain`, `timeline_resolution` 중 path-compatible answer schema task에 허용한다
- 구현된 조절 축은 우선 `required_hops`와 compatible path 간 `fanout_ambiguity`다
- `aggregate_verification`과 질문/answer contract를 안전하게 재작성할 수 없는 task family는 현재 same-task 반복만 허용한다

safe early termination rule:

```text
N = total replicas
n = completed replicas
p = passes among completed replicas
r = N - n

min_final = p / N
max_final = (p + r) / N

if max_final < lower_bound:
  reject_too_hard
elif min_final > upper_bound:
  reject_too_easy
elif min_final >= lower_bound and max_final <= upper_bound:
  accept
else:
  continue
```

이 규칙은 upper bound를 무시하는 biased early accept를 금지한다.

### Dedup and Coverage

dedup은 두 단계다.

1. online exact dedup
2. post-run near-dedup

coverage는 아래 차원으로 추적한다.

- domain
- table
- path id
- hop count
- question family
- outcome type
- tool level
- answer type
- label tier

### Persistence and Eventing

운영 상태는 `run.db`에 기록한다.

필수 테이블:

- runs
- anchors
- tasks
- ground_truth
- solver_runs
- verification_results
- accepted_examples
- rejection_reasons
- budget_ledger
- dedup_signatures
- coverage_counters
- event_log

event bus는 아래 범주의 이벤트를 낸다.

- run_start / run_end
- anchor_started / anchor_skipped / anchor_failed
- task_composed / task_rejected
- solver_started / solver_completed / solver_failed
- verification_completed
- shadow_verifier_completed
- calibration_decision
- budget_reserved / budget_settled
- checkpoint_written

## CLI Dashboard

dashboard는 source of truth가 아니라 projection이다.

표시 항목:

- run progress
- active solver replicas
- provider utilization
- acceptance rate
- rejection breakdown
- budget spent / reserved
- coverage gaps
- recent failures
- shadow verifier disagreement rate

## Configuration

```yaml
database:
  dsn: "${DATABASE_DSN}"
  schema_allowlist: ["public"]
  readonly_role: "rlvr_reader"
  statement_timeout_ms: 5000
  lock_timeout_ms: 1000
  idle_tx_timeout_ms: 5000
  solver_pool_size: 32
  control_pool_size: 8

domain:
  name: "customer_support"
  language: "ko"

providers:
  codex_oauth:
    type: "openai_compatible"
    base_url: "http://127.0.0.1:10531/v1"
    api_key_env: "OPENAI_API_KEY"
    max_concurrency: 16
    timeout_s: 120
  local_server:
    type: "openai_compatible"
    base_url: "${LOCAL_OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
    api_key_env: "LOCAL_OPENAI_API_KEY"
    max_concurrency: 16
    timeout_s: 120
  anthropic_main:
    type: "anthropic"
    api_key_env: "ANTHROPIC_API_KEY"
    max_concurrency: 8
    timeout_s: 120
  google_main:
    type: "google"
    api_key_env: "GOOGLE_API_KEY"
    max_concurrency: 8
    timeout_s: 120

models:
  composer:
    provider: "codex_oauth"
    model: "gpt-5.4-mini"
  solver_backbone:
    provider: "codex_oauth"
    model: "gpt-5.4-mini"
  solvers:
    - solver_id: "local_reasoner"
      backend: "openai_agents"
      provider: "local_server"
      model: "gpt-5.4-mini"
      replicas: 3
      memory_mode: "none"
      summarization_mode: "off"
    - solver_id: "codex_reference"
      backend: "openai_agents"
      provider: "codex_oauth"
      model: "gpt-5.4-mini"
      replicas: 1
      memory_mode: "none"
      summarization_mode: "off"
    - solver_id: "anthropic_reference"
      backend: "openai_agents"
      provider: "anthropic_main"
      model: "claude-sonnet-4.5"
      replicas: 1
      memory_mode: "none"
      summarization_mode: "off"
    - solver_id: "codex_memory"
      backend: "openai_agents"
      provider: "codex_oauth"
      model: "gpt-5.4-mini"
      replicas: 2
      memory_mode: "explicit_summary"
      summarization_mode: "explicit"

provider_resilience:
  circuit_breaker_window_s: 60
  circuit_breaker_threshold: 0.3
  probe_interval_s: 30
  release_semaphore_on_backoff: true

tool_compiler:
  max_hops: 4
  allow_aggregates: true
  allow_timelines: true
  max_list_cardinality: 20

task_composer:
  label_tier: "A"
  question_families:
    - "status_lookup"
    - "causal_chain"
    - "timeline_resolution"
    - "aggregate_verification"
  selected_tool_level: 1
  initial_required_hops: 2
  initial_condition_complexity: 1
  max_required_hops: 6
  max_condition_complexity: 5
  negative_outcome_ratio: 0.2
  max_attempts_per_anchor: 6

solver_runtime:
  max_turns: 16
  structured_output_required: true
  tracing: true
  sdk_sessions_enabled: true
  canonical_state_store: "run_db"

calibration:
  lower_pass_rate: 0.2
  upper_pass_rate: 0.8
  ci_alpha: 0.1
  canary_replica_count: 2
  post_canary_batch_size: 2
  full_replica_limit: 6
  safe_early_termination: true
  max_iterations_per_anchor: 6

verification:
  require_provenance: true
  fail_on_internal_field_leak: true
  float_precision: 6
  shadow_sample_rate: 0.1

dedup:
  exact_enabled: true
  near_dup_enabled: true
  minhash_threshold: 0.9

budget:
  max_run_usd: 250.0
  max_gpu_hours: null
  compose_phase_usd: 40.0
  solve_phase_usd: 180.0
  reserve_strategy: "phase_specific"

privacy:
  default_visibility: "blocked"
  visibility_overrides:
    courier_phone: "user_visible"
    internal_ticket_note: "blocked"

output:
  run_db_path: "./artifacts/run.db"
  accepted_jsonl_path: "./artifacts/accepted.jsonl"
  rejected_jsonl_path: "./artifacts/rejected.jsonl"
  events_jsonl_path: "./artifacts/events.jsonl"
  traces_dir: "./artifacts/traces"
```

## Output Format

### Dataset JSONL

한 line은 하나의 accepted task다.

```json
{
  "task_id": "task_000123",
  "question": "고객이 마지막으로 환불 요청을 남긴 주문의 현재 배송 상태는 무엇인가요?",
  "label_tier": "A",
  "outcome_type": "answer",
  "answer_schema": {
    "version": "v1",
    "fields": [
      {"name": "delivery_status", "type": "string", "nullable": false, "ordered": false, "canonicalizer": "lower_trim"}
    ]
  },
  "ground_truth": {
    "expected_outcome_type": "answer",
    "canonical_answer": {"delivery_status": "in_transit"}
  },
  "calibration": {
    "pass_rate": 0.5,
    "band": [0.2, 0.8]
  },
  "metadata": {
    "anchor_table": "orders",
    "required_hops": 3,
    "tool_level": 2,
    "path_id": "orders.refunds.shipments",
    "question_family": "status_lookup",
    "language": "ko"
  }
}
```

### Run Manifest

run manifest는 최소 아래를 포함한다.

- run id
- config hash
- database fingerprint
- provider / model inventory
- total anchors scanned
- tasks composed
- accepted / rejected counts
- rejection histogram
- budget spent
- coverage summary
- output paths

## Throughput and Resilience Design

### Critical Path

accepted example 하나의 critical path는 다음과 같다.

```text
anchor selection
  -> compose
  -> ground truth
  -> task validation
  -> solver replicas
  -> verification
  -> calibration decision
  -> export
```

### Rolling Orchestration

batch barrier 방식은 금지한다.

- anchor queue를 둔다
- compose workers, solver workers, verify workers를 분리한다
- shared semaphore로 provider saturation만 제어한다
- accepted/rejected가 결정되는 즉시 다음 anchor를 투입한다

### Adaptive Calibration

anchor 하나는 단발 accept/reject로 끝나지 않는다.

- compose -> truth -> validate -> solve -> verify
- 결과가 target band 밖이면 동일 anchor에서 난이도 조절 후 재시도
- 조절 순서는 `required_hops -> condition_complexity -> fanout_ambiguity`
- confidence interval이 band와 겹치면 difficulty를 바꾸지 않고 추가 샘플링
- `max_iterations_per_anchor` 도달 시 best-so-far fallback 허용

### Budget Reservation

budget은 phase-specific reservation을 사용한다.

분리 항목:

- compose_api_usd
- solve_api_usd
- optional_gpu_hours

decision gate:

```text
spent + reserved + worst_case_next > max_run_budget -> block
```

정산은 phase별 settle로 처리한다.

현재 vertical slice에서는 `TaskSpec JSON/JSONL` 실행 경로에 budget guard가 이미 연결돼 있다. 구현상 `worst_case_next`는 현재 run에 주어진 task 수로 phase budget을 균등 분할한 per-task reservation으로 근사하고, 이후 실제 pricing model이 들어오면 같은 reservation/settle 경계를 유지한 채 교체한다.
또 solve phase settle은 planned replica 전체를 가정하지 않고, calibration early stop 이후 실제로 실행된 replica 수 비율에 맞춰 정산한다.

### Memory Management

solver memory는 세 모드다.

- `none`
- `explicit_summary`
- `session_only`

`explicit_summary` 모드에서는 summarization step이 event log에 기록되어야 한다.

### DB Resource Protection

모든 solver query는 아래 안전 규칙을 강제한다.

- dedicated readonly role
- `SET default_transaction_read_only = on`
- statement timeout
- lock timeout
- fixed search_path
- parameterized SQL only

### Provider Resilience

- per-provider concurrency cap
- replica-level provider failure absorption
- provider-level cooldown
- per-model error-rate metrics
- circuit breaker trip threshold
- healthy provider로의 quota rebalance
- retry/backoff policy는 bounded하게 유지하고 semaphore release semantics를 깨지 않도록 한다

### Canary Design

초기 canary는 전체 replica보다 작은 solver subset을 먼저 돌린다.

규칙:

- canary 결과만으로 `max_final < lower_bound` 혹은 `min_final > upper_bound`가 확정되면 즉시 종료
- 그 외는 full replica까지 확장
- shadow verifier sampling은 canary와 production 양쪽에서 유지한다

## Testing Strategy

### Fixture Schemas

최소 세 종류의 fixture schema가 필요하다.

1. 단순 2-hop 고객센터 schema
2. fanout / distractor가 큰 schema
3. visibility policy가 섞인 schema

### Test Matrix

- schema introspection correctness
- path catalog correctness
- tool SQL safety
- answer_schema canonicalization
- outcome_type verification branches
- lexical leak detection
- shortcut detection
- provenance verification
- solver runtime explicit submit_result
- binary reward and shadow verifier behavior
- calibration safe early termination
- adaptive difficulty convergence simulation
- checkpoint / resume
- dedup
- coverage accounting

### Canary -> Production Gate

아래를 통과해야 production run을 연다.

- tool SQL safety tests green
- verification golden tests green
- shadow verifier disagreement rate가 threshold 이하
- calibration simulation green
- sample run accepted example 품질 통과

## Project Structure

```text
src/rl_task_foundry/
  config/
  infra/
  schema/
  tools/
  tasks/
  truth/
  solver/
  verification/
  calibration/
  pipeline/
  cli.py
```

## Dependencies

핵심 의존성:

- `openai-agents`
- `asyncpg`
- `pydantic`
- `pyyaml`
- `typer`
- `rich`
- `datasketch`
- `sqlalchemy` or `sqlite3` for run.db access helpers
- `pytest`
- `pytest-asyncio`

선택 의존성:

- `scipy` for calibration analytics
- `pandas` for offline analysis
- tracing / observability backend

## Key Design Decisions

1. solver substrate는 `OpenAI Agents SDK`를 사용한다
2. canonical runtime state는 우리 쪽 `AgentRuntime`이 소유한다
3. `answer_schema`를 1급 계약으로 둔다
4. 운영 source of truth는 `run.db`다
5. solver 수의 source of truth는 `models.solvers[*].replicas` 합계 하나다
6. privacy는 blanket PII ban이 아니라 visibility policy로 다룬다
7. early termination은 safe bound 기반으로만 허용한다
8. batch barrier 대신 rolling orchestration을 사용한다
9. verification, calibration, export는 SDK 밖에서 수행한다
10. solver와 RL rollout agent는 같은 step contract를 사용한다
11. calibration은 CI-based adaptive loop와 best-so-far fallback을 사용한다
12. solver difficulty metadata에는 `tool_level` (`L1` / `L2`) 축이 포함되지만, 동일 run 안의 adaptive calibration은 이 값을 바꾸지 않는다
13. outcome diversity (`answer`, `no_result`, `clarify`, `deny`)를 데이터 계약에 포함한다
14. 최소 두 개 이상의 provider family에서 calibration label을 측정할 수 있어야 한다
15. verifier의 정식 판정은 binary reward이고, shadow disagreement와 fieldwise diagnostics는 보조 관측값으로만 기록한다

## Open Questions

- explicit memory mode에서 summary trigger policy를 heuristic으로 둘지, 별도 summarizer tool로 둘지
- provenance fail을 hard fail로만 둘지, auxiliary label로도 export할지
- Tier B를 v1에 포함할지, v1.1로 미룰지
- near-dup threshold를 domain별로 다르게 둘지
- shadow verifier를 rule-based secondary engine으로 둘지, 독립 SQL/row-context validator로 둘지
