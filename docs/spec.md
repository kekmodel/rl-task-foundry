# RL Task Foundry — Synthesis-Agent Hybrid Design Spec

## Overview

이 프로젝트의 목표는 사람이 read-only PostgreSQL DB를 등록하면, 그 DB를 바탕으로 RLVR용 compositional task environment를 자동 생성하고, 엄격한 quality gate를 통과한 environment만 registry에 누적하는 상시 파이프라인을 만드는 것이다.

핵심 paradigm은 더 이상 `path-centric single-value lookup`이 아니다.

새 기준은 아래와 같다.

- unit of generation은 단일 task가 아니라 db-level atomic tool bundle을 참조하는 environment bundle이다
- synthesis agent는 label-first로 task contract, `solution.py`, instance space를 합성하고 runtime이 per-instance `rendered_user_prompt`를 materialize한다
- truth source는 DB-grounded label construction이고, reward path는 canonical answer + exact-match reward로 수렴한다
- solver는 compositional reasoning을 풀고, reward model은 deterministic binary exact match를 준다

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
2. label correctness와 uniqueness는 RLVR의 핵심 자산이며 절대 희생하지 않는다
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
- synthesis agent가 grounded exploration 뒤에 canonical label, proposed environment, audit artifact source를 생성한다
- cross-instance와 solver pass-rate quality gate를 통과한 environment만 registry에 커밋한다
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
- baseline qualitative artifacts는 archive로 보존한다
- production run은 새 hybrid pipeline이 proof되기 전까지 금지한다
- task generation code의 incremental polishing은 중단한다

## Modules After Transition

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
  - `submit_result(answer_text: str)`
  - no system prompt
  - environment-contract-first runtime
- `pipeline/environment_orchestrator.py`
- `calibration/`
  - pass-rate band
  - Clopper-Pearson CI
  - safe early stop
- `cli.py`

삭제된 path-centric legacy stack은 git history에만 남고 authoritative runtime surface에는 포함되지 않는다.

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
   - atomic tool generation / materialization
   - task synthesis
   - solution synthesis
   - verifier synthesis
   - instance / canonical answer materialization

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
   - multi-db orchestrator

## Environment as the Core Unit

### Bundle Unit

이 rewrite에서 environment의 실제 배포 단위는 per-environment narrow tool bundle이 아니다.

핵심 unit은 아래 두 층의 조합이다.

- database 단위 shared atomic tool bundle
- environment 단위 metadata / instances / canonical answers / audit artifacts

즉 같은 `db_id`에 속한 여러 environment는 동일한 atomic tool set을 공유하고,
environment는 그 위에 놓이는 task / verifier / rollout metadata layer가 된다.

### Filesystem Layout

```text
bundle_root/
  databases/
    <db_id>/
      atomic_tools.py
      atomic_tool_definitions.json
  environments/
    <env_id>/
      environment.yaml
      instances.jsonl
      canonical_answers.jsonl
      audit/
        solution.py
        verifier.py
        shadow_verifier.py
```

authoritative bundle 구조와 각 파일의 역할은 아래 `Environment Bundle Structure` 섹션에서 다시 고정한다.

### Environment Metadata

`environment.yaml`은 최소 아래를 포함한다.

```yaml
env_id: str
db_id: str
domain: str
category: str
difficulty_vector:
  search_cost: float
  solution_space: float
  constraint_density: float
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
  - `task`
  - `solution`
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
DifficultyVector:
  search_cost: float
  solution_space: float
  constraint_density: float
```

운영 원칙:

- 한 escalation step에서는 정확히 한 difficulty axis만 증가시킨다
- difficulty crank는 monotonic vector를 가져야 한다
- 종료 조건은 pass-rate band 수렴 또는 iteration budget 소진이다
- crank 순서는 `search_cost -> solution_space -> constraint_density` 이다

이 difficulty model은 actor가 실제로 atomic tool을 통해 겪는 비용을 직접 반영한다.

- `search_cost`
  - DB에서 필요한 정보를 모으는 데 드는 탐색 비용
  - runtime이 artifact를 분석해 `(diversity, scale)`로 측정한다
- `solution_space`
  - 가능한 답 조합의 크기
  - runtime이 artifact를 분석해 `(diversity, scale)`로 측정한다
- `constraint_density`
  - 전체 해 공간 중 유효한 해의 희소성
  - runtime이 artifact를 분석해 `(diversity, scale)`로 측정한다

label과 canonical answer contract가 먼저 difficulty를 결정하고, `rendered_user_prompt`는 그 latent task semantics를 자연스럽게 surface하는 단계다. prompt wording 자체를 비틀어서 difficulty를 만드는 것은 금지한다.

runtime은 각 axis에 대해 내부적으로 `D_a(T) = (d_a, s_a)` 를 측정한다.

- `d_a`
  - 해당 axis에서 서로 다른 reasoning primitive 수
- `s_a`
  - primitive당 반복 규모

crank rule은 lexicographic이다.

- 먼저 diversity(`d_a`)를 올린다
- diversity가 DB schema 또는 현재 artifact 구조 때문에 포화되면 scale(`s_a`)을 올린다

이 `(d, s)` 값은 solver에는 노출되지 않는다. synthesis runtime은 이 값을 내부적으로만 유지하고, synthesis agent에는 English natural-language `crank_hint`로만 전달한다. raw debug scalar dump를 prompt에 그대로 넣는 것은 금지한다.

## Actor-facing Interface

actor가 episode 동안 볼 수 있는 정보 상태는 training / evaluation / rollout에서 완전히 동일해야 한다.  
이 spec은 이를 parity 불변량으로 다룬다.

actor에게 노출되는 것은 정확히 세 가지다.

- `rendered_user_prompt`
- database 단위 atomic tool set
- `submit_result`

그 외의 task-specific 정보는 actor에게 도달하면 안 된다.

- system prompt 없음
- hidden verifier rule 없음
- hidden task metadata 없음
- canonical answer 직접 노출 없음
  - canonical answer가 client에 노출되면 actor가 이를 그대로 echo해서 reward `1.0`을 받는 shortcut이 생긴다
  - 이는 training signal을 무의미하게 만들므로 canonical answer는 server trust boundary 안에만 남아야 한다

### Parity Principle

solver의 information state는 actor가 실제 RL training에서 보게 될 것과 byte-level에 가깝게 동일해야 한다.

이 뜻은 다음과 같다.

- solver prompt는 task-agnostic constant여야 한다
- task-specific 정보의 single source of truth는 `rendered_user_prompt`뿐이다
- tool definition은 DB schema에서 기계적으로 생성된 atomic tool set뿐이다
- `submit_result`는 task-agnostic terminal action이어야 한다

solver가 task.question, answer schema, verifier internals, canonical answer, synthesis memory를 직접 보면 parity가 깨진다.

### rendered_user_prompt

`rendered_user_prompt`는 task-specific 정보를 담는 단일 문서다.

구성 요소:

- 자연어 질문
- hard constraint 설명
- tie-breaker 규칙
- `Submit Result Format:` 이후의 JSON schema text
- "`submit_result`를 호출하라"는 지시

운영 원칙:

- 자연어 질문과 constraint wording은 모두 이 prompt에 들어간다
- instance parameter는 instance 생성 시점에 prompt에 fully substitute되어 `instances.jsonl`에 저장된다
- template placeholder 형태를 runtime rollout 경로에 남기지 않는다
- 출력 필드 이름은 prompt 안의 `Submit Result Format:` block에서만 노출된다
- schema 밖의 hidden formatting rule을 두지 않는다

### submit_result

actor의 terminal action은 task-agnostic constant다.

```python
submit_result(answer_text: str) -> None
```

운영 규칙:

- parameter는 `answer_text: string` 단일 필드만 가진다
- description은 constant여야 한다
- description에 task-specific question, answer shape, exemplar answer를 넣지 않는다
- 제출값은 JSON string이어야 한다

server는 이 string을 받아:

1. `json.loads(answer_text)`
2. `canonicalize(parsed, output_schema)`
3. canonical expected answer와 exact match 비교

를 수행한다.

즉 actor는 “답안을 dict로 직접 반환”하는 것이 아니라 “JSON string을 제출”한다.

## Tool Architecture

tool set은 environment 단위가 아니라 database 단위다.

- 같은 `db_id`의 모든 environment는 동일한 atomic tool set을 공유한다
- tool set은 schema graph로부터 rule-based로 생성된다
- synthesis agent는 `tool.py`를 생성하지 않는다

### Atomicity

atomic tool의 정의는 “하나의 SQL primitive만 수행하는 tool”이다.

금지:

- 1-hop을 넘는 multi-table join
- subquery
- `GROUP BY`
- window function
- 한 번의 tool 호출로 최종 정답을 바로 산출하는 helper

허용:

- single-table point lookup
- bounded enumeration
- single-column filter
- 1-hop FK traversal
- distinct value listing
- filtered single-column aggregate
- sorted top-k over a single table
- grouped aggregate top-k over a single table

즉 한 tool 호출만으로는 정답에 도달할 수 없고, 모든 정답은 tool chaining으로만 도달해야 한다.

### Tool Family

v1 atomic tool family는 여덟 종류다.

1. T1 point lookup
   - `get_{table}_by_id`
   - `get_{table}_by_ids_batch`
2. T2 bounded enumeration
   - `count_{table}`
   - `list_{table}_ids`
3. T3 single-column filter
   - `eq`
   - `range`
   - `like`
   - `in`
4. T4 FK traversal
   - bidirectional
   - one-hop only
5. T5 distinct values
6. T6 filtered single-column aggregates
   - `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`
   - `AVG`와 non-integer `SUM`은 `atomic_tools.float_precision` 기준으로 DB에서 `ROUND`한다
7. T7 sorted top-k
   - filtered / unfiltered variant
   - actor-specified `limit` with runtime cap
   - deterministic `ORDER BY {sort_col}, {pk}` tie-break
8. T8 grouped aggregate top-k
   - filtered / unfiltered variant
   - `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`
   - actor-specified `limit` with runtime cap
   - deterministic `ORDER BY aggregate_value, group_key` tie-break

### Generation and Compression

tool set은 schema graph의 함수로 고정된다.

- naming은 schema-name 기반 기계적 규칙을 따른다
- `visibility=internal/blocked` column은 actor-facing tool surface에서 제외한다
- PK column은 T1에서만 직접 노출해 중복 도구를 줄인다
- multi-row family(`T2`, `T3`, `T4`, `T5`, `T7`, `T8`)는 actor-specified `limit`를 받고 runtime이 `bounded_result_limit` cap을 적용한다
- `max_tool_count`를 넘으면 deterministic compression을 적용한다
- drop priority는 `aggregate -> grouped_aggregate -> sorted_top_k -> like -> distinct -> range -> in` 순이다

중요한 운영 원칙:

- tool count는 난이도 조절 lever가 아니다
- 난이도는 tool 수가 아니라 chain depth와 constraint structure가 담당한다
- tool name은 task intent를 leak하지 않아야 한다

실측 참고값:

- atomic tool count는 schema, visibility profile, compression 정책의 함수다
- current authoritative family surface는 `T1~T8`이며 exact count는 real DB trial 시점의 schema snapshot으로 재산출한다
- `max_tool_count=256`은 상한이지 목표값이 아니다

추가 원칙:

- 어떤 environment에서 `solution.py`가 호출하지 않는 atomic tool은 그 environment에 대해 distractor 역할을 한다
- distractor는 별도 플래그나 분리된 저장 구조를 갖지 않는다
- 즉 distractor 여부는 tool set 전체와 solution trace를 비교하면 structural하게 식별된다

### SQL Contract

모든 atomic tool 구현은 아래 contract를 따른다.

```python
async def tool_name(conn: asyncpg.Connection, **kwargs) -> Any
```

요구:

- parameterized SQL (`$1`, `$2`, ...)
- read-only (`SELECT` only)
- PK equality 또는 explicit `LIMIT`이 있는 bounded query
- deterministic execution (`NOW`, `RANDOM` 등 금지)
- `asyncpg` 기반 live PostgreSQL access

### Completeness Boundary

ADR 0002의 informal completeness argument를 따른다.

chain으로 도달 가능한 영역:

- point lookup
- single-column filter
- FK-based multi-hop traversal
- local grouping / sorting / intersection이 수반되는 ordinary `SELECT` queries

표현 불가 영역:

- recursive CTE
- window function
- 복잡한 many-table analytic join
- `LIKE`를 넘는 full-text semantics

synthesis prompt는 task를 이 expressible subset 안에만 생성해야 한다.

## Synthesis Pipeline

synthesis agent는 아래 authoritative input을 생성한다.

- `question`
- `output_schema`
- `solution.py`
- `verifier.py`
- `shadow_verifier.py`
- `instance_space`

생성하지 않는 것:

- `tool.py`
- `tool_self_test.py`

### Stage 1: Schema and Data Exploration

- schema introspection
- sample row inspection
- grounded atomic-tool exploration
- domain/category proposal

structured output:

- `SchemaExplorationOutput`
  - `domain_hypothesis`
  - `candidate_categories`
  - `sample_observations`
  - `memory_summary`

운영 규칙:

- `candidate_categories`는 비어 있으면 안 된다
- `sample_observations`는 비어 있으면 안 된다
- category inference는 이 structured output만을 authoritative input으로 쓴다

### Stage 2: Label Construction

label construction phase output:

- `LabelConstructionOutput`
  - `canonical_answer_json`
  - `output_schema`
  - `difficulty_vector`
  - `instance_parameters`
  - `label_summary`
  - `memory_summary`

이 단계가 latent task semantics의 source of truth다.

### Stage 3: Task Synthesis

task synthesis phase output:

- `TaskSynthesisOutput`
  - `question`
  - `constraint_summary`
  - `instance_space`
  - `memory_summary`

이 단계는 label을 자연어 task로 surface할 뿐, label을 바꾸지 않는다.

### Stage 4: Code Generation

artifact generation phase output:

- `ArtifactGenerationOutput`
  - `proposed_environment`
  - `artifacts`
  - `memory_summary`

`proposed_environment`는 trust field를 포함하지 않는다.

### Runtime Flow

synthesis 진입 시점의 흐름은 아래와 같다.

1. `db_id`에 대한 atomic tool bundle이 캐시되어 있지 않으면 `AtomicToolGenerator`로 생성한다
2. 생성물은 `databases/{db_id}/` 하위에 materialize한다
3. schema/data exploration phase에서 synthesis agent는 live atomic tools를 실제로 호출해 sample rows와 grounded observations를 수집한다
4. category inference가 grounded exploration output만을 보고 category를 확정한다
5. label construction이 canonical answer, output schema, difficulty vector를 먼저 확정한다
6. task synthesis가 그 latent label을 유일한 정답이 되도록 question / constraints / instance space로 surface한다
7. code generation phase가 이미 확정된 label/task를 재현하는 `solution.py`, `verifier.py`, `shadow_verifier.py`를 생성한다
8. registration policy가 generated code가 atomic tool set 내 이름만 호출하는지 AST 검증한다
9. self-consistency와 triple oracle 검증을 수행한다
10. runtime이 per-instance `rendered_user_prompt`와 canonical answer를 materialize한다
11. environment registry에 commit한다

### Prompt Constraints

synthesis prompt는 아래를 강하게 요구한다.

- system instructions / error templates / difficulty templates / few-shot structure는 English로 고정한다
- 생성되는 `question`, constraint summary, tie-break wording, `rendered_user_prompt`만 `config.domain.language`를 따른다
- schema/data exploration phase에서만 live tool calling을 허용한다
- later phase는 grounded outputs만 보고 label-first로 진행한다

- unique canonical answer가 존재해야 한다
- 자연어 질문 안에 tie-breaker를 반드시 포함한다
  - 예: cheapest, earliest, alphabetical
- `solution.py`는 deterministic이어야 한다
  - `random`, `now`, `os.environ` 금지
- atomic tool만 호출할 수 있다
- 새로운 tool 정의는 금지한다
- live PostgreSQL을 전제로 하므로 SQLite compatibility 제약은 더 이상 두지 않는다

### Generation Retry

v1 generation retry는 grounded exploration / category inference를 재사용하면서 `label construction -> task synthesis -> code generation` block만 재시도한다.

- schema/data exploration과 category inference는 한 번만 수행한다
- label / task / artifact block만 `attempt_index=1..N`으로 재실행한다
- artifact generation 실패 diagnostics는 다음 attempt input으로 들어간다
- band 밖 quality feedback도 다음 harder retry input으로 들어간다
- canonical answer materialization에 성공한 attempt만 rollout으로 넘어간다

운영 규칙:

- `max_generation_attempts`를 가진다
- tool set이 바뀌면 solution을 invalidate한다
- task가 infeasible이면 discard 경로를 가진다
- discard는 budget을 소비한다
- `db_id x category` failure counter와 backoff window를 유지한다
- difficulty vector는 retry 과정에서 감소하면 안 된다

### Cross-Instance

cross-instance variation의 주체는 runtime이다.

agent는 `instance_space`만 정의한다.

bundle contract는 deterministic sampler 기반 multi-instance 확장을 전제로 한다.
다만 2026-04-12 기준 current implementation은 bootstrap 단계로 단일 persisted instance(`instance_0001`)만 materialize한다.
`instances.jsonl`, `canonical_answers.jsonl`, `CrossInstanceSet` contract는 이후 multi-instance expansion을 위해 이미 고정되어 있다.

## Verification and Reward Model

production reward verification은 canonical answer exact match 기반으로 단일화한다.

핵심 가정:

- 각 instance에는 canonical answer가 정확히 하나 존재한다
- 이 canonical answer는 DB-grounded label construction에서 선택되고 tie-break까지 포함해 canonicalize된다

### Submit and Reward Flow

1. actor가 `submit_result(answer_text: str)`를 호출한다
2. server가 raw string을 capture한다
3. `json.loads(answer_text)`를 수행한다
   - 실패 시 `json_decode_failed`
4. `canonicalize(parsed, output_schema)`를 수행한다
   - 실패 시 `schema_mismatch`
5. canonical submitted answer와 canonical expected answer를 exact match 비교한다
   - 불일치 시 `em_mismatch`

reward는 binary다.

- `1.0`
- `0.0`

부분 점수는 없다.

### Canonicalization

canonicalization은 schema-driven recursive function이다.

- `OutputFieldContract.ordered`
- `OutputFieldContract.sort_key`
- `OutputFieldContract.unique_elements`

를 기준으로 동작한다.

정책:

- ordered list는 순서를 보존한다
- unordered list of primitives는 값 기준으로 정렬한다
- unordered list of objects는 `sort_key`가 필수다
- tie는 `sort_key -> canonical JSON repr` fallback으로 해소한다
- `date` / `datetime`은 단일 ISO-8601 format으로 normalize한다
- string은 byte-level exact compare를 사용한다

### Label Truth

현재 authoritative truth source는 label construction 단계다.

- schema exploration은 live atomic tools로 실제 DB row를 본다
- label construction은 그 grounded evidence를 바탕으로 canonical answer를 먼저 확정한다
- task synthesis는 그 label이 유일한 정답이 되도록 question과 constraints를 역설계한다
- `solution.py`는 이미 확정된 label semantics를 재현하는 audit artifact다

즉 truth는 verifier code가 아니라 source-of-truth DB에서 선택된 label에 있다.

### Reward Function

reward function은 pure function이어야 한다.

```python
compute_reward(submitted_text, canonical_answer, output_schema) -> RewardResult
```

금지:

- DB 접근
- synthesized code 실행
- non-deterministic branch

이 함수는 RL training runtime에 그대로 들어갈 수 있어야 한다.

## RL Runtime and Environment API

environment API server는 이 harness의 최종 runtime consumer이며, 구현 자체는 이 repo의 v1 범위 밖일 수 있다.

### Recommended Architecture

- stateless request handling
- `session_id` correlation
- async server runtime
- live PostgreSQL hit via `asyncpg`
- `pgbouncer` transaction pooling + read replica 전제 가능

server가 로드하는 것은 아래다.

- db-level atomic tool bundle
- environment metadata
- per-instance rendered prompt
- per-instance canonical answer

canonical answer는 server 내부 trust boundary에 남고 client로 노출되지 않는다.

### Request / Response Model

권장 API shape:

- `ResetEpisode`
- `CallTool`
- `SubmitAnswer`

transport는 gRPC를 권장한다.

session state는 server가 아니라 client가 갖는 것을 기본으로 한다.

- `tool_call_index`
- `episode_start_ms`
- `session_id`

server는 request echo 값을 읽어 stateless하게 다음을 검증한다.

- `max_turns`
- timeout
- session consistency

reward 계산은 `SubmitAnswer` handler 내부에서 inline 실행한다.

tool result에는 optional in-process LRU cache를 둘 수 있다.

### Bundle Self-Containment

environment bundle은 self-contained여야 한다.

- server는 bundle 외의 synthesis runtime state를 요구하면 안 된다
- registration policy나 synthesis memory를 서버 runtime 의존성으로 요구하면 안 된다
- bundle만 있으면 server가 tool call과 reward calculation을 처리할 수 있어야 한다

## Environment Bundle Structure

bundle은 이 harness의 최종 deliverable이다.

```text
bundle_root/
  databases/
    <db_id>/
      atomic_tools.py
      atomic_tool_definitions.json
  environments/
    <env_id>/
      environment.yaml
      instances.jsonl
      canonical_answers.jsonl
      audit/
        solution.py
        verifier.py
        shadow_verifier.py
```

이 구조는 `export-bundle` CLI가 registry에서 materialized environment를 읽어 생성한다.

파일 역할:

- `databases/{db_id}/atomic_tools.py`
  - shared atomic tool implementation
  - 모든 env가 import하는 audit/runtime source
- `databases/{db_id}/atomic_tool_definitions.json`
  - actor-facing tool spec
  - JSON Schema draft-2020-12 기반 tool list
  - 각 tool entry는 `name`, `description`, `params_schema`, `returns_schema`를 가진다
  - server는 이를 그대로 FunctionTool spec으로 변환한다
- `environments/{env_id}/environment.yaml`
  - env metadata
  - output schema
  - rollout constraints
  - `atomic_tool_set_ref`
- `environments/{env_id}/instances.jsonl`
  - instance별 `instance_id`, `rendered_user_prompt`, params
  - 여기의 `rendered_user_prompt`는 placeholder가 남지 않은 fully rendered prompt다
- `environments/{env_id}/canonical_answers.jsonl`
  - instance별 canonical answer
- `environments/{env_id}/audit/`
  - `solution.py`
  - `verifier.py`
  - `shadow_verifier.py`
  - 서버 runtime이 아니라 audit / debug / validation 용도

rollout constraints는 environment 생성 시 freeze된다.

- `max_turns`
- `max_episode_duration_ms`
- `max_tool_rows`

## Contract Schema

### OutputFieldContract

`OutputFieldContract`는 기존 nested schema 기능을 유지하면서 canonicalization metadata를 추가한다.

- `ordered: bool = False`
- `sort_key: tuple[str, ...] | None = None`
- `unique_elements: bool = False`

unordered object list에서는 `sort_key`가 mandatory다.

### TaskContract

`TaskContract`는 actor-facing task description을 담는다.

- `question: str`
- `output_schema: OutputSchemaContract`
- `constraint_summary`
- `difficulty_vector`
- `category`
- `instance_parameters`

actor-facing `rendered_user_prompt`는 현재 `TaskContract`가 아니라 instance materialization 결과(`instances.jsonl`)에 저장된다.

### SolutionContract

`solution.py`는 reference oracle 역할을 한다.

- self-consistency 단계에서만 사용
- production actor runtime에는 절대 노출되지 않는다
- signature는 `def solve(tools) -> dict`

### VerifierContract

2026-04-12 기준 현재 verifier contract는 아래다.

```python
async def fetch_facts(answer, tools) -> dict
def facts_match_answer_claims(answer, facts) -> bool
def check_constraints(answer, facts) -> bool
def verify(answer, tools) -> bool
```

`VerifierContract`는 `facts_schema`를 포함하고, runtime probe / registration lane은 위 staged contract를 authoritative path로 검사한다.

계획상 `compute_canonical_answer(tools) -> dict` 단일 entrypoint로의 수렴은 여전히 가능하지만, 아직 구현 완료 상태는 아니다.

### ShadowVerifierContract

shadow verifier는 같은 staged signature를 따른다.

차이는 contract looseness가 아니라 independent reasoning path다.

### EnvironmentContract

`EnvironmentContract`는 per-env narrow tool list를 더 이상 갖지 않는다.

- `tools: list[ToolContract]` 제거
- `atomic_tool_set_ref: str` 추가
- `rollout_constraints` 추가

예:

- `atomic_tool_set_ref: "db://sakila"`

### GeneratedArtifactBundle

generated artifact bundle은 이제 아래만 가진다.

- `solution_source`
- `verifier_source`
- `shadow_verifier_source`

제거:

- `tool_source`
- `tool_self_test_source`

### RolloutConstraintsContract

```python
RolloutConstraintsContract:
  max_turns: int
  max_episode_duration_ms: int
  max_tool_rows: int = 1000
```

### Legacy Contracts

아래 contract는 legacy deletion으로 이미 authoritative path에서 제거됐다.

- `AnswerSchema`
- `AnswerField`
- `TaskSpec`

authoritative spec에서는 더 이상 중심 contract로 다루지 않는다.
삭제는 `docs/plan.md`의 `Milestone M-Atomic-Transition` Phase 4 (`C11`)에서 완료됐다.

## Code Registration Policy

unregistered generated Python code는 in-process로 실행하지 않는다.  
`AST preflight + subprocess isolation`은 계속 mandatory다.

다만 ADR 0002 이후 registration 대상은 바뀐다.

- generated `tool.py` registration은 제거된다
- atomic tool implementation은 schema-driven deterministic generator가 만든다
- registration policy는 `solution.py`, `verifier.py`, `shadow_verifier.py`에 집중한다

### Static Policy

AST 검사에서 아래를 강제한다.

- import allowlist
- 금지 builtins / attributes
- 금지 syntax
- signature contract
- dunder access ban
- atomic tool set 밖의 이름 호출 금지

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

#### Import Policy

허용 예:

- `datetime`
- `math`
- `json`
- `re`
- `decimal`
- `typing`
- `collections.abc`

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

persistent subprocess worker pool에서 실행되는 것은 아래다.

- `solution.py`
- `verifier.py`
- `shadow_verifier.py`
- triple oracle self-consistency evaluation

운영 방식:

- subprocess worker는 persistent pool이다
- 각 worker는 bounded read-only async DB pool을 가진다
- timeout과 memory limit은 subprocess 단위로 강제한다
- main process의 asyncpg pool을 subprocess와 공유하지 않는다

#### Lane B: Production Solver Runtime Lane

production runtime은 db-level atomic tool bundle을 main process 또는 environment API server process에서 직접 실행한다.

- per-tool-call subprocess roundtrip은 하지 않는다
- registration isolation은 untrusted generated audit code에만 적용한다
- production actor runtime은 generated verifier code를 실행하지 않고 canonical answer + pure reward function만 사용한다

## Parity Invariants

아래 불변량은 test로 강제한다.

1. solver prompt는 task-agnostic constant다
2. 같은 `db_id`의 모든 environment는 identical tool definitions를 공유한다
3. answer schema field name은 `rendered_user_prompt`의 `Submit Result Format:` block 밖에 등장하지 않는다
4. answer schema field name은 actor-facing tool definitions JSON에 등장하지 않는다
5. `submit_result` params schema는 constant `{answer_text: string}`이다
6. solver backend는 actor-facing task 정보 source로 `environment.task.question`, `environment.task.output_schema`, `environment.task.category`를 직접 사용하지 않는다
7. environment bundle은 self-contained다

## Quality Gate

environment는 아래 단계를 모두 통과해야 accepted 된다.

1. code registration policy
2. self-consistency triple oracle
3. cross-instance consistency
4. solver pass-rate band
5. registry dedup / coverage policy

운영 메모:

- shadow verifier는 triple oracle의 세 번째 독립 계산 경로로 포함된다
- quality gate의 공식 reward model은 canonicalized exact match다
- pass-rate band는 atomic tool 전환 이후 다시 calibration해야 한다
- 2026-04-12 기준 synthesis registry runner와 real-db trial path 모두 `code registration -> self-consistency -> cross-instance -> solver pass-rate band -> registry` 순 acceptance gate를 사용한다
- 남은 open item은 quality-gate 결과를 difficulty crank loop에 직접 닫는 automated closed-loop policy와 pass-rate band re-calibration이다

## Domain Scheduler and Provider Resilience

### Domain Scheduler

- DB add/remove 지원
- round-robin / priority queue
- DB 단위 독립 처리
- per-DB progress 추적
- v1 helper는 `SynthesisDomainScheduler`로 구현되며 `SynthesisDbSnapshot + category_status()`를 입력으로 받아 next `(db_id, category)` 또는 earliest backoff wait를 결정한다

### Synthesis Provider Resilience

synthesis agent도 solver와 동일하게 provider resilience를 사용한다.

- provider semaphore
- circuit breaker
- cooldown
- quota rebalance

즉 synthesis runtime/provider resilience는 legacy orchestrator reuse가 아니라 현재 `synthesis/runtime.py` 내부 구현과 `pipeline/environment_orchestrator.py` rollout path를 기준으로 동작한다.

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

v1 구현은 `output.traces_dir.parent / environments` 아래에 filesystem bundle을 쓰고,
`output.traces_dir.parent / environment_registry.db`에 sqlite index를 유지한다.

### SQLite Index

최소 기록 항목:

- env_id
- db_id
- domain
- category
- difficulty_band
- created_at
- status
- exact_signature
- tool_signature
- task_signature
- verifier_signature
- generator_version
- semantic_dedup_text
- semantic_dedup_text_version
- semantic_minhash_signature
- filesystem_path
- payload_json

운영 메모:

- sqlite index는 WAL mode + single-writer append queue를 기본으로 한다
- 조회용 index는 최소 `exact_signature`, `(db_id, category)`를 유지한다
- parallel synthesis worker는 직접 index에 동시 write하지 않고 writer lane을 통해 반영한다

### Dedup

dedup은 두 단계다.

1. exact dedup
   - v1은 `db_id + tool_signature + task_signature + verifier_signature`의 exact signature를 sqlite unique key로 사용한다
   - `instance_space`는 exact dedup key에 포함하지 않는다. environment identity는 task/tool/verifier contract 기준으로 보고, `instance_space`는 runtime variation surface로 취급한다
2. semantic dedup
   - v1은 같은 `db_id x category` 안에서 `semantic_dedup_text`의 MinHash similarity로 near-duplicate를 판정한다
   - `minhash_threshold`는 config source-of-truth다
   - v1은 commit 시 MinHash signature를 sqlite에 저장하고 lookup 시 재사용한다
   - lookup은 pair-scoped MinHashLSH candidate retrieval 뒤에 exact similarity 재계산으로 best match를 고른다
   - registry는 semantic candidate document를 durable하게 저장하고 query surface를 제공한다
   - v1 shingling은 normalized word 3-gram 기준이라 Korean phrasing variation에서는 false negative가 남을 수 있다

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

v1 registry는 read/query surface도 제공한다.

- recent environment listing
- structured coverage snapshot
- semantic dedup candidate export

v1 coverage planner는 registry inventory를 기준으로 deficit plan도 계산한다.

- source-of-truth target은 `synthesis.coverage_planner.target_count_per_band`다
- 기본 tracked band는 `low, medium, high`이고 `unset`은 opt-in이다
- planner는 zero-count cell도 포함해서 `db_id x category x difficulty band` deficit을 계산한다
- planner는 `db_id x category` pair-level aggregate deficit도 같이 계산해서 다음 scheduler priority 입력으로 재사용할 수 있어야 한다

## Manual Bundle Review

manual review는 mandatory production gate가 아니다.
quality gate와 독립적인 qualitative spot-check process다.

포함:

- exported bundle의 `environment.yaml`
- `instances.jsonl`
- `canonical_answers.jsonl`
- `databases/{db_id}/atomic_tool_definitions.json`
- `audit/solution.py`

운영 역할:

- 품질 taxonomy 작성
- prompt / policy 회귀 감지
- proof environment 검토

즉 verifier correctness의 source of truth는 아니고, 운영 guardrail이다.

### Review Rubric

exported bundle review의 “예시 수준 이상” 평가는 아래 rubric으로 기록한다.

- `compositional_structure`
  - multi-slot 또는 multi-entity coupling이 필요하다
- `constraint_density`
  - non-trivial hard constraint가 5개 이상이다
- `branching_or_threshold`
  - conditional branching 또는 numeric threshold가 존재한다
- `grounded_verification`
  - canonical answer 계산 경로와 exact-match reward path가 명확하다
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
- reward path는 canonicalized exact-match 기반 deterministic binary reward를 준다
- shadow / cross-instance / pass-rate 품질 필터가 모두 동작한다
- arbitrary DB를 registry에 추가해도 pipeline이 돌아간다
- 최근 exported bundle 10개 중 최소 7개가 “예시 수준 이상” 품질로 정성 평가된다

## Freeze Policy

이 spec 기준으로:

- 기존 task generation code는 incremental polishing을 중단한다
- rewrite는 `spec -> plan -> core contracts -> proof environment -> generalization` 순으로 간다
- proof environment가 나오기 전 production run은 금지한다
