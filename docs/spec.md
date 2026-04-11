# RL Task Foundry — Synthesis-Agent Hybrid Design Spec

## Overview

이 프로젝트의 목표는 사람이 PostgreSQL DB를 등록하면, 그 DB를 바탕으로 RLVR용 high-quality compositional task environment를 자동 생성하고, 엄격한 품질 필터를 통과한 environment만 dataset registry에 누적하는 상시 파이프라인을 만드는 것이다.

핵심 paradigm은 더 이상 `path-centric single-value lookup`이 아니다.

새 기준은 아래와 같다.

- task는 `<environment, tools, task, verifier>` 4-tuple 단위로 생성한다
- synthesis agent가 task, tool surface, solution, verifier를 함께 합성한다
- 그러나 verifier 신뢰도는 `hybrid DB-grounded constraints`로 강하게 통제한다
- solver는 compositional reasoning을 풀고, verifier는 deterministic binary reward를 준다

대표 목표 task는 다음 수준이다.

- trip planning
- roster / assignment
- bundle construction
- no-repeat recommendation
- threshold-conditioned selection
- temporal / budget / uniqueness constraints가 섞인 planning task

즉 이 시스템은 “값 하나를 조회하는 질문 생성기”가 아니라,  
“임의 DB에 대해 compositional reasoning environment를 자동 생성하고 품질 필터링하는 production pipeline”이다.

## Why A Clean Break Is Required

기존 path-centric 구조는 다음 상한을 가진다.

- 대부분 scalar / count / bool 수준의 answer shape
- question이 자연스러워져도 core contract는 단순 lookup
- conditional branching, uniqueness, threshold, multi-slot dependency를 표현하기 어려움
- reverse aggregate, multi-field record, sampling 개선을 해도 복잡도 상한 자체는 깨지지 않음

따라서 필요한 것은 incremental improvement가 아니라 paradigm 전환이다.

## Operating Principles

1. `hard to solve, easy to verify`
2. 품질 최우선. throughput보다 quality gate가 우선이다
3. verifier 신뢰도는 RLVR의 핵심 자산이며 절대 희생하지 않는다
4. arbitrary relational DB에 대응해야 하며, schema/domain 하드코딩을 금지한다
5. DB 간 task를 섞지 않는다. 한 environment는 정확히 한 DB에 속한다
6. DB는 언제든 registry에 추가될 수 있어야 한다
7. reward는 binary만 사용하고, diagnostics는 별도 채널로 기록한다

## System Goal

시스템의 장기 목표는 아래다.

- 사용자는 read-only PostgreSQL DB만 등록한다
- system이 schema를 탐색한다
- system이 domain/category를 추론한다
- synthesis agent가 environment 4-tuple을 작성한다
- self-consistency / shadow verifier / cross-instance / solver pass-rate quality gate를 통과한 environment만 registry에 커밋한다
- registry는 계속 커지고, scheduler는 DB별로 round-robin 또는 priority queue로 작업한다

## Non-Goals

- path-centric Tier A와 composition-centric Tier C를 병행 유지하는 것
- 현재 baseline dataset으로 production RL training을 시작하는 것
- fuzzy / semantic similarity verifier
- verifier correctness를 사람 수동 리뷰에 의존하는 운영
- DB 직접 write가 필요한 environment generation

## Clean Break Policy

이 rewrite는 clean break다.

- 기존 path-centric baseline은 freeze한다
- baseline review pack은 archive로 보존한다
- production run은 새 hybrid pipeline이 proof되기 전까지 금지한다
- 기존 task generation layer는 더 이상 점진 보강하지 않는다

## Current Assets To Preserve

다음 자산은 최대한 유지한다.

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
  - circuit breaker
  - checkpoint / budget / export skeleton
- `schema/introspect.py`, `schema/graph.py`, `schema/path_catalog.py`
- `verification/shadow.py`의 independent-verifier 개념
- `infra/json_chat_client.py`
- `cli.py`, `pipeline/review_pack.py`의 shell 구조

## Rewrite Targets

다음 계층은 전면 재설계 대상이다.

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

### Top-Level Structure

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
   - domain / category inference

2. `Synthesis Layer`
   - tool synthesis
   - task synthesis
   - solution synthesis
   - verifier synthesis
   - difficulty crank

3. `Registration and Runtime Safety Layer`
   - AST policy
   - function contract validation
   - runtime timeout / resource limits
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

이 네 개를 함께 생성하고 함께 품질 평가한다.

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
difficulty_level: str
created_at: str
generator_version: str
tool_hash: str
task_signature: str
status: draft | accepted | rejected | archived
quality_metrics:
  self_consistency_pass: bool
  shadow_disagreement_rate: float
  solver_pass_rate: float
  solver_ci_low: float
  solver_ci_high: float
```

## Task Complexity Target

이 시스템이 지향하는 task는 다음 성질을 가져야 한다.

- 여러 entity slot
- cross-slot uniqueness
- locality / compatibility constraints
- threshold constraints
- conditional branching
- multi-day / multi-step composition
- multiple valid solutions 가능

예시:

- 3일 itinerary
- 호텔 가격 버킷에 따라 다른 budget constraints
- 도시 / 호텔 / 식당 / 관광지 중복 금지
- 각 day의 entity가 같은 city context를 공유

즉 complexity는 더 이상 hop 수가 아니라 `constraint structure`다.

## General Agent Methodology, With RLVR Corrections

이 설계는 general agent식 synthesis 방법론의 표현력은 가져오되, verifier 신뢰도 약화는 hybrid rules로 막는다.

### Baseline General-Agent Loop

1. schema / sample exploration
2. category proposal
3. tools synthesis
4. task + solution + verifier synthesis
5. solution 실행
6. verifier 통과 여부 확인
7. 실패 시 iterate
8. 통과 시 difficulty crank

### Why Naive General-Agent Is Not Enough

그대로 쓰면 아래 문제가 생긴다.

- solution과 verifier가 같은 세션에서 공모할 수 있다
- Python verifier는 non-determinism risk가 있다
- verifier가 tool bug를 그대로 따라갈 수 있다
- multiple valid solutions handling이 애매해진다
- verifier correctness를 대규모로 증명하기 어렵다

따라서 production에서는 아래 hybrid rules를 모두 강제한다.

## Hybrid Verifier Rules

### Hybrid A — Fact Check Must Use Tools

verifier는 answer의 factual claim을 검사할 때 반드시 tool call을 통해 DB 사실을 다시 조회해야 한다.

허용:

- `hotel_name -> tools.get_infos_by_hotel(["price"], hotel_name)`
- `restaurant_name -> tools.get_city_by_restaurant(...)`

금지:

- answer literal을 하드코딩된 문자열/숫자와만 비교
- tool 호출 없이 pure Python if/else만으로 fact correctness 판정

등록 시점 정적 규칙:

- verifier 본문에 `tools.` 호출이 최소 1회 이상 있어야 한다
- direct literal comparison만 있고 tool fact check가 없으면 거부한다

### Hybrid B — Fact Check and Constraint Check Must Be Separated

verifier는 두 단계 구조를 강제한다.

```python
def verify(answer, tools):
    facts = fetch_facts(answer, tools)
    if not facts_match_answer_claims(answer, facts):
        return False
    if not check_constraints(answer, facts):
        return False
    return True
```

- Stage 1: fact check
  - tool-grounded
  - DB state와 answer claim의 일치성
- Stage 2: constraint check
  - pure Python allowed
  - uniqueness, thresholds, conditionals, budgets, ordering

이 구조로 Python 표현력은 유지하면서 factual correctness는 DB-grounded로 보존한다.

### Hybrid C — Shadow Verifier Is Mandatory

각 environment는 primary verifier와 independent shadow verifier를 가져야 한다.

요구:

- 다른 synthesis session에서 생성
- 다른 prompt / temperature
- 가능하면 다른 model family

평가:

- primary pass + shadow pass -> trusted
- primary fail + shadow fail -> trusted
- disagreement -> quality risk

`shadow_disagreement_rate > threshold`면 environment를 reject한다.

### Hybrid D — Cross-Instance Verification Is Mandatory

같은 environment 구조로 서로 다른 instance를 여러 개 생성한다.

요구:

- instance별 anchor / threshold / context가 달라야 한다
- valid solution도 instance별로 달라져야 한다
- verifier가 한 정답만 하드코딩 통과시키는 환경은 여기서 잡아낸다

production rule:

- 최소 `N`개 instance 생성
- 모든 instance에서 verifier consistency가 유지되어야 한다

## Code Registration Policy

container sandbox 대신, 등록 단계와 runtime 단계의 강한 policy로 generated code를 통제한다.

### Static Policy

AST 검사에서 아래를 강제한다.

- import allowlist
- 금지 builtins / attributes
- 금지 syntax
- 함수 signature contract

#### Import Allowlist

허용 예:

- `datetime`
- `math`
- `json`
- `re`
- `decimal`
- `typing`
- `collections.abc`
- registry가 제공하는 tool module

금지 예:

- `os`
- `sys`
- `subprocess`
- `socket`
- `pathlib` file write
- DB 직접 client import

#### Forbidden Operations

- `open`
- `eval`
- `exec`
- `compile`
- `__import__`
- process spawn
- network access
- filesystem write
- raw DB connection creation

### Function Contracts

tool / solution / verifier는 signature가 고정된다.

```python
async def tool_name(conn, **kwargs) -> Any
def solve(tools) -> dict
def verify(answer, tools) -> bool
```

generated code는 DB에 직접 연결하지 못하고 runtime이 주입한 tool / connection만 사용한다.

### Runtime Policy

- per-call timeout
- total function timeout
- memory limit
- function call count limit
- readonly DB role only
- statement timeout / idle tx timeout
- exception capture and trace logging

## Synthesis Agent Loop

### Stage 1: Schema and Domain Discovery

- schema introspection
- sample row inspection
- entity cluster inference
- category proposal

### Stage 2: 4-Tuple Synthesis

agent는 아래를 함께 만든다.

- `tools.py`
- `task.json`
- `solution.py`
- `verifier.py`

### Stage 3: Self-Consistency Iterate

- solution 실행
- verifier 통과?
- 실패하면 solution / verifier / tools를 수정
- trivial verifier / overfit verifier는 registration policy와 shadow/cross-instance gate가 차단

### Stage 4: Difficulty Escalation

기본 environment가 통과하면 constraint를 추가해 난이도를 올린다.

difficulty crank 예시:

- slot count 증가
- uniqueness scope 추가
- threshold 추가
- conditional branch 추가
- temporal dependency 추가
- budget dependency 추가

### Stage 5: Shadow and Cross-Instance

- independent shadow verifier 생성
- N instances 생성
- instance별 consistency 측정

### Stage 6: Solver Quality Filter

- 여러 solver attempts 실행
- pass-rate band 판정
- CI 기반 early stop

## Quality Gate

Environment는 아래 다섯 단계를 모두 통과해야 accepted 된다.

### Gate 1: Code Registration Policy

- AST 검사
- import/syntax/signature enforcement

### Gate 2: Self-Consistency

- synthesized solution이 synthesized verifier를 통과

### Gate 3: Shadow Verifier Agreement

- primary vs shadow disagreement rate 측정

### Gate 4: Cross-Instance Consistency

- 여러 instance에서 verifier가 stable하게 동작

### Gate 5: Solver Pass-Rate Band

- N attempts
- Clopper-Pearson CI
- safe early termination

quality filter config 예시:

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

## Domain Scheduler

### Registry Model

시스템은 여러 DB를 관리하지만, task는 DB 간 섞지 않는다.

```text
DB Registry
  - db_1
  - db_2
  - db_n
```

### Scheduler Requirements

- DB 동적 추가/제거 지원
- round-robin 또는 priority queue
- 한 번에 한 DB unit으로 synthesis
- domain별 progress 추적
- 실패한 environment 재시도 / 폐기 정책

## Environment Registry

registry는 filesystem + sqlite index의 이중 구조를 가진다.

### Filesystem

- environment directory 자체가 source of truth artifact

### SQLite Index

최소 아래를 기록한다.

- env_id
- db_id
- domain
- category
- difficulty
- pass_rate
- CI interval
- shadow disagreement
- created_at
- status
- tool_hash
- task_signature
- generator_version

### Dedup and Coverage

- dedup은 `tool set hash + task signature + verifier signature` 기준
- coverage는 domain/category/difficulty 분포를 추적

## Solver Runtime Position

OpenAI Agents SDK 기반 solver runtime은 계속 solver substrate로 유지한다.

solver가 맡는 역할:

- presented tools 사용
- reasoning
- explicit `submit_result()`
- transcript / tool trace

solver가 맡지 않는 역할:

- verifier correctness
- environment synthesis correctness
- quality gate

## Review Pack

review pack은 계속 1급 artifact다.

포함해야 할 것:

- final question
- tool set
- output schema
- constraint summary
- verifier summary
- shadow verifier status
- instance summary
- canonical solution reference

정성 평가 기준:

- 실제 user 요청처럼 보이는가
- compositional reasoning이 필요한가
- tool surface가 task와 coherent한가
- verifier summary가 이해 가능하고 설득력 있는가

## Success Criteria

성공은 아래로 판단한다.

- generated task가 더 이상 lookup task처럼 보이지 않는다
- solver가 실제 composition reasoning을 해야 한다
- verifier는 deterministic binary reward를 준다
- arbitrary DB에 대해 environment를 생성할 수 있다
- review pack 정성 평가에서 예시 수준 이상의 quality가 반복적으로 나온다

## Freeze Policy

이 spec 기준으로:

- 기존 task generation code는 incremental polishing을 중단한다
- rewrite는 `spec -> plan -> core contracts -> proof environment -> generalization` 순으로 간다
- proof environment가 나오기 전 production run은 금지한다

