# Plan 1: OpenAI Agents SDK Foundation — Implementation Plan

## Objective

이 계획의 목적은 [design spec](./spec.md) 에 정의된 RL task foundry를 실제 구현 가능한 workstream으로 쪼개는 것이다.

이번 계획은 다음 원칙을 따른다.

- solver substrate는 `OpenAI Agents SDK`
- correctness-critical logic은 전부 우리 도메인 레이어에서 구현
- solver와 RL rollout agent는 같은 step contract를 사용
- 운영 상태의 source of truth는 `run.db`
- answer contract는 `answer_schema`
- adaptive calibration, tool level, outcome diversity, shadow verifier를 초기 설계부터 포함

## Deliverable

Plan 1이 끝나면 아래가 가능해야 한다.

1. PostgreSQL schema를 읽어 path catalog와 visibility policy를 생성한다
2. canonical `L1` read-only DB tools를 생성하고, task-aware `L2` presented bundle을 만든다
3. anchor row에서 task package를 compose한다
4. `answer | no_result | clarify | deny` outcome을 포함한 deterministic ground truth를 생성한다
5. OpenAI Agents SDK 기반 multi-provider solver replica를 고정 quota로 병렬 실행한다
6. solver output을 exact / provenance / shadow verifier 기준으로 판정한다
7. adaptive difficulty loop로 target pass-rate band에 수렴시킨다
8. accepted dataset을 JSONL로 export한다
9. run state를 `run.db`에 기록하고 재시작 시 resume한다

## Non-Deliverable

이번 계획은 아래를 포함하지 않는다.

- production-grade hosted control plane
- multi-tenant web app
- 완성형 analytics dashboard
- 자동 prompt optimization
- RL training loop 자체

단, 이후 확장이 가능하도록 인터페이스와 상태 모델은 미리 준비한다.

## Design Constraints

### Constraint 1: SDK Boundary

OpenAI Agents SDK는 아래만 담당한다.

- tool calling loop
- session convenience
- tracing
- optional handoff
- tool schema transport

그 외의 모든 정답성 책임은 우리 코드가 가진다.

### Constraint 2: Rollout Compatibility

solver runtime은 future RL rollout agent와 같은 구조를 가져야 한다.

canonical step:

```text
Observation -> Policy Call -> Action -> Tool Result -> State Update
```

memory와 summary는 hidden framework behavior가 아니라 explicit state event여야 한다.

### Constraint 3: Deterministic Judge

아래는 모델에 맡기지 않는다.

- answer schema
- outcome type semantics
- ground truth SQL
- canonicalization rules
- verification
- calibration decision
- provenance policy

### Constraint 4: Durability

운영 source of truth는 SQLite `run.db`다. JSONL은 export와 event mirror일 뿐이다.

### Constraint 5: Read-Only Safety

solver 도구는 자유 SQL이 아니라 compiled read-only query only다.

### Constraint 6: Diversity

calibration label은 최소 두 개 이상의 provider family에서 측정 가능해야 한다. 단일 모델 특이성이 difficulty label에 bake되면 안 된다.

## Execution Strategy

구현 순서는 `judge-first`로 간다.

이유:

- solver runtime보다 먼저 answer contract와 verifier를 고정해야 dataset correctness가 흔들리지 않는다
- adaptive calibration도 verifier와 score contract 위에 올라간다
- SDK 선택은 교체 가능하지만 answer schema와 run state는 교체 비용이 크다

## Milestones

### Milestone A: Deterministic Backbone

- config
- run.db
- DB access
- schema explorer
- path catalog
- answer schema
- outcome contract
- ground truth

### Milestone B: Judge and Tool Surface

- tool compiler with `L1/L2`
- task composer
- validator
- provenance rules
- verification engine
- `AnswerField.visibility` 기반 internal-field leak 검출
- shadow verifier

### Milestone C: Solver Runtime and Swarm

- OpenAI Agents SDK solver runtime
- provider adapters
- fixed quota orchestrator
- provider resilience

### Milestone D: Adaptive Production Loop

- calibration loop
- checkpoint / resume
- export
- coverage / dedup
- rollout-ready runtime hooks

## Target Project Structure

```text
src/rl_task_foundry/
  config/
    models.py
    load.py

  infra/
    db.py
    storage.py
    checkpoint.py
    events.py
    budget.py
    privacy.py

  schema/
    introspect.py
    sensitivity.py
    graph.py
    path_catalog.py

  tools/
    models.py
    compiler.py
    sql_templates.py
    openai_agents_adapter.py
    model_naming.py
    naming_eval.py

  tasks/
    models.py
    composer.py
    validator.py
    provenance.py
    dedup.py

  truth/
    schemas.py
    generator.py
    canonicalize.py

  solver/
    runtime.py
    backend_openai_agents.py
    state.py
    prompts.py
    memory.py

  verification/
    compare.py
    scorer.py
    policies.py
    shadow.py

  calibration/
    banding.py
    early_stop.py
    runner.py

  pipeline/
    orchestrator.py
    export.py
    manifest.py
    coverage.py

  cli.py
```

## Task 1: Reset Package Boundary and Dependencies

### Goal

프로젝트를 새 architecture 기준으로 다시 정렬하고 dependency boundary를 고정한다.

### Work

- `pyproject.toml` 정리
- OpenAI Agents SDK dependency 추가
- legacy harness 의존 제거
- top-level package layout 재정리
- artifacts 디렉터리 구조 정의

### Acceptance Criteria

- import graph에 legacy harness import가 더 이상 없다
- CLI가 새 config path를 읽을 수 있다

## Task 2: Config System and Run Manifest Contract

### Goal

spec에 나온 모든 핵심 설정을 타입으로 표현한다.

### Work

- `database`, `providers`, `models.solvers`, `provider_resilience`, `tool_compiler`, `task_composer`, `solver_runtime`, `calibration`, `verification`, `budget`, `privacy`, `output` 모델 정의
- env expansion 지원
- config hash 계산
- database fingerprint 계산 contract 정의

### Acceptance Criteria

- `models.solvers[*].replicas` 합계가 solver 수 source of truth로 계산된다
- budget section이 compose/solve/gpu를 분리 표현한다
- `negative_outcome_ratio`, `shadow_sample_rate`, `ci_alpha`, `selected_tool_level`이 타입으로 강제된다

## Task 3: Durable Run State with run.db

### Goal

SQLite 기반 durable run state를 만든다.

### Work

- `run.db` schema 설계
- migration bootstrap
- event log table
- accepted/rejected/task/solver_run/verification/budget tables
- processed anchor / dedup / coverage tables
- atomic checkpoint write

### Files

- `infra/storage.py`
- `infra/checkpoint.py`

### Required Tables

- `runs`
- `anchors`
- `tasks`
- `ground_truth`
- `solver_runs`
- `verification_results`
- `accepted_examples`
- `rejection_reasons`
- `budget_ledger`
- `coverage_counters`
- `dedup_signatures`
- `event_log`

### Acceptance Criteria

- crash 후 restart 시 processed anchors와 accepted tasks를 복구할 수 있다
- accepted example export 여부를 DB에서 재구성할 수 있다

## Task 4: Event Bus and Budget Ledger

### Goal

orchestrator가 관찰 가능한 이벤트를 내고, budget을 phase-specific으로 추적한다.

### Work

- typed event definitions
- in-process async event bus with drop-on-lag
- event persistence adapter
- compose/solve separated reservation
- reservation id + settle API
- accept-rate abort guard

### Files

- `infra/events.py`
- `infra/budget.py`

### Acceptance Criteria

- `budget.reserve(compose_api_usd=..., solve_api_usd=..., gpu_hours=...)` 형태를 지원한다
- accepted/rejected까지의 전체 경로에 이벤트가 빠짐없이 기록된다

## Task 5: Read-Only DB Infrastructure

### Goal

solver lane과 control lane이 분리된 안전한 DB 접근을 구현한다.

### Work

- 2-lane pool
- readonly role 지원
- transaction-level read-only enforcement
- statement timeout / lock timeout / idle tx timeout
- fixed search_path

### Files

- `infra/db.py`

### Acceptance Criteria

- solver lane은 `SET default_transaction_read_only = on`으로 시작한다
- write SQL이나 side-effect SQL이 tool path에서 실행되지 않는다

## Task 6: Schema Explorer and Sensitivity Policy

### Goal

task generation과 privacy policy의 기반이 되는 schema metadata를 만든다.

### Work

- table/column/PK/FK introspection
- fanout / uniqueness / nullability 추정
- visibility policy classification
- overrides merge

### Files

- `schema/introspect.py`
- `schema/graph.py`
- `schema/sensitivity.py`
- `infra/privacy.py`

### Acceptance Criteria

- 컬럼이 `blocked / internal / user_visible` 중 하나로 분류된다
- path catalog 입력으로 필요한 graph metadata가 전부 생성된다

## Task 7: Path Catalog and Difficulty Features

### Goal

question family가 사용할 수 있는 admissible path set을 계산한다.

### Work

- reachable FK path enumeration
- path id canonicalization
- shortcut detection
- difficulty feature extraction

### Files

- `schema/path_catalog.py`

### Output

- `PathSpec`
- `PathCatalog`
- `DifficultyFeatures`

### Acceptance Criteria

- 각 path가 hop count, shortcut candidates, fanout statistics를 가진다
- task composer는 free-form join reasoning 없이 path catalog만 참조한다

## Task 8: Core Data Models and Answer / Outcome Contract

### Goal

이 프로젝트의 중심 계약을 고정한다.

### Work

- `AnswerSchema`
- `TaskSpec`
- `GroundTruth`
- `SolverResult`
- `VerifyResult`
- `AcceptedExample`
- `RolloutRecord`

### Files

- `truth/schemas.py`
- `tasks/models.py`

### Acceptance Criteria

- solver output, canonical answer, verifier가 같은 schema object를 참조한다
- `outcome_type`이 task, ground truth, verifier에 공통으로 존재한다
- answer schema 없는 task는 생성 불가다

## Task 9: Ground Truth Generator and Canonicalizer

### Goal

deterministic judge를 먼저 완성한다.

### Work

- verification SQL compiler
- typed canonicalization
- supported field types implementation
- list ordering rules
- datetime/date normalization
- outcome branch generation (`answer`, `no_result`, `clarify`, `deny`)

### Files

- `truth/generator.py`
- `truth/canonicalize.py`

### Acceptance Criteria

- 같은 task는 항상 같은 canonical answer를 생성한다
- verifier가 ground truth canonicalizer를 그대로 재사용한다

## Task 10: Tool Compiler and OpenAI Agents Tool Adapter

### Goal

schema/path metadata를 solver가 사용할 canonical DB tools로 바꾼다.

### Work

- `ToolSpec` model
- SQL template compilation
- read-only query adapter
- OpenAI Agents SDK tool registration adapter
- tool result schema enforcement
- canonical `L1` semantic bundle compilation
- `list_related_*` compiled with deterministic ordering and `max_list_cardinality`
- `label_tier="A"`에서는 `lookup/list_related/count/exists`만 생성
- `aggregate_*` and `timeline_*` are gated by both `task_composer.label_tier == "B"` and `tool_compiler.allow_aggregates` / `allow_timelines`
- `aggregate_*` excludes identifier-like numeric columns (`id`, `*_id`, PK/FK)
- `aggregate_*` must add `WHERE aggregate_column IS NOT NULL`
- `aggregate_*` for numeric-like `sum` / `avg` must emit `ROUND(..., verification.float_precision)` to stabilize scale
- `L1` naming is rule-based canonical naming
- preview/dev fallback may still compile `L2` aliases for diagnostics, but production difficulty labels must not rely on compiler-only naming
- current rule-based business-alias layer is bootstrapping fallback only and must be marked as such in metadata until model-generated naming is wired in
- naming evaluator tracks raw identifier overlap, duplicate/invalid names, and level-policy violations for generated bundles
- `L2` acceptance gate:
  - less literal than `L1`
  - not a direct path-chain restatement
  - still retains some discoverable cue

### Files

- `tools/models.py`
- `tools/compiler.py`
- `tools/sql_templates.py`
- `tools/openai_agents_adapter.py`

### Acceptance Criteria

- solver가 보는 tool set은 compiled DB tools only다
- blocked column이 tool output에 나타나지 않는다
- tool output schema가 stable하다
- canonical bundle이 semantic source of truth가 된다
- level 간 semantic equivalence가 유지된다
- evaluator가 `L2` naming quality를 수치와 violation 목록으로 출력한다

## Task 11: Task Composer, Validator, Provenance Rules

### Goal

task를 compose하되, solver 실행 전에 quality gate를 통과시킨다.

### Work

- question family template system
- anchor row materialization
- outcome type selection
- task-aware presented tool bundle generation
- lexical leak detector
- ambiguity detector
- shortcut detector
- provenance requirements model

### Files

- `tasks/composer.py`
- `tasks/validator.py`
- `tasks/provenance.py`

### Acceptance Criteria

- compose failure는 solve budget을 소비하지 않는다
- `TaskPackage`가 `TaskSpec + PresentedToolBundle`를 함께 가진다
- `TaskPackage`는 stable `L1` base option과 optional `L2` option을 함께 가질 수 있다
- 실제 solver 실행에 사용하는 tool level은 `task_composer.selected_tool_level`이 run 시작 시 고정한다
- `answer_schema`, `selected_path`, `outcome_type`, `tool_level`, provenance requirements가 task에 포함된다
- current task factory implementation is contract-first: it drafts `answer_schema/outcome_type/path` first, and production run에서는 composer model이 path context + sanitized row context를 받아 final question을 다시 합성한다
- composer는 final question + presented tool bundle을 대상으로 task package judge agent를 돌려 semantic coherence / answer leak / schema exposure / tool answerability를 점검한다
- `L2` presented bundle은 question context를 함께 받은 model-generated naming/layout을 사용한다
- model-generated `L2`가 quality gate를 통과하지 못하면 fallback alias로 downgrade하고 metadata에 이유를 남긴다
- invalid task는 solver 실행 전에 reject된다
- `generate-task-specs` CLI로 Tier A task spec JSONL을 만들 수 있다
- current automatic factory는 source round-robin으로 path/family coverage를 먼저 넓히고, 현재 slice에서 `status_lookup`, `causal_chain`, `aggregate_verification`를 우선 생성한다
- `generate-review-pack` CLI는 factory -> ground truth -> composer를 실제로 태워 review JSONL과 Markdown을 생성한다
- dataset 품질 평가는 unit test만으로 끝내지 않고, generated review pack을 사람 눈으로 샘플링 검토하는 절차를 필수로 둔다

## Task 12: OpenAI Agents SDK Solver Runtime

### Goal

single solver execution을 담당하는 canonical runtime을 만든다.

### Work

- `AgentRuntime` 인터페이스
- OpenAI Agents SDK backend
- provider-specific model adapter boundary
- session integration
- tracing integration
- explicit `submit_result()` terminal tool
- explicit memory event recording
- turn-count capture and solver termination metadata

### Files

- `solver/runtime.py`
- `solver/backend_openai_agents.py`
- `solver/state.py`
- `solver/prompts.py`
- `solver/memory.py`

### Acceptance Criteria

- solver transcript를 재생해 step-by-step replay 가능하다
- 최종 답 제출이 `submit_result()` tool trace에 명시적으로 남는다
- `submit_result()` schema mismatch는 terminal fail로 기록되고, 일반 tool argument error와 구분된다
- solver result에 `turn_count`, `termination_reason`, `termination_metadata`가 남는다
- SDK session을 꺼도 canonical state가 유지된다
- future RL rollout agent가 같은 state object를 사용할 수 있다
- 최소 두 provider family를 수용 가능한 adapter boundary가 존재한다

## Task 13: Solver Swarm Orchestrator and Provider Resilience

### Goal

여러 solver replica를 rolling 방식으로 실행하는 batch engine을 만든다.

### Work

- anchor queue
- compose workers
- solver workers
- verify workers
- provider-aware semaphore
- fixed quota by solver/model family
- timeout / retry policy
- provider circuit breaker
- quota rebalance

### Files

- `pipeline/orchestrator.py`

### Acceptance Criteria

- batch barrier 없이 rolling orchestration이 동작한다
- provider max concurrency를 넘지 않는다
- solver 수는 `models.solvers[*].replicas` 합계로 계산된다
- degraded provider에서 healthy provider로 quota를 옮길 수 있다

### Current Slice

- `TaskSpec JSON/JSONL` 입력을 받아 headless run을 실행할 수 있다
- orchestrator는 task별로 `compose -> ground truth -> solver swarm -> verify -> persist`를 관통한다
- task 간 실행도 bounded in-flight set으로 rolling 처리한다
- provider-aware semaphore는 이미 solver replica 실행에 적용된다
- provider runtime error는 task 전체 abort가 아니라 replica-level `provider_error` / `provider_timeout` result로 흡수된다
- rolling error-rate window를 기준으로 provider circuit breaker를 열고, cooldown 중인 provider quota는 healthy provider replica로 단순 rebalance한다
- checkpoint는 `processed_keys`를 통해 config-hash namespace로 연결되어, 동일 입력 재실행 시 processed task를 skip할 수 있다
- budget reserve/settle도 single-run path에 연결되어, phase budget을 per-task worst-case share로 나눠 gate를 건다
- 세밀한 probe policy, adaptive backoff, rolling P90 기반 reservation은 아직 후속 단계다

## Task 14: Verification Engine and Shadow Verifier

### Goal

solver output을 deterministic하게 판정한다.

### Work

- canonicalization reuse
- outcome-aware exact comparison
- fieldwise diagnostics
- provenance enforcement
- failure taxonomy
- shadow verifier sampling

### Files

- `verification/compare.py`
- `verification/scorer.py`
- `verification/policies.py`
- `verification/shadow.py`

### Acceptance Criteria

- value correctness와 provenance fail을 구분해서 저장한다
- internal/blocked field leak는 explicit failure reason으로 기록된다
- 정식 reward / accept 기준은 `pass_exact` 하나뿐이다
- fieldwise diagnostics는 error analysis용으로만 기록된다
- sampled verification은 `shadow_verifier_status`, `shadow_pass_exact`, `shadow_failure_reason`으로 남는다
- shadow verifier disagreement rate를 계산할 수 있다

## Task 15: Adaptive Calibration Loop and Safe Early Termination

### Goal

accepted dataset 품질을 pass-rate band로 제어한다.

### Work

- lower/upper pass-rate band
- CI-based decision
- canary phase
- post-canary micro-batch expansion
- safe early termination
- full solve escalation
- executed/planned replica count metadata
- one-axis adaptive difficulty adjustment
- best-so-far fallback
- reject reason taxonomy

### Files

- `calibration/banding.py`
- `calibration/early_stop.py`
- `calibration/runner.py`

### Core Rule

아래 두 계층을 모두 구현한다.

1. `safe early termination`
   - upper/lower bound를 동시에 존중하는 종료 조건
2. `adaptive difficulty`
   - `selected_tool_level`은 run partition으로 고정하고, band 밖이면 `required_hops -> condition_complexity -> fanout_ambiguity` 순서로 한 축씩 조절
   - CI가 band와 겹치면 difficulty를 바꾸지 않고 추가 샘플링
   - `max_iterations_per_anchor` 도달 시 best-so-far fallback

### Acceptance Criteria

- upper bound를 무시하는 early accept가 없다
- canary stage와 full stage budget accounting이 분리된다
- post-canary replica 실행은 `post_canary_batch_size` 단위 batch 경계에서 decision을 다시 계산한다
- accepted/rejected payload에 `calibration.decision`, `executed_solver_replicas`, `planned_solver_replicas`를 남길 수 있다
- 같은 anchor에서 반복 조절이 가능하다

### Current Slice

- canary -> full solve escalation과 CI-based decision은 이미 orchestrator에 연결되어 있다
- `max_attempts_per_anchor` 범위에서 same-anchor 반복 시도가 가능하다
- 현재 path difficulty 조절은 `status_lookup`, `causal_chain`, `timeline_resolution` 중 compatible answer schema task에 한해 `required_hops`와 `fanout_ambiguity` 중심으로 적용된다
- question/answer contract를 안전하게 유지할 수 없는 task family의 full rewrite loop는 후속 단계다

## Task 16: Dedup, Coverage, Export

### Goal

accepted dataset이 중복과 편향 없이 export되도록 한다.

### Work

- exact dedup
- near-dup signatures
- coverage counters
- accepted/rejected JSONL export
- accepted example training metadata export
- manifest writer

### Files

- `tasks/dedup.py`
- `pipeline/coverage.py`
- `pipeline/export.py`
- `pipeline/manifest.py`

### Acceptance Criteria

- accepted JSONL에 `training_metadata.mean_correct_solver_turns_rounded`가 포함될 수 있다
- 이 값은 `pass_exact=True`인 solver들의 평균 turn 수를 반올림한 값이다
- accepted export와 review pack에는 `question_source` / `question_generation_metadata`가 포함되어 seed fallback과 model-generated question을 구분할 수 있다
- question synthesis는 domain-configured user/agent role을 사용하고, semantic coherence validator를 통과해야 한다

### Acceptance Criteria

- exact duplicate task는 재수락되지 않는다
- coverage summary가 manifest에 기록된다
- outcome type / tool level / provider diversity를 coverage에서 볼 수 있다

## Task 17: CLI and Dashboard Projection

### Goal

headless batch 실행과 상태 관찰이 가능해야 한다.

### Work

- config validate command
- run command
- resume command
- run summary command
- lightweight dashboard projection

### Files

- `cli.py`

### Acceptance Criteria

- `run.db`만으로 run summary를 다시 그릴 수 있다
- dashboard는 projection이며 source of truth가 아님이 유지된다

### Current Slice

- `run` command는 `TaskSpec JSON/JSONL` 파일을 받아 실행한다
- `run-summary` command는 `run.db`만 읽어 accepted/rejected/skipped/task/event 수를 재구성한다
- `generate-review-pack` command는 synthesized question + tool set + answer key를 `review_pack.jsonl` / `review_pack.md`로 저장한다
- `resume`과 dashboard projection은 아직 후속 단계다

## Task 18: Test Strategy

### Test Layers

1. unit tests
2. fixture schema tests
3. deterministic judge golden tests
4. solver runtime contract tests
5. orchestration integration tests
6. canary smoke tests

### Must-Have Test Files

- `tests/test_config.py`
- `tests/test_storage.py`
- `tests/test_budget.py`
- `tests/test_db.py`
- `tests/test_schema_introspect.py`
- `tests/test_path_catalog.py`
- `tests/test_truth_generator.py`
- `tests/test_tool_compiler.py`
- `tests/test_task_validator.py`
- `tests/test_solver_runtime_openai_agents.py`
- `tests/test_verification.py`
- `tests/test_shadow_verifier.py`
- `tests/test_calibration.py`
- `tests/test_pipeline_resume.py`

### Golden Tests

반드시 golden으로 고정할 항목:

- answer canonicalization
- outcome-type verification branches
- verification SQL output
- lexical leak rejection
- shortcut rejection
- safe early termination decisions
- adaptive difficulty decisions
- `L1/L2` tool equivalence

## Implementation Order

실제 개발 순서는 아래를 따른다.

1. Task 1
2. Task 2
3. Task 3
4. Task 5
5. Task 6
6. Task 7
7. Task 8
8. Task 9
9. Task 10
10. Task 11
11. Task 14
12. Task 15
13. Task 12
14. Task 13
15. Task 16
16. Task 4
17. Task 17
18. Task 18

이 순서를 택하는 이유는 아래와 같다.

- judge correctness를 solver보다 먼저 고정한다
- adaptive calibration은 verifier contract 위에 올라간다
- orchestrator는 solver/verifier/calibration contract가 정해진 뒤 붙인다
- event bus와 dashboard는 core correctness보다 나중에 붙여도 된다

## Production Gate

아래를 통과해야 첫 production-scale run을 연다.

1. fixture schema 기준 deterministic judge tests green
2. tool compiler safety tests green
3. `L1/L2` semantic equivalence tests green
4. solver runtime explicit `submit_result()` tests green
5. calibration simulation tests green
6. shadow verifier disagreement rate가 threshold 이하
7. checkpoint / resume integration green
8. canary dry run에서 accepted/rejected decision이 수작업 검토와 일치

## Deferred Extensions

초기 구현엔 얇게 두되, 인터페이스는 열어 둔다.

- explicit summarizer agent
- handoff-based sub-solver
- richer reward attachment for RL rollout
- Tier B default enable
- offline analytics notebooks
- distributed worker executor

## Self-Review Checklist

- OpenAI Agents SDK가 solver substrate 역할만 하는가
- run.db가 source of truth로 유지되는가
- answer_schema가 모든 label path의 기준인가
- outcome diversity가 task/ground truth/verifier에 일관되게 반영되는가
- solver 수 source of truth가 하나뿐인가
- privacy policy가 visibility 기반으로 일관되는가
- safe early termination이 upper/lower bound를 모두 존중하는가
- adaptive difficulty가 한 번에 한 축만 조절하는가
- tool level (`L1/L2`)이 실제 난이도 축으로 동작하는가
- shadow verifier disagreement를 관측할 수 있는가
- solver runtime과 RL rollout runtime의 상태 모델이 같은가
- invalid task가 solver budget을 소비하지 않는가
- batch barrier가 남아 있지 않은가
