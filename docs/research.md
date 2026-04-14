# RL Task Foundry Deep Research

작성일: `2026-04-14`

분석 기준:

- branch: `quick-harbor`
- commit: `112aed7`
- 실행 기준 환경: `uv`, Python `3.14`

## 1. Executive Summary

이 프로젝트는 read-only PostgreSQL 데이터베이스를 RL task bundle로 바꾸는 합성 파이프라인이다. 핵심 아이디어는 "DB별 atomic tool bundle"을 먼저 만들고, 그 위에서 synthesis agent가 실제 DB 증거를 탐색해 label-first 방식으로 task를 합성한 뒤, solver pass-rate quality gate를 통과한 task만 durable registry에 적재하는 것이다.

현재 authoritative surface는 legacy path-centric generator가 아니라 `synthesis/`, `solver/`, `pipeline/solver_orchestrator.py`, `task_registry`, `bundle_exporter`, 그리고 이를 감싸는 CLI다. reward는 fuzzy verifier가 아니라 exact-match canonicalization 기반 binary reward로 단순화되어 있다.

가장 중요한 시스템 성질은 다음 네 가지다.

- task의 기본 단위는 "문제 하나"가 아니라 `db_id`에 종속된 `TaskBundleContract`다.
- 같은 `db_id`에 속한 모든 task는 동일한 database-level atomic tool set을 공유한다.
- synthesis 결과물은 disposable이고, quality gate와 registry가 authoritative하다.
- canonical answer가 source of truth이며, output schema는 synthesis agent가 아니라 runtime이 canonical answer에서 추론한다.

## 2. 현재 상태 평가

문서와 코드 기준으로 보면 이 저장소는 Plan 4 rewrite의 후반 단계에 있다.

- `docs/plan.md`는 path-centric baseline에서 synthesis-agent hybrid architecture로의 완전 전환을 선언한다.
- `docs/spec/*.md`는 현재 동작과 꽤 잘 맞물리는 규범 문서다.
- `docs/runbook.md`와 `docs/phase0_baseline.md`는 rewrite freeze와 artifact inspection 관점의 운영 문서다.
- 코드상으로는 proof task, real-db trial, registry runner, exporter, solver quality gate, semantic dedup까지 이미 구현되어 있다.

한 줄로 정리하면, 이 프로젝트는 "연구용 skeleton" 단계를 넘어 "end-to-end vertical slice가 실제로 닫히는 rewrite branch"에 가깝다. 다만 운영 확장성, 문서 정합성, 일부 미검증 edge path는 아직 남아 있다.

## 3. 저장소 구조

### 최상위

- `pyproject.toml`
  - 프로젝트 메타데이터, Python `>=3.14,<3.15`, 의존성, CLI 엔트리포인트 정의
- `rl_task_foundry.yaml`
  - 실제 runtime 기본 설정
- `docs/`
  - spec, plan, runbook, baseline, ADR
- `src/rl_task_foundry/`
  - 실제 시스템 코드
- `tests/`
  - 회귀 및 계약 테스트

### 패키지 맵

| 패키지 | 역할 |
| --- | --- |
| `config/` | YAML 로드, env expansion, strict Pydantic config |
| `infra/` | DB pool, run DB, checkpoint, budget, event bus, SDK helpers |
| `schema/` | PostgreSQL introspection, schema graph, path catalog, column sensitivity |
| `synthesis/` | atomic tool generation, prompting, synthesis runtime, draft validation, registry, exporter |
| `solver/` | solver runtime protocol, OpenAI Agents SDK backend |
| `pipeline/` | solver rollout orchestration, provider circuit breaker |
| `calibration/` | pass-rate banding, confidence interval, early stop |
| `cli.py` | 운영 명령 진입점 |

## 4. 시스템 아키텍처

문서와 코드가 합의하는 high-level flow는 아래와 같다.

```text
DB registry
  -> scheduler chooses (db_id, topic)
  -> single-db synthesis runtime introspects schema
  -> atomic tools are generated and materialized
  -> synthesis agent explores tools and submits grounded draft
  -> submit_draft internally triggers solver rollout
  -> quality gate accepts/rejects by pass-rate band
  -> accepted draft commits into filesystem + SQLite registry
  -> exporter builds serving bundle
```

여기서 중요한 점은 quality gate가 synthesis loop "밖"에 있는 별도 후기 단계가 아니라, `submit_draft` tool validation 과정 안으로 들어가 있다는 것이다. synthesis agent는 draft를 제출하면 즉시 solver 결과를 feedback으로 받고 같은 대화 컨텍스트에서 재시도한다.

## 5. 설정 계층

`src/rl_task_foundry/config/models.py`는 매우 strict한 계약 계층이다.

- 모든 config model은 `extra="forbid"`다.
- database lane은 solver/control pool을 분리한다.
- provider surface는 `openai_compatible`, `anthropic`, `openai`, `google`를 허용하지만, 현재 실제 backend 구현은 사실상 `openai`/`openai_compatible`만 지원한다.
- solver는 기본 설정상 6개다.
  - `gpt54m_a`~`gpt54m_d`
  - `gpt54m_memory_a`
  - `gpt54m_memory_b`
- calibration default band는 `0.25 <= pass_rate <= 0.75`
- coverage planner default는 band별 3개 task
- atomic tool cap은 기본 `300`

`config/load.py`는 `${ENV_VAR:-default}` 형식의 문자열 expansion을 지원하고, composer/solver provider/model override를 runtime에서 덮어쓸 수 있다.

운영적으로 중요한 사실:

- bare Python이 아니라 `uv` + Python `3.14` 전제가 강하다.
- 실제 shell에서 `pytest -q`는 dev dependency 부재로 바로 실패했고, `uv run pytest -q`는 정상 동작했다.

## 6. Infra 계층

### `infra/db.py`

두 개의 DB lane이 있다.

- control plane
  - timeout만 설정
- solver plane
  - `default_transaction_read_only = on`
  - optional `SET ROLE {readonly_role}`
  - 동일한 timeout 정책

즉, schema introspection/control 작업과 solver/tool 실행이 분리돼 있다.

### `infra/storage.py`

run-level durable state는 SQLite `run.db`에 저장된다.

- `runs`
- `tasks`
- `accepted_examples`
- `verification_results`
- `budget_ledger`
- `coverage_counters`
- `event_log`
- `processed_keys`
- `budget_reservations`

이 DB는 "accepted task registry"와는 다르다. `run.db`는 실행 요약/체크포인트/ledger 성격이고, registry는 `tasks/` 디렉터리 + 별도 `task_registry.db` 조합이다.

### `infra/checkpoint.py`

checkpoint는 `processed_keys` 테이블 위에 thin wrapper를 얹은 구조다.

- 메모리 set으로 O(1) membership check
- flush 시 SQLite에 durable 기록
- `SynthesisRegistryRunner`가 `(db_id, topic)` pair resume에 사용

### 기타

- `infra/privacy.py`
  - 컬럼명 패턴으로 `blocked/internal/user_visible` visibility 추정
- `infra/sdk_helpers.py`
  - OpenAI Agents SDK 공통 어댑터 역할
  - tool result normalization
  - strict tool schema wrapping
  - artifact write helper
- `infra/events.py`
  - bounded queue + drop-on-lag event bus
- `infra/budget.py`
  - phase-specific reserve/settle ledger

## 7. Schema 계층

### Introspection

`schema/introspect.py`는 PostgreSQL 메타데이터를 직접 읽는다.

- `pg_class`, `information_schema.columns`, `pg_index`, `pg_constraint`, `pg_stats`
- row estimate, PK, unique, FK, `n_distinct`를 수집
- visibility는 `schema/sensitivity.py`를 통해 컬럼별로 부착

결과는 `SchemaGraph`로 투영된다.

- `TableProfile`
- `ColumnProfile`
- `ForeignKeyEdge`

여기서 `fanout_estimate`를 FK source/target row estimate 비율로 계산하는데, 이후 difficulty 추정과 경로 이해에 활용된다.

### Path Catalog

`schema/path_catalog.py`는 simple FK path를 열거한다.

- `max_hops` 범위 내 simple path 탐색
- shortcut candidate 계산
- difficulty proxy
  - required_hops
  - fanout_max
  - fanout_product
  - cardinality_estimate
  - has_unique_join
  - has_nullable_hop
  - shortcut_count

현재 authoritative synthesis runtime이 path catalog를 직접 critical path에서 쓰는 것은 아니지만, schema reasoning과 difficulty research를 위한 보조 도구로 남아 있다.

## 8. Atomic Tool Architecture

### 설계 개념

문서상 atomic tool family는 `T1~T8`로 설명되지만, 코드 구현은 더 응축된 네 개의 runtime family로 정리된다.

- `GET`
- `FIND`
- `CALC`
- `RANK`

문서의 `T1~T8`은 사실상 이 네 family의 parameterized subtype으로 흡수됐다.

- point lookup -> `GET`
- bounded enumeration / single-column filter -> `FIND`
- filtered aggregate -> `CALC`
- sorted/grouped top-k -> `RANK`

### 생성 로직

`synthesis/atomic_tools.py`의 `AtomicToolGenerator`는 schema graph에서 deterministic하게 tool bundle을 만든다.

- `get_<table>`
- `find_<table>_by_<column>`
- `calc_<table>`
- `rank_<table>_by_<column>`

선택 기준:

- PK 없는 테이블은 `GET` 미생성
- FK 컬럼은 적극적으로 `FIND`/`RANK` 후보로 채택
- `n_distinct`와 row estimate를 이용해 text column usefulness 판단
- numeric/date 계열은 filter/aggregate/rank에서 특별 취급

### 출력 형태

각 tool은 다음 계약을 가진다.

- actor-facing `name`, `description`, `params_schema`, `returns_schema`
- display/audit용 SQL template
- runtime metadata

중요한 구현 세부:

- multi-row surface는 seeded deterministic ordering 사용
- `find`와 `rank`만 `_shuffle_seed`를 받아 ordering tie-break에 사용
- `calc`와 `get`은 shuffle 무관
- generated source는 실제 async Python module로 materialize된다

### Materialization

`AtomicToolMaterializer`는 DB별로 아래 파일을 남긴다.

```text
artifacts/databases/<db_id>/
  atomic_tools.py
  atomic_tool_definitions.json
```

이 번들은 registry commit과 exporter에서 재사용된다.

## 9. Core Contracts

`synthesis/contracts.py`는 현재 시스템에서 가장 중요한 typed boundary다.

### 주요 타입

- `DifficultyVectorContract`
- `OutputFieldContract`
- `OutputSchemaContract`
- `TaskContract`
- `TaskBundleContract`
- `TaskQualityMetrics`
- `RolloutConstraintsContract`

### 중요한 계약 규칙

- topic은 이제 enum이 아니라 free-form string이다.
- `TaskBundleContract.topic == TaskContract.topic`
- `TaskBundleContract.difficulty_vector == TaskContract.difficulty_vector`
- unordered object list는 반드시 `sort_key`를 가져야 한다.
- list root면 `primary_output_format="json_array"`여야 한다.

즉, 이 프로젝트는 문자열 기반 free-form topic으로 유연성을 확보하면서도, answer shape와 bundle metadata에 대해서는 꽤 강한 구조적 엄격성을 유지한다.

## 10. Canonicalization and Reward

`synthesis/canonicalize.py`는 reward truth path의 핵심이다.

flow:

1. solver가 `submit_result(answer_text)` 호출
2. JSON parse
3. schema-driven canonicalization
4. stored canonical answer와 exact equality 비교

reward status taxonomy:

- `matched`
- `json_decode_failed`
- `schema_mismatch`
- `em_mismatch`

이 모듈은 list/object canonicalization이 꽤 엄격하다.

- unordered list는 canonical sort
- `unique_elements=True`면 dedupe
- date/datetime는 ISO normalization
- enum은 whitelist 강제
- object는 unexpected key를 허용하지 않음

따라서 verifier complexity를 runtime code generation으로 푸는 대신, output schema와 canonicalization의 엄격함으로 문제를 닫는다.

## 11. Prompt Rendering

`rendered_prompt_builder.py`의 actor-facing prompt는 고정 shape다.

```text
<entity>
{"pk": 123}
</entity>

{user_request}

# Submit Result Format
{json_schema}
```

중요한 점:

- `<entity>` block은 anchor entity PK만 포함
- schema는 canonical answer에서 runtime이 자동 추론
- synthesis agent가 output schema를 별도로 창작하지 않는다

이 결정은 label과 schema drift를 줄이는 데 핵심적이다.

## 12. Synthesis Prompting

`synthesis/prompts.py`는 synthesis agent에게 상당히 강한 행동 규율을 부여한다.

핵심 메시지:

- label-first
- research first
- anchor neighborhood를 여러 path로 탐색
- readable path vs id-only path를 분리
- rejection은 종료 신호가 아니라 feedback
- too easy / too hard feedback 이후에는 정확히 한 difficulty axis만 조정

또한 prompt는 "coverage hint로서의 topic"과 "authoritative semantics로서의 label"을 분리한다. 즉, scheduler가 준 topic을 억지로 맞추기보다 실제 grounded label을 우선한다.

## 13. `SubmitDraftController`: 실제 synthesis 심장부

가장 복잡하고 중요한 모듈은 `synthesis/submit_draft_tool.py`다.

이 컨트롤러는 단순 schema validation기가 아니다. 사실상 synthesis loop의 judge이자 repair controller다.

### 수행 역할

- payload schema validation
- anchor stability enforcement
- question/entity block 정합성 확인
- placeholder/schema leakage 방지
- ungrounded string 탐지
- id-only answer 차단
- single-tool-derivable label 차단
- count semantics에 대한 aggregate evidence 요구
- temporal/global ranking misuse 방지
- difficulty weakening 방지
- too-easy / too-hard feedback 생성
- solver orchestration 호출
- phase monitor 기록

### 이 컨트롤러가 강제하는 중요한 제약

- 첫 submission 이후 retry는 같은 `anchor_entity`를 유지해야 한다.
- rejection 뒤에는 적어도 한 번의 새 atomic tool call이 필요하다.
- label은 실제 tool output에서 직접 관측된 값만 사용해야 한다.
- anchor entity를 canonical answer에 반복하면 안 된다.
- `_id` 체인만으로 된 답은 금지된다.
- single atomic tool call 결과를 거의 그대로 제출하는 task는 금지된다.

즉, synthesis agent가 "그럴듯한 문제"를 만드는 것이 아니라 "groundedness와 difficulty band를 만족하는 문제"를 만들도록 좁은 행동 공간을 강제한다.

## 14. Synthesis Runtime

`synthesis/runtime.py`의 `SynthesisAgentRuntime`는 single-db runtime이다.

중요한 구조적 성질:

- runtime 인스턴스는 하나의 `db_id`에만 bind된다.
- synthesis conversation은 `_conversation_lock`으로 serialize된다.
- graph, atomic tool bundle, tool executor를 cache한다.
- category discard/backoff 상태를 메모리로 관리한다.

### 실제 synthesis 흐름

1. `db_id` bind 확인
2. `(db_id, topic)` backoff 확인
3. schema introspection
4. atomic tool bundle 생성/물질화
5. compact schema summary/tool surface summary 생성
6. `SubmitDraftController` 생성
7. backend에 atomic tools + submit controller bind
8. single-agent conversation 실행
9. accepted draft 반환 또는 artifact generation error 발생

### provider resilience

- provider별 `ProviderCircuitBreaker` 보유
- synthesis backend 실패 시 failure 기록
- cooldown 중 provider는 건너뜀

### category backoff

- 같은 `(db_id, topic)`에서 discard가 누적되면 temporary backoff
- threshold와 backoff duration은 config에서 제어

이 설계는 robustness는 높이지만, throughput은 runtime 인스턴스당 낮다. scale-out은 여러 runtime/process를 병렬로 띄우는 방향이 필요하다.

## 15. Solver Runtime and Quality Gate

### Solver backend

`solver/backend_openai_agents.py`는 OpenAI Agents SDK thin adapter다.

- atomic tools를 SDK `FunctionTool`로 감싼다.
- `submit_result` tool을 별도 singleton으로 추가한다.
- solver가 최종 답을 `submit_result`로 제출하지 않으면 실패 처리한다.
- transcript/tool trace를 artifact로 기록한다.

### Solver orchestrator

`pipeline/solver_orchestrator.py`는 accepted 여부를 판단하는 실제 quality gate executor다.

- task bundle마다 solver 목록을 순회
- solver별 distinct shuffle seed 주입
- provider semaphore로 concurrency 제어
- reward는 `compute_reward`로 계산
- batch 단위 실행 후 early stop 가능

`evaluate_rollout_summary(...)`는 pass-rate와 confidence interval, early stop decision을 합쳐 최종 status를 낸다.

- in band -> `accept`
- below lower bound -> `reject_too_hard`
- above upper bound -> `reject_too_easy`

### 중요한 설계 포인트

quality gate는 registry commit보다 앞에 있고, accepted metrics는 `accepted_draft_with_quality_metrics(...)`를 통해 draft에 역주입된다. 즉, registry는 "이미 solver 검증을 통과한 draft"만 받는다.

## 16. Scheduler, Orchestrator, Registry Runner

### Scheduler

`synthesis/scheduler.py`는 DB-major round robin이다.

- DB를 번갈아 방문
- DB 안에서는 topic round robin
- backed-off topic은 skip
- 모든 후보가 backoff면 earliest wait time 반환

### Orchestrator

`synthesis/orchestrator.py`는 multi-db thin wrapper다.

- registry entry별 runtime lazily 생성
- scheduler decision을 runtime 호출로 연결
- 실제 draft 생성은 single-db runtime에 위임

### Registry runner

`synthesis/runner.py`는 bounded registry loop다.

- registry file 로드
- checkpoint 기반 resume
- `(db_id, topic)` pair 단위 처리
- accepted draft만 processed checkpoint로 mark
- phase monitor에 quality gate / registry commit 기록

중요한 구현 의미:

- quality-rejected pair는 checkpoint에 찍히지 않으므로 다음 bounded run에서 재생성 가능
- accepted 또는 duplicate commit된 pair만 "processed"로 간주된다

## 17. Task Registry

`synthesis/task_registry.py`는 이 프로젝트의 durable memory다.

### 저장 형태

- filesystem task directory
- SQLite `task_registry.db`
- MinHash 기반 semantic dedup index
- difficulty-band coverage accounting

### exact identity

exact signature는 아래 조합으로 결정된다.

- `db_id`
- `topic`
- `tool_signature`
- `task_signature`

따라서 label signature 자체는 commit identity의 직접 재료가 아니다.

### semantic dedup

semantic dedup text는 다음 surface로 만든다.

- db/domain/topic
- question
- output schema shape
- constraint summaries
- difficulty vector

MinHash LSH는 `(db_id, topic)` scope 안에서만 작동한다. 즉, cross-topic 또는 cross-db near-dup는 현재 차단하지 않는다.

### registry filesystem layout

registry 내부 task directory는 아래 파일을 가진다.

```text
tasks/<task_id>/
  task.yaml
  task.json
  instance.json
  canonical_answer.json
  tools.py
  registry_metadata.json
```

여기서 `tools.py`는 internal registry artifact다. exported serving bundle은 task-local `tools.py`를 복사하지 않고, database-level bundle만 복사한다.

## 18. Bundle Export

`bundle_exporter.py`는 serving bundle을 만든다.

export shape:

```text
bundle_root/
  databases/<db_id>/atomic_tools.py
  databases/<db_id>/atomic_tool_definitions.json
  tasks/<task_id>/task.yaml
  tasks/<task_id>/task.json
  tasks/<task_id>/instance.json
  tasks/<task_id>/canonical_answer.json
```

registry는 bookkeeping을 더 갖고 있고, exporter는 runtime-serving에 필요한 파일만 추린다. 이 구분은 spec과 구현이 잘 맞는다.

## 19. Proof Task and Real DB Trial

### Proof task

`proof_environment.py`는 deterministic vertical slice다.

- synthetic fixture schema/seed SQL 생성
- compositional itinerary task draft 구축
- solver rollout
- quality gate
- registry commit
- bundle export

즉, proof task는 "설계 설명용 예제"가 아니라 실제 end-to-end regression fixture다.

### Real DB trial

`real_db_trial.py`는 실제 DB 하나와 topic 하나에 대해 single-task trial을 돈다.

- synthesis runtime 실행
- accepted draft면 registry commit
- bundle export
- failure면 structured summary 반환
- 중요한 debug artifact 위치를 summary에 포함

이 경로는 production-like trial/debug lane으로 볼 수 있다.

## 20. CLI Surface

현재 CLI는 운영자가 다음 행동을 바로 수행할 수 있게 되어 있다.

- `validate-config`
- `bootstrap-run-db`
- `run-synthesis-registry`
- `show-task-registry`
- `plan-synthesis-coverage`
- `export-bundle`
- `run-proof-task`
- `run-real-db-trial`
- `check-db`
- `show-layout`
- `run-summary`

즉, 이 저장소는 단순 라이브러리가 아니라 operator-facing toolchain까지 포함한다.

## 21. 테스트 현황

이번 분석에서 확인한 테스트 결과:

- bare shell: `pytest -q`
  - 실패
  - 원인: dev dependency `sqlglot` 미설치
- authoritative run: `uv run pytest -q`
  - `171 passed`

테스트가 덮는 영역은 꽤 넓다.

- atomic tool determinism / AST validity / SQL parseability
- schema sensitivity 분류
- DB session safety
- checkpoint / storage
- prompt rendering
- canonicalization
- submit_draft rejection heuristics
- synthesis runtime acceptance path
- solver orchestrator scoring / shuffle seed
- registry commit / migration / dedup text
- proof task
- real-db trial
- CLI reporting

문서 baseline과 비교하면 `docs/phase0_baseline.md`의 `124 passed`에서 현재는 `171 passed`까지 수트가 커졌고, 여전히 green 상태가 유지되고 있다.

## 22. 이번 분석 중 확인하고 수정한 실제 결함

분석 과정에서 registry 조회 경로의 실결함을 하나 재현했고 바로 수정했다.

- 위치: `TaskRegistryWriter.semantic_dedup_candidates`
- 문제: SQLite query는 `topic` 컬럼을 SELECT하는데, 코드가 `row["category"]`를 읽고 있었다.
- 실제 재현: committed draft 뒤에 `semantic_dedup_candidates()` 호출 시 `IndexError: No item with that key`
- 조치:
  - 구현을 `row["topic"]` 기준으로 수정
  - regression test 추가

이 버그는 기존 테스트가 mock registry를 사용해 CLI만 검증하면서 실구현 경로를 직접 호출하지 않아 숨어 있었다.

## 23. 남아 있는 리스크와 관찰

### 1. 문서 잔존 drift

`docs/spec/*.md`는 현재 구현과 대체로 잘 맞지만, `docs/runbook.md`의 일부 exit criteria는 이전 verifier vocabulary를 여전히 남기고 있다. 예를 들어 runbook은 hybrid verifier A/B/C/D를 언급하지만, spec과 코드의 authoritative acceptance path는 solver pass-rate + exact match reward다.

### 2. backend surface와 config surface의 불일치

config는 `anthropic`, `google` provider type을 허용하지만, synthesis/solver backend 구현은 현재 `openai`/`openai_compatible`만 지원한다. 설정 가능 범위와 실행 가능 범위가 완전히 일치하지 않는다.

### 3. runtime throughput 제약

`SynthesisAgentRuntime`는 single-db bind + single conversation lock 구조다. groundedness와 state consistency에는 유리하지만, throughput을 높이려면 runtime 인스턴스를 많이 띄우는 외부 orchestration이 필요하다.

### 4. scheduler cold-start semantics

테스트가 보여주듯 runtime이 아직 생성되지 않은 DB는 cached backoff state가 없으므로 cold start 시 backoff가 무시된다. 현재는 의도된 trade-off처럼 보이지만, 대규모 운영에서는 fairness와 load-shedding 정책을 다시 볼 필요가 있다.

### 5. registry writer concurrency

`TaskRegistryWriter`는 single-writer 가정이다. SQLite `BEGIN IMMEDIATE`와 filesystem rename을 조합하지만, 다중 writer coordination을 제공하지 않는다.

### 6. strict grounding trade-off

submit controller는 매우 공격적으로 label을 제한한다. 이건 task quality엔 좋지만, real DB에서 readable surface가 부족할 경우 reject loop가 길어질 수 있다. privacy default가 `blocked`인 점까지 합쳐지면, 실제 도메인 DB에선 topic diversity보다 "answerable readable path" 확보가 먼저 병목이 될 가능성이 높다.

## 24. 확장 포인트

현재 구조에서 확장하기 좋은 지점은 아래다.

- 추가 synthesis backend
- 추가 solver backend
- richer semantic dedup scope
- registry multi-writer 전략
- more informative schema summaries/tool surface summaries
- cold-start aware scheduler state persistence
- coverage planner와 registry를 결합한 smarter selection heuristic

반대로 쉽게 건드리면 안 되는 핵심 invariant는 아래다.

- canonical answer is source of truth
- output schema는 runtime이 추론
- database-level atomic tool sharing
- solver-facing prompt/tool parity
- quality gate before registry commit
- exact-match reward only

## 25. 결론

이 프로젝트는 "LLM이 DB를 보고 task를 즉석 생성한다"는 느슨한 아이디어를 꽤 엄격한 계약 시스템으로 다듬은 구현이다. 설계상 가장 인상적인 부분은 synthesis creativity를 넓히기보다, grounded evidence와 quality gate 중심으로 synthesis agent를 강하게 제약했다는 점이다.

코드 구조도 그 철학을 그대로 반영한다.

- schema/infra는 비교적 순수하고 deterministic하다.
- 복잡성은 `submit_draft`와 synthesis runtime에 집중된다.
- registry/export는 durable boundary를 분명히 나눈다.
- 테스트는 "정말 이 파이프라인이 legacy 없이 닫히는가?"를 꽤 넓게 검증한다.

현재 시점에서 이 저장소는 rewrite 아키텍처가 이미 실동하는 상태이며, 다음 단계는 기능 추가보다 운영 정합성, scheduler/state persistence, backend 다양화, real DB에서의 grounded readable path 확보 전략을 다듬는 쪽이 더 중요해 보인다.
