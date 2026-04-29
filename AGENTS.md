# AGENTS.md

이 저장소의 최상위 목표는 현재 evaluator를 통과하는 것이 아니라, 실제로 일반화 가능하고 검증 가능한 고품질 RL task dataset을 생성하는 것이다. 원칙과 평가 메트릭을 제외한 composer, solver, tools, prompt, pipeline, architecture, SDK agent 종류/수/context 공유 방식은 모두 변경 가능하다.

## 제1원칙

평가 통과보다 일반화 가능한 실제 dataset 품질을 우선한다. 어떤 변경이 측정 성공률을 올리더라도, 그 이유가 실제 task 품질 개선이 아니라 evaluator, solver band, benchmark DB, rubric 문구, 또는 이미 본 실패 케이스를 이용한 것이라면 그 변경은 무효다.

## 불변 원칙

1. 직접 누설 금지: 알려진 정답, row id, anchor, 특정 DB 리터럴, 테이블명, 컬럼명, 관찰된 bad draft 토큰을 prompt, validator, scoring, generation policy에 넣지 않는다.
2. 구조로만 일반화: 실패 케이스는 schema metadata, PK/FK, profile, query spec, visibility metadata, live evidence 위의 DB-agnostic structural rule로 일반화될 때만 수정 근거가 된다.
3. Precision-100 rejector만 허용: runtime reject logic은 available evidence에서 구조적으로 증명 가능한 조건만 reject할 수 있다. 의미/품질 판단은 hard validator가 아니라 rubric 평가에 둔다.
4. source-of-truth 분리: durable policy는 한 surface에만 둔다. feedback은 named policy와 현재 실패 evidence를 상기할 수 있지만 두 번째 durable prompt가 되면 안 된다.
5. 품질이 yield보다 우선: low-quality accepted는 hard failure다. accept 수, pass rate, throughput 증가로 보상할 수 없다.
6. 다양성은 필수 품질이다: 하나의 안전한 task shape, path, answer surface, request template만 반복하는 dataset은 고품질이 아니다.
7. 억지 생성보다 거절: DB 또는 DB region이 stable identity, visible answer surface, diversity, fanout, verifiable task structure를 충분히 제공하지 못하면 task를 조작해서 만들지 말고 구조적 이유를 보고하고 거절한다.
8. 모든 평가는 버전 기록: rubric version, solver band, solver model, solver count, DB, prompt version, architecture node, qualitative judgment를 실험 노드마다 기록한다.
9. 단계 목표 달성 시 정지: phase target을 metric 기준으로 달성하면 자동 개선을 멈추고 사용자에게 promote/freeze/continue/branch/revise 중 무엇을 할지 물어본다.
10. 독립 평가: promotion 후보는 builder 자신이 단독 평가하지 않는다. 사용자가 subagent 평가를 승인했거나 요청한 경우, `docs/experiments/evaluator_subagent.md` 프로토콜로 독립 DQS evaluator subagent를 만든다. 승인 없이 subagent를 만들 수 없으면 멈추고 사용자에게 평가 승인을 요청한다.
11. LLM judge/validator 허용: LLM 기반 judge, reviewer, semantic validator는 목표 달성에 도움이 되면 사용할 수 있다. 단, versioned evaluation component로 기록해야 하며, precision-100 structural rejector인 것처럼 가장하면 안 된다. LLM 판단은 rubric/semantic quality signal이고, hard structural reject는 원칙 3을 따른다.

## 평가 메트릭

기본 metric은 `DQS-v1`이다. 자세한 기준은 `docs/experiments/rubric_dqs_v1.md`를 따른다.

Hard gates:

- `accepted_low_quality == 0`
- DB-specific literal/token heuristic 없음
- precision-100가 아닌 hard rejector 없음
- severe mode collapse 없음
- 평가 정책 변경은 별도 experiment node로 기록
- promotion 전 독립 DQS 평가 리포트 기록
- LLM judge/validator 사용 시 모델, prompt, context scope, decision authority 기록

DQS는 정성 라벨의 정량 집계이며 promotion 판단을 돕는 지표다. 단일 숫자만으로 승격하지 않는다.

## 실험 그래프 운영

- Git commit graph가 실험 탐색 그래프다.
- Branch가 실험 노드이고 worktree는 활성 작업공간이다.
- `main`은 안정 snapshot이다. 실험을 직접 누적하지 않는다.
- `baseline/current` 또는 registry의 current baseline이 연구 기준선이다.
- 실험 노드는 `exp/<node-id>-<slug>` 형식을 사용한다.
- 실험 노드마다 agent topology를 기록한다: agent 종류/수, shared context 여부, 독립 evaluator context 분리 여부, LLM judge/validator의 권한.
- 후속 실험은 부모 노드 commit에서 새 branch/worktree로 분기한다.
- 실패 노드도 삭제하지 말고 `abandoned`, `local_minimum`, `regressed`, `infra_fail` 등으로 기록한다.
- greedy merge 금지. 좋은 노드는 candidate/frontier로 남기고, 반복 평가 후 baseline으로 promote한다.

## 실험 전 필수 확인

실험/아키텍처 변경을 시작하기 전 최소한 아래 문서를 확인한다.

- `docs/experiments/first_principles.md`
- `docs/experiments/rubric_dqs_v1.md`
- `docs/experiments/evaluator_subagent.md`
- `docs/experiments/registry.yaml`
- `docs/experiments/tree.md`

현재 v1 평가 DB set:

- `mimiciv_demo`
- `postgres_air`
- `pagila`
