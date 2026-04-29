# First Principles

이 문서는 실험 그래프의 불변 원칙을 정의한다. spec, composer, solver, tools, prompt, pipeline, architecture, SDK agent 종류/수/context 공유 방식은 모두 변경 가능하지만, 이 원칙과 평가 메트릭은 평가 해킹을 막기 위한 고정점이다.

## 제1원칙

이 프로젝트는 현재 evaluator를 통과하는 것이 아니라, 실제로 일반화 가능하고 검증 가능한 고품질 RL task dataset을 생성하는 것을 최적화한다.

어떤 변경이 측정 성공률을 올리더라도, 그 이유가 실제 task 품질 개선이 아니라 evaluator, solver band, 특정 benchmark DB, rubric 문구, 또는 이미 본 실패 케이스를 이용한 것이라면 그 변경은 무효다.

## 프로젝트 목표

일정 품질 이상의 relational DB가 주어졌을 때, 그 DB의 schema, 관계, profile, live evidence에 적응해 고품질 RL task dataset을 자동 생성한다.

모든 DB에서 억지로 task를 만들어야 하는 것은 아니다. DB 또는 DB region이 high-quality task를 만들 구조를 제공하지 못하면, 시스템은 task를 조작해서 accept시키지 말고 구조적 이유를 보고하고 거절해야 한다.

## 불변 원칙

### 1. 직접 누설 금지

알려진 정답, row id, anchor, 특정 DB 리터럴, 테이블명, 컬럼명, 관찰된 bad draft 토큰을 prompt, validator, scoring, generation policy에 넣지 않는다.

금지 예:

- 특정 DB의 특정 컬럼 이름을 hardcode해서 reject한다.
- 특정 anchor나 row id에서 올바른 정답을 알려준다.
- 이미 본 실패 draft의 문자열 패턴을 token heuristic으로 막는다.

허용 예:

- PK/FK, nullability, uniqueness, fanout, visibility metadata를 사용한다.
- query spec과 live evidence를 비교해 구조적으로 증명 가능한 오류를 reject한다.

### 2. 구조로만 일반화

실패 케이스는 DB-agnostic structural rule로 일반화될 때만 수정 근거가 된다. 특정 DB에서 발견한 문제는 table/column/value 이름을 제거해도 성립해야 한다.

판단 질문:

- DB 이름을 지워도 이 수정이 설명되는가?
- 테이블/컬럼/값이 바뀌어도 같은 원칙으로 작동하는가?
- 정답을 알려주는가, 아니면 구조적 오류 계열을 막는가?

### 3. Precision-100 rejector만 허용

Runtime rejector는 자신이 검증하는 structural predicate에 대해 false positive가 없다고 주장할 수 있을 때만 reject 권한을 가진다.

배경: 이 원칙은 rule-based code가 benchmark DB, 이미 본 실패 케이스, 특정 모델 약점에 맞춘 편협한 힌트가 되는 것을 막기 위한 평가 해킹 방지 장치다. 코드로 막는 로직은 "현재 실험 샘플에서는 맞았다"가 아니라 "임의의 DB에서도 이 structural predicate가 참이면 reject가 항상 타당하다"를 만족해야 한다. 이 보장을 못 하면 hard reject가 아니라 rubric/semantic quality signal로만 사용한다.

허용:

- duplicate projected answer rows가 query result로 증명됨
- hidden filter/order가 request/contract/label surface에 없는 row-set control임이 query spec으로 증명됨
- list difficulty-up retry가 기존 evaluated row set에 새 narrowing `where` predicate를 추가했음이 query spec diff로 증명됨

금지:

- 자연스러움, 흥미로움, 좋은 difficulty를 hard validator로 추측
- 특정 문자열 포함 여부로 source role 또는 품질 판단

LLM 기반 judge나 semantic validator는 허용된다. 다만 이것은 rubric/semantic quality signal이지 precision-100 structural rejector가 아니다. LLM 판단을 생성 pipeline의 filter, judge, reviewer, evaluator로 사용할 수 있지만, 다음을 지켜야 한다.

- model, prompt, context scope, decision authority를 experiment node에 기록한다.
- hard structural reject 권한을 주려면 해당 predicate가 LLM 판단과 무관하게 구조적으로 증명되어야 한다.
- LLM judge의 판정은 accepted/rejected 정성평가와 DQS 집계에 사용할 수 있다.
- LLM judge를 현재 evaluator나 known benchmark DB에 맞춘 hidden scorer로 쓰면 평가 해킹이다.

### 4. Source-of-truth 분리

Durable policy는 한 surface에만 존재해야 한다.

- system/developer prompt: 역할, 정책, 판단 원칙
- user context: 현재 run의 schema/profile/evidence/context
- tool schema/description: callable contract와 argument constraints
- feedback: 이미 존재하는 named policy와 현재 실패 evidence 상기

Feedback은 새 정책을 만들 수 없다. 자세해질 수는 있지만 두 번째 instruction source가 되면 안 된다.

### 5. 품질이 yield보다 우선

Low-quality accepted는 hard failure다. accept count, pass rate, throughput이 좋아져도 보상할 수 없다.

좋은 rejected는 나쁘지 않을 수 있다. good-but-too-easy/hard rejected는 calibration 또는 difficulty 문제일 수 있으며, low-quality accepted보다 훨씬 덜 위험하다.

### 6. 다양성은 필수 품질

하나의 안전한 task shape, path, answer surface, request template만 반복하면 dataset으로는 실패다. 개별 task가 맞아도 mode collapse가 있으면 promotion할 수 없다.

### 7. 억지 생성보다 구조적 거절

DB 또는 DB region에 다음이 부족하면 생성 불가를 보고한다.

- stable identity
- visible answer surface
- enough fanout
- enough diversity
- low enough null/blank rate
- verifiable relation paths
- non-trivial task affordance

거절은 가능한 한 DB 전체가 아니라 region/path/anchor pool 단위로 진단한다.

### 8. Versioned evaluation

Rubric, solver band, solver model, solver count, DB set, architecture, prompt version, qualitative judgment는 모두 실험 노드에 기록한다. Evaluation policy 변경은 별도 실험 노드다.

Spec과 runtime contract 변경도 architecture 변경이다. Spec은 현재 설계 snapshot이지 불변 공리가 아니며, 바꿀 수 있다. 단, 변경 이유와 영향은 실험 노드에 기록해야 하고, 이 문서의 원칙과 DQS-v1 hard gate를 우회하는 새 규범으로 쓰면 안 된다.

Agent topology도 기록한다. 여러 SDK agent, subagent, planner/verifier/naturalizer 분리, shared context, isolated context, blind-ish evaluator context는 모두 자유롭게 설계할 수 있지만 평가 재현성을 위해 명시해야 한다.

### 9. 적대적 리뷰

Promotion 후보는 반드시 다음 질문을 통과해야 한다.

1. 테이블/컬럼/값 이름이 바뀌어도 유효한가?
2. 다른 고품질 DB에서도 작동할 원칙인가?
3. 알려진 정답이나 bad draft 패턴을 넣었나?
4. accepted 수는 늘었지만 accepted 품질이 약해졌나?
5. solver band/model 변경으로 generator 실패를 숨겼나?
6. 다양성이 실제로 늘었나, 표면적으로만 늘었나?
7. feedback이 두 번째 durable prompt가 되었나?
8. validator가 의미 추측으로 reject하고 있나?
9. 강한 AI가 이 metric을 해킹하려면 어디를 공격할까?

## Phase stop rule

단계별 목표를 metric 기준으로 달성하면 자동으로 다음 개선을 밀어붙이지 않는다. Codex는 멈추고 사용자에게 다음 중 무엇을 할지 물어야 한다.

1. 현재 baseline freeze/promote
2. 다음 phase로 진행
3. 다른 노드에서 새 분기
4. metric/rubric 수정
