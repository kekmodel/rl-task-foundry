# DQS-v1 Rubric

`DQS-v1`은 실험 노드를 비교하기 위한 기본 평가 메트릭이다. 단일 숫자가 최종 진실은 아니다. DQS는 rubric label을 정량 집계해 비교를 돕고, promotion은 hard gate와 적대적 리뷰를 함께 통과해야 한다.

## Hard gates

다음 중 하나라도 실패하면 promotion 불가다.

- `accepted_low_quality > 0`
- DB-specific literal/token heuristic 사용
- precision-100가 아닌 hard rejector 추가
- severe mode collapse
- 평가 정책 변경 미기록
- source-of-truth 원칙 위반
- DB/region readiness 부족을 무시하고 억지 accepted 생성
- promotion 후보에 독립 DQS evaluator report 없음
- LLM judge/validator의 model, prompt/version, context scope, decision authority 미기록
- LLM semantic judgment를 precision-100 structural rejector로 사용하거나 표기

## Production viability gates

이 gate는 DQS 품질 판단을 대체하지 않는다. Low-quality accepted를 줄이기 위해
의도적으로 느린 구조를 만들면 실패다. 반대로 빠르지만 품질이 낮은 accepted도 실패다.

- single real-db trial wall-clock timeout: `trial_timeout_s <= 300`
- promotion 평가마다 accepted/rejected/provider issue 전체 trial의 `elapsed_seconds`를 기록
- accepted 데이터 1개당 평균 productive loop time이 300초를 넘으면 v1 production 후보가 아님
- productive loop time은 DB 서버 기동, pool 생성, schema/profile warm-up 같은 DB별 1회성 준비 시간을 제외
- quality reject trial은 accepted 1개를 만들기 위한 반복 생산 비용이므로 포함
- provider outage, rate limit, authentication failure처럼 명확한 provider issue trial은 평균 계산에서 제외하지만, 생산성 보고에는 별도 집계

## Draft labels

### ACCEPTED_GOOD

Accepted 되었고 dataset에 넣을 만한 고품질 task.

모두 만족해야 한다.

- DB evidence로 검증 가능
- user_request가 자연스럽고 사용자 관점임
- request, label, query evidence, answer_contract가 정렬됨
- 정답 row set/order가 유일하거나 tie 의미가 명시됨
- hidden row-set control 없음
- user-visible하지 않은 handle/ID 노출 없음
- 단순 current-record lookup이 아닌 학습 가치 있음
- competent solver가 허용 tool로 풀 수 있음
- DB-generalizable pattern임
- dataset 다양성을 해치지 않음

### ACCEPTED_BORDERLINE

Accepted 되었지만 약한 품질 우려가 있음. 반복되면 위험하다. Promotion에는 제한적으로만 기여한다.

예:

- 자연어가 약간 기계적이나 의미는 명확함
- task는 valid하지만 diversity에 거의 기여하지 않음
- source role은 명확하지만 request가 다소 어색함

### ACCEPTED_LOW_QUALITY

Accepted 되었지만 dataset에 들어가면 안 되는 task. Hard failure다.

예:

- request와 label이 다른 것을 묻고 답함
- hidden filter/order가 row set을 결정함
- user가 알 수 없는 handle/ID가 정답 표면임
- duplicate projected rows로 정답이 모호함
- DB/table/column jargon이 자연어 request에 노출됨
- trivial one-row detail lookup
- 특정 DB 리터럴 patch 덕분에만 맞음

### REJECTED_GOOD_TOO_EASY

좋고 검증 가능한 task지만 solver 기준 너무 쉬워서 reject됨. 데이터 자체는 나쁘지 않다.

### REJECTED_GOOD_TOO_HARD

좋고 검증 가능한 task지만 solver/model/tool search 한계로 reject됨. 어려운 좋은 문제일 수 있다.

### REJECTED_LOW_QUALITY

나쁜 draft가 올바르게 reject됨. 긍정 신호다.

### REJECTED_INFRA

provider/tool/timeout 등 인프라 실패. 품질 판단에서 분리한다.

### REJECTED_UNKNOWN

판단 근거 부족. 실험 신뢰도를 낮춘다.

## Diversity rubric

Batch 단위로 `0..5` 점수를 준다.

평가 축:

- task shape 다양성: scalar aggregate, grouped aggregate, ordered list, filtered list, comparison, related-row reasoning, cross-record reasoning
- anchor/root entity 다양성
- relation path 다양성
- answer surface 다양성: date/time, category/status, numeric metric, text/name, boolean, count, aggregate, related attribute
- operation 다양성: filter, order, limit, group, aggregate, compare, join/follow relation
- difficulty-up 방식 다양성
- natural request framing 다양성
- usable DB region coverage

점수:

- 0: severe mode collapse
- 1: 매우 작은 표면 변형, core pattern 동일
- 2: field variation은 있으나 shape/path가 거의 동일
- 3: 최소 2개 축에서 의미 있는 다양성
- 4: task shape, path, answer surface 전반의 강한 다양성
- 5: 여러 usable DB region과 task family를 폭넓게 커버

Severe mode collapse 예:

- accepted task 대부분이 같은 shape + path pattern
- 대부분이 single-record detail lookup
- 다른 usable region이 있는데 한 table/relationship만 반복
- request template이 거의 동일

## DQS 계산

```text
DQS =
  3 * accepted_good
+ 1 * accepted_borderline
+ 1 * rejected_good_too_easy
+ 1 * rejected_good_too_hard
+ 1 * rejected_low_quality
+ diversity_score
- 100 * accepted_low_quality
- 2 * rejected_unknown
- 1 * rejected_infra
```

DQS는 ranking aid다. 다음을 모두 만족해야 promotion 가능하다.

- hard gates 통과
- DQS가 baseline보다 개선되거나, 품질 유지 상태에서 yield/diversity 개선
- 적대적 리뷰에서 평가 해킹 가능성이 설득력 있게 발견되지 않음
- 독립 evaluator subagent의 전수 정성평가 report가 기록됨

## Independent evaluation

Builder가 만든 실험을 builder가 단독으로 promotion 판단하지 않는다. Promotion 후보는 `docs/experiments/evaluator_subagent.md` 프로토콜에 따라 독립 DQS evaluator가 평가해야 한다.

Evaluator는 구현 의도보다 artifact, generated drafts, solver traces, DB evidence, rubric을 우선한다. Builder는 evaluator report에 반론을 쓸 수 있지만, 단독 override할 수 없다. Accepted low-quality 판정 충돌은 사용자 판단 또는 재평가가 필요하다.

LLM 기반 judge/validator는 DQS 평가를 보조하거나 semantic quality signal을 만들 수 있다. 단, 사용한 모델, prompt/version, context scope, decision authority를 evaluation policy 또는 node metadata에 기록해야 하며, 구조적으로 증명되지 않은 판단을 precision-100 hard reject로 승격할 수 없다.

## v1 target

v1은 safe trustworthy generator 기준선이다.

평가 DB set:

- `mimiciv_demo`
- `postgres_air`
- `pagila`

권장 평가 정책:

- batch size: 20
- repetitions: 2
- solver count: 8 또는 해당 evaluation policy에 기록된 값
- trial timeout: 300 seconds
- rubric: `DQS-v1`

v1 달성 기준:

- 모든 DB, 모든 repetition에서 `accepted_low_quality == 0`
- 각 DB 평균 `accepted_good >= 30%`
- 각 DB 평균 `accepted_good + rejected_good >= 70%`
- 각 DB 평균 `diversity_score >= 3.5`
- severe mode collapse 없음
- DB readiness diagnosis가 불가능 region/DB를 구조적으로 설명함
- accepted 데이터 1개당 평균 productive loop time `<= 300s`

여기서 `rejected_good`은 `rejected_good_too_easy + rejected_good_too_hard`이다.
