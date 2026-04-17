# Prompt Tuning Log — qwen3.5-plus synthesis agent

## Context

- **목표**: label이 다양한 구조적 차원(field width, filter, cardinality, cross-item rule, composite)으로 점진적으로 확장되면서 solver pass-rate band(0.25-0.75)에 안착한다.
- **제약**: prompt는 DB 특화 금지. Sakila가 아니라 임의의 DB에 그대로 사용 가능해야 한다. 구체 테이블/컬럼 이름은 prompt에 나오지 않고, 오직 구조적 범주(Hop/Width/Filter/Cardinality/…)만 사용한다.
- **모델**: `qwen3.5-plus` via `opencode_zen`. Thinking mode 켜짐. `tool_choice="auto"` (qwen thinking mode가 `required`를 거부하므로 우회 불가).
- **스택 사전조건**: S1~S6 6-stage 리팩터 완료. reasoning_content replay hook on qwen (openai-agents SDK custom hook). `build_tool_call_budget_instruction` 로 tool-call-budget 프롬프트 주입. solver는 `instructions=None` 유지 (편향 없는 측정기 역할).
- **Smoke config (`/tmp/rl_tf_smoke.yaml`)**: `max_generation_attempts=3`, synthesis `max_turns=20`, solvers=3, calibration band=[0.25, 0.75].

## History

### Iteration 1 — 2026-04-17

**Hypothesis.** 기존 프롬프트는 7종 escalation 메뉴에서 "하나 골라라" 형태라 agent가 순서를 몰라 같은 강도의 변경을 반복한다. 숫자 순서가 있는 **Escalation Ladder**로 바꾸고 "rejection = 사다리 한 칸 올리기"로 규칙을 고정하면 누적 난이도가 단조 증가할 것이다.

**Change.**
- `build_synthesis_agent_instructions`의 `# Workflow`에서 7종 표를 제거하고 "Rung 1 draft를 submit하라"로 단순화.
- 새 섹션 `# Escalation Ladder` 추가. Rung 1(Hop) → Rung 2(Width) → Rung 3(Filter) → Rung 4(Cardinality) → Rung 5(Cross-item rule) → Rung 6(Composite filter). 각 rung은 이전 rung의 구조를 보존하고 한 차원만 더 붙인다. Rung 이름은 구조적 범주만 사용, 구체 field/filter는 agent가 현재 DB의 schema+data distribution에서 뽑는다.
- 새 섹션 `# After Rejection` 추가. "rejection은 탐색 신호 아님. 2 atomic call 안에 submit_draft 재호출하라."
- anchor는 locked 유지, label/question만 바꾼다.

**Trial.** `artifacts/smoke_iter01` (task id `b6pg13z2f`), config `/tmp/rl_tf_smoke.yaml`.

**Findings.** Regression. `MaxTurnsExceeded` with **0 submissions**. Agent burned all 20 tool calls on city/country exploration (`get_city`×6, `find_city_by_country_id`×5, `rank_city_by_country_id`×3, `calc_city`×3, others). 6-rung 상세 열거가 qwen thinking mode에게 "전 계단을 미리 계획해야 한다"는 신호로 해석돼 탐색 depth가 폭발. 이전 smoke_007(3-rung 없는 prompt)이 submit 2회를 내던 것에 비해 역행.

**Next direction.** 사다리 구조 자체는 유지하되 **상세 나열을 제거**하고, 프롬프트에는 "Rung 1만 지금 제출하라"만 명시. Rungs 2~6은 rejection feedback 경로로 reveal하거나 그냥 구조적 차원 이름만 한 줄로 나열. qwen reasoning 모드는 "미래 상세 계획이 주어지면 그것을 먼저 reason through하는" 경향이 있어서 상세를 넣을수록 지연됨.

---

### Iteration 2 — 2026-04-17

**Hypothesis.** Iter01은 rung별 상세가 qwen thinking mode의 "전 계단 미리 plan" behavior를 유발했다. Rung 번호와 per-rung 상세를 제거하고, 프롬프트의 즉시 지시를 "minimum viable draft를 submit해라"로 단일화하면 반드시 첫 submit에 도달하고, 그 뒤 escalation axis는 간단한 bullet list로만 참조용 제공하면 agent가 rejection feedback을 통해 필요할 때 하나씩 적용할 것이다.

**Change.**
- `# Workflow` 재구성. "Your ONLY immediate target is the minimum viable draft. Ignore later escalations until you receive a rejection." 문장을 최상단에 박아서 multi-step planning 경향을 차단.
- `# Escalation Ladder`를 `# Escalation Axes`로 rename하고 per-rung 상세 전부 제거. Width / Filter / Cardinality / Cross-item rule / Composite — 5개 axis만 한 줄씩 bullet으로. 각 axis는 구조적 범주만 기술.
- Rung 번호 완전 삭제. 이제 사다리는 agent 머리 속에서 rejection 경로에 의해 자연스럽게 형성됨.

**Trial.** `artifacts/smoke_iter02` (task id `bob780mh1`). anchor = `inventory_id=2866`.

**Findings.** Submit 2회 성공(submit_007과 같은 수준 회복), 3번째 attempt에서 18 atomic call 탐색 후 0 submit → MaxTurnsExceeded. 두 번의 submission 모두 `reject_too_easy`. 즉 iter01의 regression은 치유됐지만 pass_rate band 진입에 실패. 추가로 **중대 발견**: `submit_draft_messages._too_easy_retry_guidance()`가 시스템 프롬프트와 완전히 별개의 구식 문구("Pick ONE structural change: (a) FK hop, (b) filter, (c) list")를 rejection마다 agent에게 내보내고 있었다. agent가 두 개의 충돌하는 지시를 동시에 받고 있었고, 이게 escalation 품질 저하와 3번째 attempt 탐색 폭주의 공통 원인일 가능성이 높다.

**Next direction.** rejection feedback 본문을 새 Escalation Axes와 어휘+의미 양쪽으로 일치시킨다. 핵심은 "ADD, not REPLACE" semantics를 rejection 메시지 안에도 명시해서 agent가 "필드 하나 바꾸기"로 빠지지 않도록 하는 것.

---

### Iteration 3 — 2026-04-17

**Hypothesis.** 시스템 프롬프트와 rejection feedback이 서로 다른 어휘로 서로 다른 escalation 전략을 지시하고 있어서 agent가 일관된 규율을 잃는다. 두 신호를 하나의 Axes 어휘로 통일하고 "ADD, not REPLACE" 규칙을 rejection 메시지에 직접 박으면 escalation 품질이 올라가고 반복 rejection 후 탐색 폭주도 완화될 것이다.

**Change.**
- `submit_draft_messages._too_easy_retry_guidance()` 재작성. (a) FK hop / (b) filter / (c) list 3항 목록을 Width / Filter / Cardinality / Cross-item rule / Composite 5항 axis로 교체.
- "ADD exactly one new structural dimension without removing any existing structure. Replacing a field on the same path is not an escalation and will be rejected." 문장을 rejection 첫 줄에 추가.
- 테스트 `test_submit_draft_too_easy_feedback_preserves_readable_path`를 새 Axes 어휘에 맞춰 업데이트.

**Trial.** `artifacts/smoke_iter03` (task id `btjljq0ax`). anchor = `rental_id=11765`. 두 번의 attempt 모두 `difficulty_crank_invalid`로 분류됨.

**Findings.** 진전. 이번엔 모든 attempt가 crank 단계에 도달했고(즉 submit_draft 실제로 호출), 에러 타입이 `SynthesisPhaseExecutionError(MaxTurnsExceeded)` → `SynthesisArtifactGenerationError`로 바뀜. Phase 시퀀스: (1) submit_draft rejected reject_too_easy → (2) submit_draft feedback `no_new_grounded_observation` ← **새 진단 신호** → (3) submit_draft budget_exhausted reject_too_easy → synthesis failed. 2번째 submit은 grounding 검증에 실패 — agent가 새 dimension(필드, 필터 값)을 label에 넣었지만 해당 값이 tool 결과에 관찰되지 않았다. 즉 "ADD, not REPLACE" 지시는 수용됐지만 "escalation 이전에 새 축의 값을 tool call로 확인"이라는 선행 조건이 프롬프트에 없어서 환각 값을 넣었다.

**Next direction.** `# After Rejection` 섹션에 grounding 규칙 추가. rejection 이후 `submit_draft` 호출 전에 "새 축이 참조할 값을 surface하는 1~3개 atomic call"을 의무화. 현재의 "within 2 atomic calls" 상한을 "1-3 calls targeted at the new axis"로 바꾸고, "ungrounded values → `no_new_grounded_observation`" 피드백 코드를 prompt에 노출해서 agent가 왜 거절되는지 사전에 알도록 한다.

---

### Iteration 4 — 2026-04-17

**Hypothesis.** rejection 후 escalation의 실패는 **grounding 누락** 때문. agent는 "add Filter"까지는 따라가지만 그 filter의 실제 값(예: "movies with rating > X"의 X)을 tool call로 확인하지 않고 상상으로 박는다. `# After Rejection` 섹션을 grounding 중심으로 재작성해서 "새 축에 대해 1-3 atomic call을 먼저 하라, 그 다음 submit_draft"로 순서를 강제하면 `no_new_grounded_observation` reject가 사라질 것이다.

**Change.**
- `# After Rejection` 3항목 재구성. (1) too_easy 후 "ground the new axis first: 1-3 atomic calls that surface the exact value(s) the escalated label will reference", (2) "every value in the escalated label must come from a tool response observed in the current conversation. Ungrounded values are rejected as `no_new_grounded_observation`", (3) "new axis에 무관한 탐색 금지".
- rejection 이전의 "within 2 atomic calls" 상한을 "1-3 targeted calls"로 완화 + 목적화.
- 전체 182 pytest green 유지.

**Trial.** `artifacts/smoke_iter04` (task id `bjc9jy8tt`). anchor = `actor_id=166`. submit 단 1회.

**Findings.** Regression. iter03가 2 attempts × crank_invalid 였던 것이 iter04에서는 submit 1회 후 MaxTurnsExceeded. 이유: 마지막 attempt에서 `find_film_actor_by_last_update` 10회 + `get_film_actor` 6회의 반복 tool call 루프가 발생. "ground the new axis first: 1-3 atomic calls"라는 허용 문구가 qwen thinking mode한테 **"탐색해도 된다"는 재해석 여지**를 제공. 규칙의 미묘함 자체가 reasoning 모델에게 독. 즉 iter03의 `no_new_grounded_observation` 진단은 진짜 문제였지만 해결책으로 추가한 허용이 더 나쁜 탐색 폭주를 유발.

**Next direction.** grounding 요구는 유지하되 **완전히 압축**한다. "Every escalated label value must come from an earlier tool response. Only if no usable value exists, make ONE call, then resubmit." 형태로 단 한 문장 + 단 1 call 상한. 반복 탐색 금지를 같은 문장에 포함.

---

### Iteration 5 — 2026-04-17

**Hypothesis.** iter04의 "1-3 calls" 문구가 qwen reasoning mode의 탐색 폭주를 유발했다. 허용 상한을 **1로 하드리밋**하고 "repeating same tool = no new evidence"를 명시적으로 금지하면, 반복 탐색 루프를 끊으면서 grounding 원칙은 유지될 것이다.

**Change.**
- `# After Rejection`을 단 한 문단으로 압축. "Every value in the escalated label must appear in an earlier tool response. Pick an axis whose value you already have; only if none is usable, make one call to fetch it, then resubmit."
- "Repeating the same tool with near-identical parameters never surfaces new evidence" 문구를 같은 문단에 추가 (iter04의 10회 반복 tool call 패턴 차단 목적).
- 전체 182 pytest green 유지.

**Trial.** `artifacts/smoke_iter05` (task id `bmr7fropl`). anchor = `film_id=546`. submit 1회.

**Findings.** Regression 지속. iter04와 동일하게 submit 1회 + 나머지 attempt MaxTurnsExceeded. 이번엔 반복 tool loop는 줄었지만 **탐색 폭이 넓어짐**(address / category / city / country / film / inventory / language / store 등 19 calls). qwen reasoning mode는 조건문("only if none is usable") 자체를 decision tree 탐색으로 처리한다. 단일 문장이어도 "if... then else..." 구조이면 qwen이 각 tool마다 "이게 쓸만한가?"를 판정하느라 부담이 늘어난다. 즉 iter03~iter05 구간에서 **After Rejection 섹션을 건드릴수록 나빠지는 패턴**이 확인됨.

**Next direction.** grounding 책임을 `# After Rejection`에서 빼서 `# Label Rules`의 한 줄 bullet으로 이동. 이유는 Label Rules는 agent가 submit 직전 label을 점검할 때 참조하는 원칙 리스트라 conditional reasoning 부담이 적고, After Rejection은 iter03의 최소 형태("within 2 atomic calls, resubmit") 그대로 두는 것이 qwen thinking mode에 가장 안전.

---

### Iteration 6 — 2026-04-17

**Hypothesis.** After Rejection을 건드릴수록 qwen이 조건문 tree 탐색에 빠진다. grounding 요건을 **Label Rules에 bullet 하나 추가**하는 형태로 옮기면, submit 직전 label 점검 루틴 안에 자연스럽게 들어가고 rejection path는 최소한으로 보존된다.

**Change.**
- `# After Rejection` 섹션을 iter03의 최소 형태로 되돌림. "A rejection is not a signal to explore more. Within 2 atomic calls of rejection feedback, call submit_draft again."
- `# Label Rules`에 다섯 번째 bullet 추가. "Every value referenced by the label — including filter thresholds, categorical filters, and cardinality targets — must already appear in a prior tool response. Ungrounded values are rejected as `no_new_grounded_observation`."
- rejection feedback 본문은 iter03 상태 그대로(Axes + ADD not REPLACE).

**Trial.** `artifacts/smoke_iter06` (task id `bvwmfwup7`). anchor = `payment_id=15711`. submit 2회.

**Findings.** Submission 회복 + grounding 성공. 두 submit 모두 `no_new_grounded_observation` 없이 순수 `reject_too_easy`만 떴다. 즉 iter06 변경으로 ungrounded value 차단이 작동하고 있다. 하지만 **escalation 강도**는 여전히 부족 — 2회 submit 모두 너무 쉬움. 마지막 attempt는 MaxTurnsExceeded(18 tool calls, 다양한 payment filter 축 탐색: by_customer × 4, by_amount, by_date, by_staff). agent는 Filter axis를 **시도하고 있지만** solver 3명(qwen3.5-plus thinking mode) 전원을 떨구는 강도에는 도달 못 함. 구조적 한계 발견: composer와 solver가 같은 강한 reasoning 모델이면 band [0.25, 0.75] 진입이 어렵다.

**Next direction.** 프롬프트 측면에서 escalation 축의 **강도 우선순위**를 명시해서 Width / 단일 Filter 단계를 건너뛰게 유도. Cardinality(answer shape 변경) / Cross-item rule / Composite 을 우선 축으로 프레임. 이건 DB 특화 없이 구조적 강도 ranking만으로 가능.

---

### Iteration 7 — 2026-04-17

**Hypothesis.** 같은 reasoning 모델이 composer/solver 양쪽이라 Width나 단일 Filter 수준 escalation으론 pass_rate를 band로 끌어내리지 못한다. Escalation Axes를 **강도 순서대로 재배열**하고 "Width와 단일 Filter는 첫 escalation부터 피해라"를 명시하면 agent가 Cardinality/Composite 부터 시도할 것이다.

**Change.**
- `# Escalation Axes`의 bullet 순서를 Cross-item rule → Cardinality → Composite → Filter → Width로 재배열(강→약).
- 각 bullet 첫 줄에 strength 특성 주석(e.g., "Cardinality — ... Changes answer shape.").
- Axes 섹션 하단에 추가 문장: "Width and a single Filter alone rarely shift pass_rate enough. The first escalation after a too_easy rejection should add Cardinality or a Composite filter unless the label already has one."
- Cross-item rule의 prerequisite "requires Cardinality already present"를 bullet 설명에 포함.

**Trial.** `artifacts/smoke_iter07` (task id `b9govvnwn`). **Blocked by quota**: Alibaba Qwen API가 429 `insufficient_quota`를 반환 — 계정 쿼터 소진. 7회 연속 reasoning-heavy trial이 누적돼 quota cap 도달.

**Findings.** 프롬프트 변경 효과 측정 불가. qwen3.5-plus billing 리셋 전까지 후속 iteration은 blocked.

**Next direction.** (a) 다른 모델(gpt-5.4-mini, claude-haiku-4-5 등 non-thinking)로 전환해서 프롬프트 검증 계속, (b) billing 리셋 대기, (c) 지금까지 얻은 학습을 바탕으로 설계 결정 정리 후 재개. 판단은 다음 세션에서.

---

## Cross-Iteration Summary (iter 1-7)

### 행동 변화 요약

| Iter | Submits | 3번째 attempt | 주요 증상 |
|------|---------|----|-----------|
| 01 | 0 | 20-call 탐색 | Rung 상세 6단계 → qwen 과계획 |
| 02 | 2 | 18-call 탐색 | Axes bullet list 복귀, baseline |
| 03 | 2 | crank × 2 | rejection text 정렬, `no_new_grounded_observation` 관측 |
| 04 | 1 | 19-call loop | "1-3 calls" 허용 → 탐색 폭주 |
| 05 | 1 | 19-call 넓은 탐색 | 조건문 문장도 qwen에겐 tree 탐색 유발 |
| 06 | 2 | 18-call 탐색 | grounding을 Label Rules로 이전 → ungrounded 차단 성공 |
| 07 | N/A | — | quota 소진, 측정 불가 |

### 확정된 설계 결정 (DB-agnostic)

1. **qwen thinking-mode는 system prompt의 nuance를 decision-tree로 재해석한다**. "within N calls" 같은 상한, "only if... then..." 같은 조건문을 주면 각 tool call마다 조건을 재평가하느라 탐색 폭주. 단순·명령형 문장으로 유지.
2. **시스템 프롬프트와 rejection feedback 본문은 어휘+의미 완전 정렬 필수**. 두 신호가 다르면 agent는 일관된 규율을 잃는다 (iter03 핵심 발견).
3. **grounding 책임은 `# Label Rules`에 bullet 하나로 둬야 한다**. `# After Rejection`에 넣으면 조건문이 돼 과탐색 유발.
4. **`# After Rejection`은 최소 형태 고정**: "rejection ≠ 탐색 신호, 2 atomic calls 내 재submit" 외에는 건드리지 않는다 (iter04/iter05가 증명).
5. **Escalation Axes는 강도 우선순위로 나열해야 Cardinality/Cross-item 등 강한 축이 선택된다** (iter07 가설, 미검증).

### 구조적 한계 (prompt로 해결 불가)

같은 reasoning 모델(qwen3.5-plus thinking)이 composer + solver 양쪽에 쓰이면 composer가 solver 3명을 동시에 떨구는 task를 설계하기 매우 어렵다. Band [0.25, 0.75] 진입은 composer와 solver의 상대적 능력 차가 있어야 자연스럽다. 후속 조치 후보:
- composer = qwen3.5-plus, solver = 약한 모델(gpt-5.4-nano 등) 혼합
- solver 수를 3 → 5~10으로 늘려 band 해상도 증가
- calibration band 완화(현재 [0.25, 0.75] → [0.1, 0.9])

### 다음 세션 작업 우선순위

1. **billing 확인 후 qwen3.5-plus 재개** 또는 **non-thinking 모델로 전환**.
2. **iter07 프롬프트(강도 순 escalation)** 효과 검증.
3. 검증 후 **iter 03/06/07 변경사항 커밋**(지금까지 프롬프트 변화는 unstaged).
4. 이후 composer-vs-solver 비대칭 설정 실험.

---

---

## Metrics Template (per iteration)

- **Attempts observed**: N / max_generation_attempts
- **Submissions**: M / N (how many attempts actually reached submit_draft)
- **Pass-rate trajectory**: [0.0, 0.67, 0.33, ...] per submitted attempt
- **Terminal status**: accepted | reject_too_easy (band still above) | reject_too_hard (band still below) | MaxTurnsExceeded | other
- **Ladder climb observed**: which rungs were visible in each submission's label structure
- **Regression signals**: agent over-exploring after rejection, repeating same rung, weakening label, etc.
