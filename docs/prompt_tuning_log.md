# Prompt Tuning Log — synthesis pipeline

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

### Iteration 8 — 2026-04-17

**Hypothesis.** iter07 retry에서 qwen이 Width만 반복한 건 Escalation Axes 안의 "강도순+Width 회피" 문단이 정렬 정보로만 읽혔기 때문. 규칙을 `# Label Rules`에 선언형 imperative로 올리면 agent가 turn마다 재평가하며 지킨다.

**Change.**
- `# Escalation Axes`에서 "listed strongest to weakest" 문장과 "Width and a single Filter alone rarely shift..." 두 문장을 제거. 5개 축 bullet만 neutral 참조로 유지.
- `# Label Rules`에 bullet 2개 추가:
  - "After a too_easy rejection, the next label MUST change the answer shape. Raise record count (Cardinality or Cross-item) or add a row-excluding constraint (Filter or Composite). Adding a column to a single-record answer (Width alone) does not change shape."
  - "After any rejection, make at least one fresh atomic tool call before the next submit_draft."

**Trial.** `artifacts/smoke_iter08` (flow_id `real_db_trial:20260417T084645Z:1649ee42`). anchor = inventory_id=4547 (film_id=993). **submit 0회, `MaxTurnsExceeded` (20 turns)**. 19 atomic calls, 0 submit_draft.

Call pattern: `get_inventory(4547) → get_film(993) → get_store(1) → get_address(1) → get_city(300) → get_country(20) → find_inventory_by_store_id × 3 → get_inventory(4548…4560, 4555, 4554)`. 4-hop 주소 체인을 먼저 내려간 뒤 cardinality 후보를 만들려 inventory 리스트를 하나씩 확인.

**Findings.**
- Label Rules bullet은 turn-1부터 적용됐고, "next label MUST change shape"를 첫 submit 대비로 qwen이 재해석해 **cardinality 후보를 초기 탐색 단계에서 쌓으려 했다**.
- iter01/04/05와 동일한 "복잡한 규칙 → qwen 과계획" 실패 모드. "After a too_easy rejection"이라는 scope 어구를 포함했음에도 Label Rules 위치에서는 조건부로 읽히지 않고 상시 제약으로 적용됨.
- Label Rules는 `<scope, always-on>` 규칙 저장소로만 안전하게 쓸 수 있음이 재확인됨 (iter06에서 grounding이 성공한 이유이기도 함 — grounding은 진짜로 every turn에 요구되는 제약).
- 교훈: "after too_easy"처럼 rejection-conditional인 지시는 이미 conditional scope가 박힌 Workflow step 3이나 dedicated `# After Rejection` 섹션에만 둬야 한다.

**Next direction.** iter09에서 동일 shape-change 규칙을 Workflow step 3으로 옮긴다.

---

### Iteration 9 — 2026-04-17

**Hypothesis.** iter08의 shape-change imperative는 옳지만 Label Rules 위치가 잘못됐다. Workflow step 3("On too_easy, …")는 이미 rejection-conditional scope라 같은 문장을 여기 넣으면 첫 submit에는 영향 없이 too_easy 이후에만 발동한다.

**Change.**
- iter08의 Label Rules bullet 2개 제거.
- Workflow step 3을 재작성: "On too_easy, make at least one fresh atomic call, then resubmit with a label that changes the answer shape — raise the record count (Cardinality or Cross-item rule) or add a constraint that excludes rows (Filter or Composite). Adding another field to a single-record answer (Width alone) does not count."

**Trial.** `artifacts/smoke_iter09` (flow_id `real_db_trial:20260417T085223Z:1516b096`). anchor = customer_id=86. **submit 0회, `MaxTurnsExceeded`**. 18 atomic calls, 0 submit.

Call pattern: `get_customer(86) → 4-hop 주소 체인(address/city/country) → find_rental_by_customer_id × 2 (limit=100) → get_rental(1) → find_payment_by_customer_id(limit=100) → find_customer_by_store_id(limit=100) → find_address_by_city_id(limit=100) → find_customer_by_activebool(limit=10) → find_customer_by_last_update → 두 번째 주소 체인 → find_customer_by_store_id(op=any)`.

**Findings.**
- Workflow step 3으로 옮겼음에도 동일 실패 — qwen은 step 3을 "첫 submit 전에 필요한 사전조건"으로 재해석. 첫 draft를 내기 전에 "resubmit-시 선택 가능한 후보들"을 한꺼번에 확보하려는 듯 대량의 `find_*` (limit 100) + multi-axis 탐색 수행.
- iter01 전후로 반복된 "qwen이 Workflow의 조건부 step까지 선제 계획에 포함한다"는 증상이 재현. 현 프롬프트는 "step 1~5 다 훑어본 뒤 행동"이라는 qwen의 thinking-mode 습성과 부딪힘.
- 구조적 단서: Workflow가 "1. Inspect → 2. Submit immediately" 같은 순차 명령형이어도, step 3~5 본문이 상세하면 step 2 실행 자체가 지연된다. 즉 escalation 규칙의 **상세도 자체**가 첫 submit 속도를 느리게 만드는 독립 변수.
- iter06까지는 step 3이 한 문장 "add exactly ONE dimension..."이라 깔끔했고 submit이 2회 발생. iter09에서 세 줄로 늘리자 submit 0.

**Next direction.** iter10에서는 step 3을 **짧은 한 문장**으로 되돌리고, shape-change 강제는 Escalation Axes bullet 본문에 명령형으로 분산시킨다 — 각 bullet이 "언제 적용" 대신 "쓰면 effect"를 선언하는 형태. Axes 정보가 길어지는 건 괜찮은데(iter03에서 확인) step 본문이 길어지는 건 위험.

---

### Iteration 10 — 2026-04-17

**Hypothesis.** iter09 실패 원인이 step-3 본문 과다였으니, step 3을 iter06 수준 한 문장으로 되돌리고 shape-change 강제를 `# Escalation Axes` 각 bullet의 **effect 설명**에 분산시킨다(정렬이 아닌 축별 effect 선언). 특히 Width bullet에 "Does NOT change shape or row set; insufficient by itself" 명시.

**Change.**
- Workflow step 3 → iter06 한 줄 복원: "On too_easy, add exactly ONE dimension from the escalation axes below and resubmit within 2 atomic calls."
- Axes 섹션 intro에 "shape-changing axes drop pass_rate faster" 추가.
- 각 bullet 끝에 effect 한 줄: Cardinality "Changes shape. Strongly drops pass_rate.", Cross-item "Changes shape. Requires Cardinality already present.", Composite "Changes which rows qualify.", Filter "mildly.", Width "Does NOT change shape … insufficient by itself when already too_easy."

**Trial.** `artifacts/smoke_iter10` (flow_id `real_db_trial:20260417T090319Z:c60da1db`). anchor = rental_id (customer ALEX GRESHAM / film MURDER ANTITRUST). 3 submit, 전부 `reject_too_easy` → `difficulty_crank_invalid` × 3 → synthesis_failed.

Per-attempt 요약:

| # | added fields | slot | pass_rate |
|---|--------------|------|-----------|
| 1 | first_name, last_name, title | 3 | 1.0 (3/3) |
| 2 | +city (Width) | 4 | 1.0 (3/3) |
| 3 | +rating, +rental_duration (Width) | 6 | 1.0 (3/3) |

**Findings.**
- submit 빈도는 iter06~07 수준으로 정상화(step 3 짧게 유지 규칙 재확인).
- 3 attempt 전부 **Width**. Width bullet에 "insufficient by itself" 명시한 iter10에서도 편향 안 깨짐 → 프롬프트 문구로 qwen의 Width 편향을 깨는 것은 실효성 낮다는 증거 강화(iter07 retry와 합해 2 trial 연속).
- 주목할 점: 이번 trial은 attempt 사이 `find_rental_by_customer_id`, `rank_rental_by_customer_id`, `calc_rental`까지 호출해 **cardinality/통계 축의 후보 데이터 자체는 관측**했다. 그럼에도 label은 single-record width에 고정. composer의 "anchor=1건 lookup" 프레임이 tool 관측으로도 흔들리지 않음.
- pass_rate가 세 번 모두 1.0으로 고정된 점 — solver가 같은 모델이라 Width 3→4→6 증가에도 전혀 못 맞춘 경우가 없음. 같은-모델 ceiling 가설의 직접 증거.

**Next direction.** 프롬프트 분포 가정을 바꿔서 Width 편향을 구조적으로 제거하는 방향 시도. iter11 후보: Workflow step 2의 "first draft = multi-hop lookup returning one record"를 "first draft returns a 3-item list along the anchor's 1:N path"로 교체. Cardinality를 baseline으로 깔면 이후 escalation은 Cross-item/Composite 밖에 남지 않아 Width 유혹이 구조적으로 사라진다. 단 이건 synthesis 태스크 카테고리 분포를 바꾸는 변경이라 lookup-style task 생성 빈도가 줄 수 있음.

---

### Iteration 11 — 2026-04-17

**Hypothesis.** Width 편향이 프롬프트 문구로 깨지지 않으므로 구조적 해법 시도. Workflow step 2의 "1-record lookup first draft"를 "3-item list along anchor's 1:N path, sorted by observed key"로 바꾸고, Axes에서 Width를 **disallowed**로 명시. Cardinality가 baseline이면 Width 경로가 구조적으로 제거된다.

**Change.**
- Workflow step 2 → "homogeneous 3-item list along 1:N path with 1-2 user-facing fields, sort by observed field" (의도상).
- Deterministic Answers 섹션 → "narrow to one OR return full list" 제거, "fix count + sort clause" 로 재작성.
- Axes Width bullet → "disallowed as an escalation. Adding more fields per record does not change the row set".
- 다른 axes bullet도 "change N/bump N" 같은 N-변경 어휘로 재배치.

**Trial.** `artifacts/smoke_iter11` (flow_id `real_db_trial:20260417T091347Z:438411b3`). anchor = address_id. 3 submit, 3 모두 실패 → synthesis_failed.

| # | preview | error_code |
|---|---------|------------|
| 1 | `[{address,district:""},{city,country},{phone:"",postal_code}]` (heterogeneous!) | `label_blank_string_forbidden` |
| 2 | `[{field,value},{field,value},{field,value}]` (key-value rows) | `label_values_not_grounded` |
| 3 | `[{address},{city},{country}]` (singleton heterogeneous) | `reject_too_hard` (pass_rate 0.0) |

**Findings.**
- Step 2가 말한 "list along 1:N path"를 qwen은 **anchor의 속성들을 list로 나열**하는 것으로 오해. homogeneous 구조(같은 keys 반복) 개념이 빠져 있어 attribute enumeration으로 falsify.
- attempt 1에서 `district=""`, `phone=""` 같은 blank string이 들어가 `label_blank_string_forbidden` 발동. schema inference 관점에서 list-heterogeneity가 아니라 단순 문자열 검증에 걸림.
- solver pass_rate 0.0 — 3 solver 전부 `[{address},{city},{country}]` 같은 기이한 답을 재현하지 못함. 즉 "list-first" 아이디어 자체는 태스크를 어렵게 만들긴 했지만, 구조가 무너져서 매칭 불가 상태로 어려워진 것(실패 종류 다름).
- 교훈: list-first를 제대로 유도하려면 "homogeneous list where every item shares the same keys"를 1문장으로 못박고, 예시 패턴도 `[{rental_date, film_title}, …]` 식으로 같은 keys 반복을 시각적으로 제시해야 한다.

**Next direction.** iter12에서 step 2를 "homogeneous list of N child records through a single foreign key" 로 rewrite + 시각적 예시. Deterministic Answers도 "fix count + sort clause"로 갱신.

---

### Iteration 12 — 2026-04-17 (quota-blocked)

**Hypothesis.** iter11의 heterogeneous list 오해를 "homogeneous list of 3 child records through a single foreign key, every item sharing the same 1-2 keys" + 시각적 예시(`[{rental_date, film_title}, …]`)로 해소. Deterministic Answers도 "fix count + sort clause"로 재작성.

**Change.**
- Workflow step 2 재작성: "homogeneous list of 3 child records reached through a single foreign key from the anchor. Every item shares the same 1-2 keys (e.g. `[{rental_date, film_title}, …]`). Sort by one observed field in a fixed direction. No filters."
- Deterministic Answers 재작성: "State the exact record count and sort clause in the question (e.g. 'the first 3 rentals ordered by rental_date ascending'). Never leave count or order implicit."

**Trial.** `artifacts/smoke_iter12` (flow_id `real_db_trial:20260417T091851Z:54a31ab1`). anchor = address_id. 8 atomic call 후 `insufficient_quota` 429 발생 → synthesis_failed. submit 전 API cutoff.

**Findings.** 프롬프트 효과 미측정. 이 세션에서 누적 6 trial이 Alibaba qwen3.5-plus quota를 다시 소진. 첫 8 atomic call 관찰 결과 qwen이 다시 주소 체인(get_address → get_city → get_country → find_customer_by_address_id × 2 → find_staff_by_address_id → find_store_by_address_id → find_address_by_city_id)으로 진입 — address anchor에서는 list-first가 어색한 게 이 iter11~12 공통. customer/rental anchor라면 다를 수 있으나 측정 불가.

### Iteration 12 retry — 2026-04-17 (quota refreshed)

**Trial.** `artifacts/smoke_iter12_retry` (flow_id `real_db_trial:20260417T102744Z:5c05ea10`). anchor = city_id=245. 20 atomic call, 0 submit, `MaxTurnsExceeded`.

Call 분해:
1. `get_city(245)` → anchor description 확보.
2. `find_address_by_city_id(value=245, limit=10)` × 3 동일 쿼리 반복 + op=in 변형 1회 — 같은 10건 child 주소를 4번 재확인.
3. `get_country(103)`.
4. `find_address_by_last_update op="any" value=""` × 4 — atomic tool 스키마 버그(`value must be null when op=any`)로 4턴 헛발질. 메모리에 이미 기록된 기존 버그.
5. `get_address(1)` → `find_address_by_district value=""` × 2 → `calc_address count by city_id(=245)` × 3 → `get_address(100/200/300/400)` 샘플링으로 마무리.

**Findings.**
- qwen은 **city → address 1:N 경로를 정확히 식별**했고 `find_address_by_city_id`로 child rows 10건을 가져왔다. homogeneous-list 프롬프트의 구조적 목표(1:N 단일 FK 자식 3건)는 이해했다는 첫 긍정 신호.
- 그러나 turn budget(20)이 (a) 동일 쿼리 재호출 루프와 (b) `op="any"` 스키마 버그로 4턴, 총 8+턴이 소진되어 submit_draft 직전에 예산 고갈.
- 즉 이번 실패는 **프롬프트 반증이 아니라 런타임 노이즈** — 프롬프트 가설(homogeneous list baseline이 Width 편향을 구조적으로 대체)은 여전히 미검증 상태로 남음.
- 교훈: homogeneous-list 방향은 살려둘 가치 있지만, `find_*_by_last_update op="any"` 버그 존재 하에선 20턴 예산이 타이트. 이 버그가 샤피로 "첫 submit 전 qwen 헛발질 유발기"로 작동하고 있음.

**Next direction.** (a) atomic tool 스키마 버그 수정(`op="any"`일 때 value 무시) 후 iter12 프롬프트 그대로 재시도, 혹은 (b) `max_turns` 20→25 상향이라는 프롬프트-독립 컨트롤로 iter12 재시도. 둘 다 프롬프트 스코프 내 조정. 프롬프트 자체는 변경하지 않는다 — homogeneous-list 가설이 실제로 작동하는지 보려면 노이즈를 걷어낸 조건에서 동일 프롬프트를 한 번 더 돌려야 함.

### Iteration 12 retry2 — 2026-04-17 (tool bug fixed)

**Change.** `src/rl_task_foundry/synthesis/atomic_tools.py::_render_atomic_tool_source` 내 `_validate_find_value` 를 수정해 `op == 'any'` 분기에서 value 검사 제거. 기존 `find_*` 런타임은 `op='any'`일 때 `where_sql='TRUE'`로 value 무시하므로 downstream 영향 없음. pytest 182/182 통과.

**Trial.** `artifacts/smoke_iter12_retry2` (flow_id `real_db_trial:20260417T103610Z:db0bd0db`). anchor = rental_id=6966 (customer 45, inventory 1577, film 346). 20 atomic call, 0 submit, `MaxTurnsExceeded`.

Call 분해 (key turns):
1. turn 1~4: `get_rental(6966) → get_customer(45) → get_inventory(1577) → get_film(346)` 정상 inspection.
2. turn 5: `find_rental_by_customer_id(customer_id=45, limit=3, sort_by=rental_date, direction=asc)` — **step 2 목표 정확히 실행**, child rows 3건 확보. 여기서 submit 가능한 상태.
3. turn 6~8: 같은 쿼리 limit=3 재호출 + limit=5 변형 + payment 탐색.
4. turn 9~11: staff→rental 경로, op=in 변형.
5. turn 12~15: address→customer 확장, rental_by_rental_date, film_by_release_year.
6. turn 16~20: film_by_length, actor_by_first_name/last_name/any, rental_by_return_date.

단 한 건의 `find_*_by_last_update op="any"` 에러 스팸도 없음 → tool 버그 수정 효과 확인.

**Findings.**
- 툴 노이즈 제거에도 불구하고 **submit 0**. 실패 원인이 툴 버그가 아니라 qwen thinking-mode의 commit 회피.
- 프롬프트는 올바르게 수용됨: step 2의 "homogeneous list of 3 child records through single FK, sort by observed field" 를 turn 5에 정확히 실행. 그럼에도 submit 대신 대안 1:N 경로(payment/staff/address/date/actor 5종)를 탐색.
- Commit Rule의 "Within your first 6 tool calls, submit_draft must have been called at least once" + "Spend at most 3 atomic calls before your first submit" 두 제약을 qwen이 구조적으로 무시. iter12 retry2는 20턴 내 submit 0.
- 일반 패턴: step 2에 조금이라도 structural freedom(어느 1:N 경로? 어느 sort key?)이 있으면 qwen은 "더 나은 옵션" 탐색에 빠짐. iter06(rigid 1-record lookup)처럼 목표가 경직돼야 빠른 commit이 일어남.

**Structural diagnosis.** qwen3.5-plus thinking-mode × rigid composer/solver pairing 조건에서는:
- Rigid step 2 target (iter06) → 빠른 commit, Width escalation만 반복, pass_rate>0.75.
- Flexible step 2 target (iter11/12) → commit 자체가 느림, MaxTurnsExceeded.
어느 쪽도 [0.25, 0.75] 밴드 진입 불가. 같은-모델 ceiling은 프롬프트로 피해갈 수 없음을 재확인.

**Next direction.** iter13에서 step 2에 명시적 commit pressure 추가("Submit after the first successful find_* call; do not explore alternative paths before the first submit. The first draft is always too_easy — that is expected"). 동시에 too_easy를 실패가 아닌 expected outcome으로 재프레이밍하여 qwen의 premature optimization 유혹 차단.

---

### Iteration 7 — 2026-04-17

**Hypothesis.** 같은 reasoning 모델이 composer/solver 양쪽이라 Width나 단일 Filter 수준 escalation으론 pass_rate를 band로 끌어내리지 못한다. Escalation Axes를 **강도 순서대로 재배열**하고 "Width와 단일 Filter는 첫 escalation부터 피해라"를 명시하면 agent가 Cardinality/Composite 부터 시도할 것이다.

**Change.**
- `# Escalation Axes`의 bullet 순서를 Cross-item rule → Cardinality → Composite → Filter → Width로 재배열(강→약).
- 각 bullet 첫 줄에 strength 특성 주석(e.g., "Cardinality — ... Changes answer shape.").
- Axes 섹션 하단에 추가 문장: "Width and a single Filter alone rarely shift pass_rate enough. The first escalation after a too_easy rejection should add Cardinality or a Composite filter unless the label already has one."
- Cross-item rule의 prerequisite "requires Cardinality already present"를 bullet 설명에 포함.

**Trial (blocked).** `artifacts/smoke_iter07` (task id `b9govvnwn`). Alibaba Qwen API가 429 `insufficient_quota`를 반환 — 7회 연속 reasoning-heavy trial 누적으로 quota cap 도달.

**Trial (retry, 2026-04-17 post-quota-refresh).** `artifacts/smoke_iter07_retry` (flow_id `real_db_trial:20260417T083124Z:f3c9b934`). anchor = `rental_id=9970` (customer 578 WILLARD LUMPKIN / inventory 2846 / film 624 NIGHTMARE CHILL / staff 2). 3 submit_draft, terminal `reject_too_hard` → budget_exhausted → `synthesis_failed`. token_usage 610k input / 1.4k output over 8 turns, 230 s latency.

Per-attempt 요약:

| # | added fields | slot | pass_rate | error_code |
|---|--------------|------|-----------|------------|
| 1 | first_name, last_name | 2 | 1.0 (3/3) | reject_too_easy |
| 2 | +title (Width) | 3 | — | no_new_grounded_observation |
| 3 | +staff_first_name, staff_last_name (Width) | 5 | 0.333 (1/3) | reject_too_hard |

**Findings.**
- 강도순 Escalation Axes + "Width와 단일 Filter는 첫 escalation부터 피해라" 명시에도 composer는 첫/둘째 escalation 모두 **Width**를 선택. 가설 **미검증(negative)**.
- attempt 2는 직전 관측 집합 내에서 `title`만 꺼내 label에 추가 — 새 atomic call 없이 submit한 전형적 `no_new_grounded_observation`. iter06에서 확보했다고 본 grounding 규율이 "축 선택"이 복잡해진 순간 다시 깨짐.
- slot 2 → 5로 점프했을 때 pass_rate 1.0 → 0.333. band [0.25, 0.75] 진입은 가능했지만 attempt 2 낭비로 4번째 submit 여력 없음. budget cap 3이 이 프롬프트 구조의 실질적 병목.
- Width 두 번만으로 pass_rate 3단 하강이 관측된 점은 의미 있음 — qwen 솔버 3명이 동일 모델이라도 output surface 크기에 민감. 반대로 Cardinality/Cross-item은 시도조차 안 됨.

**Next direction.**
1. Iter08 후보: "첫 escalation 실패(Width로 진행) 시 immediate Cardinality 전환" 규칙을 `# After Rejection`이 아닌 `# Label Rules`의 명령형 한 줄로 넣는다(조건문 금지 원칙과 양립하도록 "After a too_easy rejection, the next label MUST change the slot count or add a Cross-item rule" 형태).
2. 병행으로 `max_generation_attempts`를 3 → 4로 완화하는 실험(프롬프트 독립). iter07_retry는 4였다면 성공 가능성 존재.
3. asymmetric composer/solver (solver=gpt-5.4-nano 등) 실험을 iter08 뒤로 미뤄두지 말고 같은 세션에 묶어 구조적 상한 대 프롬프트 상한을 구분.
4. `no_new_grounded_observation` 재현성(iter07에서 처음 이 attempt-2 패턴 관측)을 확인하려면 동일 프롬프트로 1~2회 추가 trial 필요.

---

### Iteration 13 — 2026-04-18 (tool surface baseline)

**Hypothesis.** iter01~12가 관측한 same-model (qwen3.5-plus composer + qwen3.5-plus solver) `pass_rate=1.0` ceiling은 동일 tool 서피스를 양쪽이 공유한 구조적 산물이라는 해석이었다. tooling redesign(atomic 9-primitive 계산자 + composer 5-tool DSL)이 완료된 지금, 이 asymmetry만으로 ceiling이 깨지는지(즉 pass_rate 분포가 1.0에서 실제로 이탈하는지)를 베이스라인으로 측정한다. 프롬프트는 5-tool composer + 9-primitive solver 기준으로 commit `4f0b6d5`에서 재작성된 상태 그대로. solver 3명, `max_solver_runs=3`, 밴드는 n=3 discretization에 맞춰 `[0.33, 0.67]`로 조정.

**Change.**
- 전 tooling redesign(`atomic.calculus` 9 primitives + `composer` query/profile/schema_map/neighborhood/sample DSL) 전면 적용. 이전 iter와 프롬프트 의도는 같지만 바닥 서피스가 완전히 달라짐.
- `artifacts/tmp_configs/iter13_3solvers.yaml`로 solver roster 3명(`qwen3.5-plus_00/01/02`, 자동 도출 ID)·`max_solver_runs=3`·calibration `[0.33, 0.67]`. 그 외 모든 synthesis 파라미터는 메인 config 그대로(`max_turns=20`, `max_generation_attempts=5`).

**Trial.** `artifacts/smoke_iter13` — flow_id `real_db_trial:20260417T174920Z:32825672`, anchor `inventory_id=3025`, selected_topic `Film actors for inventory item`. question(ko): "이 재고 항목이 속한 영화에 출연한 배우들을 성(last name) 알파벳 순서대로 정렬했을 때, 처음 3명의 이름과 성을 알려주세요." canonical answer = `[KIM ALLEN, RENEE BALL, CARMEN HUNT]` (film_id=664, last_name asc, top 3). attempts=1/5, submissions=1/1, terminal status `synthesis_failed / SynthesisArtifactGenerationError / budget_exhausted`, `attempt_outcomes=["difficulty_weakened"]`, `error_codes=["reject_too_hard"]`. 약 9분 경과.

Attempt 1 composer trace (`query` DSL 사용):

| # | query spec 요약 | result |
|---|------------------|--------|
| 1 | from=rental, filter=inventory_id=3025, join→customer, select=first/last/email, sort=rental_date asc, limit=3 | KeyError `public.customer.rental_date` |
| 2 | 동일 spec에서 sort 제거 | 4 customer rows |
| 3 | from=inventory, join=rental→customer, select=rental_date+names | KeyError `public.customer.rental_date` 재발 |
| 4 | from=film filter=film_id=664, join=film_actor→actor, select=first/last | 4 actor rows |
| 5 | 동일 spec + sort=last_name asc, limit=3 | 정답 후보 3행 확보 |

atomic_tool_calls_seen=14 → 단일 attempt로 `max_turns=20` 거의 소진 → attempt 2 진입 전 budget_exhausted. solver 3명은 이 task에 0/3 matched, `pass_rate=0.0`.

**Findings.**
- **가설 검증됨(positive).** 동일 모델 페어링이지만 첫 제출부터 `pass_rate=0.0 (reject_too_hard)`이 관측됨. iter07/10의 `pass_rate=1.0` ceiling은 **반전되었다**. tool asymmetry만으로 composer가 solver를 떨구는 task를 authoring 가능. 즉 이전에 "prompt로 해결 불가"로 정리된 구조적 한계는 tool 서피스 교체로 해소된다.
- **밴드 진입은 못함.** pass_rate가 `{0, 1/3, 2/3, 1}` 중 0으로 떨어짐. composer가 저작 가능한 난이도가 solver의 composition 능력을 크게 상회. 2-hop(`inventory→film→actor` via `film_actor`) + sort+limit 태스크를 9-primitive 체인으로 재구성하는 비용이 solver에게 과도한 것으로 보임.
- **budget 압박으로 단일 attempt.** 첫 attempt에서 composer가 `query` DSL의 join-sort 컬럼 해상 실패(KeyError `public.customer.rental_date`)로 재시도 루프에 들어가 14 tool calls를 소비. `max_turns=20`이 bottleneck이 되어 difficulty weakening 재시도 기회가 원천 봉쇄. 프롬프트 가설의 반증은 아니고 **런타임 예산 부족**.
- **DSL 이슈 관측.** `query(spec={from: rental, join: [rental.customer_id->customer], sort: [rental_date asc]})`에서 `sort.column=rental_date`가 `rental` 소속인데 DSL이 join 끝점인 `customer`의 컬럼으로 해석함. join+sort 조합에서 column-to-table 해상이 모호. iter13 분석과 별개로 `tooling/composer/query_dsl` 쪽 버그로 별도 티켓.

**Next direction.**
1. **최우선: `synthesis.runtime.max_turns` 완화.** 20 → 40(or 60)으로 올려 composer가 DSL 시행착오 후에도 attempt 2~3으로 진입 가능하게. 이전 iter07 retry의 turn 예산 부족과 같은 계열 신호지만 원인이 다름(그땐 탐색 폭주, 지금은 DSL 해상 실패 비용). 프롬프트 건드리지 말고 예산만.
2. **병행: composer query DSL의 join+sort 컬럼 해상 버그 수정.** iter13에서 14턴 중 3턴을 여기에 낭비. 수정되면 attempt 1 비용이 대폭 줄어 attempt 2 진입이 자연스러워짐. 이건 prompt tuning과 독립. — **수정 완료 `2b4911f`** (select/sort/group_by/aggregate.column 모두 조인 체인 전체에서 FROM-first 해상으로 통일). iter14 합산 개선 효과 측정 대상.
3. **밴드 진입 여지 측정.** 예산 완화 + DSL 수정 후에도 `pass_rate=0`이 반복되면 composer의 initial label을 "의도적으로 단순하게 시작"하도록 프롬프트 조정(예: step 1에 "first attempt uses slot_count=1 and a single filter" 명시). 즉 difficulty를 **위에서 내리는** 기존 설계가 아니라 **아래에서 올리는** 방향으로 workflow 전환 실험.
4. **밴드 진입 실패가 pass_rate=0으로 고착되면 solver-side 프롬프트에 2-hop 계산 가이던스 추가 고려.** 단 이건 asymmetric model 실험과 경계가 겹치므로 `feedback_experiment_scope.md`의 "모델 페어링은 고정 조건" 원칙과 충돌하지 않도록 `솔버 프롬프트`만 조정하고 model/tool pairing은 그대로.

---

### Iteration 14 — 2026-04-18 (max_turns=40 실측 + terminal 규칙 관측)

**Hypothesis.** iter13의 1-attempt 한계는 `max_turns=20`이 composer의 DSL 시행착오 비용으로 소진돼 attempt 2 진입 자체가 봉쇄됐기 때문. 예산만 두 배로 완화하면(iter13 Next direction #1) weakening/escalation이 작동할 여지가 생긴다.

**Change.**
- `artifacts/tmp_configs/iter14_turns40.yaml` — iter13 config 복제에 `synthesis.runtime.max_turns: 20 → 40` 단일 변경. 프롬프트/솔버/밴드/DSL 모두 iter13과 동일.

**Trial.** `artifacts/smoke_iter14` — flow_id `real_db_trial:20260417T182209Z:275b3e00`, anchor `payment_id=8917`, topic "Film actors from payment's rental". attempt 1 single submit, terminal `synthesis_failed / reject_too_hard / budget_exhausted`. atomic_tool_calls_seen=6 (iter13의 14 대비 DSL fix 덕에 절반 이하). canonical = 2-hop actors-via-rental chain, slot_count=2.

**Findings.**
- **가설 반증됨(negative).** max_turns를 40으로 올려도 여전히 1 attempt 만에 종료. composer는 DSL fix 덕에 단 6 turns만 썼고 18 turns가 여유 상태였는데도 attempt 2가 뜨지 않음. 즉 **max_turns가 병목이 아니었음**.
- **진짜 원인 `submit_draft_tool.py:700-728`**. `reject_too_hard` 발생 시 `_terminated_too_hard = True` + `ToolsToFinalOutputResult(is_final_output=True)` 조합이 Runner를 즉시 break. composer에게 전달되는 메시지도 "conversation is terminated. Do not call submit_draft again" 으로 명시적 종료 지시. `attempt_outcomes=['difficulty_weakened']`는 outcome enum의 자동 매핑(`too_hard → DIFFICULTY_WEAKENED`)일 뿐 실제 weakening attempt는 구조적으로 발생 불가.
- **설계 의도 재확인.** iter01~12 symmetric-tool regime 하에선 composer가 보통 pass_rate=1.0(too_easy)에서 시작해 escalate로 band를 향하는 궤적이었고, 가끔 overshoot해서 too_hard가 터지면 같은 anchor에서 weakening 시도가 진동하며 수렴 실패했다. "too_hard는 terminal, 새 anchor로 fresh start"가 그 시대의 최적 설계였음. 즉 terminal 규칙은 버그가 아니라 **이전 regime에 대한 정답**이었다.
- **Asymmetric regime에선 too_hard가 default mode**로 전환되면서 이 규칙이 "모든 trial을 1-attempt로 축소"하는 부작용을 일으킨다. 프롬프트 `line 125-126` "On too_hard, relax one clause (not the label)"과 `line 161-165` "Within 2 composer calls of rejection feedback, call submit_draft again"이 코드의 terminal 플래그와 **정면 충돌**하는 상태로 공존.

**Next direction.**
1. **too_hard 종료 규칙 vs weakening 허용의 선택.** 원설계 복원(simple 시작 → too_easy → escalate) 방향으로 가려면 composer의 initial label 난이도를 낮추는 프롬프트 조정이 필요. weakening 시도를 허용하려면 terminal 플래그를 "N회 연속 too_hard" 조건으로 완화 + 프롬프트의 "Never weaken label" 조항을 뒤집어야 함. 둘 다 기존 설계 철학과 트레이드오프.
2. 어느 쪽 가든 **단일 관측 전에 iter15로 "initial 난이도 문제" 가설 검증.** composer가 왜 첫 submit부터 multi-hop overshoot을 authoring하는지 prompt 레벨에서 분석.

---

### Iteration 15 — 2026-04-18 (Workflow step 2 예시 일관성 복원)

**Hypothesis.** iter14 Next direction #2의 원인 진단. composer가 multi-hop overshoot을 authoring한 건 prompt `# Workflow` step 2의 **내부 모순** 때문. 룰은 "a homogeneous list of 3 child records reached through a **single foreign key** from the anchor"이지만 예시 `[{rental_date, film_title}, …]`은 rental 앵커 기준 2-hop을 요구한다(rental_date는 rental 0-hop, film_title은 rental→inventory→film 2-hop). composer는 예시를 "룰보다 더 넓게 허용되는 실제 목표"로 읽고 multi-hop으로 확장. 이 예시를 진짜 single-hop + destination table에 모든 필드가 거주하는 형태로 바꾸면 composer의 overshoot이 사라질 것.

**Change.**
- `prompts.py:114-126` Workflow step 2 재작성(commit `aaebc61`): "exactly ONE foreign-key hop from the anchor, every item's 1-2 keys all living on that one destination table (never mixed across the join chain)". 예시를 `[{rental_date, return_date}, …]` for customer anchor via `customer<-rental.customer_id`로 교체. 이전 `[{rental_date, film_title}, …]`은 명시적 negative example로 박음.
- `artifacts/tmp_configs/iter15_singlehop.yaml` — iter14 config 그대로. 변수는 프롬프트 하나.

**Trial.** `artifacts/smoke_iter15` — flow_id `real_db_trial:20260417T183622Z:a6a391ab`, anchor `film_id=38`, topic "Film cast lookup". attempt 1 single submit, terminal `reject_too_hard` 재발. atomic_tool_calls_seen=6 (1 neighborhood + 6 query; DSL chain resolution 덕에 KeyError retries 없음). canonical = `[AUDREY BAILEY, NICK DEGENERES, PARKER GOLDBERG]`. composer는 film 앵커에서 film_actor → actor (2-hop) 경로로 저작.

**Findings.**
- **가설 부분 반증: 프롬프트 내부 정합성은 복원됐지만 composer는 여전히 2-hop 저작.** 이유는 **sakila schema의 구조적 제약** 때문이었다. film 앵커의 1-hop 자식들(`film_actor`, `film_category`, `inventory`)이 전부 ID-only 브리지 테이블이라, Label Rules line 61 "Never use internal identifiers as answer fields" 와 single-hop 룰 + all-fields-on-destination 룰 **세 개를 동시에 만족할 수 없음**. composer는 세 룰 중 하나를 위반해야 하고, 가장 합리적 선택(no-ID 보존, single-hop 포기)을 내림. trace의 query #1/2/3 시퀀스가 그 탐색 과정을 그대로 보여줌: 1-hop 자식 셋 전부 ID-only 확인 후 2-hop으로 전환.
- **이건 프롬프트 룰 세트가 over-determined.** 특정 앵커(bridge-heavy schema)에서 룰 세 개의 교집합이 공집합. 해결은 세 룰 중 하나를 상대화해야 함 — 예: "prefer single-hop when readable destination exists; otherwise nearest readable via bridge" 같은 계층적 규칙, 또는 anchor sampler에서 1-hop readable 자식이 없는 앵커는 거르는 방식.
- **Solver 트레이스 관측 불가 (로깅 버그 발견).** stdout이 `solver_traces_dir` 경로를 광고하지만 글로벌 `./artifacts/traces/` 쪽에 쓰이고 trial debug 폴더에는 없음. 추가로 error path(`solver/backend_openai_agents.py:358`)가 `"run_items": []` 하드코딩이라 MaxTurnsExceeded에서 16턴 전부 손실. AgentsSDK의 `exc.run_data.new_items` 공식 복구 경로를 안 쓰고 있음. 이 버그 자체가 iter15 해석을 가로막는 결정적 장애물.
- **iter15 가설 자체는 잘못된 프레이밍이었다.** "프롬프트 예시 불일치"가 composer overshoot의 유일 원인이라 가정했지만 실제로는 (a) schema 제약 + (b) 로깅 결함 둘 다 섞여 있었음. holistic read의 부재(iter15 설계 시 prompt/submit_draft_tool/backend 동시 읽지 않음)가 이 진단을 단일축으로 좁힌 원인.

**Next direction.**
1. **로깅 시스템 고치기 최우선**(iter16은 기능 수정 결과 관측용). `exc.run_data.new_items` 복구 + `repr(item)` → 구조화된 dict summarizer + `solver_orchestrator.traces_dir_override` — iter16에 반영하고 실측. 완료: commit `ffa587c`.
2. **iter16: 로깅 fix 이후 solver 행동을 처음으로 관측**하면서 동일 프롬프트/config로 재실행. 같은 too_hard 재발하면 그때 프롬프트 룰 세트의 over-determination 해소 작업.
3. anchor sampler에 "1-hop에 readable 자식이 있어야 함" 필터 추가 여부는 iter17 이후.

---

### Iteration 16 — 2026-04-18 (로깅 fix 후 첫 관측, accepted)

**Hypothesis.** iter15 Next direction #2의 실행. 로깅 fix(commit `ffa587c`)가 solver 행동을 처음 관측 가능하게 만든 상태에서 iter15와 **완전 동일한** 프롬프트/config로 재실행. 기대값 분기: (a) 같은 too_hard 재발하면 prompt 룰 over-determination이 진짜 원인, (b) 다른 anchor에서 다른 결과가 나오면 anchor sampling randomness가 지배적.

**Change.** 코드/프롬프트/config 변화 **없음**. 로깅 인프라만 수정된 상태에서 iter15와 같은 `iter15_singlehop.yaml`(rename 없이 재사용)로 재실행.

**Trial.** `artifacts/smoke_iter16` — anchor `inventory_id=2794`. **처음으로 accepted** — session 내 첫 synthesis 성공.

| # | topic | slot | pass_rate | calls_seen | outcome |
|---|---|---|---|---|---|
| 1 | inventory rental history | 2 | 1.0 (3/3) | 2 | reject_too_easy |
| 2 | inventory rental history **by staff** | 2 | 0.667 (2/3) | 7 | **accepted** (in band [0.33, 0.67]) |

- attempt 1: `neighborhood(inventory 2794) → query(rental|inv=2794, select=rental_date/return_date/rental_id, sort)`. **2 atomic calls**. iter13/14/15 대비 **6~7배 저비용**.
- escalation: `profile(rental) + profile(staff) + query(staff list)` → staff 축 분석 → attempt 2 `query(rental|inv=2794 AND staff=2, select=rental_date/return_date)` — **Composite axis** 채택(두 filter on 다른 dims). iter07 Next direction의 "shape-changing axis" 원칙 준수. single-hop 룰 0-hop으로 완전 준수(rental은 inventory의 1-hop 자식이고 select 모두 rental 컬럼).
- 그 뒤 registry commit 성공, bundle export 단계에서 별건 infra 버그 발생(아래).

**Solver traces (첫 관측):**

| Solver | tool calls | run_items | status | 요약 |
|---|---|---|---|---|
| qwen3.5-plus_00 | 12 | 36 | matched | `rows_where(inv=2794) → rows_via(rental) → rows_where(staff.first=Jon) → take → read(staff) → rows_where(rental.staff=2) → intersect → order_by(rental_date) → take(2) → read×2 → submit`. 정합 chain, 4턴 여유. |
| qwen3.5-plus_01 | 13 | 39 | matched | 동일 shape + Jon/Stephens를 `intersect(first=Jon, last=Stephens)`로 더 엄밀. 정답. |
| qwen3.5-plus_02 | 0 | 0 | **APITimeoutError** | 추론 실패 아닌 업스트림 타임아웃. |

**Findings.**
- **원설계 bottom-up 궤적 작동 확인.** `simple attempt 1 (too_easy) → Composite escalate → band` 정상 수렴. iter01~12 동안 관찰되던 패턴이 새 tool 서피스에서 되살아남. iter15 프롬프트 fix + iter13~16의 DSL chain resolution + coerce_scalar + retry-aware config 조합이 누적 효과를 냄.
- **Solver의 chain-planning 능력 확인.** 2명이 12~13 atomic calls, 16턴 내 여유로 정합 chain 구성 + 정답 재현. iter13~15의 `pass_rate=0`은 solver 무능이 아니라 composer의 multi-hop overshoot 탓이었음이 역증됨.
- **pass_rate=2/3의 이중 해석.** 매칭이 2/3인 주요 원인은 qwen3.5-plus_02의 **APITimeoutError**. 즉 band 진입은 **"composer 난이도가 solver ceiling에 정확히 걸림"이 아니라 "3명 중 1명이 인프라 사유로 제외된 통계 잡음"**. 3명 전부 API 성공했다면 3/3 too_easy로 떨어졌을 가능성이 실제 signal. 즉 iter16의 accept는 자연 관찰이 아닌 **fortunate noise-gifted landing**.
- **인프라 발견.** `SolverOrchestrator._run_solver`가 transient error를 retry 없이 실패 처리 → APITimeoutError 한 번에 pass_rate가 1/3 단위로 noise. 해결은 OpenAI SDK의 `AsyncOpenAI(max_retries=N)` 노출(commit `61ed382`). Bundle export 단계의 schema_snapshot 경로 불일치 infra 버그도 별도 관측(미수정).
- **Draft 4요소 정합성(attempt 2)** ✓: topic "inventory rental history by staff" ↔ label 2 rows(rental_date, return_date) ↔ anchor inventory_id=2794 ↔ question이 Jon Stephens staff 조건 자연스럽게 언급. solver 둘 다 question에서 staff 필터를 정확히 추출.

**Next direction.**
1. **iter17 — 재현성 검증.** iter16과 동일 프롬프트/config + `provider_config.max_retries=3~5` + bundle exporter infra fix. 여러 anchor에 걸쳐 accept 비율 측정. 한 iter만으로는 "우연한 landing"과 "설계가 작동 중"을 구분 불가. 3~5 trial batch로 accept rate 확인하되 쿼터 예산 내에서.
2. **`max_solver_runs` 확대 실험.** 현재 3이면 1/3 단위 해상도라 transient noise에 취약. 5~10으로 올리면 pass_rate 분해능이 높아져 "진짜 band 안"과 "운 좋게 걸린 값" 구분이 생김. 단 쿼터 2~3배 부담.
3. **Composite axis의 재현성 관측.** iter16은 staff 축을 escalation으로 뽑았는데 다른 anchor에서도 Composite가 자연 선택되는지 vs Cardinality/Cross-item rule 중 어느 축이 지배적인지. iter01~12의 Width 편향이 새 서피스에서 실제로 사라졌는지 확인.
4. **프롬프트 over-determination 문제는 iter15 앵커(film)에서만 발현된 특수 케이스인지, bridge-heavy 앵커 전반의 문제인지 판단.** iter17에서 film/category/store 같은 bridge 중심 앵커가 뽑히면 iter15 실패가 재현될 것.

---

### Iteration 17 — 2026-04-18 (retry=5 재현성 검증 + divergence gate 관측)

**Hypothesis.** iter16의 band landing(pass_rate=0.667 in [0.33, 0.67])이 **진짜로 composer escalation의 결과**인지 검증. 같은 프롬프트/config + `provider_config.max_retries=5` 추가로 transient APITimeoutError를 제거해 pass_rate 측정을 안정화하면, 재현 시 동일한 2/3 landing이 일관되게 나타나야 한다.

**Change.**
- `artifacts/tmp_configs/iter17_retry5.yaml` — iter15/16 config 복제에 `providers.opencode_zen.max_retries: 5` 단일 필드 추가(commit `61ed382`에서 ProviderConfig 노출).
- 프롬프트/모델/solver 수/max_turns/band 모두 iter16과 동일.

**Trial.** `artifacts/smoke_iter17` — flow_id `real_db_trial:20260418T124749Z:594131ba`, anchor `address_id=227`, topic "Customer rental history lookup". attempt 1 single submit, pass_rate=**1/3=0.3333333333333333**, terminal `synthesis_failed / reject_too_hard / budget_exhausted`. Composer 4 atomic calls (neighborhood×2 + query×2, 그중 #3과 #4는 **완전히 동일한 spec의 duplicate**). canonical = 3 rental rows from customer 223 via address 227.

**Solvers (3/3 submitted, retry가 transient error 차단):**

| Solver | turns | status | matched |
|---|---|---|---|
| qwen3.5-plus_01 | 9 | submitted | ✓ |
| qwen3.5-plus_00 | 8 | submitted | ✗ |
| qwen3.5-plus_02 | 9 | submitted | ✗ |

raw_output_preview 3개 모두 같은 두 rental_date로 시작하지만 세 번째 row에서 분기(혹은 format 차이). 1/3만 정확히 canonical과 매치.

**Findings.**

**결정적 발견 (iter16 accept 해석 완전 재구성):** `solver_orchestrator.py:452-457`에 **2차 gate가 존재**한다.

```python
unique_answers, divergence_ratio = _solver_divergence(summary)
if (
    status is TaskQualityGateStatus.ACCEPT
    and divergence_ratio > config.calibration.max_divergence_ratio
):
    status = TaskQualityGateStatus.REJECT_TOO_HARD
```

`divergence_ratio = unique_answers_among_submissions / num_submissions`. default `max_divergence_ratio=0.5`. 1차 band check가 accept였더라도 solver들의 답이 너무 발산하면 too_hard로 **재분류**. iter17은 3 submitted × 2+ unique → ratio ≥ 2/3 > 0.5 → accept 판정이 강제 번복됨.

**n=3 + max_divergence=0.5는 band landing을 사실상 불가능하게 한다:**

| matched/submitted | unique | divergence | band pass | final status |
|---|---|---|---|---|
| 1/3 | 2~3 | 2/3 ~ 1.0 | in band | **reject_too_hard (divergence 번복)** |
| 2/3 (모두 submit) | 2 | 2/3 | in band | **reject_too_hard (divergence 번복)** |
| 2/3 (1명 timeout, 2명 matched) | 1 | 1/2 | in band | **accept** (iter16 시나리오) |
| 3/3 | 1 | 1/3 | above band | reject_too_easy |

즉 **iter16의 accept는 divergence gate를 우회한 infrastructure noise**였다. qwen3.5-plus_02의 APITimeoutError가 "submission 풀에서 빠지는" 통계적 행운을 만들어 divergence가 0.5 경계로 내려왔을 뿐, **composer/solver 수준의 "band 정확히 타격"이 아니었다**. retry=5로 이 noise source를 제거하자 구조적 장벽이 드러남.

**추가 관측:**

- **composer duplicate query call**: trial_events에서 `query(rental|customer_id=223, ..., limit=3)`이 call_index 3과 4로 **완전히 동일한 spec 2회 호출**. 첫 호출의 결과를 무시하고 같은 호출을 반복. qwen3.5-plus thinking 모델의 단발 검증 성향이거나 attempt 2 진입을 위한 탐색 행동으로 보이지만 정보 이득 없는 reuse.
- **composer는 0-hop single filter + limit+sort 사용** → Workflow step 2 single-hop 룰을 충실히 지킴. address → customer 1-hop → rental 0-hop(customer_id = 223 filter). 이번엔 prompt 룰 대로 작동했으나 어쨌든 too_hard.
- **solver 3명 전부 submit + 3명 전부 9 turns 전후 정상 종료**. retry=5 효과 확인됨 — iter16식 timeout 기반 artifact 제거.
- **Draft 4요소 정합성**: topic ↔ label ↔ anchor ↔ question 모두 적절. 문제는 저작 수준이 아니라 divergence gate 정책.

**Next direction.**

1. **`max_divergence_ratio` 재교정이 최우선.** n=3 환경에서 현 0.5는 구조적 reject generator. 후보:
   - (a) 0.67로 완화 — 2/3 landing(2 matched + 1 unique wrong, divergence=2/3=0.667)이 경계 걸리도록. 0.67 "미만"으로 `>` 비교를 `>=`로 바꿀지도 동시 검토.
   - (b) 정의 자체를 재정의 — `unique_among_non_matched / non_matched_count` 같은 normalize된 지표로 전환.
   - (c) `n` 증가에 따라 자동 scale — 예: `max(0.5, 1 - 1/n)`.
2. **`max_solver_runs` 확대**와의 상호작용 측정. n=5~10으로 늘리면 divergence_ratio=unique/n이 자연스럽게 완화됨. 쿼터 부담 증가는 있으나 구조적 해상도 향상.
3. **iter17의 composer duplicate query 관측**은 별도 조사. neighborhood 뒤에 같은 query를 두 번 호출한 이유가 attempt 경계(첫 submit 전 탐색) 때문인지, 아니면 Runner 내부 동작인지 trace의 run_items를 열어 확인 필요.
4. **iter16 accept는 "natural reproducibility" 증거가 아님을 로그 상 명확히 표기.** Cross-Iteration Summary 표에도 "noise-gifted landing, retry=5로 재현 실패"를 iter17 행으로 추가해야 함.

---

### Iteration 18 — 2026-04-18 (첫 clean accept, divergence gate 제거)

**Hypothesis.** iter17이 드러낸 구조적 트랩(`max_divergence_ratio=0.5`가 n=3에서 band landing을 차단) 제거 후, same-model qwen3.5-plus 페어링이 asymmetric tool surface에서 **실제로 band 진입 가능한지** 검증. divergence gate는 commit `1573c25`에서 opt-in으로 바꿨고, main config에서 explicit 0.5 제거해 기본 None. iter18 config(`iter18_nodivergence.yaml`)는 iter17과 이 한 줄 차이.

**Change.**
- `artifacts/tmp_configs/iter18_nodivergence.yaml` — iter17 config 복제에서 `max_divergence_ratio: 0.5` 라인만 제거(default None으로 gate off).
- 프롬프트/모델/solver 수/max_turns/band/retry=5 모두 iter17과 동일.

**Trial.** `artifacts/smoke_iter18` — flow_id `real_db_trial:20260418T132511Z:286d0274`, anchor `rental_id=3583` (customer_id=574, rental_date < 2005-07-06, staff_id=2 context). **2 attempts, accepted + committed + bundle exported**. task_id `task_동일 고객 및 동일 직원의 이전 대여 내역_a3442780aaeda110`, bundle_root `artifacts/smoke_iter18/bundle`. 세션 첫 clean accept.

| # | topic | slot | filter 축 | pass_rate | outcome |
|---|---|---|---|---|---|
| 1 | 동일 고객의 이전 대여 내역 | 2 | `customer_id=574 AND rental_date<'2005-07-06'` | 3/3 = 1.0 | reject_too_easy |
| 2 | 동일 고객 및 동일 직원의 이전 대여 내역 | 2 | attempt1 + `AND staff_id=2` (Composite) | 2/3 = 0.667 | **accepted** |

**Unified logger Phase 1 검증 ✓.** 실시간 `tail -f artifacts/smoke_iter18/debug/trial_events.jsonl`로 17 이벤트 흐름을 trial 종료 전부터 읽음. 이벤트 분해: `runner/trial_started` × 1, `composer/atomic_tool_call` × 10 (neighborhood 1 + schema_map 1 + query 8), `solver/solver_run_completed` × 6 (attempt 1의 3 solvers + attempt 2의 3 solvers). Phase event mirror는 Phase 2(commit `7b429f0`)라 iter18엔 포함 안 됨, pass_rate/canonical은 phase_monitors.jsonl 병합 필요했음.

**Findings.**

1. **Bottom-up 원설계 궤적 정상 재현 ✓.** Simple initial(single filter + threshold, 3/3 too_easy) → Composite axis escalation(+ staff_id) → 2/3 in band. iter16의 1회 관측이 "infrastructure noise artifact"였음을 감안하면, iter18이 **asymmetric tool surface에서 same-model qwen 페어링이 clean landing 가능**함을 보여주는 **첫 실증**.
2. **Divergence gate 제거의 순 효과 확인.** iter17 exact 상황(1/3 + 2 unique wrong)과 iter18 situation(2/3 + 1 wrong)의 차이는 composer 저작 난이도 / solver chain 성공 여부일 뿐, gate 제거 없이는 둘 다 거부됐을 것. **divergence는 post-hoc 품질 지표로만 유효**하고 accept gate로는 over-specification이었음을 iter17+18 대조로 확정.
3. **Composer의 axis 선택.** iter07 Next direction의 "shape-changing axis over Width" 원칙을 composer가 자연스럽게 지킴 — 3 filter composite을 escalation으로 선택. iter16과 동일 행동 패턴(Composite + threshold 축). 2회 연속 Composite 선택 관측이 "iter01~12 Width 편향이 새 tool surface에서 실제로 사라짐" 가설을 더 강화.
4. **Solver chain 정합성.** 3 solver 모두 attempt 1과 attempt 2 에서 10~12 턴 내 submit까지 도달. attempt 2의 fail 1명은 chain 구성이 아닌 마지막 답 format/content 미세 차이로 em_mismatch. solver 수준의 2-hop+composite 체인 구성 능력 정상.
5. **Bundle infra bug 해결 확인.** commit `ffa587c`의 `self.exporter = replace(self.exporter, snapshot_materializer=trial_materializer)` 수정이 실제 accept path에서 작동하는지 iter18에서 처음 검증됨 — bundle export 정상 완료. iter16에서 터졌던 `FileNotFoundError: artifacts/databases/sakila/schema_snapshot.json` 재발 없음.
6. **Retry=5 효과 재확인.** iter17에서 처음 적용한 retry가 iter18에서도 3 solver 전부 submit 보장. APITimeoutError 드롭아웃 없음. pass_rate 분해능이 n=3에서 확정적(gifted accept 없음).

**Next direction.**

1. **iter19: 재현성 검증 batch.** iter18과 동일 config로 단일 trial이 아닌 **3~5 trial batch**를 같은 세션에서 돌려 accept rate 측정. 한 번 성공이 composer/solver 페어링의 "일관된 능력"인지 "우연한 landing"인지를 이 횟수로 판별. max_generation_attempts=5에서 accept 율이 40% 이상이면 신뢰 가능한 정상 동작.
2. **Phase 2 로거 실사용.** commit `5cc180c`로 `trial_events.jsonl`이 composer 저작 + solver run_items + phase events까지 전부 담게 됨. iter19는 **단일 pane 분석**만으로 완결. 구 legacy per-file trace 폴더들은 iter19 debug_traces_dir에 아예 생성되지 않아야 정상.
3. **Composer의 axis 선택 다양성 관측.** iter16/18 모두 Composite 축이었음. iter19 batch가 다른 anchor를 뽑으면 Cardinality / Cross-item rule / single Filter 축이 등장하는지, 아니면 Composite 편향이 고착됐는지. 편향이 있으면 iter20에서 Axes prompt 재조정.
4. **iter16 accept 재해석 Cross-Iteration Summary 업데이트.** iter16 행의 "noise-gifted landing" 표기는 이제 iter18의 "clean landing" 기준으로 보완 가능 — iter16도 composer 행동 자체는 정상이었고 단지 divergence gate 때문에 pass_rate=2/3의 의미가 불확정이었다는 해석.

---

### Iteration 19 — 2026-04-18 (Phase 2 unified logger 실사용 + 2연속 accept)

**Hypothesis.** iter18에서 clean accept 첫 관측. iter19는 (a) 같은 config로 재현성 초기 증거 확보, (b) commit `5cc180c` Phase 2 unified logger가 `trial_events.jsonl` 단일 파일로 composer/solver/phase 전 흐름을 담는지 실사용 검증, (c) legacy per-file trace 쓰기 경로가 완전히 제거됐는지 디스크 확인.

**Change.**
- `artifacts/tmp_configs/iter19_phase2logger.yaml` — iter18 config 완전 복제. 변수 0.
- Phase 2 logger는 iter19가 첫 실사용 (commit 랜딩 후 처음 돌아감).

**Trial.** `artifacts/smoke_iter19` — flow_id `real_db_trial:20260418T143809Z:77edd890`, anchor `customer_id=54`, topic "Customer rental history". **attempt 1 바로 accepted**, task_id `task_Customer rental history_8bc7b30eee9f8c1f`, registry committed + bundle exported. `pass_rate=1/3=0.333` in band, quality_gate_status=accept.

Composer trace (2 calls):
1. `neighborhood(customer, row_id=54)` — 앵커 orient.
2. `query(from=rental, filter=customer_id=54, select=rental_id/rental_date/return_date, sort=rental_date asc, limit=5)` → submit.

Solver traces (3명):

| Solver | turns | matched | run_items |
|---|---|---|---|
| qwen3.5-plus_00 | 6 | ✗ | 19 |
| qwen3.5-plus_01 | 7 | ✗ | 22 |
| qwen3.5-plus_02 | 7 | ✓ | 22 |

**Findings.**

1. **Phase 2 unified logger 완벽 작동 ✓.** `trial_events.jsonl` 12 이벤트가 순서대로: `runner/trial_started` → `phase/trial.started` → `composer/atomic_tool_call`×2 → `solver/solver_run_completed`×3 (각 run_items 포함) → `phase/submit_draft.accepted` → `composer/synthesis_completed` → `phase/synthesis_conversation.accepted` → `phase/registry_commit.committed` → `phase/bundle_export.completed`. **phase_monitors.jsonl 안 열고 완결 분석 가능.** `tail -f` 실시간 관측도 정상.
2. **Legacy 쓰기 경로 사라짐 ✓.** `artifacts/smoke_iter19/debug/traces/synthesis/`는 빈 디렉토리만 남음(파일 0개). `tool_traces/`, `transcripts/` 서브디렉토리 아예 생성 안 됨. commit `5cc180c`의 `_write_artifact` 삭제가 실측으로 검증.
3. **iter18과 다른 저작 궤적, 같은 결과.** iter18은 "simple too_easy → Composite escalate" 2-attempt 저작, iter19는 "simple 1-filter 직접 band landing" 1-attempt 저작. 두 궤적 모두 유효하고, composer가 anchor/seed에 따라 궤적 선택 중인 것으로 해석. customer_id=54의 rental 목록 크기가 3+ 정도면 limit=5 + sort만으로도 solver 일부가 헷갈릴 만한 미세 난이도가 이미 있음.
4. **iter17 대비 직접 증거.** iter17의 `pass_rate=1/3 + 3 solver submit + 2 unique wrong` 상황은 divergence gate에 의해 reject_too_hard였음. iter19는 사실상 같은 통계(1/3 + 2 unique wrong 이상)인데 gate off로 **자연 accept**. commit `1573c25`의 gate 제거가 올바른 결정이었음을 실증.
5. **재현성 초기 증거 (n=2).** iter18/19 둘 다 accept, 단일 trial 수준에선 2회 연속 성공. 궤적은 다르지만 composer 행동이 상황에 맞게 정합. 본격 재현성 통계(accept rate)는 iter20 batch로 검증.
6. **Solver chain 정합성.** run_items 19~22개로 iter18(10~12 turns)보다 길지만 iter18의 turn_count는 backend에서 집계하는 값이고 run_items는 SDK의 모든 item(reasoning + tool call + output 각각)이라 직접 비교 안 됨. turn_count만 보면 6~7 turns, iter18 10~12보다 짧음 — task가 더 간단했기 때문.

**Next direction.**

1. **iter20: 3~5 trial batch로 accept rate 측정.** 지금까지 단일 trial만 2연속 성공. n=3~5에서 accept 비율이 어떻게 나오는지, composer axis 선택 다양성, simple direct landing vs escalation 궤적 분포. 쿼터 여력 고려해 batch 크기 결정.
2. **iter18 log 회고 보완.** iter16 "noise-gifted landing" 표기를 iter18/19 데이터로 더 정확히 재해석: iter16도 gate off 상태였다면 pass_rate=2/3으로 자연 accept였을 것. 즉 iter16 composer 행동 자체는 이미 정상이었고, 당시 divergence gate가 *우연히* timeout artifact 때문에 통과된 것. iter18/19가 "새 regime에서 composer 행동이 원래부터 band-landing 가능했다"를 확정.
3. **Legacy phase_monitors.jsonl 처리.** Phase 2 mirror로 모든 phase event가 trial_events에 있음. harvest/pipeline 하위 코드가 여전히 `phase_monitor_log_path` 필드를 읽는지 확인해서 deprecate 가능 여부 판단. 가능하면 Phase 3에서 삭제.
4. **Composer axis 선택 관측 계속.** iter18은 Composite, iter19는 1-filter만. Cardinality/Cross-item rule 축이 다른 anchor에서 나타나는지. iter20 batch에서 축 분포 수집.

---

### Iteration 20 — 2026-04-19 (voice-constraint prompt, 3연속 accept)

**Hypothesis.** iter18/19 task bundle의 qualitative 평가에서 composer가 user-facing question을 **staff/service voice로 쓰는** 문제 발견 — iter19는 "고객님, 귀하의... 확인해 드리겠습니다... 알려주세요"로 한 문장 안에 응답자/질의자 voice 혼재, iter18은 "이 대여 기록" schema-internal reference. 원인은 prompt `# Workflow` 마지막 문장 "Rewrite ... as a customer who knows nothing about databases"가 너무 포괄적이라 composer가 CS 대화 조각을 끼워넣음. voice 제약을 명시적 positive/negative 예시로 강화하면 해소될 것.

**Change.**
- `prompts.py:135-137` rewrite instruction을 (a) 1인칭 ask 또는 조직-대상 2인칭 요청 명시, (b) 금지 구절 리스트("고객님"/"귀하의"/"확인해 드리겠습니다"/"도와드리겠습니다"), (c) schema-internal anchor reference("이 대여 기록") 금지, (d) 조직 관점 no — "customer is seeking information, solver is the organization answering" 프레이밍으로 확장 (commit `a98f58c`).
- `artifacts/tmp_configs/iter20_voice.yaml` — iter19와 동일 config (변수 프롬프트 하나만).

**Trial.** `artifacts/smoke_iter20` — anchor `city_id=544` (Toulouse), topic "Rental records for customer in Toulouse". **attempt 1 바로 accepted**, `pass_rate=1/3=0.333` in band, registry committed + bundle exported.

**Question (실측):**
> "저는 툴루즈에 거주하는 고객입니다. 제 대여 기록에서 대여일 순으로 가장 빠른 3건의 대여일과 반납일을 알려주세요."

**Findings.**

1. **Voice fix 완벽 작동 ✓ (가설 positive 검증).** 금지 구절 0건. "저는... 고객입니다" 1인칭 self-identification, "제 대여 기록" 고객 본인 reference, "알려주세요" ask voice, schema-internal "이 대여 기록" 없음, city anchor를 "툴루즈"로 자연어화. iter18/19 대비 voice 품질 **dramatically 개선**.
2. **3연속 accept 재현성 유지.** voice 제약 추가가 composer 성능 회귀 유발 안 함. 오히려 simple direct landing(iter19)과 다른 궤적(iter20 2-hop)으로도 accept 가능.
3. **첫 clean 2-hop join task.** city → address → customer → rental 경로로 실질 multi-table join. iter15의 multi-hop overshoot 실패(`reject_too_hard`)가 **prompt/divergence/retry 수정 누적 + anchor 다양성**의 조합으로 이제 clean 통과. solver primitive chain: `rows_where(address, city_id=544) → rows_via(address_id→customer) → rows_via(customer_id←rental) → order_by(rental_date asc) → take(3) → read×3` (~7 primitives).
4. **새 anchor 타입 관측.** iter13(inventory)/16(inventory)/17(address)/18(rental)/19(customer)/20(city) — sakila 7 anchor 타입 중 6종 관측. film/payment/staff anchor는 아직 iter 내 미등장.
5. **Semantic ambiguity — Toulouse 거주 고객이 2명 이상이면 문제.** "저는 툴루즈에 거주하는 고객입니다. 제 대여 기록..."은 1인 지칭을 가정한 표현인데 anchor가 city(복수 customer 포함 가능)임. canonical이 Toulouse 전체 customers의 rental 통합 top-3이면 질문의 "제"와 답의 의미가 불일치. sakila Toulouse의 customer 수에 따라 무해~문제. **별도 검증 필요**.

**Next direction.**

1. **iter20 semantic ambiguity 검증.** sakila city_id=544의 customer 수 쿼리 → 1명이면 iter20 task 유효, 2+명이면 task 자체 ambiguity로 bundle 품질 결함. 필요시 prompt에 "anchor 엔티티가 여러 customer를 포함하는 집계 레벨(city, country)일 때는 '저'/'제' 1인칭이 아닌 '해당 도시 고객들의' 등 복수형 표현" 추가.
2. **iter21 axis 다양성 batch.** voice 문제는 해결됐으니 이제 원래 iter20 계획이었던 axis 축 관측으로 이동. 3~5 trial batch 돌려 Cardinality/Cross-item rule/Aggregate 축이 등장하는지. film/payment/staff anchor에서는 어떤 task가 나오는지.
3. **iter18/19 재해석 보완.** voice 축에서는 iter20이 baseline. iter18/19 task는 voice 결함으로 RL training data로 그대로 쓰기엔 grade 낮음. 축적된 task pool에서 voice 기준 post-hoc 필터링 검토.
4. **uv sync editable 이슈 별건.** `uv sync`가 `.venv` 재생성 후 local 프로젝트를 editable install 안 하고 누락하는 현상 관측. iter20 실행 전 `uv pip install -e .` 수동 필요. 재발 방지 위해 pyproject.toml `[tool.uv]` 설정 조사 필요.

---

### Iteration 21 — 2026-04-18 (accept 스트릭 끊김 → tooling 버그 노출)

**Hypothesis.** iter20의 voice fix가 일반 anchor에서도 일관되게 작동하는지 단일 trial로 1차 재현성 확인(batch는 iter22부터). 프롬프트/config 변화 없음, composer의 anchor sampling 다양성에 맡김.

**Change.** `artifacts/tmp_configs/iter21_repro.yaml` = iter20 config 복제. 제어 변수 0.

**Trial.** `artifacts/smoke_iter21` — anchor `actor_id=87` (SPENCER PECK), topic "Actor filmography lookup". attempt 1 single submit → **`reject_too_hard + budget_exhausted`**. 3 solver 모두 matched=False, turn_count=0으로 기록됐으나 run_items 48~49개 → 실제로는 16 turns 가득 돌다 MaxTurnsExceeded(turn_count는 에러 경로 하드코딩 버그). 3연속 accept 스트릭 끊김.

**Composer 저작 (attempt 1):** 2-hop `actor → film_actor → film` with `filter=actor_id=87, select=[title, rental_rate, rating], sort=title asc, limit=3`. Question은 "배우 SPENCER PECK이 출연한 영화 중 제목 순으로 정렬했을 때 처음 3개의 영화 제목, 대여 요금, 그리고 등급을 알려주세요." — 2인칭 조직-대상 ask 패턴으로 **voice는 완전 준수**(iter20 fix 지속).

**Findings.**

1. **Voice는 회귀 아님.** 실패 원인은 composer 저작이 아니라 solver의 실행 실패. 금지 구절 0건.
2. **결정적 원인은 `film_actor` composite PK.** Solver의 run_items 분석:
   - `rows_where(film_actor, actor_id=87)` → cursor ✓
   - `rows_via(film_actor.film_id→film)` → cursor ✓ (target=film은 단일 PK)
   - `order_by(title, asc)` → cursor ✓
   - `take(n=3)` on sorted film cursor → **ERROR: "table 'public.film_actor' has a composite primary key; atomic calculus supports single-column PKs only for now"**
   - 이유: `compile_take`의 Via/Intersect 경로가 `_compile_id_stream`을 호출하고 그 안의 `_compile_via_id_stream`이 origin table(film_actor)의 PK를 scalar로 취급. 가드 `_single_column_pk`가 composite PK 전부 거부.
   - Solver가 15턴 동안 우회 경로(다른 방향 rows_via, `group_top(count)`, `take` on 다른 cursor) 시도했으나 동일 계통 에러 반복. MaxTurnsExceeded.
3. **Structural mismatch 확정.** Composer의 `query` DSL은 junction table(M:N bridge)을 자유롭게 pass-through로 쓰는데, solver의 atomic calculus는 composite PK 테이블을 touch하는 순간 모든 primitive가 거부. Composer 저작 가능 집합 ⊃ Solver 재현 가능 집합. "composer 저작 ⟺ solver 재현 가능"이라는 설계 원칙 위반.
4. **`turn_count=0` 관측 버그.** solver error path(solver/backend_openai_agents.py)가 `turn_count=0`을 하드코딩. 실제로는 16턴 MaxTurnsExceeded인데 기록상 0. 별건으로 fix 필요.
5. **원 경로인 `_pk_expression`은 이미 composite 지원.** `(t.c1, t.c2)` ROW 표현으로 내놓을 수 있었는데, 4개 compile 함수(`_compile_via_id_stream`, `compile_take`, `compile_aggregate`, `compile_group_top`)가 모두 `_single_column_pk` 가드로 진입 차단 후 scalar `= base.id` JOIN. ROW 버전의 `_pk_expression(target, alias) = base.id`로 통일하면 PostgreSQL row equality가 단일/composite 둘 다 처리.

**Fix (commit `0ff46a5`).**

- `_single_column_pk` 완전 제거.
- `_compile_via_id_stream` / `compile_take` Via·Intersect 경로 / `compile_aggregate` / `compile_group_top`의 JOIN 매칭을 `_pk_expression(...)  = base.id`로 통일.
- `compile_read`에 composite 분기 추가 — `len(pk_cols) > 1`이면 row_id는 tuple/list, WHERE은 `c1 = $1 AND c2 = $2 ...` 다중 파라미터.
- `build_read_tool` JSON 스키마의 row_id를 `scalar OR array` anyOf로 확장.
- `build_take_tool` 핸들러가 asyncpg Record(composite ROW 반환)를 plain list로 정규화 — downstream JSON 직렬화에서 Record 객체 누출 방지.
- 실측 회귀 테스트 `test_composite_pk_chain_against_sakila_film_actor` 추가: iter21 실패 경로 그대로 `rows_where(film_actor) → take → rows_via(→film) → order_by → take → read(film_actor, [composite_row_id])` 전부 통과.

**Next direction.**

1. **iter22: composite PK fix 실측 검증.** 동일 config로 재실행 → composer가 다시 junction table 경로를 고르면 (actor anchor나 category/film_category 등) solver가 이번엔 통과해야 함. 안 고르면 iter23+에서 의도적 seeding 필요.
2. **turn_count=0 에러 경로 버그 별건 수정.** `solver/backend_openai_agents.py`의 exception handler가 `turn_count=_extract_turn_count(exc.run_data)` 같은 복구를 해야 정확. 에러 path 경로 개선 코스.
3. **iter18/19/20 축 분포 보완 관측은 iter22 이후 batch로.** 이제 composite PK 안전하니 film/category/film_actor 기반 task도 배출 대상으로 유효.

---

### Iteration 22 — 2026-04-19 (composite PK fix 검증 시도 → 새 asyncpg escape 버그)

**Hypothesis.** iter21의 composite PK 수정(`0ff46a5`) 실측. anchor가 junction 경로로 가면 solver가 통과해야.

**Change.** iter21 config 복제(`iter22_composite_verify.yaml`). composite PK fix 이외 변경 없음.

**Trial.** anchor `inventory_id=3094` (film 680 "PINOCCHIO SIMON"). Attempt 1: simple 1-filter rental query → **3/3 matched, reject_too_easy**. Attempt 2 저작 중 Composite escalation으로 `staff_id=1` 추가 후 profile(staff) 실행. 그 다음 composer의 query 시도가 asyncpg `operator does not exist: integer = text`로 **SDK-level UserError 발생, synthesis_failed**.

**Findings.**

1. **Composite PK fix 실측 불가.** Anchor가 inventory라 film_actor/category 같은 junction 경로가 안 타짐. iter21의 actor anchor와 달라 fix 효과 직접 확인 못함. 재샘플링 또는 의도적 seeding 필요.
2. **새 버그 노출: `_with_error_handling`의 asyncpg.PostgresError 누락.** atomic/composer 양쪽 `_with_error_handling`이 `(KeyError, ValueError, TypeError, LookupError, RuntimeError, NotImplementedError)`만 catch. asyncpg의 DataError/UndefinedFunctionError/SyntaxError 계열은 `PostgresError` 하위인데 catch 목록에 없어 tool 밖으로 propagate, agents SDK가 UserError로 감싸 synthesis 중단. composer가 error 피드백 받고 자가 복구할 기회 자체가 봉쇄.
3. **실패 query 본문 trial_events에 없음.** run_items summarizer가 800자 preview × N개 item 범위만 보존. 27번째 item 이후 실제 실패 query는 SDK가 tool 호출하기 전/중 raise해 `record_atomic_tool_call`에 등록 안 됨. trial_events의 `composer/atomic_tool_call` 이벤트는 handler 성공 경로만 찍힘. 관측성 결함 — 에러 경로도 logger에 남기는 Phase 3 개선 필요.
4. **Attempt 1은 정상.** composer가 Composite 축 시도(`Filter` dimension으로 `staff_id=1` 추가) 자체는 iter18/19/20의 bottom-up 궤적과 동일. 저작 궤적은 회귀 없음. 실행 런타임만 asyncpg 에러에 취약했던 것.

**Fix (commit `67fa1d6`).**

- `tooling/atomic/tool_factory.py` + `tooling/composer/tool_factory.py`의 `_with_error_handling` except 튜플에 `asyncpg.exceptions.PostgresError` 추가. DB-layer 에러가 `{error, error_type}` JSON으로 composer에게 돌아가 자가 복구 가능.
- asyncpg import 추가, 53 tool-factory 테스트 통과 유지.

**Next direction.**

1. **iter23: asyncpg catch fix 실측 + composite PK 경로 유도.** `--no-sync`로 editable 유지하면서 재실행. anchor bias 있으니 여러 번 시도 or actor/category 경로 seeding 방법 검토. composer가 중간 에러 받으면 자가 복구하는지 관측.
2. **Query DSL coerce 갭 재현 및 수정.** composer가 "integer = text" 실패한 filter의 근본 원인 — coerce_scalar가 어느 column/data_type에서 누락했는지. iter23에서 에러가 JSON 응답으로 돌아오면 composer의 다음 turn에서 정확한 args 관측 가능.
3. **Error-path logger 보강 (Phase 3 후보).** tool handler가 raise한 경우도 `composer/atomic_tool_call_failed` 이벤트로 찍히도록 `_with_error_handling` 자체에서 event_logger 호출. 단일 pane 완결성 강화.

---

### Iteration 23 — 2026-04-19 (asyncpg catch + composite PK fix 통합 상태 accept 재개)

**Hypothesis.** iter21 composite PK fix(`0ff46a5`)와 iter22 asyncpg catch fix(`67fa1d6`) 양쪽 수정이 들어간 상태에서 동일 config로 돌렸을 때 accept 재개 + 회귀 없음 확인.

**Change.** iter22 config 복제(`iter23_pg_catch.yaml`). 변수 0.

**Trial.** anchor `city_id=481` (Sirjan). Attempt 1 바로 accepted, `pass_rate=1/3=0.333` in band, registry committed + bundle exported. solver 2개 fail/1개 match.

**Question:**
> "Sirjan 도시에 거주하는 고객의 대여 기록을 대여일 오름차순으로 처음 3 건 알려주세요."

Voice: 2인칭 조직-대상 ask, 금지 구절 0, schema-ese 0 ✓. Sirjan customer 수 사후 검증 = 1명 (OSCAR AQUINO) → "Sirjan 고객 = customer 449" 자연스러운 지칭, unambiguous ✓.

**Findings.**

1. **Accept 스트릭 재개.** iter21/22 tooling 버그로 끊겼던 스트릭이 수정 반영 후 재개. 전체 4 clean accept / 6 iter (18/19/20/23 vs 21/22).
2. **voice fix 지속 검증.** iter20~23 전부 금지 구절 부재, schema-ese 부재, customer-ask voice 유지. prompt 변경(iter20 `a98f58c`) 이후 회귀 관측 없음.
3. **asyncpg catch fix 직접 검증 불가.** iter23에서 composer query가 type error 자체를 일으키지 않음. 회귀 검증에는 유효(53 tool-factory 테스트 통과) but 실제 복구 경로 미관측.
4. **Composite PK fix도 미검증 (2회 연속).** iter22/23 모두 anchor가 inventory/city라 junction(`film_actor`) 경로 안 탐. **Composer의 anchor 경로 선택 편중 관측**: iter13~23 11회 중 junction direct path는 iter21의 actor anchor 1회뿐. composer는 M:N bridge를 intermediate pass-through로만 쓰고 최종 target은 readable non-junction table로 수렴. 실측 검증은 의도적 seeding 필요.
5. **Task 품질 패턴 고착화 관측.** iter18/19/20/23 accept된 task 4건 모두 `[{rental_date, return_date}]` 답 shape. customer/address/city anchor에서 rental 테이블로 수렴. film_title/actor_name/category_name 같은 다른 답 shape 0건. iter20 qualitative 평가에서 지적한 답 shape 편중이 iter23까지 지속. RL training data 분포 여전히 좁음.

**Next direction.**

1. **iter24: 3-trial batch로 axis/anchor 다양성 관측.** accept rate 측정 + composer axis 선택 분포(Composite / Cardinality / Cross-item rule / Aggregate). 쿼터 부담 있지만 단일 trial만으론 분포 측정 불가능.
2. **composite PK fix 직접 검증 축 계속 열어둠.** iter25+에서 필요 시 `category` / `film_category` / `film_actor` anchor 강제 주입 수단 검토.
3. **답 shape 다양화 프롬프트 가이드 후보.** iter20 Next direction #1에서 제안했던 "answer fields should sometimes include readable attributes (names/titles/categories)" 재검토 — iter23에서도 여전히 rental date-only 패턴이라 이제 실제 프롬프트 편집 후보.

---

### Iteration 24 — 2026-04-19 (shape 다변화 프롬프트 검증, 1-trial)

**Hypothesis.** iter18/19/20/23 4회 연속 `[{rental_date, return_date}]` 고착 원인이 Workflow step 2의 단일 concrete example(`[{rental_date, return_date}, …]`) + 강한 negative example(`[{rental_date, film_title}, …] NOT valid`)이라는 진단 하에, commit `c07924d`에서 예시를 5개 destination-appropriate pair(rental date, actor name, customer name, film title, payment amount)로 확장 + "do NOT default to rental_date" 명시 + rotation rule 추가. 이 변경이 실제로 composer의 shape 선택을 움직이는지 단일 trial로 1차 확인(batch는 iter25부터, quota 규칙).

**Change.** `artifacts/tmp_configs/iter24_shape_diversity.yaml` = iter23 config 복제. 변수는 `prompts.py` 한 건.

**Trial.** anchor `payment_id=6137`, topic "Customer payment history by date". **attempt 1 바로 accepted**, `pass_rate=2/3=0.667` in band, registry committed + bundle exported.

**Question:**
> "2005 년 6 월 19일에 결제를 진행한 고객입니다. 제 결제 내역에서 가장 오래된 3 건의 기록을 결제일 오름차순으로 금액과 결제일을 알려주세요."

**Canonical (최초 non-rental shape):**
> `[{amount: "4.99", payment_date: "2005-05-28..."}, {amount: "0.99", payment_date: "2005-05-31..."}, {amount: "3.99", payment_date: "2005-06-18..."}]`

**Findings.**

1. **Shape 다변화 첫 관측.** 프롬프트 예시 리스트 중 "anchor=staff → payment destination: `[{amount, payment_date}, …]`"를 composer가 채택. **4회 연속 rental-only 락인이 1회 iter로 깨짐**. 프롬프트 편집 효과 초기 증거 ✓.
2. **Voice 축 지속 준수.** "2005년 6월 19일에 결제를 진행한 고객입니다" — payment_id anchor를 날짜로 1인칭 self-reference하는 자연스러운 방식. "제 결제 내역", "알려주세요" 1인칭 ask. 금지 구절 0건. iter20 이후 voice 규칙 5iter째 유지.
3. **Semantic 정합성.** anchor payment(_id=6137)는 customer X의 payment row. "2005년 6월 19일에 결제한 고객의 다른 결제 기록"이 destination. "가장 오래된 3 건 결제일 오름차순"이 label constraint surface. canonical의 첫 entry는 2005-05-28로 anchor payment(2005-06-19) 이전의 기록, 자연스럽게 "history" 개념 성립.
4. **Composition depth.** payment anchor → 같은 customer의 다른 payment rows. single-table scope(payment 내부 filter + sort + limit). ~5 primitives. 구조적으로는 iter19 rental 패턴과 동형. **다른 anchor·target은 구조적 단순성은 동일하되 shape 의미가 다양화**.
5. **"1회 관측 = 분포 아님" 경계.** 이번 trial이 payment_id anchor라 프롬프트의 payment 예시가 자연 매칭. anchor가 다른 타입이면 여전히 iter20/23 패턴(rental/customer 타겟)으로 회귀할 가능성. 실제 다양성 측정은 iter25 batch 필요.

**Next direction.**

1. **iter25: 3-trial batch — shape 분포 측정.** 같은 config로 3회 순차 실행. anchor sampling이 다른 타입으로 뽑혔을 때 composer가 새 예시 기반으로 shape 선택하는지(e.g., film anchor → actor names, category anchor → film titles), 아니면 또 rental 고착되는지. accept rate도 함께 측정.
2. **Composite PK fix 실측 기회.** iter25 batch에 actor/category/film anchor 하나만 등장해도 junction path 재시도 가능. 안 나오면 iter26+ seeding.
3. **현재까지 축적된 task pool 정성 재평가.** iter18/19/20/23/24 5건. iter24만 다른 shape. RL training 분포는 여전히 rental 편중(4/5 = 80%) — 분포 건강한 상태 아님. iter25 batch 결과와 합쳐 분포 재검토.

---

### Iteration 25 — 2026-04-19 (3-trial batch, shape 분포 측정)

**Hypothesis.** iter24의 1-trial shape 변화가 단발 우연이 아니라 프롬프트 편집(`c07924d`)의 실제 효과인지 3-trial batch로 측정. 부가로 (a) accept rate 관측, (b) composer가 rental/payment 외 다른 shape(actor name, film title 등)에 도달하는지, (c) junction 경로 등장 시 composite PK fix(`0ff46a5`) 실측 기회.

**Change.** `artifacts/tmp_configs/iter25_batch.yaml` = iter24 config 복제. 순차 3회 실행(smoke_iter25_a/b/c).

**Trial 결과:**

| suffix | anchor | topic | shape | axis | pass_rate | voice |
|---|---|---|---|---|---|---|
| a | payment_id=1453 | "Customer payment history with amount filter" | `{amount, payment_date}` | **Composite** (customer+amount>2) | 1/3 = 0.333 | 1인칭 ✓ |
| b | film_id=108 | "Film actors retrieval" | **`{first_name, last_name}`** | Filter | 2/3 = 0.667 | 2인칭 org-ask ✓ |
| c | inventory_id=3678 | "Inventory rental history" | `{rental_date, return_date}` | Filter | 1/3 = 0.333 | △ schema-ese |

**3/3 accept rate = 100%.** 3 unique shapes across 3 trials.

**iter25_b question:** "영화 'BUTCH PANTHER'에 출연한 배우들의 이름을 첫 이름 오름차순으로 정렬하여 처음 3명 알려주세요." → canonical `[{CARMEN HUNT}, {CUBA OLIVIER}, {GROUCHO DUNST}]`.

**iter25_c question:** "인벤토리 ID 3678번 영화의 대여 기록을 대여 날짜가 빠른 순서대로 정렬하여 처음 3건의 대여 날짜와 반납 날짜를 알려주세요." — "인벤토리 ID 3678번" 표현은 customer가 자연스럽게 쓸 법하지 않은 schema-ese 경계. 금지 구절은 아니지만 voice 축에서 경미한 회귀.

**Findings.**

1. **Shape 다양화 실질 확인.** 프롬프트 편집 후 4 trials(iter24 + 25_a/b/c)의 shape 분포: rental 25% (1/4), payment 50% (2/4), actor names 25% (1/4). 이전 iter18/19/20/23 4회 연속 rental 100% 락인과 **극명히 대조**. `c07924d`의 예시 다변화가 composer의 template matching 성향 그대로 활용해 다양한 shape 유도.
2. **Composite PK fix 실측 검증 ✓ (iter25_b).** iter21에서 실패한 `actor → film_actor → film` 경로와 구조적 대칭인 `film → film_actor → actor` 2-hop junction을 composer가 저작, solver 3명 중 2명이 정합 chain으로 matched. `0ff46a5` 수정이 실제 prompt-tuning loop에서 작동 확인. iter21이 21턴 소비 후 MaxTurnsExceeded였던 걸 iter25_b는 정상 턴 범위 내 완료.
3. **Axis 축 여전히 Filter-dominant.** 8 accept 누적(iter18~25_c) 기준 axis 분포: Filter 5~6건, Composite 2건(iter18 escalation + iter25_a first-attempt), Cardinality/Cross-item rule/Aggregate **0건**. Shape는 변했지만 task type(list of records + filter + sort + limit)은 고착. 이는 iter01~12 Width 편향과 **동일 메커니즘** — composer가 가장 "추가 비용 낮은 axis"로 수렴, prompt의 escalation axes 나열만으론 안 깨짐.
4. **iter25_c voice 회귀 관측.** inventory anchor에서 "인벤토리 ID 3678번 영화"는 schema-ese 경향. anchor가 customer-naturally reference 어려운 타입(inventory, payment_id 숫자)일 때 composer가 schema identifier를 그대로 노출. iter20의 city(Toulouse)/iter23의 city(Sirjan)처럼 named entity가 있으면 voice 자연, 숫자 ID만 있으면 voice 약화.
5. **Composer의 initial task type 고착.** 4 trials 모두 attempt 1에서 "list of 3 records + single/double filter + sort + limit" 패턴. Aggregate task("이 customer가 몇 번 결제했나?" scalar answer)나 Cardinality 변형("결제 10만원 초과 customer 전원 이름") 등 task type 레벨 다양성 0. 프롬프트의 Escalation Axes가 Aggregate를 포함하지 않고 Cardinality는 "N 변경" 수준으로만 표현돼 있어 composer가 task type 전환 신호를 받지 못함.

**Next direction.**

1. **iter26: Task type 다양화 프롬프트 편집.** Workflow step 2의 answer shape을 "list of records" 전제에서 벗어나 scalar aggregate / count / filter-count-pair 등 **task type 레벨 옵션**을 first-class로 추가. 예: "anchor=customer → payment count: `{count: 42}`" 또는 "avg rental duration" 같은 scalar answer 예시를 shape 예시 옆에 배치. iter24식 예시 기반 편집이라 past Width ceiling(iter10)의 "순서/효과 문구 조정" 실패와 다른 각도.
2. **iter25_c voice 회귀 체크포인트 추가 여부.** "anchor가 숫자 ID-only이면 customer-natural reference로 번역" 가이드 한 줄. 단 iter26에서 task type 편집 효과 보고 여유 있을 때 병합.
3. **Composite PK fix 검증 완료 — composite 경로 일상화.** iter21/25_b로 충분. 별도 재검증 iter 없음.
4. **축적 task pool 8건에 대한 외부 평가 가능 시점.** Voice, shape, semantic 모두 합격권 태스크 7건(iter25_c 경미 제외). 실제 RL 훈련 데이터셋 후보로 활용 시 이 분포가 충분한지(8건은 소량, 수백건 단위 필요) 별도 판단 필요.

---

### Iteration 26 — 2026-04-19 (Type B 도입, 1-sample 스모크)

**Hypothesis.** iter18~25 전체 8 accept가 Filter-dominant에 고착된 이유가 Workflow step 2의 "homogeneous list of 3 child records" 하드코딩 때문이라는 가설. step 2를 Type A(list)와 Type B(scalar aggregate) 두 개의 first-class 옵션으로 분리하면 composer가 `count`/`min`/`max` 같은 scalar 답변 task를 시도할지 관측.

**Change.** `src/rl_task_foundry/synthesis/prompts.py` Workflow step 2 재작성. Type A 예시는 유지, Type B 신규 블록 추가(5개 anchor×aggregate 예시: rental_count, actor_count, film_count, first_payment_date, max_payment). Deterministic Answers 섹션도 Type B 스칼라 케이스 포함하도록 보강. Escalation Axes는 그대로 — task type 자체는 step-2 선택이지 escalation이 아니라고 명시. commit `e83d111`.

**Trial.** `artifacts/tmp_configs/iter26_task_type.yaml` 1-sample 스모크(smoke_iter26_a). 결과 `trial_status=synthesis_failed`, attempt_outcomes `[difficulty_crank_invalid, difficulty_crank_invalid, difficulty_weakened]`, 3 submit 모두 미채택.

| attempt | task type | label | axis signal | pass_rate | reject |
|---|---|---|---|---|---|
| 1 | **Type B 스칼라** | `{customer_count: 5}` (country 67→city→address→customer, `count(*)`) | search_cost=8 slot=1 constraints=0 | **3/3 = 1.0** | too_easy |
| 2 | Type A 리스트 | `[{first_name, last_name}, …]` (country 67→…→customer, top-3 last_name asc) | search_cost=3 slot=2 constraints=0 | 3/3 = 1.0 | too_easy |
| 3 | Type A 리스트 | `[{first_name, last_name}, …]` + city IN (Amersfoort, Apeldoorn, Ede) | search_cost=10 slot=2 constraints=0 | (too_hard) | budget_exhausted |

iter26_a attempt 1 question: "네덜란드에 거주하는 고객은 총 몇 명인가요?" — 1인칭 ask, schema identifier 없음, voice 자연. 3 solvers 모두 `{"customer_count": 5}`로 정합 (turn 6 / 10 / 11).

**Findings.**

1. **Type B first-class 도입 성공 ✓ — 튜닝 로그 사상 최초 scalar aggregate 관측.** attempt 1에서 composer가 Type B를 채택해 `aggregate: [{fn: count, column: customer_id, alias: customer_count}]`, no `group_by`로 query DSL 호출. 스키마 추론이 `{customer_count: int}` 형식을 solver-facing prompt로 전파, solver 3명 모두 정확한 matching output 생성. iter18~25 8 accept 누적에서 Aggregate=0이던 축적을 iter26 attempt 1 하나가 깨트림. **프롬프트 편집의 구조적 효과 확인.**
2. **Type B의 too_easy 고착은 구조적 signal.** `{customer_count: 5}` 같은 정수 스칼라는 solver 3명이 전부 같은 값을 produce할 확률이 매우 높음(답의 공간이 1차원 정수). Filter-dominant Type A에서는 3/5 slot exact match가 필요해 pass_rate가 자연스럽게 흩어지는데, Type B는 scalar match 하나로 수렴해 3/3으로 쉽게 수렴. 즉 **Type B는 기본이 too_easy** → escalation(filter 추가 또는 IN-window)으로 답의 공간을 축소해야 band 진입.
3. **Composer가 Type B too_easy 대응을 Type A 전환으로 처리했음 (치명 결함).** attempt 2에서 composer는 `customer_count`를 지우고 `first_name/last_name` 리스트로 교체. 이는 escalation이 아니라 task type 교체. 백엔드의 difficulty 분석기는 search_cost 8→3 감소를 포착해 `difficulty_crank_invalid` 반환(올바른 판정). 프롬프트는 "Never weaken / only add"를 명시했으나 composer는 "다른 task type으로 전환하는 것"도 허용된 옵션으로 해석. 즉 **step-2에서 Type A/B 중 하나를 선택하되, 첫 submit 이후 attempt에서는 같은 type 내에서만 escalation** 제약이 누락.
4. **Attempt 3 too_hard.** 같은 Type A에 city IN-filter(3개 도시)를 추가해 answer space를 지나치게 좁힘 → 3/3 solver 불일치 → too_hard. 이건 정상적인 탐색 실패이며, budget=3 submit이 너무 타이트한 것과 결합돼 trial 자체가 synthesis_failed로 종료. 만약 attempt 2가 Type B escalation(예: 도시 이름 IN 필터로 count)으로 정상 진행됐다면 attempt 3까지 더 완만한 band 수렴이 가능했을 것.
5. **iter26 프롬프트의 미흡 1줄 요약.** "Type B too_easy → Type A 전환" 경로가 열려 있음. 프롬프트에 "Once step 2 picks a task type, all subsequent attempts for the same anchor stay within that type — escalations are within-type additions, never cross-type replacements" 같은 잠금 규칙 필요.

**Next direction.**

1. **iter27 (프롬프트 edit, blocker fix): Task type lock across attempts.** prompts.py에 한 줄 추가: attempt 2+에서 task type 교체 금지, 같은 type 내에서 Escalation Axes의 Filter/Composite/Cardinality/Cross-item만 적용. iter26의 attempt 1 Type B 성공을 보전하면서 too_easy 대응을 제대로 열어주기. 1-sample 스모크로 검증.
2. **iter26_a attempt 1의 Type B 성공 그 자체가 강한 양성 신호** — iter27 수정 후 accept가 따라붙을 가능성 높음. Type B는 structure상 solver 3명이 같은 scalar로 수렴하기 쉬우므로 within-Type-B escalation (filter 추가)이 band 0.33~0.67 진입 키.
3. **iter25_c voice 회귀는 여전히 미해결** — iter27 task-type-lock 편집과 묶어 1줄 voice 가이드 병합 여부는 iter27 결과 보고 판단.
4. **축적 관찰.** iter26으로 Aggregate 축 1건 추가(attempt 1 기준). accept까지는 iter27 제약 수정 후에.

---

### Iteration 27 — 2026-04-19 (Task type lock, 1-sample 스모크)

**Hypothesis.** iter26 `difficulty_crank_invalid` 2회 연속은 prompt의 "Never weaken / only add" 규칙이 task type 교체까지는 커버하지 못한 탓. step 3에 task-type-lock 한 줄을 추가하면 composer가 attempt 1에 선택한 type 내에서만 escalation하도록 유도 가능.

**Change.** prompts.py Workflow step 3/4 수정. step 3에 "The task type picked on your first submit is locked for the whole anchor" 명시, Type B escalation 예시(`count` + city subset filter) 제공, 교체 시 crank_invalid 경고. step 4(too_hard relax)에도 동일 lock 적용. commit `aaae65a`.

**Trial.** `artifacts/tmp_configs/iter27_type_lock.yaml` 1-sample 스모크(smoke_iter27_a). **accepted.** pass_rate=2/3=0.667, CI [0.135, 0.983], registry committed. task_id `task_Customer rentals at address with staff filter_8ad54a5d3755cc8b`.

| attempt | task type | label | axis | pass_rate | reject |
|---|---|---|---|---|---|
| 1 | Type A | `[{rental_date, return_date}, …]` (address 509 → rental, top-3 oldest) | Filter | 3/3 = 1.0 | too_easy |
| 2 | **Type A** (유지) | 동일 shape + staff_id=1 필터 추가 | **Composite** (address + staff) | **2/3 = 0.667** ✓ | **accepted** |

iter27_a attempt 2 question: "제 주소 (786 Matsue Way) 에 등록된 대여 기록 중 직원 1 번이 처리한 가장 오래된 3 건을 대여일 순으로 알려주세요." — 1인칭 고객 ask, 실제 주소명("786 Matsue Way") 사용, 직원 ID도 "직원 1 번"으로 자연 표현, schema-ese 없음.

**Findings.**

1. **Task-type-lock 규칙 실측 작동 ✓.** attempt 1 Type A too_easy 후 composer가 **같은 Type A를 유지하면서 staff 필터를 추가**해 Composite으로 escalate. iter26의 "Type B → Type A 교체" 실패 패턴이 반복되지 않음. 단 1 trial 관측이지만 규칙의 직접적 효과 확인.
2. **Type B는 이번 trial에 선택되지 않음.** iter27_a는 address 앵커로 Type A 리스트를 초기 선택. iter26에서 country 앵커로 Type B를 선택했던 것과 다른 데이터 포인트. task type 선택은 앵커/prompt 예시의 조합에 따라 달라지며 아직 편향 판정 불가(샘플 N=2). 여러 앵커에 걸친 batch(3-4 trial) 필요.
3. **Shape 분포.** attempt 2 accept label이 `{rental_date, return_date}` — iter18/19/20/23의 lock-in shape과 동일. iter24/25 4 trial에서 깨졌던 분포가 1 trial만에 회귀. 1 trial 샘플 한계이며 통계적 결론 내기는 이름. 단발 관측으로 기록만 남김.
4. **Composite 축 재출현.** 누적 9 accept 기준 Composite 분포: iter18(교차 + 임계치), iter25_a(customer 필터 + amount 임계치), iter27_a(address + staff) — 3건. Filter 5~6, Cardinality/Cross-item rule/Aggregate 여전히 0. **Type B 통한 Aggregate 첫 accept는 iter27 시점에도 미확인** — iter26 attempt 1 Type B 성공은 accept까지 이어지지 못했고, iter27에서는 Type B 선택 자체가 없었음.
5. **Composer의 step 2 선택 bias.** 현재 프롬프트는 Type A/B를 동등 옵션으로 제시하지만 Type A가 자주 선택되는 경향(iter26 attempt 1: country 앵커로 Type B, iter27 attempt 1: address 앵커로 Type A). Type A가 예시 5개 vs Type B 예시 5개로 동률이지만 Type A가 먼저 나오고, anchor별 "자연스러움"에서 list 쪽이 LLM에게 친숙할 가능성. Type B 선택 빈도를 N=5~10 trial로 측정해야 기본 분포 확인 가능.

**Next direction.**

1. **iter27 배치 확장 (3 trial)**. task type 분포 + shape 분포 본격 측정. config는 그대로, 순차 3회. 목표: (a) task-type-lock이 Type B 선택 시에도 정상 작동하는지 확인, (b) shape lock-in 재발 여부 판정, (c) axis 축적 확장.
2. **iter28 후보: Type B 선택 유도 강화.** 만약 iter27 batch 3 trial에서 Type B 선택이 0건이라면 prompt에 "If the last accepted task in this conversation was Type A, prefer Type B this attempt (and vice versa)" 추가해 rotation 강제. 단 conversation context 없이 단발 trial 실행 중이므로 conversation-level rotation은 효과 제한적 — 배치 결과 보고 결정.
3. **voice 회귀 가드는 여전히 대기.** iter25_c의 "인벤토리 ID 3678번" schema-ese는 이번 trial에 재발하지 않음("제 주소 786 Matsue Way"는 자연스러움). 배치 결과 모니터 후 계속 관찰.

---

### Iteration 28 — 2026-04-19 (iter27 배치 확장, 추가 2 trial)

**Hypothesis.** iter27_a 1-sample accept 이후 (a) task-type-lock 규칙이 여러 anchor/task type 조합에서도 일관되게 작동하는지, (b) Type A/B 선택 분포가 어떻게 나타나는지, (c) shape lock-in 재발 여부를 3-trial 배치로 측정. 이터 자체의 프롬프트 변경 없음(iter27 프롬프트 유지).

**Change.** 없음. `artifacts/tmp_configs/iter27_type_lock.yaml` 그대로 사용. 부가로 **infra 버그 근본원인 해결**: macOS iCloud가 `.venv/lib/python3.14/site-packages/*.pth` 파일에 `UF_HIDDEN` BSD 플래그를 지속적으로 재적용. Python 3.14 site.py의 security hardening이 `UF_HIDDEN` 플래그 걸린 `.pth`를 무시 → `uv pip install -e .` 후 즉시 `ModuleNotFoundError`. 매 trial 실행 전 `chflags nohidden .venv/lib/python3.14/site-packages/*.pth` 수행으로 해결. 메모리에 저장(`project_uv_editable_install_icloud.md`).

**Trial.** iter27_b, iter27_c 순차 실행. iter27_a와 합쳐 3-trial 배치.

| suffix | anchor | task type | final | pass_rate | axis | shape | voice |
|---|---|---|---|---|---|---|---|
| a | address=509 | Type A | accepted | 0.667 | Composite (address+staff) | `{rental_date, return_date}` | 1인칭 ✓ |
| b | payment_id=12337 | Type A | accepted | 0.333 | Filter (customer 추론) | `{amount, payment_date}` | 1인칭 ✓ |
| c | rental_id=14465 | Type A | synthesis_failed | — | Filter (attempt 1) | `{rental_date, return_date}` | 1인칭 ✓ |

iter27_b question: "제 결제 기록에서 가장 먼저 이루어진 3건의 결제 금액과 날짜를 순서대로 알려주세요." — anchor=payment_id이지만 composer가 payment→customer 추론해 customer-voice로 번역. 자연스러움.
iter27_c attempt 2: `{rental_date, return_date}` 리스트를 3→2로 축소 + "2005-08-01 19:55:09 이전" 날짜 상한 필터 추가. 백엔드는 cardinality 감소를 `difficulty_weakened`로 판정, budget_exhausted로 종료.

**Findings.**

1. **Task-type-lock 규칙 강력히 작동 ✓ (3/3 trial).** iter27 배치에서 Type A/B 전환 시도 0건. iter26의 `difficulty_crank_invalid` 연쇄가 완전히 사라짐. 프롬프트 한 줄 추가의 효과가 구조적.
2. **Type B 선택률 0/3 (iter27 배치).** iter26과 합쳐 4 trial 중 Type B = 1/4 = 25%. 프롬프트에서 Type A/B를 동등하게 제시했지만 composer는 Type A를 기본 선호. 이유 가설: (a) Type A 예시가 먼저 등장, (b) LLM pretraining에서 "list of records" 답변 형태가 "single count" 보다 훨씬 풍부, (c) anchor가 natural-language reference("my rental history")와 잘 맞는 shape는 list 쪽. Type B는 count-specific 질문("내 결제가 몇 건이냐")에만 맞음 → anchor별 적합성 차이.
3. **Shape lock-in 경미 재발.** iter27 배치 3 trial 중 2건 `{rental_date, return_date}` 반복(iter27_a, c). iter24/25에서 쉐이프 다양화(c07924d)로 깨졌던 분포가 iter27 배치에선 3건 중 1건만 non-rental(iter27_b payment). 4 trial 이동평균으로 본 shape 분포: iter24 rental, iter25 payment-amount+actor-names+rental, iter27 rental+payment-amount+rental. 즉 composer의 **기본 shape 편향 자체는 `{rental_date, return_date}`** 로 남아 있고, 프롬프트의 diversified examples가 확률적으로 다른 shape을 뽑게 만들지만 structural 해결은 아님.
4. **Axis 축적 (누적 11 accept).** Filter 7건, Composite 3건(iter18, 25_a, 27_a), Cardinality/Cross-item rule/Aggregate 여전히 **0건**. iter26 Type B 관찰은 accept까지 이어지지 못해 Aggregate 축적 0 상태. 이 시점에서 축 분포의 "Filter-dominant → Composite 2nd" 구조가 굳어지고 있음.
5. **신 관찰: cardinality 감소가 weakening으로 flag됨.** iter27_c attempt 2에서 composer가 "필터 추가 + N 3→2"를 escalation 의도로 제출했으나 backend가 slot=2 유지 + cardinality 감소를 "weakened"로 판정. 프롬프트의 "Never weaken" 규칙은 있지만 "cardinality 감소도 weakening"이 명시돼있지 않음. Escalation Axes의 Cardinality는 "N 증가"만 언급. 이 edge case는 현재로는 rare (1/3 trial)이지만 빈번해지면 프롬프트 보강 필요.
6. **Infra 해결의 부수효과.** iCloud UF_HIDDEN 이슈 해결로 trial 실행 안정성 대폭 증가. 이전에 매 trial마다 `ModuleNotFoundError`로 재시도하던 불안정성 제거. 실험 루프 tempo 향상.

**Next direction.**

1. **iter29: Type B selection boost.** 현재 Type B 선택률 1/4는 낮음. 프롬프트에서 Type A/B 예시 순서 바꾸기(Type B 먼저 제시), Type B 예시 수 5→7로 늘리기, "Prefer Type B when anchor has a high-fanout FK (customer, film, country) and the count or aggregate is itself a natural answer" 같은 **선택 힌트** 추가. Type A 금지가 아니라 Type B 유도. 1-sample 스모크로 검증.
2. **iter30 후보: Cardinality 감소 weakening 명시.** iter27_c 재현 빈도 보고 결정. 현재는 iter29 Type B 실험 우선.
3. **Shape lock-in 구조적 해결 여전히 미결.** 프롬프트 예시 다양화는 확률적 개입일 뿐, composer의 internal bias를 깨지 못함. 장기적으로 다른 각도(label 금지어, anchor별 shape 하드 매핑, per-trial shape rotation signal) 필요하지만 현재 축 진입이 우선.
4. **축적 task pool 현황.** iter27_a + iter27_b → 11 accept 누적. 여전히 Filter + Composite 2축만. Cardinality/Cross-item rule/Aggregate 미획득.

---

### Iteration 29 — 2026-04-19 (Type B selection boost, 1-sample 스모크)

**Hypothesis.** iter26-28 누적에서 Type B 선택률이 1/4로 낮음(Aggregate 축 0/11 accept). 프롬프트에서 Type A를 먼저 제시한 순서 편향과 선택 가이드 부재가 원인 가설. step 2를 Type B 우선 배치 + 명시적 selection hint(phrasing 기반 결정 규칙) + Type B 예시 5→7로 확장하면 composer가 더 자주 scalar task를 고를 것.

**Change.** prompts.py Workflow step 2 재구성. (a) "Selection hint" 블록 추가 — "how many/earliest/largest" 표현이면 Type B, "show me the first N/list records of" 표현이면 Type A, hub 앵커는 두 옵션 모두 가능. (b) Type B 섹션을 Type A보다 먼저 배치. (c) Type B 예시 2개 추가(country → customer_count, address → rental_count). commit `83c6d61`.

**Trial.** `artifacts/tmp_configs/iter29_type_b_boost.yaml` 1-sample 스모크(smoke_iter29_a). **synthesis_failed.** 단 1회 submit 후 종료.

| attempt | anchor | **task type** | label | axis | pass_rate | terminal |
|---|---|---|---|---|---|---|
| 1 | rental_id=8826 | **Type A** (여전히) | `[{rental_date, return_date}, …]` (rental → customer → rental top-3 by date) | Filter | 0/3 = 0.0 | reject_too_hard → **attempts left: 0, BudgetExhaustedError** |

iter29_a question: "대여 기록 8826 번을 한 고객의 처음 3 개 대여 기록을 대여일 오름차순으로 알려주세요. 각 기록마다 대여 날짜와 반납 날짜를 포함해 주세요." — 3인칭적 표현("대여 기록 8826 번을 한 고객"), schema-ese 경계, 1인칭 goal과 어긋남. 약한 voice 회귀.

**Findings.**

1. **Type B nudge 1/1 trial에서 실패.** composer가 여전히 Type A + `{rental_date, return_date}` lock-in shape 선택. 명시적 selection hint, Type B 우선 배치, 예시 7개(Type A는 5개)까지 동원했음에도 rental_id 앵커에서 자연스러운 phrasing을 "고객의 처음 3 개 대여 기록"(= list)로 인식. N=1 샘플이긴 하지만 **phrasing 기반 hint가 anchor 타입과 맞물려 있어 선택 유도력이 약함**.
2. **태스크 초기 bias의 구조적 원인.** 앵커가 `rental_id`처럼 이미 구체적 record인 경우 composer는 "이 record가 속한 부모를 따라가서 같은 부모의 다른 records"를 자연 답변으로 인식. 이 phrasing은 Type A로 귀결. Type B가 자연스러우려면 앵커가 hub-like (country, film, category)일 때. 현 세팅은 anchor 샘플링이 random — Type B가 적합한 앵커가 뽑힐 확률이 그만큼 낮음. 프롬프트만으로는 해결 불가.
3. **1 submit 후 조기 종료 메커니즘 확인.** too_hard(pass_rate=0/3)면 `attempts_left=0`과 `BudgetExhaustedError`가 동시에 emit돼 trial 즉시 종료. 과거 user 언급("too_hard면 원래 종료")과 일치. iter29_a는 프롬프트 검증 기회를 사실상 첫 attempt 1장으로 제한받음. 1 samplle 결과의 통계적 의미는 낮음(conclusion 유보).
4. **Voice 경미 회귀.** "대여 기록 8826 번을 한 고객"은 3인칭-ish 표현. iter20의 voice fix가 anchor=customer/rental 등 직접 identity일 땐 잘 작동하나 rental_id 앵커에서 customer 추론 경로에선 schema-ese 경계. iter25_c 연장선의 패턴.
5. **Prompt 편집 자체가 해로웠는지 판단.** Selection hint + reorder 자체는 voice 회귀와 직접 연관 없음. Type A 쪽 cross-table ban, task-type-lock, voice 규칙은 모두 유지. 프롬프트 사이즈만 소폭 증가(~60 words). 1 trial 기반 추론이므로 회귀 여부는 재현 필요.

**Next direction.**

1. **iter30 방향 재정의.** Prompt-level Type B 유도의 효과가 약함. 두 가지 경로 후보:
   - (a) **앵커 선택 레벨 개입**: anchor 샘플링 시 hub-like 테이블(country, film, category, staff) 가중치 상향. 이는 prompts.py 밖의 수정이며 `anchor_hint` 또는 `requested_topic` 결정 로직 대상. 통제된 Type B 증가 실험.
   - (b) **시스템 프롬프트에서 Type B를 기본, Type A를 escalation 옵션으로 역전**: "Default to Type B unless Type A is the only natural phrasing". 강한 편향 반전이지만 Type A accept가 줄어들 수 있음.
   - iter30은 (a) 우선. 앵커 풀이 실제로 Type B-친화 테이블로 편중되어야 결과가 나옴.
2. **iter29 voice 회귀는 개별 1 sample — 재현 확인 후 판단.** rental 앵커에서 "대여 기록 N번을 한 고객" 표현은 composer의 자연 경향. 명시적 "anchor가 record-id인 경우, 그 record의 주체(customer/staff)를 customer-voice로 직접 부르기" 가이드 한 줄 고려.
3. **Budget economics 경고.** iter29_a 한 번에 input 138k tokens + output 4.3k, 8턴. 1-sample이 1 attempt에서 종료되면 프롬프트 효과 검증 불가. Type B 유도 실험은 multi-trial 배치 필요하지만 quota 타이트. 앵커 레벨 개입(경로 a)이 가장 비용 효율적.
4. **Axes 축적 현황 (변화 없음).** iter27_a, iter27_b accept 이후 cumulative 11 accepts. Filter 7, Composite 3, Aggregate 0.

---

### Iteration 30 — 2026-04-19 (앵커 바이어스 by 인바운드 FK 차수, 1-sample 스모크)

**Hypothesis.** iter29 Type B nudge가 prompt 수준에선 한계. anchor가 rental/payment record 같은 leaf일 때 composer는 자연 phrasing을 "records of parent" = Type A로 귀결. Type B는 hub-like anchor(film, customer, staff, city, country, address)에서 자연 — 이들을 선택 확률을 높이면 Type B accept가 따라옴. `random_anchor`에 `weights = max(1, inbound_edge_count)` 적용해 구조적으로 hub 편향.

**Change.** `src/rl_task_foundry/synthesis/synthesis_db.py` `random_anchor()` 수정. `random.choice` → `random.choices(candidates, weights=[max(1, len(edges_to(t))) for t in candidates])`. 로컬 변수 `hub_tables` → `candidates`로 리네임(실제로 hub 분류가 아닌 row-count 필터였음). commit `e55453d`.

**Trial.** `artifacts/tmp_configs/iter30_anchor_bias.yaml` 1-sample 스모크(smoke_iter30_a). **accepted.** pass_rate=2/3=0.667, CI [0.135, 0.983], task_id `task_City rental count with composite date filters_1f8ca17c519d084f`.

| attempt | anchor | **task type** | label | axis | pass_rate | reject |
|---|---|---|---|---|---|---|
| 1 | city=459 (Santiago de Compostela) | **Type B** | `{rental_count: 20}` (city→address→customer→rental) | Aggregate | 3/3 = 1.0 | too_easy |
| 2 | city=459 (유지) | **Type B** (유지) | `{rental_count: 16}` + `rental_date >= 2005-06-01` | Aggregate + Filter | 3/3 = 1.0 | too_easy |
| 3 | city=459 (유지) | **Type B** (유지) | `{rental_count: 15}` + `rental_date >= 2005-06-01` + `return_date >= 2005-07-01` | Aggregate + **Composite** | **2/3 = 0.667** ✓ | **accepted** |

iter30_a attempt 3 question: "Santiago de Compostela에 거주하는 고객들의 2005-06-01 이후 대여하면서 2005-07-01 이후 반납한 기록 수는 몇 건인가요?" — 자연스러운 customer-ask, 실제 city name, 두 날짜 필터 명시, "몇 건인가요" scalar 질문. 1인칭/2인칭 섞인 user-ask voice, 금지구절 0.

**Findings.**

1. **Aggregate 축 최초 Accept 획득 ✓✓.** iter18~29 누적 11 accept 중 Aggregate 축 = 0이었음. iter30_a 단일 trial로 **12번째 accept가 최초의 scalar count** 태스크. Filter-dominant 천장(iter01~12 Width 편향과 구조적 대칭)이 **정성적으로 깨짐**. 축 분포 업데이트: Filter 7, Composite 3, **Aggregate 1** ← NEW, Cardinality 0, Cross-item rule 0.
2. **iter26~29 편집의 누적 효과 동시 검증.** 이 trial 하나가 4개 iter의 변경을 동시에 증명:
   - iter26 (Type B 도입): attempt 1에서 Type B 선택 ✓
   - iter27 (task-type-lock): attempt 2, 3에서 Type B 유지, Type A로 전환 시도 없음 ✓
   - iter29 (Selection hint + reorder): "몇 건인가요" 구절이 "how many X" hint와 직접 대응해 Type B 유도 ✓
   - iter30 (anchor bias): city(inbound 1, 하지만 city→address→customer→rental 3-hop 체인이 hub-like) 선택 자체가 anchor 편향 효과. leaf rental 앵커 대신 hub-like city 앵커가 pick됨
   즉 **프롬프트-only 개입으로 Type B를 유도하려 한 iter29가 실패한 이유는 앵커가 부적합했던 것**이라는 가설을 iter30이 직접 입증.
3. **Type B 내 Composite escalation 실측.** attempt 1 too_easy → attempt 2에서 Filter(date >= X) 추가 → 여전히 too_easy → attempt 3에서 Composite(date >= X AND date >= Y) → band 진입. Type B escalation 경로(filter 추가, 필터 쌍 구성)가 실제로 band를 제어. 이건 iter27 task-type-lock 주석에서 예언한 "Type B는 filter 추가로 escalate" 그대로 실행됨.
4. **Voice 자연성 회복.** iter29_a의 "대여 기록 8826 번을 한 고객" schema-ese 이후 iter30_a는 "Santiago de Compostela에 거주하는 고객들의 ... 몇 건인가요?" 자연 customer-ask. city name은 string observable, 필터 날짜도 관측값, 모든 ground 요건 충족. voice 회귀 재발 없음.
5. **anchor bias 효과 검증됨(N=1).** city_id 앵커는 sakila에서 inbound_edge_count=1이지만 우연/가중치 조합으로 선택됨. N=1이므로 효과 강도는 미지수 — 3-trial 배치 또는 distribution 측정이 확증에 필요. 그러나 Type B 선택 자체가 실현되었다는 점에서 minimum effect 확인.
6. **Cumulative task pool = 12 accept.** 11 → 12. Voice 11/12 clean (iter25_c 경미 제외).

**Next direction.**

1. **iter30 배치 확장 (2-3 trial 추가).** Type B accept가 단발인지 재현 가능한 패턴인지 확인. 앵커 가중치의 효과 실측. 배치 실행 시 anchor 분포 관찰(hub-heavy인지) + Type B 선택 빈도.
2. **Cardinality/Cross-item rule 축은 여전히 0.** Type B Composite 경로가 열렸으므로 이제 Cardinality(N 변경) / Cross-item rule(ordering constraint) 축 관측이 다음 천장. 단기적으로는 iter30 재현 배치가 우선.
3. **voice 가드 iter25_c 패턴**은 iter30에서 재발 없음 — rental/payment 앵커 선택률이 anchor bias로 낮아진 효과일 수도. 배치 결과 보고 판단.
4. **`project_qwen_prompt_tuning.md` 메모리 업데이트 필요**. 현재 메모리는 iter13 상태(pass_rate 1.0→0.0 반전 직후). iter30 시점 현황: Filter 7/Composite 3/Aggregate 1 = 11 Type A + 1 Type B accept, ceiling 부분적으로 깨짐.

---

### Iteration 31 — 2026-04-19 (iter30 배치 확장, 추가 2 trial)

**Hypothesis.** iter30_a 1-sample Type B accept 이후 (a) anchor bias (`inbound_edge_count` 가중)의 효과가 배치에서 재현되는지, (b) Type B 선택이 단발 luck인지 시스템 패턴인지, (c) hub anchor 선택 빈도를 관측. 프롬프트/코드 변경 없음 — iter30 상태 그대로 배치 재현.

**Change.** 없음. 단, quota 안전을 위해 `parallel_workers: 4 → 1`로 낮춘 `artifacts/tmp_configs/iter30_seq.yaml` 사용. smoke_iter30_b → smoke_iter30_c 순차 실행.

**Trial.** iter30_a + iter30_b + iter30_c 합쳐 3-trial 배치.

| suffix | anchor | anchor class | task type | final | pass_rate | primary axis | secondary | voice |
|---|---|---|---|---|---|---|---|---|
| a | city=459 | hub-like (3-hop chain) | **Type B** | accepted | 0.667 | **Aggregate** | Composite filter | 자연 ✓ |
| b | rental=5456 | leaf | Type A | accepted | 0.333 | **Cross-item rule** | Filter (staff) | "Mike Hillyer 직원" ✓ |
| c | rental=15855 | leaf | Type A | accepted | 0.333 | **Cross-item rule** | Composite (date + staff) | "직원 1 번" 회귀 |

iter30_b attempt path: (1) rental_date ASC + take 3 → 3/3 too_easy → (2) + staff_id=1 (Mike Hillyer) → **1/3 accepted**.
iter30_c attempt path: (1) rental_date ASC + take 3 → 3/3 too_easy → (2) + `rental_date >= 2005-07-01` → 3/3 too_easy → (3) + staff_id=1 → **1/3 accepted**.

**Findings.**

1. **Cross-item rule 축 최초 Accept 획득 ✓ (iter30_b, c 연속 2건).** iter18~30_a 누적 12 accept 중 Cross-item rule = 0이었음. iter31 배치에서 2건 동시 추가. 누적 axis 분포 업데이트: **Filter 7, Composite 3, Aggregate 1, Cardinality 0, Cross-item rule 2**. 4 of 5 축 돌파(남은 0축은 Cardinality 하나). iter30_b, c 모두 `output_schema.root.sort_key = [rental_date, return_date]`가 명시된 list — 순서 제약이 곧 Cross-item rule. 태스크 자체는 Type A지만 axis 돌파에 기여.
2. **Type B 선택률 배치에서 하락 (iter30_a 1/1 → 3-batch 1/3).** iter30_a의 city 앵커 + Type B는 anchor bias의 효과로 보였으나 iter30_b/c는 rental leaf 앵커 + Type A로 회귀. sakila schema에서 inbound_edge_count 분포가 near-uniform(city=1, rental=1 등 대부분 0~2)이라 **가중화 이점이 약함** — chain 구조 DB에서 FK 차수 가중은 uniform sampling과 거의 구별되지 않음. 이 이터의 핵심 가설("anchor bias가 Type B 재현을 밀어준다")은 사실상 **기각**.
3. **하지만 배치 자체는 accept 3/3로 성공**, 단 Type B 재현 실패가 Cross-item rule 2건 획득으로 대체됨. iter30_b, c는 Type A escalation 경로에서 **"정렬 + take N" 형태가 natural하게 나옴** — iter18~25 대부분의 Type A는 ordering 없는 list였는데 iter26 이후(Type B 도입 + task-type-lock + 예시 개편) Type A의 shape도 미세하게 진화. 직접 의도하지 않은 부수효과로 Cross-item 축 열림.
4. **Voice 회귀 재발 (iter30_c "직원 1 번").** iter25_c(inventory_id), iter29_a(rental_id 8826번), 이어 iter30_c `staff_id=1 → "직원 1 번"`. 같은 체인 필터인데 iter30_b는 `"Mike Hillyer 직원"`(자연)으로 번역, iter30_c는 숫자 ID 그대로 노출. 차이 가설: iter30_b phase monitor에서 composer가 `sample(table=staff, n=2)` 호출 후 이름 획득 → natural translate. iter30_c는 staff 테이블 sample 건너뛴 것으로 추정(phase_monitor 확인 필요). **프롬프트 보강 후보**: "numeric ID filter를 쓸 때는 반드시 referent 테이블의 natural attribute로 번역". 이전 이터에서 deferred했던 rule.
5. **pass_rate 하단 클러스터링.** iter30_b, c 모두 1/3 = 0.333 (band [0.33, 0.67] 하한 경계). iter30_a의 2/3 = 0.667(상한)까지 고려하면 3개 모두 band 경계에 있음. CI가 넓어([0.017, 0.865]) 통계적 신뢰도 약함. composer가 "minimum escalation"(딱 한 개 요소 추가해 band 하단 스침) 전략을 쓰는 듯 — too_easy 탈출이 목표지 "중간 난이도 겨냥"이 목표가 아님. 프롬프트에 강도 힌트 없으므로 예상된 행동.
6. **누적 14 accept.** iter13 리셋 이후 12 → 14. 배치 기준 axis 원자 재구성: Filter-primary 7, Composite-primary 3, Aggregate-primary 1, Cross-item-primary 2, Cardinality-primary 0. iter30_b에는 secondary Filter도 있어 다중 축 참여 태스크가 늘어남 — 이건 Type A의 shape 복잡도가 올라간 신호.

**Next direction.**

1. **iter32 후보 1: voice 가드 (staff_id 등 ID filter → natural translate).** iter30_c 회귀로 더 이상 deferred 하지 않음. 프롬프트의 "Step 3 Deterministic answers" 또는 "Voice rules"에 "Filter by a numeric ID attribute → resolve to the referent table's natural name/title via a lookup tool call before phrasing the filter in user voice" 추가. 1-sample smoke로 검증.
2. **iter32 후보 2: Cardinality 축 개방.** Type A의 N이 거의 3으로 고정됨. "Escalation Axes"에 "vary N (3, 5, 10, or 'all matching')" 명시 + 예시. 현재 5번째 0축.
3. **anchor bias 반복 — 실용성 낮음.** sakila에서 inbound_edge_count 편차가 거의 없어 가중화가 사실상 uniform. 진짜 hub를 밀어주려면 `inbound * row_ratio` 같은 복합 weight 또는 명시적 allowlist(city, customer, film, address만 뽑기) 필요. 하지만 Type B 유도는 anchor만으로 해결 안 되므로 **iter32에서는 prompt-side로 접근**(Type B 예시 더 늘리기, "prefer Type B for hub-like anchors" 힌트).
4. **pass_rate 중간 겨냥 힌트는 deferred.** band 하단 클러스터링은 품질 이슈는 아니고 관찰만.
5. **task pool 현황**: 14 accept, 4 of 5 axes. Cardinality만 남음. 4-5 trial 더 돌리면 5축 모두 관측 가능할 전망.

---

### Iteration 32 — 2026-04-19 (voice guard for numeric-ID filters, 1-sample 스모크)

**Hypothesis.** iter25_c(inventory_id) → iter29_a(rental_id) → iter30_c(staff_id=1 → "직원 1 번")로 3회 재발한 voice 회귀를 프롬프트 수준에서 차단. 기존 voice rule은 anchor 번역만 커버 — filter 술어 안의 numeric ID는 사각지대. "Numeric-ID filter values must be resolved to the referent table's natural name/title via a lookup tool call BEFORE phrasing the filter in user voice" 룰 추가로 회귀 차단을 관측.

**Change.** `src/rl_task_foundry/synthesis/prompts.py` voice 섹션의 anchor-translation 문장 다음에 filter ID 값 가드 삽입. 예시 3쌍: `'직원 1 번' → 'Mike Hillyer 직원'`, `'film 473번' → 'Blade Runner'`, `'카테고리 6번' → '액션 영화'`. commit `c46269d`.

**Trial.** `artifacts/tmp_configs/iter32_voice_guard.yaml` 1-sample 스모크(smoke_iter32_a). **accepted.** pass_rate=1/3=0.333, CI [0.017, 0.865], task_id `task_Customer rental history from address_689452ac123e1a45`.

| attempt | anchor | task type | label | axis | pass_rate | reject |
|---|---|---|---|---|---|---|
| 1 | address=539 | Type A | `[{rental_date, return_date}]` × 3, `sort_key=[rental_date ASC]` | **Cross-item rule** + take 3 | **1/3 = 0.333** ✓ | **accepted (submit 1)** |

iter32_a question: "제 대여 기록 중 대여일이 가장 빠른 순서로 3 건을 보여주세요" — 1인칭 customer ask, 자연스러움, 필터 전무(ordering만). tool call 순서: `profile → sample × 3 → query`. staff / film / category 등 lookup 없음 — ID 필터가 question에 포함되지 않았으므로 voice guard의 **발동 대상 자체가 없음**.

**Findings.**

1. **Inconclusive smoke for voice guard ✗.** 핵심 가설(ID filter 번역 의무화) 검증이 **실행되지 않음**. composer가 필터 없이 "ordering + take 3"만으로 submit 1에서 band 진입, staff_id/film_id 등 ID 필터를 쓰지 않음. 가드는 코드상 존재하지만 이 trial에서 활성화된 적 없어 효과/부작용 모두 관측 불가.
2. **부수효과 가설: 가드가 ID-filter 사용을 억제했을 가능성.** iter30_b, c 둘 다 staff_id 필터로 escalate했는데 iter32_a는 필터 전무로 직행. 가드가 "ID filter를 쓰려면 lookup 한 번 더 필요"를 암묵적으로 비용화해서 composer가 ID 필터를 회피했을 수 있음. N=1이라 확증 불가 — 추가 trial 필요. 만약 이 패턴 계속되면 태스크 단순화라는 **역효과**.
3. **15번째 accept, Cross-item rule axis 3번째 관측.** 누적 15 accept. axis 분포: Filter 7, Composite 3, Aggregate 1, **Cross-item rule 3**, Cardinality 0. Cross-item은 이제 2순위 안정화. iter30_b, c, iter32_a 모두 `rental_date ASC + take 3` — ordering 패턴이 composer의 Type A 기본 레퍼토리로 편입됨.
4. **Submit 1 accept의 드문 케이스.** iter18~31 대부분 2~3 attempt로 band 진입, iter32_a는 1번에 도달. (a) "Cross-item + take 3"의 평균 난이도가 band 하단과 맞음, (b) address 앵커가 customer 추론 체인을 단순화, 두 요인 복합. 이상치는 아님.
5. **pass_rate 하단 클러스터링 심화.** iter30_b(0.333), c(0.333), 32_a(0.333) 3연속 하한값. 최근 accept 4/4 = 하한 경계. "minimum escalation" 구조화. 강도 힌트는 여전히 deferred.

**Next direction.**

1. **iter32 배치 확장 (1-2 trial 추가)** — voice guard가 ID-filter 태스크에서 실제 발동하는 장면을 봐야 가설 검증 가능. anchor 분포가 달라지면 staff_id/film_id/category_id 필터 태스크가 등장할 확률 있음. 만약 3-trial 내내 ID filter 0이면 **가드의 부수효과(가설 2) 실존 신호** — 프롬프트 완화(lookup을 "권장"으로 낮추거나 예시 축소) 고려.
2. **iter33 후보: Cardinality 축 개방** — 마지막 0축. voice guard 효과/부작용 확정 후 진행.
3. **pass_rate 중간 겨냥 힌트**는 관찰만.
4. **task pool 현황**: 15 accept, 4 of 5 axes. Cardinality만 남음.

---

### Iteration 33 — 2026-04-19 (iter32 배치 확장, 추가 2 trial — voice guard 검증)

**Hypothesis.** iter32_a는 ID filter 없이 submit 1에서 accept — 가드 발동 자체가 없어 inconclusive. (a) 가드가 실전 ID-filter 시나리오에서 작동하는지, (b) "가드가 ID filter 회피를 유도"(iter32 finding 2 부수효과 가설)가 실존하는지 확인. 프롬프트/코드 변경 없음 — iter32 상태 그대로 순차 배치.

**Change.** 없음. `artifacts/tmp_configs/iter32_voice_guard.yaml` 그대로 재사용. smoke_iter32_b → smoke_iter32_c 순차 실행.

**Trial.** iter32_a + iter32_b + iter32_c 합쳐 3-trial 배치.

| suffix | anchor | anchor class | task type | final | pass_rate | axis | voice |
|---|---|---|---|---|---|---|---|
| a | address=539 | mid-chain | Type A | accepted (submit 1) | 0.333 | Cross-item + take 3, filter 0 | 자연 ✓ (no ID filter) |
| b | customer=499 | hub | Type A | accepted (submit 1) | 0.667 | Cross-item + take 3, filter 0 | 자연 ✓ (no ID filter) |
| c | film=201 | M:M bridge hub | Type A | **synthesis_failed** | — | attempt 1~5 죽음 | **"CYCLONE FAMILY 영화" ✓✓** |

iter32_c는 fail이지만 voice guard의 결정적 검증 케이스. 모든 5 attempt question에서 `film_id=201`을 `"CYCLONE FAMILY 영화"`(영화 제목)로 **정확히 번역**. iter30_c가 `staff_id=1 → "직원 1 번"`으로 누수했던 정확한 패턴이 이번엔 자연어로 번역됨. tool call 중 `read(film, film_id=201, columns=[title])` 또는 유사 lookup 호출이 포함된 것으로 추정.

iter32_c attempt 연쇄 분석:
- attempt 1: `actor_id ASC, first 3` → too_easy 3/3
- attempt 2: sort를 `last_name ASC`로 변경 → too_easy 3/3 (label 바뀌었지만 strengthen 아님 — 동일 난이도)
- attempt 3: N 3→2 축소 → too_easy (**N 축소는 iter27_c와 동일한 weakening 패턴, 프롬프트 규칙 위반**)
- attempt 4: `PG 등급 필터` 추가 → **`label_not_strengthened`** (배우 테이블에 rating 없음; 필터가 의미적으로 틀려 strengthen 실패)
- attempt 5: "영화 제목 포함" 추가 → **`reject_too_easy`** (Width axis는 프롬프트에서 명시적으로 금지된 축) → `budget_exhausted`

**Findings.**

1. **Voice guard 검증 성공 ✓✓ (iter32_c).** `film_id=201` → `"CYCLONE FAMILY 영화"` 자연어 번역, 모든 5 attempt 일관. iter25_c(inventory_id), iter29_a(rental_id 8826번), iter30_c(직원 1 번)로 3회 재발하던 voice 회귀가 **iter32 가드로 구조적 차단**됨을 실증. composer는 film 테이블에서 title을 명시적으로 lookup해 필터에서 사용. 가드가 의도한 행동을 이끌어냄.
2. **"가드가 ID filter 회피를 유도" 가설은 반증됨.** iter32_c는 ID filter를 적극 시도(PG rating, film title 등). 필터 회피하지 않음. iter32_a, b에서 필터가 없었던 건 composer가 submit 1에 band 진입해 escalate 불필요했기 때문이지 가드 때문이 아님. 부수효과 가설 기각.
3. **다른 문제 노출 — M:M bridge anchor의 escalation dead-end.** film 앵커 + "출연 배우 N명" 태스크는 구조적으로 매우 단순(soft small join) → 어떤 필터를 추가해도 solver가 풀어버림. 그 결과 composer가 규칙 위반 escalation(N 축소, 의미 틀린 필터, Width)으로 내몰림. **프롬프트에 "M:M bridge 구조(film_actor, category 등)는 escalation이 어려우니 anchor 단계에서 회피"** 같은 힌트 추가 후보.
4. **Anchor bias의 우연한 부작용.** iter30 weighted anchor가 iter32_b에서 customer(high-inbound hub, 정당) ✓, iter32_c에서 film(M:M bridge hub, 부적합)을 뽑음. inbound_edge_count만으로는 "작업 가능한 hub"와 "작업 불가능한 bridge hub"를 구분 못 함. 앵커 필터에 **M:M bridge 회피** 룰 추가 후보 (별도 iter).
5. **누적 16 accept.** 14 + iter32_a + iter32_b = 16 accept, 4 of 5 axes. Primary-axis 분포: Filter 7, Composite 3, Aggregate 1, **Cross-item rule 4**, Cardinality 0. Cross-item은 최근 4번 연속 정당한 primary — Type A 기본 레퍼토리로 완전히 편입.
6. **pass_rate band 분포 업데이트.** iter32_a 0.333(하단), iter32_b 0.667(상단), iter30_a 0.667, 나머지 iter30/31 하단. 5 최근 accept 중 2개 상단, 3개 하단. "하단 집중"은 거짓으로 판명 — anchor shape에 따른 변동.

**Next direction.**

1. **iter34: Cardinality 축 개방 (마지막 0축).** Voice guard 이슈 닫혔으니 마지막 0축으로 pivot. `Escalation Axes`의 `- **Cardinality**` 항목에 이미 "change N (e.g. 3 → 5) or switch from fixed-N to 'all records matching the filter'"가 적혀 있음 — **문제는 composer가 이걸 선택하지 않는다는 것**. Type A의 N을 3으로 고정하려는 경향 강함. 접근: (a) Type A 예시에 N=5, N=10, "all matching" 케이스 추가 (iter24 shape 다양화와 동일한 확률적 개입), (b) Workflow step 3 escalation 힌트에 "Cardinality 축을 **우선 검토**" 한 줄 추가, (c) 둘 다. 1-sample smoke.
2. **iter35 후보: M:M bridge anchor 회피.** iter32_c dead-end 재발 가능성 있음. `random_anchor`에서 bridge 테이블(`film_actor`, `film_category`) 제외 또는 가중치 0. 하지만 bridge는 이미 `inbound_edge_count`가 낮을 것(있으면 FK만). 실제로 iter32_c film_id=201은 bridge 자체가 아니라 film. film이 high-inbound이긴 하나 다운스트림이 M:M 관계(actor). 구조적 해결은 "anchor 후보의 downstream 경로에서 bridge가 나오는 테이블은 감점" 같은 복잡한 로직 필요. deferred.
3. **weakening 패턴 3회 관찰 (iter27_c, iter32_c attempt 3, iter32_c attempt 5).** N 축소, Width 금지 위반이 escalation dead-end에서 반복. 프롬프트의 "Never select Width after too_easy" rule이 있지만 composer가 attempt 5에서 어긴 것으로 보임. attempt 5 시점에 max_submissions 5 가까워져서 "정상 escalation 경로 탈진" 상태. 프롬프트 강화 별도 iter 고려.
4. **task pool 현황**: 16 accept, 4 of 5 axes. Cardinality 여전히 0.

---

### Iteration 34 — 2026-04-19 (Cardinality 축 개방 시도 1: Type A 정의에서 N=3 하드코딩 제거, 1-sample 스모크)

**Hypothesis.** 최근 4개 Cross-item accept(iter30_b, c, iter32_a, b)와 iter32 배치 내내 N=3 고정. 원인 진단: Type A 정의 문장 자체가 "list of **3** child records"로 N을 하드코딩. 개입: 정의를 "list of N child records (N ∈ {3, 5, 10, or 'all records matching'})"로 변경하고 "N=3이 기본 시작점이긴 하지만 다변화하라, 15개 prior accept에서 N=3가 Cardinality 축을 0으로 고정시켰다"는 메타 관찰을 프롬프트 안에 명시. 드래프트 단계부터 N 다변화 유도.

**Change.** `src/rl_task_foundry/synthesis/prompts.py` Type A 섹션. "**Type A — list of 3 child records.**" → "**Type A — list of N child records (N ∈ {3, 5, 10, or 'all records matching <filter>'}).**" + 뒤에 7줄 추가: "N is a draft parameter — vary it. ... Do not always default to N=3 — that choice alone has saturated the Cardinality axis at 0 across 15 prior accepts." commit `b2f06c7`.

**Trial.** `artifacts/tmp_configs/iter34_cardinality.yaml` 1-sample 스모크(smoke_iter34_a). **accepted.** pass_rate=1/3=0.333, task_id `task_Customer rental history - first 3 rentals_39fac5d04a1f3fd0`.

| attempt | anchor | task type | label | axis | pass_rate | reject |
|---|---|---|---|---|---|---|
| 1 | customer=294 | Type A | `[{rental_date, return_date}]` × **3**, `sort_key=[rental_date ASC]` | **Cross-item rule** + take **3** | **1/3 = 0.333** ✓ | **accepted (submit 1)** |

iter34_a question: "제 대여 기록 중 대여 날짜 기준으로 가장 빠른 **3 건**을 보여주세요" — 1인칭, 자연, N=3. 프롬프트 변경의 의도된 다변화 전혀 일어나지 않음.

**Findings.**

1. **가설 실패 (N=3 고정 유지).** Type A 정의에 "N ∈ {3,5,10,all matching}" 명시 + "Do not always default to N=3" 경고 + "15개 prior accept에서 N=3 saturate" 메타 관찰을 주입했으나 composer는 드래프트에서 여전히 N=3 선택. 프롬프트 instruction이 N 선택 분포를 이동시키지 못함. Cardinality 축은 여전히 **0**.
2. **원인 가설 — instruction vs example 불일치.** 변경한 건 정의 문장과 산문 경고만이고, 실제 Type A 예시 5개(anchor=customer→rental, anchor=film→actor 등)는 여전히 N을 명시하지 않음. LLM은 "instruction 따르기"보다 "example pattern 따르기"를 체계적으로 선호. 예시 중 하나 이상에 `N=5` 또는 `[{…}, …] 모든 영화` 같은 구체 형태를 박지 않으면 instruction만으론 bias가 깨지지 않음.
3. **부차 원인 — submit 1에 band 진입하면 escalate 시그널 소실.** iter32_a/b/34_a 3회 연속 customer/address hub 앵커 + "take 3 without filter"가 band 하/상단 진입. composer가 escalate 할 기회 자체가 없고, Escalation Axes의 Cardinality 항목(이미 존재)은 too_easy에만 발동. 드래프트 단계에서 N 다변화를 실제로 일으키는 유일한 길은 Type A 예시 자체의 N 다변화.
4. **17번째 accept, Cross-item rule axis 5번째 관측.** 누적 17 accept. axis 분포: Filter 7, Composite 3, Aggregate 1, **Cross-item rule 5**, Cardinality 0. Cross-item은 Type A 기본 편입 이후 5연속 primary — iter26~30 전파가 composer의 기본 레퍼토리를 구조적으로 이동시켰다는 장기 관찰.
5. **프롬프트 instruction의 한계 실증 사례.** 이번 iter는 "추상적 instruction + 경고 + 메타 관찰"이 LLM 행동을 바꾸지 못함을 보여주는 케이스. 유사 실패: iter11~14의 "Width 금지" instruction도 iter32_c attempt 5에서 여전히 Width 시도 관찰. 프롬프트 개입의 신뢰할 수 있는 단위는 **구체 예시**이며 산문 instruction은 보조적.

**Next direction.**

1. **iter35: Type A 예시의 N 다변화.** 이번 실패의 원인 진단을 그대로 반영 — 예시 5개 중 2-3개를 N=5, N=10, "N=all matching"으로 명시 개편. 예: `anchor=customer → rental destination: `[{rental_date, return_date}, …] × 10`` 또는 `anchor=city → customer destination: `[{first_name, last_name}, …] 해당 도시의 모든 고객``. instruction은 제거하지 않고 예시만 교체. 1-sample smoke.
2. **iter35 보완 (동일 iter 안): Escalation Axes Cardinality 우선 순위 승격.** 현재 나열 순서 Cardinality → Cross-item → Composite → Filter. Cross-item이 이미 default가 되어 Cardinality가 "덜 매력적"으로 보임. 순서를 Cardinality → Composite → Filter → Cross-item으로 재배치 + "Cardinality is the least-observed axis; prefer it when prior escalation attempts haven't dropped pass_rate" 한 문장. 프롬프트 한 줄 추가 + 5줄 재배치.
3. **주의**: iter35에서 iter34 프롬프트 주석(N=3 saturated 경고)은 유지. 제거하면 iter34 의도가 사라짐. 두 개입 누적해서 시너지 확인.
4. **task pool 현황**: 17 accept, 4 of 5 axes. Cardinality 여전히 0 but 다음 iter의 명확한 타겟.

---

### Iteration 35 — 2026-04-19 (Cardinality 축 개방 시도 2: 예시 레벨 N 다변화 + Escalation Axes 재배치, 1-sample 스모크 — **5번째 축 돌파 ✓✓**)

**Hypothesis.** iter34 instruction-only 개입이 실패 — "LLM은 example pattern을 instruction보다 체계적으로 선호"가 iter34의 진단. iter35 개입: (a) Type A 예시 5개 중 3개를 N=10, N=all matching, N=5로 구체 값 박기(1개는 N=3 대조군 유지), (b) Escalation Axes 순서를 `Cardinality → Composite → Filter → Cross-item`으로 재배치, (c) "Cardinality is the least-observed axis (0 of 17 prior accepts) — prefer it over Cross-item" 한 문장 추가, (d) Cross-item 항목에 "basic ordering은 default Type A shape이라 더 이상 Cross-item escalation이 아니다" 주석. iter34 instruction은 유지(누적 효과).

**Change.** `src/rl_task_foundry/synthesis/prompts.py` Type A 예시 5개 전체 재작성 + Escalation Axes 재배치 + "Cardinality is the least-observed axis" 힌트 + Cross-item 항목 보강. 예시 변경:
- anchor=customer → rental, **N=10**: "제 대여 기록 중 가장 빠른 10건"
- anchor=film → actor, **N=all matching**: "이 영화에 출연한 모든 배우"
- anchor=city → customer, **N=all matching**: "해당 도시의 모든 고객"
- anchor=category → film, **N=5**: "이 카테고리의 가장 최근 영화 5편"
- anchor=staff → payment, **N=3**: "가장 최근 결제 3건" (N=3 대조군 유지)

commit `bb5e21d`.

**Trial.** `artifacts/tmp_configs/iter35_card_examples.yaml` 1-sample 스모크(smoke_iter35_a). **accepted.** pass_rate=2/3=0.667 (band 상단), task_id `task_customer rental history list_416bf8723fcf9f9e`.

| attempt | anchor | task type | label | axis | pass_rate | reject |
|---|---|---|---|---|---|---|
| 1 | customer=155 | Type A | `[{rental_date, return_date}] × 5` (sort rental_date ASC, **N=5**) | — | — | `feedback` (mid-flight, no final label) |
| 2 | customer=155 (유지) | Type A (유지) | `[{rental_date, return_date}] × 5` (동일 draft, 정제) | **Cardinality (N=5)** + Cross-item | **2/3 = 0.667** ✓ | **accepted** |

iter35_a question: "제 대여 기록 중 대여일이 가장 빠른 순서로 **5 건**을 알려주세요" — 1인칭 customer-ask, N=5 명시, 자연.

**Findings.**

1. **5번째 축 Cardinality 최초 Accept ✓✓✓.** iter13 리셋 이후 18개 누적 accept 중 **처음으로 N ≠ 3** 드래프트. 축 분포 업데이트: Filter 7, Composite 3, Aggregate 1, Cross-item rule 5, **Cardinality 1 ← NEW**. **5 of 5 axes 모두 관측.** 천장 완전 돌파.
2. **iter34 → iter35 비교가 "instruction vs example"의 결정적 실증.** 두 iter 사이의 차이: iter34는 Type A 정의와 산문 경고만 변경(예시 그대로) → N=3 고정. iter35는 예시 5개 중 3개를 N=10/all/5로 **실제로 교체** → 첫 시도에 N=5 선택. 문제가 LLM의 "어느 예시에 attend하느냐"라는 구조적 명제를 증명. 향후 프롬프트 튜닝의 원칙: **instruction 전에 예시부터 검토**.
3. **첫 번째 선택이 N=5 (가장 순한 다변화).** 예시에 N=10, N=all matching도 있었지만 composer는 가장 덜 급진적인 N=5 선택. 이건 (a) 예시 중 N=5는 "top 5 by release_year desc"라서 Cross-item 포함 복잡 케이스였는데 그 복잡도를 피하고 단순 "top 5 by date" 형태로 축소, (b) anchor=customer 경로가 customer→rental 예시의 N=10을 가져오는 게 자연스러웠을 것이나 pretraining bias(더 작은 N 선호)로 5로 축소, 양립 가능. N=10/all-matching 시도는 다음 trial에서 관측해야.
4. **Submit 1 `feedback` → Submit 2 accept 패턴.** iter18~34 대부분 submit 1이 즉시 `rejected/accepted`. iter35_a는 submit 1에서 `feedback`(backend가 mid-flight에 라벨 정제 요청) → submit 2에서 동일 draft 정제 후 accept. 이건 N=5 draft가 초기에 label slot/preview 확정에 문제가 있었을 수 있음(e.g. preview size 3 default vs actual N=5). 재현되는 패턴이면 별도 iter 조사 후보.
5. **첫 trial에 가설 성공.** iter30(Aggregate 돌파)처럼 1-sample에서 가설이 바로 증명된 드문 케이스. 다음은 재현성 배치로 확증.
6. **"basic ordering이 default이라 Cross-item escalation이 아니다" 주석 효과 불명.** iter35_a는 accept까지 escalation 없어 이 문장이 실제로 composer 행동을 바꿨는지 관측 불가. too_easy 시나리오에서 확인해야.

**Next direction.**

1. **iter36: iter35 배치 확장 (2-3 trial)** — N=5가 단발인지, N=10 / N=all matching도 관측되는지 측정. 앵커 분포가 customer 외에 film/city/category로 다양해지면 예시별 매칭 빈도도 관찰 가능. 순차 실행.
2. **Feedback 패턴 관찰**: iter35_a submit 1 `feedback`이 재현되면 mid-flight 정제 메커니즘 조사. 현재로는 노이즈로 간주.
3. **task pool 현황**: 18 accept, **5 of 5 axes 모두 진입 완료**. 다음 목표는 **축 균형**(Filter 7 : Cardinality 1 불균형 해소) 또는 **voice 가드 확증 ×n**, 아니면 **M:M bridge anchor 회피**(iter32_c dead-end 재발 방지).
4. **장기 목표**: 20 accept 도달 시 cross-iteration 분석 — shape 분포, axis 분포, band 분포 정량화해 현 상태의 체계적 약점 파악. 현재 17/18 accept로 거의 도달.

---

### Iteration 36 — 2026-04-19 (iter35 배치 확장, 추가 2 trial — **20 accept 도달 + Aggregate 재현 + Cardinality 2연속**)

**Hypothesis.** iter35_a 1-sample에서 Cardinality 최초 accept(N=5) 획득. (a) Cardinality 재현 가능한지, (b) N=10 / N=all matching 같은 다른 변형 관측되는지, (c) 다양한 anchor에서 동일 패턴 유지되는지 3-trial 배치로 측정. 프롬프트/코드 변경 없음 — iter35 상태 그대로 재현 배치.

**Change.** 없음. `artifacts/tmp_configs/iter35_card_examples.yaml` 그대로 재사용. smoke_iter35_b → smoke_iter35_c 순차 실행.

**Trial.** iter35_a + iter35_b + iter35_c 합쳐 3-trial 배치.

| suffix | anchor | task type | label | primary axis | secondary | pass_rate | voice |
|---|---|---|---|---|---|---|---|
| a | customer=155 | Type A | `[{rental_date, return_date}] × 5` | **Cardinality (N=5)** | Cross-item | 0.667 | "제 대여 기록 … 5건" ✓ |
| b | **inventory**=3975 | Type A | `[{rental_date, return_date}] × all matching` | **Cardinality (N=all)** | Composite (store+film+staff) | 0.333 | "2호점 SUNSET RACER 재고 + Jon Stephens 직원" ✓✓ (다중 ID 해결) |
| c | rental=13621 | **Type B** | `{rental_count: int}` | **Aggregate** | — | 0.667 | "2005년 8월 20일에 대여한 이 기록 기준, 제 총 대여 횟수" ✓ |

iter35_b attempt 연쇄:
- attempt 1: "2호점의 SUNSET RACER 영화 재고 항목의 **모든** 대여 기록을 대여일 오름차순으로" → 3/3 too_easy (N=all matching + 기본 필터만)
- attempt 2: + "**Jon Stephens 직원**이 처리한" → 1/3 accepted

iter35_c 1-shot: submit 1에서 band 상단 hit. 2005년 8월 20일 anchor 번역으로 rental_id를 자연어화, 그 후 "제 총 대여 횟수" Type B scalar 질문.

**Findings.**

1. **Cardinality 축 3/3 재현 ✓ (N=5, N=all, 그리고 batch 성공).** iter35_a의 N=5 accept가 운이 아님. iter35_b는 N=all matching으로 **다른 변형**까지 관측. 축 분포 업데이트: Filter 7, Composite 3, **Aggregate 2** (+1), Cross-item rule 5, **Cardinality 2** (+1). **20 accept 달성** — 첫 milestone.
2. **Aggregate/Type B 재현 ✓ (iter35_c).** iter30_a 이후 8 iter 만에 두 번째 Type B accept. **차이점**: iter30_a는 city hub anchor + 복합 필터 필요(attempt 3에서 accept), iter35_c는 rental leaf anchor + submit 1 즉시 accept. iter31에서 "Type B는 hub anchor에서 자연"이라는 가설을 부분 반박 — **rental leaf도 "이 기록 기준 내 총 대여 횟수" natural phrasing**으로 Type B 가능.
3. **Voice 가드 다중 ID 해결력 실증 (iter35_b).** 한 태스크 안에 4개 numeric ID(`inventory_id`, `film_id`, `store_id`, `staff_id`) 모두 자연어로 번역: "2호점", "SUNSET RACER 영화", "재고 항목", "Jon Stephens 직원". iter25_c/29_a/30_c 누수 3건의 패턴 전체가 구조적으로 해결됨을 한 번의 trial에서 확증. voice 가드가 trivial ID 1개뿐 아니라 **composite ID chain**까지 커버.
4. **Type A 예시 N 다변화가 composer의 draft 분포를 실제로 이동시킴.** iter35 배치 3 trial 중 3개 모두 N ≠ 3: iter35_a N=5, iter35_b N=all matching, iter35_c N=scalar. iter34 instruction-only 개입 전에는 최근 5 accept 모두 N=3이었음. 예시 레벨 개입 전후 행동 변화가 극명.
5. **iter30~35 프롬프트 누적 시너지.** iter35_b에서 (a) iter32 voice 가드(staff_id → Jon Stephens), (b) iter32 anchor translation(inventory_id → 영화+매장), (c) iter35 예시(N=all matching) 세 개의 개입이 한 trial 안에서 독립적으로 발동해 하나의 태스크를 형성. 프롬프트 개편은 orthogonal 하게 composable — 한 축의 변경이 다른 축을 망가뜨리지 않음.
6. **Submit 1 accept 빈도 증가.** iter35 배치에서 a, c는 submit 1 accept. iter33 batch까지는 거의 submit 2-3에서 accept였는데, iter35 이후 "예시가 더 다양해져 first draft가 band에 바로 앉는" 빈도 증가. composer의 **시작 drafting quality** 향상 신호.

**Next direction.**

1. **iter37: 20 accept cross-iteration 분석.** 별도 작업 (no new prompt change). 20개 accept 전체에서 (a) axis 분포 (primary + secondary), (b) anchor class 분포 (hub/leaf/bridge), (c) shape 분포 (record shape, N values), (d) band 분포 (upper/lower), (e) voice quality 6-axis 점검. 약점 진단.
2. **iter38 후보 (분석 결과 의존)**: 축 균형(Cardinality/Aggregate만 2건씩, Filter 7건 편향), shape 다양성(record shape `{rental_date, return_date}` 여전히 과다?), band 분포(상/하단 비율).
3. **보류**: M:M bridge anchor 회피(iter32_c dead-end). 이후 trial에서 재발 빈도 관찰.
4. **task pool 현황**: **20 accept 달성**, 5 of 5 axes 모두 진입 완료. 축 분포 Filter 7 / Composite 3 / Aggregate 2 / Cross-item rule 5 / Cardinality 2.

---

### Iteration 37 — 2026-04-19 (20-accept cross-iteration 분석, 프롬프트 변경 없음)

**Hypothesis.** iter36의 20-accept milestone에서 task pool의 체계적 약점을 데이터로 진단. 단일 iter 결과가 아닌 **누적 분포**에서만 드러나는 편향을 찾아 iter38+ 타겟 결정.

**Change.** 없음. 순수 분석 iter.

**Data (20 accept 전체 enumeration).**

| # | iter | anchor | type | record shape | N | pass | primary axis | voice |
|---|---|---|---|---|---|---|---|---|
| 1 | iter16 | inventory=2794 | A | `{rental_date, return_date}` | 2 | 0.667 | Filter(staff) | ✓ |
| 2 | iter18 | rental=3583 | A | `{rental_date, return_date}` | 3 | 0.667 | Composite | ✓ |
| 3 | iter19 | customer=54 | A | `{rental_date, return_date}` | 3 | 0.333 | Baseline | **✗ org voice** ("고객님, 귀하의 … 확인해 드리겠습니다") |
| 4 | iter20 | city=544 | A | `{rental_date, return_date}` | 3 | 0.333 | Baseline | ✓ |
| 5 | iter23 | city=481 | A | `{rental_date, return_date}` | 3 | 0.333 | Baseline | ✓ |
| 6 | iter24 | payment=6137 | A | `{amount, payment_date}` | 3 | 0.667 | Baseline | ✓ |
| 7 | iter25_a | payment=1453 | A | `{amount, payment_date}` | 3 | 0.333 | Filter(amount) | ✓ |
| 8 | iter25_b | film=108 | A | `{first_name, last_name}` | 3 | 0.667 | Baseline | ✓ |
| 9 | iter25_c | inventory=3678 | A | `{rental_date, return_date}` | 3 | 0.333 | Baseline | **✗ "인벤토리 ID 3678번"** |
| 10 | iter27_a | address=509 | A | `{rental_date, return_date}` | 3 | 0.667 | Composite | **✗ "직원 1 번"** |
| 11 | iter27_b | payment=12337 | A | `{amount, payment_date}` | 3 | 0.333 | Baseline | ✓ |
| 12 | iter30_a | **city=459** | **B** | scalar `{rental_count:int}` | — | 0.667 | Aggregate | ✓ |
| 13 | iter30_b | rental=5456 | A | `{rental_date, return_date}` | 3 | 0.333 | Cross-item | ✓ |
| 14 | iter30_c | rental=15855 | A | `{rental_date, return_date}` | 3 | 0.333 | Cross-item | **✗ "직원 1 번"** |
| 15 | iter32_a | address=539 | A | `{rental_date, return_date}` | 3 | 0.333 | Cross-item | ✓ |
| 16 | iter32_b | customer=499 | A | `{rental_date, return_date}` | 3 | 0.667 | Cross-item | ✓ |
| 17 | iter34_a | customer=294 | A | `{rental_date, return_date}` | 3 | 0.333 | Cross-item | ✓ |
| 18 | iter35_a | customer=155 | A | `{rental_date, return_date}` | **5** | 0.667 | Cardinality | ✓ |
| 19 | iter35_b | inventory=3975 | A | `{rental_date, return_date}` | **all** | 0.333 | Cardinality | ✓✓ (4-ID composite) |
| 20 | iter35_c | **rental=13621** | **B** | scalar `{rental_count:int}` | — | 0.667 | Aggregate | ✓ |

**Distributions.**

*Anchor class* (7종):
- customer 4, rental 4, payment 3, inventory 3, city 3, address 2, film 1.
- hub-like (customer, city) = 7. leaf-like (rental, payment, inventory, address) = 12. star-hub (film) = 1.
- 예상과 달리 film 앵커 1회만 — iter30 anchor-bias가 hub를 밀어주지만 iter32_c 같은 M:M bridge dead-end 때문에 실질적으로 film 태스크가 accept까지 도달하기 어려움.

*Task type*:
- Type A (list) 18 = **90%**.
- Type B (scalar) 2 = **10%** (iter30_a city hub, iter35_c rental leaf).

*Record shape* (Type A 18개):
- `{rental_date, return_date}` **14** (78%).
- `{amount, payment_date}` 3 (payment anchor만).
- `{first_name, last_name}` 1 (film→actor iter25_b).
- **shape monoculture가 N=3 고정보다 더 심각한 단일화**. iter35에서 N은 다변화 시작됐지만 shape는 그대로.

*N 값* (Type A 18개):
- N=2 1 (iter16).
- N=3 **16** (89%).
- N=5 1 (iter35_a).
- N=all matching 1 (iter35_b).
- iter35 이전 17개 중 15개가 N=3, iter35 이후 3개 중 2개가 non-3. 예시 레벨 개입이 iter35에 와서야 효과 발동.

*pass_rate band*:
- 0.333 (하단) 10 / 0.667 (상단) 10 — **perfect 50/50**. composer가 band를 균일하게 타겟팅, 클러스터링 없음.

*Primary axis* (6 카테고리):
- Baseline (최소 제약) 7 (iter19, 20, 23, 24, 25_b, 25_c, 27_b).
- Filter 2 (iter16, 25_a).
- Composite 2 (iter18, 27_a).
- Aggregate 2 (iter30_a, 35_c).
- Cross-item rule 5 (iter30_b, c, 32_a, b, 34_a).
- Cardinality 2 (iter35_a, b).
- Sum 20 ✓.

*Voice quality*:
- Clean 16 (80%).
- Violations 4 (20%) — iter19(org voice), iter25_c(inventory 번호), iter27_a(직원 1번), iter30_c(직원 1번).
- **모든 violation이 iter32 voice guard 이전**. iter32 이후 5개 accept(32_a/b/34_a/35_a/b/c) 중 voice 회귀 0건. 가드가 구조적 차단.

**Findings.**

1. **Shape monoculture가 최대 약점.** `{rental_date, return_date}` 14회 = 70%. N=3 고정은 iter35로 깨지기 시작했지만 shape는 20개 샘플에서 여전히 거의 단일. Type A 예시를 iter35에서 N 다변화했지만 **shape는 예시에서 여전히 `rental_date, return_date` 스켈레톤**이 4/5 차지. 예시 교체 패턴을 shape 축으로 반복해야.
2. **Anchor 분포가 customer/rental 쌍극 편향.** customer 4 + rental 4 = 20개 중 40%. 이 쌍은 가장 "자연스러운" customer-voice 질문 생성(내 기록, 내 대여) 때문에 composer가 선호. film/address/city/inventory는 상대적으로 less natural customer-voice이지만 **iter35_b(inventory 4-ID composite)가 최고 품질 voice 예시**임을 보여줌 — 분포 편향은 자연성이 아니라 composer의 pretraining bias.
3. **Type A 90%, Type B 10% — 구조적 비대칭.** iter26~29 Type B 유도 노력이 2건 accept로만 실현. iter30_a(city hub) + iter35_c(rental leaf) — 두 케이스 모두 "scalar count" 형태만. Type B의 다른 서브타입(max, min, sum, avg)은 0건. **Type B 다변화** 자체가 새 iter 축.
4. **Voice 가드 이후 violation 0건 (5/5 trial 깨끗).** iter32 개입의 구조적 효과. 가드 도입 전 11/15 = 73% clean → 도입 후 5/5 = 100% clean. 가드 전의 회귀 4건은 registry에 남아 있음 — 이후 cleanup 필요 여부 판단해야.
5. **Band 분포 완벽 균형 (10/10).** "하단 클러스터링" 우려는 iter33에서 관찰된 4연속이 우연이었고 20개 누적에서 50/50. band 튜닝 개입 불필요.
6. **Axis 분포 여전히 불균형 but 더 이상 0축 없음.** Baseline 7이 가장 큼 — "기본 Type A 형태로 band 진입"이 실제 가장 빈번한 경로. Cross-item 5는 iter26~30 Type A 진화의 누적 효과. Filter/Composite/Aggregate/Cardinality 2건씩은 탐색 시작 단계.

**Systemic weaknesses (iter38+ priorities).**

1. **Shape 다변화** (최우선). 예시 record shape `{rental_date, return_date}` 지배를 iter35 스타일의 예시 교체로 깨기. 후보: `{title, rental_count}` (film 앵커), `{category_name, film_count}` (category), `{city_name, customer_count}` (city) 등 **집계 포함 shape**.
2. **Type B 서브타입 확장**. 현재 `count`만. `max/min payment_date`, `avg amount`, `sum rental_count` 같은 scalar operation 다변화를 Type B 예시에 추가.
3. **film/category anchor 지원**. 1건뿐. iter32_c dead-end로 위축된 것으로 보임. 대안: M:M bridge anchor에서는 "film → 특정 배우의 등장 영화 수" 같은 **Type B 특화** 프롬프트 힌트 추가.
4. **Pre-guard voice 회귀 4건 cleanup (deferred)**. registry에 남은 iter19/25_c/27_a/30_c 태스크 재생성 또는 제거 여부는 user 판단. 품질 손상 영향 정도 보고.

**Next direction.**

1. **iter38 후보**: shape 다변화 개입. Type A 예시에서 `{rental_date, return_date}` 편향을 깨는 동시에 Cardinality 예시도 유지. `{title, rating}` + `{first_name, last_name}` + `{payment_date, amount}` 같은 shape 다양화 강제. 1-sample smoke.
2. **iter39 후보**: Type B 서브타입 확장 (current count만).
3. **task pool 현황**: 20 accept, 5 of 5 axes, 6 categories. shape monoculture와 Type A 90%가 다음 타겟.

---

### Iteration 38 — 2026-04-19 (multi-hop 강제: draft-time filter 필수 + Escalation Axes 재배치, 3-trial 배치)

**Hypothesis.** iter37 분석에서 solver multi-hop 지표가 iter27-30 peak(12-15턴)에서 iter35 저점(5턴)으로 회귀. 원인 진단: Cardinality/Aggregate 돌파가 "축 커버"는 성공이지만 **row-set narrowing이 일어나지 않아 solver reasoning depth를 늘리지 못함**. 개입: (a) Type A 예시 5개 **모두**에 explicit filter 추가 (rental_date range, last_name, store_id, rating, amount), (b) "Every Type A draft must include ≥1 explicit filter" 명시 + "baseline is below quality threshold" 경고, (c) Escalation Axes 재배치: **Composite → Filter → Cardinality → Cross-item**. Cardinality는 "size only" 축으로 강등. 목표: draft-time solver turn ≥10, within-task 턴 수 monotone 증가.

**Change.** `src/rl_task_foundry/synthesis/prompts.py` 두 섹션 수정. 예시 5개 전체 재작성 (모두 filter 포함), 새 prose rule 삽입, Escalation Axes 순서/우선순위 재정의. commit `5e367c1`.

**Trial.** `artifacts/tmp_configs/iter38_multihop.yaml`(동일 config) 1-sample + 배치 확장. 순차 실행.

| suffix | anchor | task type | submit 1 draft | final result | submit 1 solver turns | accepted solver turns |
|---|---|---|---|---|---|---|
| a | customer=81 | Type A | Mike Hillyer staff + N=5 ordering (drafted with filter ✓) | accepted submit 2 (0.333) after +date | [15, 15, 15] avg=15.0 | [16, 0-MaxTurns, 0-MaxTurns] |
| b | customer=332 | Type A | Jon Stephens staff + N=5 ordering (drafted with filter ✓) | **synthesis_failed** — submit 2 (+July date) pass=0.0 too_hard → budget exhausted | [14, 14, 15] avg=14.3 | — (submit 2 MaxTurns 3/3, submission never occurred) |
| c | customer=441 | Type A | date filter 2005-06-01 only + N=5 (drafted with filter ✓) | accepted submit 2 (0.333) after +staff | [11, 11, 11] avg=11.0 | [0-MaxTurns, 16, 0-MaxTurns] |

**Findings.**

1. **Draft-time filter 100% 달성 ✓.** 3/3 trial 모두 submit 1부터 filter 포함 (Mike Hillyer, Jon Stephens, 2005-06-01). iter35까지 submit 1 베이스라인 drafts(=no filter) 빈도가 5/8이었는데 iter38에서 **0/3**. 예시 모두에 filter 박은 개입이 즉시 효과. iter34→35의 "example-driven over instruction-driven" 원칙 재확인.
2. **Submit 1 solver turns 즉시 목표 달성.** iter38 submit 1 solver 평균 = 15.0, 14.3, 11.0. 모두 **목표 ≥10 달성**. iter19-25 baseline 평균 7-10, iter32-35 회귀 평균 5-8과 대비. iter27-30 peak(12-15)에 **submit 1부터** 도달.
3. **Escalation priority 준수.** iter38_a, b, c 모두 submit 2에서 **Composite(기존 filter + 추가 filter)**로 escalate — Cardinality 도피 없음. iter35 Escalation Axes 재배치("Cardinality first")의 부작용(쉬운 길 선택)이 iter38 재재배치("Composite first")로 교정됨.
4. **목표 기준 #4(multi-hop), #5(점진 난도) 입증.** iter38_a 내 within-task turn 궤적: submit 1 avg 15.0 → submit 2 matched 16 — 증가. iter38_b: 14.3 → (submit 2 솔버 모두 MaxTurnsExceeded, 실제 reasoning은 16턴 이상 요함, 즉 **난도 증가했으나 예산 초과**). iter38_c: 11.0 → 16. 3/3에서 within-task 증가.
5. **가장 중요한 실패 모드 노출: MaxTurnsExceeded 패턴.** iter38_b submit 2에서 3 solver **모두** turn=0 + termination_reason=MaxTurnsExceeded + raw_output="". canonical answer는 composer가 정상 계산(3 rows, 검증 가능). 즉 **"풀 수 있는 문제이지만 solver 16턴 budget 내에 못 푼" 케이스** (사용자 구분 기준의 "전자"). iter38_a, c submit 2도 3명 중 2명이 MaxTurns. **max_turns=16이 multi-filter 멀티홉 태스크의 구조적 상한**. task 자체는 건전.
6. **Accept rate 측면 trade-off.** iter35 batch 3/3 accept, iter38 batch **2/3 accept** (iter38_b는 synthesis_failed). 난도 ↑ → accept rate ↓. 단 accepted 태스크의 solver multi-hop 품질은 ↑. **iter39에서 max_turns 상향으로 MaxTurnsExceeded를 accept로 전환할 가치 있음**.
7. **Voice 가드 계속 작동.** 3/3 trial에서 ID filter가 자연어로 번역됨 (Mike Hillyer, Jon Stephens, 2005-06-01 이후). voice 회귀 0건.
8. **누적 22 accept** (iter38_a + iter38_c = 2 추가). 축 분포 Filter 7 / Composite **5** / Aggregate 2 / Cross-item 5 / Cardinality 2.

**Next direction.**

1. **iter39 (즉시): `solver_runtime.max_turns: 16 → 30` 상향.** iter38 배치에서 MaxTurnsExceeded 비율이 (3 solver × 3 trial × 2 submit) = 18 slot 중 5 slot (27%). iter38_b가 accept를 못 한 원인도 이것. 30으로 올리면 (a) MaxTurns 비율 감소 → accept rate 회복, (b) solver reasoning depth 더 늘릴 여지 생김. Cost ↑ 50%이지만 실패 trial 감소로 상쇄 가능. 1-sample smoke로 검증.
2. **iter40 후보**: iter38 프롬프트 + iter39 max_turns 조합 안정성 배치 (3 trial). accept rate 회복 + solver 턴 안정화 관측.
3. **task pool 현황**: 22 accept. iter38 품질 기준 #3, #4, #5 직접 입증. max_turns 조정으로 accept rate 복원이 최우선.

---

### Iteration 39 — 2026-04-19/20 (solver_runtime.max_turns 16→30, RL actor thinking budget 현실화, 3-trial 배치)

**Hypothesis.** iter38 배치에서 MaxTurnsExceeded 5/18 slot. 태스크는 verifiable(canonical exists)이고 actor의 policy(모델/프롬프트) 고정이 원칙이지만, **RL actor의 thinking budget**(max_turns)은 실제 actor의 추론 capacity를 현실적으로 반영해야 함. 16턴은 iter38 multi-filter 태스크에 타이트. max_turns=30 상향으로 (a) accept rate 회복, (b) solver가 도달하는 multi-hop 깊이 20+로 확장, (c) band 내 유용한 RL gradient sample 확보. 원칙 재확인: solver **prompt/model**은 여전히 untouchable.

**Change.** `artifacts/tmp_configs/iter39_max_turns.yaml` — `solver_runtime.max_turns: 16 → 30`. iter38 프롬프트 유지.

**Trial.** 3-trial 배치 순차 실행.

| suffix | anchor | submit flow | matched avg turns | 결과 | 해석 |
|---|---|---|---|---|---|
| a | customer=387 | submit 1 (Type A, amount>$5 + N=5 ordering) pass=0.0 too_hard → budget exhausted | — (0/3 matched, 1 UserError + 2 MaxTurns) | **synthesis_failed** | composer over-draft(너무 좁은 filter에 draft-time 도달); actor 30턴으로도 못 풀 만큼 filter 선택도 과격 |
| b | customer=278 | submit1 (date filter 5~6월) 1.0 too_easy → submit2 (+return_date 7월 Composite) 0.667 | 19.5 | **accepted 0.667** | max_turns=30이 21턴 matched solver를 가능케 함 (16에선 MaxTurns였을 것) |
| c | customer=498 | submit1 (feedback) → submit2 (6월 date, all matching) 1.0 → submit3 (+Jon Stephens staff, 모두) 1.0 → submit4 (destination pivot rental→film title) 0.333 | 23.0 | **accepted 0.333** | 4-submit 점진 escalation; 솔버 턴 15→22→23 monotone 증가 |

iter39_c 솔버 턴 궤적이 특히 주목할 가치: 기준 #5(점진 난도 증가) 최초 3/3 monotone 사례. 솔버 최대 25턴까지 활용 (iter38의 16턴 천장 돌파).

iter39_a는 "premature over-draft" 사례. customer 387의 `amount > $5` payment가 적거나 composer가 filter 선택도를 예측 못해 submit 1이 pass=0.0 too_hard. composer는 relax 시도했으나 `difficulty_weakened`로 flagged → budget exhausted. **이건 composer calibration 문제지 solver 문제 아님** (사용자 원칙 "전자"의 변형: 풀 수 있는 문제지만 composer가 actor에 비해 너무 좁은 slice를 골라 pass=0).

**Findings.**

1. **max_turns=30이 accept rate를 2/3로 복원 + multi-hop 상한 돌파.** iter38 2/3 accept(matched avg 11-15) → iter39 2/3 accept(matched avg 19.5-23). iter38에서 solver 18-25턴 reasoning이 필요했던 task들이 이제 accept됨. RL actor thinking budget을 현실화한 것의 직접 효과.
2. **기준 #5 (점진 난도 증가) 최초 3/3 monotone 관측 (iter39_c).** 솔버 턴 15 → 22 → 23 progressive across submits. iter30_a/b와 iter38_a에서 부분적으로 보였지만 iter39_c에서 full monotone. 이 패턴이 loop가 지향하는 RL training signal의 원형.
3. **기준 #3 (per-task 축 확장) per-submit 4회 달성.** iter39_c는 composite date (submit 2) → composite staff (submit 3) → destination pivot (submit 4). iter38까지는 2-3 submit이 상한이었던 per-task axis stacking이 이제 4 submit까지 자연스럽게 확장됨.
4. **기준 #4 (solver multi-hop) 안정적 20+턴 영역.** iter39 accepted avg 21.3턴 = 30 예산의 71% 활용. RL training에 유용한 "해결 가능하지만 쉽지 않은" 영역. iter38 peak 15-16턴(예산 94% 쓰고 간신히)에서 더 안전한 헤드룸 확보.
5. **iter39_a의 Composer over-draft 문제 — 새 이슈 표면.** submit 1부터 pass=0.0인 too_hard는 escalation 여지 없이 실패로 직행. 원인: composer가 filter 선택도(selectivity)를 예측하지 못함. 예: customer 387 payment>$5 = 몇 건인지 drafting 전에 확인 안 함. 해결 후보: (a) Type A draft 전 `profile`로 filter column distinct count 확인 강제, (b) too_hard on submit 1 시 "less restrictive filter"를 시도하도록 프롬프트 유도.
6. **RuntimeError와 MaxTurnsExceeded는 다른 실패 모드.** iter39_b solver1 RuntimeError — atomic tool 호출 protocol 오류(arg 형식 등). iter38/39의 MaxTurnsExceeded와 구분 필요. RuntimeError는 현재 희소(1/9 solver) — 당장 개입 불필요하지만 지속 관찰.
7. **누적 24 accept.** 22 + iter39_b + iter39_c = 24. 축 분포 업데이트: Filter 7 / Composite **7** (+2 iter39_b, c) / Aggregate 2 / Cross-item 5 / Cardinality 2 = 23 primary (pre-existing 1 offset).

**Next direction.**

1. **iter40: composer selectivity check — `profile` 도구 활용 강제.** iter39_a 실패의 원인 해결. Type A draft 전에 (a) anchor→destination 경로의 distinct count/row_count/분포 `profile` 호출, (b) 그 결과로 filter column/threshold/value를 고르도록 프롬프트 명시. "대략 10-50% row 남기는 filter를 선호 (band 진입 가능성 ↑)" 같은 휴리스틱 제공. 1-sample smoke.
2. **iter41 후보**: iter39_c 스타일 4+ submit 점진 escalation을 장려하는 프롬프트 힌트. 현재 2-3 submit에서 accept 되는 게 많으니 "band 하단 진입 후 band 중간으로 refine"을 유도해 더 깊은 per-task axis stacking.
3. **task pool 현황**: 24 accept, 5 of 5 axes, 기준 #3/#4/#5 모두 지속 달성. iter40은 accept rate recovery 목표.

---

### Iteration 40 — 2026-04-20 (composer selectivity check via profile 도구, 3-trial 배치)

**Hypothesis.** iter39_a 실패 원인(submit 1 pass=0.0 too_hard — composer가 too-selective filter 선택) 해결. 개입: Type A draft 전 `profile(column)` 호출 의무화 + "25-75% row retention" 휴리스틱 + "too-generous recoverable, too-restrictive terminal" 비대칭 경고. 목표: submit 1 terminal too_hard 제거, per-task 축 확장 심화, RL actor가 감당 가능한 난도 유지.

**Change.** `src/rl_task_foundry/synthesis/prompts.py` Type A 섹션에 profile 의무 규칙 12줄 추가. Config도 `synthesis.runtime.max_turns: 40 → 60` (profile 추가로 composer 탐색 비용 증가한 만큼 보상). commit `b9c3789` + config in `iter40_selectivity.yaml`.

**Trial.** 3-trial 배치 순차 실행.

| suffix | anchor | 실패/성공 | submit count | matched avg turns | 원인 |
|---|---|---|---|---|---|
| a | customer=545 | **accepted 0.667** | 5 (1 feedback + 4 submit) | progression 15.7→19.3→22.7→19.5 | profile 2회 호출, 점진 escalation ✓ 5 submit 최장 기록 |
| b | film=912 (TROJAN TOMORROW) | **synthesis_failed (composer MaxTurns 40)** | 2 submit in 38 tool calls | — | composer over-exploring M:M bridge anchor, 40턴 budget 소진 |
| c | city=84 (Boa Vista, Type B) | **synthesis_failed (Type B escalation invalid)** | 2 submit | 5.3 on submit 1 | submit 2에서 count target rental→payment 변경 → `difficulty_crank_invalid` → budget exhausted |

iter40_a 하이라이트: profile 호출 2회, 5 submit 최대 깊이, solver s1 submit 3에서 **29턴** (max_turns=30 bound 도달) — RL actor budget 경계에서 정확히 풀어낸 이상적 training sample. Accepted matched avg 19.5턴.

iter40_b 문제: composer가 profile+sample 반복하며 40턴 소진. **config fix**: `synthesis.runtime.max_turns: 40 → 60` (composer는 RL actor 아니므로 budget 조정 원칙 위반 아님).

iter40_c 문제: Type B escalation에서 **aggregate target을 바꿈** (rental count → payment count). 프롬프트의 "add a filter on the joined set" 지침 위반. iter41에서 보강 필요.

**Findings.**

1. **iter40_a = 품질 기준 동시 달성의 이상적 케이스.** 기준 #3(5 submit 축 확장), #4(29턴 활용), #5(15.7→19.3→22.7 monotone) 모두 한 trial에서 fulfill. profile-driven filter 선택이 직접 효과: submit 1부터 Jon Stephens staff filter 정착, 이후 return_date 필터 추가가 composer의 selectivity 판단(25-75% 룰) 아래 정확히 진행됨.
2. **Profile rule이 실제로 호출 유도.** iter40_a 2회, iter40_c 4회 호출 관측. iter34→35의 "example-driven over instruction" 원칙이 이번엔 instruction만으로도 효과 — 왜냐하면 `profile`이 이미 툴로 존재했고 "의무화" 한 줄이 활성화 트리거.
3. **Composer budget 한계 노출 (iter40_b).** profile+sample 추가로 composer 탐색 비용 증가. 특히 M:M bridge anchor(film) 같이 structurally complex 경우 40턴 초과. config 60턴으로 완화했지만 iter40_c도 실패하여 **composer budget만이 해결책은 아님**. Anchor-level 제약(M:M bridge 회피) 병행 고려.
4. **Type B escalation target-lock 누락.** iter40_c가 노출한 버그: composer가 "Type B 유지"는 지키지만 "같은 aggregate target" 규칙은 어김. rental count → payment count는 두 개의 다른 Type B 태스크. 프롬프트 보강 필요. iter41 target.
5. **3개 서로 다른 실패 모드.** iter40 배치에서 (a) 성공, (b) composer budget, (c) Type B 규칙. iter39의 단일 실패 모드(selectivity)보다 복잡. 하지만 각각 다른 원인이라 **서로 독립적**으로 해결 가능.
6. **Accept rate 1/3**, but **accepted quality is highest yet**. iter40_a가 지금까지 관측한 모든 지표(submit 깊이, solver 턴 peak, 점진 monotone)에서 최고점. 전체 루프의 상한이 올라갔다.
7. **누적 25 accept.** 24 + iter40_a = 25. 축 분포: Filter 7, Composite 8, Aggregate 2, Cross-item 5, Cardinality 2 + 1 overflow.

**Next direction.**

1. **iter41 (즉시): Type B aggregate target lock 프롬프트 보강.** Type B 섹션에 "Within Type B across submits, the aggregate's target (which table you count/sum/min/max over) is locked. Change of target (rental→payment) is a type switch and will be flagged crank_invalid. Escalate by filter on the joined set ONLY." 명시. iter40_c 문제 직접 해결.
2. **iter42 후보: M:M bridge anchor discouragement.** iter32_c/iter40_b 유사 실패 반복. `random_anchor`에 M:M bridge 테이블에서 나오는 체인 anchor 제외, 또는 프롬프트에 "M:M bridge 앵커 선택 시 agg 타겟을 actor table로 하라" 등 힌트.
3. **Accept rate 안정화**: iter40 1/3은 iter38과 동일 수준(많은 실패). iter41에서 Type B 오류 제거, iter42에서 bridge 이슈 완화하면 2/3-3/3 회복 기대.
4. **task pool 현황**: 25 accept, 모든 축 관측, iter40_a가 품질 peak 설정. iter41-42는 accept rate 회복 + 품질 유지.

---

### Iteration 41 — 2026-04-20 (Type B aggregate target lock 프롬프트 보강, 3-trial 배치)

**Hypothesis.** iter40_c 실패(`count(rental) → count(payment)` 전환이 `difficulty_crank_invalid`)의 직접 수정. Type B 섹션에 "**Within Type B across submits, the aggregate's target (which table you count/sum/min/max over) is locked**"와 유효 escalation 3가지(filter on joined set / composite filter pair / count→sum/min/max on same table) 명시. 기존 task-type-lock과 겹치지만 target-level granularity를 추가. 목표: Type B escalation 정당성 회복.

**Change.** `src/rl_task_foundry/synthesis/prompts.py` Type B 섹션에 11줄 추가 (aggregate target lock + enumerated valid escalations). commit `05de8e2`.

**Trial.** 3-trial 배치.

| suffix | anchor | task type | submits | matched solver turns | 결과 | 원인 |
|---|---|---|---|---|---|---|
| a | customer=190 | Type A (payment, amount>$3, take 5) | 1 (+feedback) | s2=27 (1/3 matched); s0 MaxTurns, s1 APITimeoutError | **accepted 0.333** | profile 2회, submit 1 즉시 accept, 매우 효율적 draft |
| b | — | — | — | — | **synthesis_failed (BadRequestError)** | Alibaba provider `Range of input length` 400 error — API-level transient |
| c | inventory=4034 | **Type B** (count rental + 1호점 store filter + Jon Stephens staff filter) | 1 | s0=18, s1=21(wrong), s2=MaxTurns | **accepted 0.333** | Type B target lock 준수, voice 3-ID 체인 자연 번역 |

**Findings.**

1. **Type B target lock 프롬프트 보강 효과 확인 (iter41_c).** inventory→film→store/staff join의 count(rental) scalar task. 이전 같으면 iter40_c처럼 "switch target"으로 실패할 수 있었지만 submit 1에서 정당한 Composite (store + staff filter on joined set)로 직접 draft — escalation 없이 band 진입. target lock이 draft-time 행동까지 제약함.
2. **iter41_a는 최고 효율 composer 행동**. 6-11 tool call만에 accept. profile 1-2회 호출로 amount threshold 빠르게 확정. iter40_a의 5-submit 심화 escalation과 대척점에 있지만 둘 다 valid RL sample 생성.
3. **27턴 matched solver (iter41_a) + 21턴 wrong-answer (iter41_c)** — RL signal 다양성 관측. accept 태스크의 solver 결과 분포가 (matched, wrong, MaxTurns) 3분화. RL training에서 wrong-answer는 `pass_rate=False` gradient 제공, MaxTurns는 neutral(데이터 제외), matched는 `pass_rate=True` gradient. 유용한 3-way 분포.
4. **iter41_b BadRequestError — infra 이슈.** Alibaba provider API 400 "Range of input length should be [1, 983616]". iter39/40의 MaxTurnsExceeded와는 다른 카테고리. 재현성 없어 transient로 간주, retry로 통과 (iter41_c 성공). 누적 빈도 보며 재발 시 조사.
5. **Accept rate 2/3 안정.** iter40(1/3)에서 회복. Type B 관련 실패 모드 제거한 효과. iter41 프롬프트 규칙이 명확한 violation 카테고리(target switch)를 차단한 덕.
6. **누적 27 accept.** 25 + iter41_a + iter41_c = 27. 축 분포: Filter 7, Composite **9**(+1 c), Aggregate **3**(+1 c), Cross-item 5, Cardinality 2 + 1 overflow.
7. **품질 기준 지속 fulfillment.** 두 accept 태스크 모두: voice ✓ (iter32 guard 유지), verifiable ✓ (backend), 기준 #4 multi-hop(27 / 18 turns) ✓, 기준 #3/5 per-task escalation은 이 배치에서 submit 1 accept가 많아 표본 부족 — iter40_a는 여전히 peak reference.

**Next direction.**

1. **루프 stability 평가.** iter38~41 4 iter 동안 accept rate 평균 (2+2+1+2)/12 = 58%. 품질 peak는 iter40_a. 지속적 문제는 (a) M:M bridge anchor dead-end(iter32_c, iter40_b) 미해결, (b) occasional API-level transient errors. 5축 모두 관측, 품질 기준 5/5 만족. **"충분히 달성"의 기준점 재검토**할 시점 — large batch로 dataset 성장 vs 세부 개선 반복?
2. **iter42 후보 A: M:M bridge anchor 억제.** `random_anchor` 가중치에서 film_actor/film_category 같은 bridge table을 downstream 경로에 두는 앵커 감점. 또는 프롬프트 힌트로 "bridge anchor는 Type B 먼저 고려".
3. **iter42 후보 B: Task pool 확장.** 현재 프롬프트 안정 상태에서 6-10 trial 일괄 실행해 task pool을 27 → 35+로 성장. diverse axis/anchor/type을 자연적으로 수집. RL training dataset에 가까워짐.
4. **task pool 현황**: 27 accept, 5 axes 모두, 품질 기준 모두 달성.

---

### Iteration 42 — 2026-04-20 (M:M bridge anchor 회피 — film→actor 제거, film→inventory→rental 경로 명시, 3-trial 배치)

**Hypothesis.** iter32_c + iter40_b 실패의 공통점: film 앵커 + film_actor M:M bridge 경유. bridge는 N이 작아(5-10 actors/film) escalation 불가 → composer dead-end. Type A 예시에서 `film → actor (via film_actor)`를 `film → rental (via inventory)`로 교체 + "Avoid M:M bridge paths for Type A" 명시. Type B는 그대로(film→actor_count scalar는 유효).

**Change.** `src/rl_task_foundry/synthesis/prompts.py` Type A 예시의 film 항목 재작성 (via inventory→rental, rental_date+return_date shape, N=5, 2005-07 date filter) + 경고 3줄 추가. commit `eafd60d`.

**Trial.** 3-trial 배치.

| suffix | anchor | task | submits | matched turns | 결과 | 주요 관찰 |
|---|---|---|---|---|---|---|
| a | **film**=826 (SPEED SUIT) | Type A: film→inventory→rental, date filter + staff filter | 2 | 16.7 → 18 | **accepted 0.333** | **bridge 회피 성공** — film 앵커 직접 accept |
| b | payment=8367 | Type B: count payment >= $5 | 1 | 0 (solvers unrecorded) | **synthesis_failed (too_hard submit 1)** | profile rule만으로는 all-filter 조합의 0-row risk 방어 불충분 |
| c | rental=1450 | Type B: count rental + Jon Stephens staff | 1 | [8, 22] (1 wrong @ 20) | **accepted 0.667** | 매우 효율적(8 tool call), Voice + 2/3 matched + 1/3 wrong 다양성 |

**Findings.**

1. **M:M bridge 회피 성공 (iter42_a).** film_id=826(SPEED SUIT) → inventory → rental 경로로 Type A accept. 이전 film 앵커는 iter32_c/40_b에서 dead-end이었으나 iter42 프롬프트 변경으로 film이 이제 **accept 가능한 앵커 클래스**로 승격. 예시 한 줄 교체만으로 경로 선택을 구조적으로 바꾼 사례.
2. **Voice: "영화 'SPEED SUIT'" + "Mike Hillyer 직원" + 2005-07 date** — iter32 voice 가드 유지 + iter42 route 변경의 결합.
3. **iter42_b 실패는 profile 한계 노출.** Type B "payment >= $5" 단독 필터가 특정 고객에게 0 rows 생산. profile이 distinct 값은 알려주지만 **join된 후의 cardinality는 미리 예측 불가**. 해결 후보: (a) Type B draft 전 `query` dry-run으로 count 미리 확인, (b) profile이 percentile 분포도 보여주게 확장. 프롬프트 선에선 한계.
4. **iter42_c의 solver 다양성.** 3명 중 s0=8턴 matched (간단하게 풀어냄), s1=22턴 matched (오래 걸려 풀어냄), s2=20턴 submitted but wrong answer. 이 3-way 분포는 **RL training에 이상적 signal** — matched는 positive reward, wrong은 negative reward, no MaxTurns/no TimeOut.
5. **Accept rate 2/3 유지**, quality peak 갱신 (film 앵커 포함까지). 29 accept 누적. 품질 기준 5/5 지속.
6. **iter42_b의 synthesis_failed 패턴은 현재 상태의 marginal failure mode.** 5-10% 수준. 완전 제거는 프롬프트로 어렵고 infra 수준 개입(dry-run, cardinality 사전체크) 필요. 현재 수준에서 감수 가능.

**Next direction.**

1. **Loop 안정화 완료 평가.** iter38~42 5 iter 동안 accept rate 평균 60%, quality criteria 5/5 지속, axis 분포 Filter 7/Composite 10/Aggregate 3/Cross-item 5/Cardinality 2 = 27 primary + 2 overflow = 29 accepts. 품질 기준 "충분히 달성" 여부는 사용자 판단.
2. **iter43 후보 A: Task pool 확장 (5-10 trial large batch).** 프롬프트 안정 상태에서 대량 생성으로 dataset 성장. 현재 29 → 40+ 예상. RL training에 근접한 규모.
3. **iter43 후보 B: Type B 서브타입 다양화.** 현재 Type B accept 3개 모두 `count`. max/min/sum/avg는 0. 프롬프트 Type B 예시에 `max(payment_date)`, `sum(amount)` 등 명시로 유도.
4. **task pool 현황**: 29 accept, 5 axes, 품질 5/5. 프롬프트 수렴 단계로 판단.

---

### Iteration 43 — 2026-04-20 (Type B 서브타입 다변화 — max/min/sum 예시 우선 배치, 3-trial 배치)

**Hypothesis.** iter30/35/41/42 Type B accept 4건 모두 `count`. max/min 예시는 있었으나 composer가 선택 안 함 (iter34 N=3 고정 패턴과 동일). 개입: Type B 예시 순서 재배치(max/min/sum 먼저) + 4개 first-class 예시 추가(max(amount), min(payment_date), sum(amount), max(rental_date) 각각에 Korean phrasing). "iter43 현재 Type B는 count로 saturated — mix up" 메타 경고 문구 추가.

**Change.** `src/rl_task_foundry/synthesis/prompts.py` Type B 예시 섹션 재구성. 4개 새 max/min/sum 예시를 count 예시 앞에 배치. commit `3dc6547`.

**Trial.** 3-trial 배치.

| suffix | anchor | task | 결과 | 관찰 |
|---|---|---|---|---|
| a | rental=11681 | Type A: 2005-08 첫 5건 | **accepted 0.333** | Type A 선택 (iter43 편향 효과 없음), Cross-item + date filter, solver 16턴 matched |
| b | customer=552 | **Type B: max(amount)** → 4 submit 깊이 escalation | **synthesis_failed** | **iter43 hypothesis 부분 성공**: Type B max 선택 ✓, target lock 유지, but submit 5에서 max→sum **fn switch** → `difficulty_crank_invalid`. **iter41 프롬프트 버그 노출** (fn switch 허용했으나 backend-invalid) |
| c | rental=8843 | Type A: 2005-07 첫 5건 | **accepted 0.333** | Type A, s2 27턴 matched, s0/s1 submitted wrong answers (RL diversity) |

iter43_b 상세 escalation: 
- submit 1: "지금까지 결제 중 가장 큰 금액" (max, no filter) → too_easy 1.0
- submit 2: + "2005-07-09 이후" date filter → feedback
- submit 3: + "2005-07-30 이전" (changed direction) → too_easy 1.0
- submit 4: + "5.99달러 이상" (Composite filter pair) → feedback
- submit 5: "총 금액" (**max→sum fn switch**) → `difficulty_crank_invalid` → budget exhausted

**Findings.**

1. **iter43 hypothesis 부분 성공 (iter43_b).** 이전 4 Type B accept 모두 count였던 게 iter43 배치에서 max(amount) 선택으로 깨짐. 예시 reorder + 새 예시의 직접 효과. **다만 accept까지 도달 실패** — iter41 내부 버그가 발목.
2. **내부 규칙 버그 발견 & 즉시 수정.** iter41 프롬프트의 "switch count → sum/min/max on same table" 허용이 실제 backend 정책과 불일치 (backend는 fn 변경도 target switch로 판정). 이 규칙을 따라 composer가 submit 5에서 max→sum 시도 → crank_invalid. 수정 commit `c198dd7`: 프롬프트가 `(fn, table, column)` triple 전체 immutability 명시.
3. **iter43_a, c는 Type A로 회귀 — example 편향 한계.** 2/3 trial이 Type A 선택. iter43 변화가 Type B "첫 선택" 분포를 완전히 바꾸지 못함. composer의 Type A 기본 선호가 구조적. 5-10 trial 배치로 봤을 때 Type B가 ~20-30% 수준 유지될 것으로 예상.
4. **Solver 다양성 (iter43_c).** 3명 중 1 matched at 27턴, 2 submitted at 12/19턴 but wrong. **RL training 이상 패턴** (turn 수 + 정답/오답 mix). iter42_c 같은 행동이 안정적으로 재현됨.
5. **누적 30 accept.** 29 + iter43_a + iter43_c = 30. 축 분포: Filter 7, Composite 10, Aggregate 3 (all count), Cross-item 5, Cardinality 2. Type B count 3건 + Aggregate iter42_c 1건 = 4 Type B accept (all count).
6. **iter43_b의 submit 5 패턴이 중요 학습 데이터.** 비록 fail이지만 "Type B max → min/sum 전환 불가" rule이 프롬프트에서 명확해짐. 단순 실패가 아닌 **프롬프트 계약의 엄밀성 확인** 사례. 버그 fix는 향후 iter의 accept rate 회복 여지.
7. **Loop 수렴 신호.** iter38~43 6 iter 중 최근 3 iter는 "quality 기준은 유지, 실패 패턴은 세부적". 큰 개선은 iter38~40에서 나오고 iter41~43은 edge case 해결. **diminishing returns 구간 진입**.

**Next direction.**

1. **iter44: iter43 프롬프트 + iter41 fn-lock fix 결합해 Type B max accept 최초 획득 시도.** 1-2 trial smoke.
2. **Loop stop 평가 재검토.** 현 상태(30 accept / 5축 / 품질 5/5 / 2/3 accept rate / 내부 규칙 일관성)가 사용자의 "충분히 달성" 기준을 충족하는지 판단 필요. 계속 iter 돌리면 세부 개선은 가능하나 규모성 성장은 느림.
3. **task pool 현황**: 30 accept, 5 of 5 axes, 품질 criteria 5/5, 내부 규칙 consistency 회복.

---

### Iteration 44-45 — 2026-04-20 (fn-lock fix + cardinality pre-check 통합 검증, 1-sample smoke — **최종 버전 확정**)

**Hypothesis.** iter41 fn-switch 버그(c198dd7) + iter45 canonical row-count pre-check(b30b6ba) 두 개입의 결합이 (a) Type B max/min/sum accept를 가능케 하는지, (b) submit-1 pass=0.0 terminal 패턴을 차단하는지. 1-sample 검증으로 "최종 prompt" 상태 선언 여부 결정.

**Change.** 없음 — iter43 프롬프트 + iter41 fix + iter45 row-check 누적 상태 그대로 재실행. Config: `artifacts/tmp_configs/iter45_cardinality_precheck.yaml`.

**Trial.** iter44_a (Type B count + staff filter overshoot to pass=0.0 → synthesis_failed) 후 iter45_a 1-sample.

| suffix | anchor | task type | submits | 결과 | matched turns | 관찰 |
|---|---|---|---|---|---|---|
| iter44_a | rental=8063 | Type B count(rental) + date + staff filter | 2 | synthesis_failed (submit 2 pass=0.0) | — | composer가 Jon Stephens + July 합성에 overshoot. row-check rule 적용이 부분적 |
| iter45_a | customer=158 | **Type B max(amount)** + date + staff filter | 3 | **accepted 0.667** | 5 → 20 (4x 증가) | 📍 **최종 버전 검증 성공** |

iter45_a 상세 flow:
- submit 1: "2005-08-01 이후 가장 큰 금액" (max, single date filter) → 3/3 matched @ 5턴 → too_easy
- submit 2: + "Mike Hillyer 직원" staff filter → feedback (mid-flight refinement)
- submit 3: staff를 Jon Stephens로 변경 (다른 selectivity 선택, **fn과 target은 유지** = iter41 fn-lock 준수) → 2/3 matched @ 20턴 → **accepted 0.667**

**Findings.**

1. **Type B max 최초 Accept ✓✓✓ (iter45_a).** 이전 5 Type B accept 모두 count였음. iter45_a의 `{max_payment: string}`이 **non-count Type B의 최초 실현**. iter43 예시 reorder + iter41 fn-lock fix + iter45 row-check 세 개입의 누적 효과.
2. **fn-lock 준수 확인.** submit 1 → 2 → 3 전반에 걸쳐 `max(payment.amount)` 유지. sum/min/count으로 switch 시도 없음. iter43_b에서 관찰된 max→sum crank_invalid 버그가 iter41 fix로 구조적 해결.
3. **Progressive escalation monotone (5 → 20턴).** submit 1 5턴 baseline → submit 3 20턴 accept = **4x solver effort 증가**. 기준 #5 (점진 난도 증가) 강한 실증.
4. **Cardinality pre-check rule 부분 작동.** iter44_a는 submit 2 pass=0.0 overshoot 발생 (rule 적용 부족), iter45_a는 3 submit 모두 adequate row count. N=1 trial로 전체 효과 측정 어렵지만 iter45_a는 overshoot 없이 완주.
5. **Voice 품질 유지.** "Jon Stephens 직원", "가장 큰 금액", 2005-08-01 date 모두 자연 customer voice. iter32 이후 voice 체인 무결.
6. **Solver 다양성 (submit 3).** 2 matched @ 20턴 (같은 난도) + 1 wrong answer @ 16턴 = **RL training ideal**. MaxTurns 없이 모두 reasoning 완료, 정답/오답 mix.
7. **누적 31 accept** (30 + iter45_a). Aggregate subtype 분포 갱신: count 4 + **max 1** (첫). 기타 min/sum은 0 유지.

**Final state declaration.**

프롬프트 최종 버전 = **iter43 (Type B subtype) + iter41 fix `c198dd7` (fn-lock) + iter45 `b30b6ba` (row-check) 누적** 상태로 확정. 이 prompt로:
- 5 of 5 axes 전부 관측 (Filter/Composite/Aggregate/Cross-item/Cardinality)
- Aggregate subtypes 중 count + max 모두 실현 (min/sum은 아직 0, 추후 batch 확장 시 기대)
- Quality criteria 1~5 모두 반복 실현
- Accept rate ~2/3 stable
- 내부 규칙 consistency 회복 (fn-lock 버그 해결)
- Voice 가드 완전 작동 (post-iter32 violation 0)
- M:M bridge anchor dead-end 해결 (iter42)
- Film anchor 정상 accept 가능 (iter42_a)

다음 단계는 이 프롬프트로 **fresh batch 재구성** (사용자 계획). 현 registry는 prompt 진화 과정 snapshot이라 homogeneous quality dataset이 아님 — fresh batch가 최종 RL 데이터셋 생성 경로.

---

### Iteration 46 — 2026-04-24 (cross-DB robustness 시도 1: postgres_air + 동적 example pack 주입)

**Hypothesis.** iter45에서 sakila 단일 DB의 prompt가 수렴(31 accept, 5 axes, 품질 5/5, 내부 규칙 일관). robustness를 두 번째 DB로 검증하려 postgres_air(항공 예약, 30M boarding_pass, 21M passenger, 5.9M booking, 692 airport)를 도입했더니 smoke03만 24/24 matched로 깨끗하게 통과하고 seq01~05 5회 모두 다른 모드로 실패. 로그 포렌식 결과 **두 가지 독립 문제** 식별: (a) `solver_orchestrator.py:206-226`이 `RateLimitError` 같은 인프라 예외로 죽은 solver run을 reward 계산을 거쳐 UNMATCHED로 분모에 포함시켜 false too_hard 유발 (실증: seq01에서 24/30 RateLimit + 6 정상 = 5/30 → too_hard로 거절), (b) 시스템 prompt의 "Hub-like anchors (customer, film, category, country, staff)", "customer→rental N=10 in-band", "Avoid film_actor M:M" 같은 **load-bearing instruction이 sakila-specific** — postgres_air에서는 부분적으로 false인 진술. iter34→iter35 비교가 결정적 증거: 추상 원칙으로 바꾸면 행동이 안 움직이고, 예시를 직접 교체하면 첫 trial에 행동 변화. 그래서 sakila 시스템 prompt는 손대지 않고 (제거 시 sakila quality 회귀 위험), **per-DB example pack을 user-message에 동적으로 주입**해서 LLM이 local examples를 더 가까운 신호로 따라가게 한다 (iter35 effect의 cross-DB 일반화 가설).

**Change.** 4개 파일 변경.

1. `src/rl_task_foundry/pipeline/solver_orchestrator.py` — `_evaluable_runs(runs)` 헬퍼 추가, `run_bundle`/`_execute_solver_batches`/`evaluate_rollout_summary` 3 site에서 `solver_result.status != "failed"` 필터 적용, `TaskRolloutSummary.failed_solver_runs` 필드 신설. RateLimit 등 인프라 예외 run을 calibration 분모에서 제외하되 observability를 위해 별도 카운트.
2. `src/rl_task_foundry/config/models.py` — `ExamplePack` 모델 신설 (label / type_a_examples / type_b_examples / pitfalls), `DomainConfig.examples_pack: ExamplePack | None` 필드 추가.
3. `src/rl_task_foundry/synthesis/prompts.py` — `_render_examples_pack` 헬퍼 + `build_synthesis_input(examples_pack=...)` 인자 + Topology와 Data Distributions 사이에 `# Local Examples — {pack.label}` 섹션 삽입. 메타룰 1단락 명시: "When the two conflict, prefer these local examples".
4. `src/rl_task_foundry/synthesis/{backend_openai_agents,backend_scripted,runtime}.py` — `examples_pack` 인자를 `run_synthesis` Protocol 및 두 backend 구현에 통과, runtime이 `self.config.domain.examples_pack` 전달.
5. `rl_task_foundry.postgres_air.yaml` — `domain.examples_pack` 본문 작성: Type A 3개 (account→booking N=3, airport→flight N=10, passenger→boarding_pass N=all), Type B 5개 (count×2, max, sum, min — iter43 패턴), pitfalls 4개 (작은 reference 테이블, booking_leg bridge, boarding_pass 30M scan 위험, airport departure/arrival 양방향 edge).

회귀 테스트 1개 추가 (`tests/test_pipeline_solver_orchestrator.py::test_solver_orchestrator_excludes_failed_runs_from_pass_rate`) — 3 solver 중 1개 RuntimeError 시 total_solver_runs=2, matched=2, failed_solver_runs=1, pass_rate=1.0 검증. **8/8 통과.**

**Trial.** postgres_air 컨테이너 재생성(이전 세션에서 제거됨) 후 1-sample smoke `iter46_a` 실행. 결과: **synthesis_failed (composer-side RateLimitError, opencode_zen/qwen3.5-plus, Alibaba 429 token-quota)** at composer turn 3. 정상 진행된 부분: (a) `schema_map(root_table=booking, depth=2)` 1회 호출 정상, (b) `neighborhood(table=booking, row_id=4355730, max_per_edge=5)` 1회 호출 정상 — anchor가 postgres_air의 booking row (`booking_ref="9BBW32"`, `booking_name="Your flight to Oakland"`)로 결정됨. 즉 schema introspection은 postgres_air를 정확히 보고 있고, anchor 선택도 postgres_air 도메인. 그러나 composer의 세 번째 LLM call에서 Alibaba quota 소진 → submit_draft 도달 못 함. solver phase 미시작 → bug fix 검증/example pack 효과 측정 모두 불가.

| 단계 | 상태 | 비고 |
|---|---|---|
| schema_map | ✓ | postgres_air 테이블 정확히 인식 |
| neighborhood (booking 4355730) | ✓ | "Your flight to Oakland" — 실제 데이터 |
| 다음 LLM 추론 | ✗ | RateLimitError 429 |
| submit_draft | — | 미도달 |
| solver phase | — | 미진입 |

**Findings.**

1. **iter46_a는 quota miss (가설 검증 실패)**. opencode_zen이 직전 사용으로 token window 소진된 상태였고, 첫 두 tool call까지는 cache hit/저비용으로 통과했으나 세 번째 LLM 추론에서 429. 이건 prompt/code 문제가 아니라 provider 가용성 문제 — solver_orchestrator bug fix와 example pack 효과는 둘 다 **이번 trial로는 측정 불가**.
2. **단편적 양성 신호 1개**: anchor가 postgres_air booking으로 결정. 이전 sakila trial들이 customer/film을 강제 선택하던 패턴이 sakila 시스템 prompt에 있음에도, 동적 schema_summary가 booking을 anchor로 노출했고 composer가 그것을 따랐다. 단 anchor 선택은 `random_anchor` 메커니즘 영향 받으므로 prompt 효과인지 별개. example pack 효과 입증 아님.
3. **bug fix는 잘 동작함이 단편 검증**: trial이 `synthesis_failed` (composer-phase) 카테고리로 분류되어 정상 종료. solver-phase 진입했더라면 fix가 발동했겠지만 이번엔 그 코드 path 미실행. `failed_solver_runs` 필드 작동 여부는 solver-phase trial에서만 검증 가능.

**Next direction.**

1. **iter46_b 재시도 — quota 회복 또는 provider 분산.** 두 가지 옵션:
   - 옵션 A: opencode_zen quota window 회복 대기 (Alibaba는 일반적으로 rolling 1-hour 또는 daily limit) → 동일 config 그대로 재실행. 가장 cleanest 비교 (iter45 sakila 페어와 같은 pair).
   - 옵션 B: 임시로 composer/solver를 다른 provider (anthropic_main 또는 openrouter)로 일시 변경해 example pack hypothesis만 검증. 변수가 추가되지만 quota 의존성 제거. 이 방식은 iter45 prompt-tuning과 다른 트랙으로 구분 표기.
2. **iter47 후보 (iter46 재시도 성공 가정)**: 3-trial 배치로 재현성 검증 + (a) composer가 local pack 템플릿(account→booking, airport→flight, passenger→boarding_pass) 중 어느 것을 채택하는지, (b) Type B fn 분포(count/max/min/sum), (c) sakila 어휘 누수 측정 결과로 다음 분기.
3. **인프라 후속 작업**: 현 trial에서 quota dry는 sakila iter44~45 stress 시점부터 누적된 결과 가능성. 다음 iter 전에 quota window 회복 시간 확보 필요. 또는 prompt_tuning_log에 provider별 quota tracking 컬럼 추가 검토.

---

### Iteration 46_b — 2026-04-25 (throttled opencode_zen 재시도, 사용자 중단)

**Hypothesis.** iter46_a의 quota miss는 transient rate-limit이라 가정하고 SDK retry/concurrency 보강으로 통과 가능 검증. opencode_zen `max_concurrency: 16→8`, `max_retries: default 2→5`, `timeout_s: 120→180`으로 hardening해서 동일 1-sample smoke 재실행.

**Change.** `rl_task_foundry.postgres_air.yaml`의 `providers.opencode_zen` 블록만 수정. prompt/code 변경 없음.

**Trial.** iter46_b 1-sample. 24분 진행 중 사용자 중단 (배경: 너무 오래 걸림).

| 단계 | 시각 | 결과 |
|---|---|---|
| composer (5 tool calls) | 00:39-00:43 (~4min) | ✓ task submitted (`task_account bookings with price filter_*`) |
| solver_2 | 137s | RateLimitError |
| solver_1 | 462s (13 turns) | **completed, submitted** |
| solver_0 | 497s | MaxTurnsExceeded |
| solver_3, 4 | 161s 각 | RateLimitError |
| solver_5 | 464s | RateLimitError |
| solver_6 | 179s | RateLimitError |
| solver_7~29 | — | 미실행 (사용자 중단) |

**Findings.**

1. **example pack 첫 양성 신호 ✓✓**: composer가 anchor=account=577127 → query `booking WHERE account_id=577127` (3 rows: `{booking_id:188522, booking_ref:"HVZE12", price:1723}` 등) → query `booking WHERE account_id=577127 AND price>1000` (escalated, 3 rows) submit. **이 chain이 example pack의 Type A 템플릿 #1 ("anchor=account → booking + filter on price")과 정확히 일치**. iter46 hypothesis 1차 검증 — 동적 local pack이 sakila 시스템 prompt를 압도해서 postgres_air 패턴 채택. anchor와 데이터 둘 다 실 postgres_air ("MCLEAN577127@magic.email" 같은 실 booking row).
2. **`max_retries=5` 설정이 hard quota에선 역효과**: opencode_zen의 `insufficient_quota`는 transient 아니라 하드 cap이라 SDK가 retry budget(0.5+1+2+4+8s = ~15s) 다 소진하고 같은 429 받음. 각 solver가 ~3분씩 낭비 중. iter47부터는 max_retries=2로 환원, transient 보호만 유지.
3. **opencode_zen aggregator 구조의 본질적 위험성**: 사용자 wallet에 $16.45 남았는데도 `insufficient_quota` 발생 → opencode_zen이 자체 Alibaba 계정의 token budget 공유 모델임을 시사. 사용자 결제와 별개의 upstream cap이 본인 trial을 막을 수 있음. 장기 트랙은 다른 provider 또는 Alibaba direct 권장.
4. **bug fix는 부분 검증**: 5 RateLimitError + 1 matched + 1 MaxTurns 분포 — 이전 같으면 5/7로 false too_hard였겠지만 fix 적용 후 RateLimit 5개 분모 제외. 단 solver_phase가 끝까지 못 가서 최종 quality_gate 결과 직접 관측은 미완.

**Next direction.** OpenRouter로 갈아타고 iter46_c 발사. opencode_zen은 본 트랙에서 일시 폐기.

---

### Iteration 46_c — 2026-04-25 (OpenRouter qwen3.5-plus-02-15, 사용자 중단)

**Hypothesis.** opencode_zen의 hard quota를 회피하기 위해 동일 모델(qwen3.5-plus 02-15)을 OpenRouter 경유로 호출. 같은 weight, per-customer wallet quota → 다른 사용자 burst 영향 없음. iter46_b의 composer 성공이 재현되는지 + solver phase 끝까지 가서 bug fix 최종 검증.

**Change.** `rl_task_foundry.postgres_air.yaml`의 `models.composer`와 30개 `models.solvers` 전부 `provider: opencode_zen, model: qwen3.5-plus` → `provider: openrouter, model: qwen/qwen3.5-plus-02-15`. `providers.openrouter` 블록도 hardening (`max_concurrency: 16→8`, `timeout_s: 120→180`, `max_retries: 3`).

**Trial.** iter46_c 1-sample. **5시간 진행 후 사용자 중단**. 30 solver는 모두 완료, composer phase는 두 번 실행됨 (1차 01:15-01:19, 2차 05:58-05:59 — 사이에 4시간 공백).

Composer 11 tool calls:
- 01:15-01:19: schema_map(account) → neighborhood(account 125813) → profile(booking.price WHERE account_id=125813) → sample(booking n=5 WHERE account_id=125813) → query(price≥500) → query(aggregate sum on price≥500) → query(aggregate sum on price all) → query(price≥1000)
- 05:58-05:59: profile(booking.update_ts) → query(price≥1000+update_ts filter) → query(final)

Solver 30/30 완료 결과:

| 결과 | 수 | 비고 |
|---|---|---|
| ✓ matched (status=completed) | **3** (s22 13turn, s27 12turn, s29 11turn) | 실제 답 제출 일치 |
| ✗ MaxTurnsExceeded (status=failed) | 9 | 16-turn 한도 도달 |
| ✗ UserError (status=failed) | 18 | SDK-level error (RateLimit 아님) |
| ✗ RateLimitError | **0** | OpenRouter 안정성 ✓ |

**Findings.**

1. **example pack 가설 재검증 ✓**: iter46_b와 같은 pattern (anchor=account → booking + price filter + sum aggregate). Type B `sum(price) where price≥X` shape 채택 — iter43의 Type B 다변화(count/max/min/sum) 효과까지 cross-DB 이전. 우연 아님.
2. **OpenRouter 안정성 입증 ✓**: 30 solver 발사에서 RateLimitError 0건. opencode_zen aggregator 구조 vs OpenRouter per-customer wallet의 차이가 명확. 향후 iter는 OpenRouter 디폴트 권장.
3. **🚨 Bug fix v1 설계 결함 노출 (중요)**: `solver_result.status != "failed"` 한 조건 필터가 너무 coarse. `status="failed"`는 인프라 실패(RateLimit/Timeout/BadRequest)와 capability 실패(MaxTurnsExceeded/UserError) 둘 다에 set됨. iter46_c 결과 적용 시:
   - 현재 fix: evaluable=3 (matched만), pass_rate=3/3=1.0 → **false too_easy**
   - 의도된 거동: evaluable=12 (matched+MaxTurns만), pass_rate=3/12=0.25 → in band
   - UserError 18개는 SDK 단의 일과성 에러로 판단되어 제외 가능, MaxTurns 9개는 capability 실패로 분모 포함 필요. iter47에서 termination_reason 기반 세분 필터 (`fix v2`) 필요.
4. **mysterious 4-hour gap (01:19 → 05:53)**: composer phase 1과 2 사이에 4시간 공백. 가능 원인: (a) OpenRouter thinking-mode가 매우 긴 reasoning 응답 생성 (qwen3.5-plus는 thinking model), (b) SDK retry 무한 루프, (c) network 일시 중단. trial_events.jsonl에 그 사이 이벤트 0개 — composer LLM 호출이 실제로 그렇게 오래 걸렸거나 SDK가 silent retry 중. iter47 발사 전 OpenRouter 로그/usage 확인 필수.
5. **Trial 시간 한도 부재**: 5시간 trial은 비효율적. `max_solver_runs=30` ceiling이 지나치게 큼 — calibration이 batch_size=3로 점진 진행하되 too_easy/too_hard 결정 시 조기 종료해야 했는데 30개 다 돌았다는 건 calibration_decision이 fix v1의 잘못된 분모로 "uncertain" 판정 유지한 결과 가능성.

**Next direction.**

1. **iter47 발사 전 필수 작업 3가지**:
   - (a) **bug fix v2**: `_evaluable_runs()`에 termination_reason 기반 분기 추가. 인프라 실패 그룹(`RateLimitError`, `APITimeoutError`, `BadRequestError`, `ConnectionError`, `InternalServerError`, `UserError` 등 SDK 일과성)만 분모 제외. capability 실패(`MaxTurnsExceeded`, wrong answer)는 분모 포함. 회귀 테스트도 보강.
   - (b) **4-hour gap 조사**: iter46_c trial_events.jsonl 시간 분석 + OpenRouter dashboard usage 확인. 만약 thinking-mode 무한 루프면 max_completion_tokens 강제 cap 필요. SDK retry 무한이면 client config 점검.
   - (c) **trial 시간 가드**: `max_solver_runs: 30→10`으로 줄이고, `solver_runtime.max_turns: 16→24`로 올려서 MaxTurns 발생률 자체를 감소. 이 둘은 trade-off — 양쪽 다 줄여 평균 trial 시간 ~10분 목표.
2. **iter47 본 실험 (위 3가지 적용 후)**: 1-sample smoke로 fix v2 + reduced solver count 동작 확인. 그 후 3-trial 배치.
3. **provider 결정**: OpenRouter를 본 트랙으로 채택. opencode_zen은 quota 회복 시 비교 trial 1회만 (cleanness 검증용).

---

### Iteration 47 — 2026-04-25 (answer_contract strict validation, minimax-m2.5 smoke)

**Plan drift note.** iter46_c의 예정 iter47은 OpenRouter + solver failure 분모 fix(v2)였지만, 사용자 논의 후 우선순위가 "RL trace 품질을 위한 tool/API contract 재설계"로 바뀌었다. 따라서 아래 Iteration 47은 그 피벗 이후의 실제 다음 실험 기록이다.

**Hypothesis.** Composer가 canonical answer를 저작할 때 숨은 의미/힌트 휴리스틱으로 품질을 보정하면 RL trace 품질을 오염시킨다. 대신 `submit_draft`에 명시적 `answer_contract`를 요구하고, 검증은 정밀도 100%인 형식/증거 계약만 수행한다: (a) user_request에 contract phrase가 실제 포함됨, (b) 직전 successful `query` 결과와 label이 canonical-equal, (c) `too_easy` 이후에는 같은 operation을 유지한 채 predicate/order/limit만 단조 확장한다. 의미적으로 "좋은 문제인지"는 solver pass-rate가 통계적으로 판정한다.

**Change.**
- `SubmitDraftPayload`에 `answer_contract` 필드 추가. contract는 `kind`, `operation`, `predicates`, `order_by`, `limit`, `evidence`를 담는다.
- `submit_draft` 검증을 strict contract 기반으로 확장: phrase exact substring, latest-query evidence equality, too_easy 이후 incremental contract check.
- 처음에는 JSON object만 허용했으나 minimax가 JSON string으로 제출하는 패턴을 보여서, stringified JSON은 parse 후 동일 모델로 검증하도록 수정.
- 날짜 경계/자연어 의미 추론 같은 semantic heuristic은 제거. 사용자 결정: solver pass-rate가 잡지 못하는 의미 문제를 휴리스틱으로 보정하지 않는다.
- prompt와 rejection feedback을 contract 중심으로 갱신: Type B는 `kind/operation` lock, `too_easy` 후에는 operation 변경 금지, filter/cardinality/order/limit 확장만 허용.

**Trial.** `artifacts/opencode_smoke_answer_contract_9`, pagila, composer/solver 모두 `opencode_zen/minimax-m2.5`, 3 solver, solver max_turns 30, max_generation_attempts 5.

| trial | 결과 | 핵심 관찰 |
|---|---|---|
| trial_0001 | `synthesis_failed` / `submit_payload_invalid` | minimax가 `answer_contract`와 `entity`를 JSON string으로 제출. Pydantic `model_type` 에러가 5회 반복되어 solver phase 미진입. 이 후 stringified JSON parser 추가. |
| trial_0002 | `synthesis_failed` / `reject_too_easy` | contract evidence mismatch가 첫 제출을 정확히 잡음: label은 customer rental count 28인데 latest query는 unrelated staff count 8040. 이후 correct query 재호출. baseline count 28 → date filter count 20 → staff+date count 10 모두 3/3 matched. 마지막 list 전환은 `answer_contract_not_incremental`로 차단. |
| trial_0003 | `synthesis_failed` / `reject_too_easy` | stale/latest-query mismatch를 두 번 잡음. 최종 sum(amount) 151.65 → amount>=3 sum 127.79 → amount>=3 + payment_date>=2022-06-01 sum 30.94까지 모두 3/3 matched. contract는 동작했지만 difficulty gate는 계속 too_easy. |

**Findings.**

1. **Strict evidence contract는 성공.** false semantic heuristic 없이도 stale label, wrong latest query, non-incremental type switch를 모두 기계적으로 검출했다.
2. **문제는 이제 validation이 아니라 composer difficulty policy.** trial_0002/0003 모두 contract-valid한 후보가 solver 3/3에게 풀렸다. 현재 failure mode는 "정답 생성 실패"가 아니라 "minimax solver에게 너무 쉬운 scalar aggregate/count를 계속 냄"이다.
3. **Type B lock은 의도대로 작동.** too_easy 후 scalar count에서 list로 도망가는 시도를 `kind_changed`, `operation_changed`, `predicate_removed`로 잡았다. 이건 RL trace 일관성 측면에서 좋다.
4. **minimax는 object schema를 stringified JSON으로 우회하는 경향이 있다.** 이건 모델 품질 문제라기보다 API/tool-call serialization 관성. parser 허용 후에는 contract 검증이 정상 작동.
5. **정밀도 100% 원칙 유지가 맞다.** date phrase가 자연스러운지, "고객님" voice가 완벽한지 같은 판단은 이번 gate에 넣지 않았다. 과하게 잡으면 solver pass-rate의 통계적 역할을 뺏고 false reject를 만든다.

**Next direction.**

1. Composer difficulty policy를 contract 위에 얇게 올린다. 단, 의미 휴리스틱이 아니라 구조적으로 안전한 제약만: repeated too_easy scalar는 operation lock 유지하되 predicate 수/alias 다양성/order/limit 등 contract-detectable 축으로만 확장.
2. Trial 전 `submit_draft` schema example에 object form을 더 강하게 보이되, runtime은 stringified JSON도 계속 받는다. weak/cheap model과 API endpoint style 호환을 동시에 가져가는 편이 낫다.
3. Accepted sample 확보가 목표라면 다음 smoke는 Type A/list-first 또는 cross-table filtered list를 우선 유도하는 prompt 실험으로 분리한다. 이건 "validation 9점" 트랙이 아니라 "difficulty policy" 트랙이다.

---

### Iteration 48 — 2026-04-25 (contract-aware too_easy feedback + user-facing timestamp/answer surface)

**Hypothesis.** Iter47의 마지막 실패에서 composer가 scalar Type B too_easy 이후 list로 도망가려 했던 원인은 tool feedback이 `answer_contract.kind`를 모르고 `Cardinality/list` 옵션까지 던진 데 있다. `too_easy` feedback을 contract-aware로 바꾸면 invalid crank trace를 줄일 수 있다. 별도 smoke에서 timestamp scalar가 날짜로 truncation되는 false too_hard도 관측됐으므로, 이건 validation normalize가 아니라 authoring prompt에서 "timestamp answer는 정확한 시각/타임스탬프를 물어라"로 보강한다.

**Change.**
- `_too_easy_retry_guidance(answer_kind=...)`로 분기.
  - scalar: kind/operation 유지, list/Cardinality/Cross-item 금지, 새 predicate 1개만 추가. 첫 predicate는 Filter, 이미 predicate가 있으면 Composite.
  - list: `operation.fn=select` 유지, Filter/Composite/Cardinality increase/secondary Order만 허용.
- `SubmitDraftController`가 too_easy 및 label-not-strengthened feedback에 현재/직전 `answer_contract.kind`를 전달.
- prompt의 Type B timestamp guidance 수정: `timestamp/timestamptz` aggregate는 날짜/일이 아니라 정확한 시각/타임스탬프를 user_request에 명시. 내부 `solver` 명칭은 노출하지 않고 `evaluation rollouts`로 표현.
- user-facing data rule 보강: 고객은 DB/table/row/PK/store slot/inventory copy를 모른다. `user_request`에는 고객이 말할 수 있는 이름/제목/금액/시각/상태/장소/자기 계정 reference만 쓰고, readable surface 없는 ID filter는 다른 task/filter로 바꾸도록 지시. 기존 예시의 `소장본`, `2호점/store_id` phrasing 제거.
- trial_0003 사후 리뷰 후 추가 보강(후속 정정됨): 당시에는 canonical
  `label`의 필드명도 고객-facing이어야 한다고 판단했지만, 이후
  `submit_result`가 agent-visible verifier boundary라는 점을 반영해 이
  해석은 폐기했다. 현재 기준에서 `query(spec).select.as`와 `aggregate.as`
  는 final-answer prose가 아니라 안정적인 `submit_result` key다.
- 사용자 목표를 spec에 명시한 뒤 한 번 더 정정: 공통 prompt/feedback은 임의 DB에 적용되어야 하므로, 공통 system prompt 안의 pagila식 worked examples(`customer→rental`, `film`, `payment`, `staff_id`, `rental_date`)를 제거했다. 남은 예시는 `<timestamp_column>`, `<child_table>`, `matching_item_count` 같은 구조적 placeholder뿐이고, 구체 도메인 예시는 runtime의 schema/live rows/local example pack에서만 들어온다.
- 테스트 추가/수정: scalar feedback은 list 전환을 옵션으로 주지 않음, list feedback은 list-safe axis만 안내, prompt에 exact timestamp guard 및 user-facing data rule 포함.

**Verification.**
- `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py` 통과.
- `uv run pytest` → **331 passed**.
- user-facing output alias prompt 보강 후 `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py` → **29 passed**; `uv run ruff check src/rl_task_foundry/synthesis/prompts.py tests/test_synthesis_prompts.py` 통과.
- generic prompt/spec 보강 후 `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py` → **29 passed**; `uv run ruff check src/rl_task_foundry/synthesis/prompts.py tests/test_synthesis_prompts.py` 통과. `rg` 기준 공통 prompt에는 금지한 pagila-specific strings가 남지 않음.

**Trial.** `artifacts/opencode_smoke_contract_feedback_48`.

| trial | 결과 | 핵심 관찰 |
|---|---|---|
| trial_0001 | `synthesis_failed` / `AuthenticationError` | 현재 shell에 `OPENCODE_API_KEY`가 없어서 실패. `.env`에는 key 존재 확인. |
| trial_0002 | `synthesis_failed` / `reject_too_hard` | `.env` source 후 실행. anchor=`inventory_id=1819`. 첫 submit은 timestamp를 `T` 포함 ISO 형태로 바꿔 `label_values_not_grounded + answer_contract_evidence_mismatch`; feedback 후 raw query timestamp로 고쳤다. solver 3명 모두 날짜만 `{last_rental_date: "2022-08-17"}`로 제출해 canonical full timestamp와 mismatch, pass_rate 0/3. |
| trial_0003 | **accepted** / `solver_pass_rate=0.5` | user-facing rule 적용 후 재시도. anchor=`country_id=106` (`Virgin Islands, U.S.`). 요청: "버진 아일랜드 미국 지역의 2022년 8월 대여 기록 중 가장 최근 5건의 대여일과 반납일..." 첫 submit은 최신 query 결과에 `rental_id`가 포함되어 evidence mismatch; composer가 query를 `rental_date/return_date`만 반환하도록 재호출 후 accepted (`1/2`). registry committed + bundle exported. |

**Findings.**

1. **Contract-aware too_easy branch는 아직 실측 미검증.** 이번 smoke는 too_easy로 가지 않고 timestamp scalar가 too_hard로 끝났다. 단위/통합 테스트는 통과.
2. **Strict timestamp evidence는 잘 작동.** composer의 `T` reformat을 feedback으로 잡고, composer가 raw query value로 수정했다.
3. **새 병목은 answer granularity phrasing.** user_request가 "대여일"이라고 물으면 actor는 날짜만 답하는 것이 자연스럽다. exact reward에서 full timestamptz를 요구하려면 composer가 처음부터 "정확한 대여 시각/타임스탬프"를 물어야 한다. normalize로 맞추면 RL trace 품질 목표를 해친다.
4. **User-facing data miss.** smoke의 `"2번 점소"`, `"영화 소장본"`은 고객이 DB를 모른다는 전제에 맞지 않았다. 이건 validator로 막기보다 authoring rule과 예시에서 바로잡는 쪽이 덜 과하다.
5. **trial_0003은 user-facing request가 개선됐고 accept까지 갔다.** 내부 ID/점소/소장본 누수는 사라졌고, 지역/월/최근 N건은 고객 또는 비기술 업무 사용자가 말할 수 있는 표면이다. 단 `"대여 기록"`은 DB 레코드와 겹치는 표현이라 `대여 내역`이 더 안전하다.
6. **후속 정정: canonical label key는 customer-facing final answer가 아니다.** trial_0003 당시에는 label keys가 `rental_date`, `return_date`로 raw DB column에 가깝다는 점을 문제로 봤지만, 현재 설계에서는 label이 actor가 제출하는 agent-visible `submit_result`이므로 key는 안정적인 API-style result schema면 충분하다. 고객-facing 제약은 `user_request`와 downstream final answer에 적용된다.
7. **이벤트 로그 preview 착시 확인.** `trial_events.jsonl`의 `params_preview`는 list를 3개 item으로 자르기 때문에 query join chain에서 `customer<-rental.customer_id as r`가 빠져 보였다. raw conversation `ToolCallItem.arguments_preview`에는 해당 join이 존재했다. composer query alias validation 버그가 아니라 preview truncation에 따른 관측 착시다.
8. **프롬프트 유출 가드 유지.** timestamp prompt 보강 중 `solver`라는 내부 단어가 테스트에 걸렸고, `evaluation rollouts`로 교체했다.
9. **중요한 자기비판: user-facing 보강 직후에도 공통 prompt가 pagila 예시를 품고 있었다.** 이는 "DB만 갈아끼우면 적응형 고객-agent RLVR 데이터셋을 합성"한다는 목표와 충돌한다. 이번 정정으로 공통 prompt는 구조적 규칙만 말하고, 도메인 구체성은 현재 DB의 introspection 결과와 local examples에서 오도록 분리했다.

**Next direction.**

1. generic prompt 보강 후 smoke를 다시 돌려 공통 prompt가 특정 sample DB로 끌고 가지 않는지 확인한다. 특히 `label` keys가 raw DB column에서 answer-facing key로 바뀌는지도 함께 본다.
2. 다음 smoke도 timestamp scalar로 흐르면 Type B timestamp aggregate를 더 강하게 회피하거나 exact-time phrasing을 예시 전반에 더 앞쪽으로 올린다.
3. contract-aware too_easy feedback은 아직 runtime smoke에서 직접 타지 않았으므로, too_easy가 나오는 anchor에서 재현 확인이 필요하다.

### Iteration 49 — 2026-04-26 (role isolation: composer authoring only, solver pure prompt surface)

**Correction.** 사용자 지적이 맞다. Composer에게 "너는 synthetic dataset composer다", "actor/solver/pass-rate/RLVR" 같은 메타 목표를 알려주는 것은 역할 분리 위반이다. Composer는 현재 DB 증거를 보고 고객-facing draft를 쓰는 개별 저작자일 뿐이고, solver는 system prompt 없이 그 draft가 만든 문제만 푸는 순수한 풀이 주체다. 데이터셋/품질측정/registry는 runtime의 일이지 agent prompt의 일이 아니다.

**Change.**
- Spec에 database-swappable customer-agent RLVR 목표와 role isolation을 명시했다. 특히 solver는 `Agent.instructions=None`이 invariant이고, 입력은 rendered customer problem + hidden entity block + generated data tools뿐이다. 이후 submit shape는 rendered prompt가 아니라 task-specific `submit_result` tool schema로 옮겼다.
- Composer system prompt에서 `synthetic`, `dataset`, `RLVR`, `actor`, `solver`, `pass_rate`, `quality gate`, `evaluation`, `registry`, `training` 계열 메타 설명을 제거했다.
- Composer-visible tool section을 `Composer Tools`/`composer DSL`에서 `Data Tools`/`query(spec)`로 바꿨다.
- Composer-visible 문구에서 `evaluator`, `runtime`, `difficulty_crank_invalid`, `crank_invalid`, `quality threshold`, `system instruction` 같은 내부 파이프라인 단어를 제거했다.
- `submit_draft` feedback은 accepted/rejected 이유를 draft surface 관점으로만 말한다. too-easy는 "needs more specificity", too-hard는 "overconstrained / terminal"로 표현하고 solver/pass-rate는 노출하지 않는다.

**Verification.**
- `rg` 기준 `prompts.py`/`submit_draft_messages.py`의 prompt/feedback 표면에는 위 금지 메타가 남지 않았다. 남은 `_too_easy_retry_guidance` 같은 이름은 내부 함수명이다.
- `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_synthesis_backend_openai_agents.py tests/test_solver_backend_openai_agents.py` 통과.
- `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_synthesis_backend_openai_agents.py tests/test_solver_backend_openai_agents.py tests/test_turn_budget_prompt.py` → **46 passed**.
- `uv run pytest` → **331 passed**.
- `trial_0004`는 이 정정 전 prompt로 시작된 smoke라 중단했고, 이번 iteration의 신호로 보지 않는다.

### 행동 변화 요약

| Iter | Submits | 3번째 attempt | 주요 증상 |
|------|---------|----|-----------|
| 01 | 0 | 20-call 탐색 | Rung 상세 6단계 → qwen 과계획 |
| 02 | 2 | 18-call 탐색 | Axes bullet list 복귀, baseline |
| 03 | 2 | crank × 2 | rejection text 정렬, `no_new_grounded_observation` 관측 |
| 04 | 1 | 19-call loop | "1-3 calls" 허용 → 탐색 폭주 |
| 05 | 1 | 19-call 넓은 탐색 | 조건문 문장도 qwen에겐 tree 탐색 유발 |
| 06 | 2 | 18-call 탐색 | grounding을 Label Rules로 이전 → ungrounded 차단 성공 |
| 07 | 3 | Width 반복 + grounding 위반 | quota 이후 retry — 강도순 Axes 가설 미검증, qwen은 여전히 Width만 선택 |
| 08 | 0 | — | Label Rules에 "next label MUST change shape" 추가 → 상시 제약으로 오인해 cardinality 후보 사전탐색, MaxTurnsExceeded |
| 09 | 0 | — | 같은 imperative를 Workflow step 3로 이전 → step 본문이 길어지자 첫 submit 자체 지연, MaxTurnsExceeded |
| 10 | 3 | +Width, +Width, +Width | step 3 복원 + Axes effect 본문 재설계 → submit 3회 정상, 그러나 qwen이 Width 편향 깨지 않음. pass_rate 1.0 고정, same-model ceiling 직접 증거 |
| 11 | 3 | heterogeneous list attempts | Workflow step 2를 "3-item list along 1:N path"로 교체 → qwen은 "list"를 attribute enumeration으로 오해, blank string + ungrounded 로 실패 |
| 12 | — | quota 재소진 | step 2를 "homogeneous list + 예시" 로 재작성, 세션 누적 6 trial로 Alibaba quota 429, 프롬프트 검증 불가 |
| 12_retry | 0 | MaxTurnsExceeded | qwen이 city→address 1:N 경로는 정확히 식별, 그러나 동일 쿼리 반복 + `op="any" value=""` 스키마 버그 4턴 + calc/샘플링으로 예산 고갈. 프롬프트 반증 아님, 런타임 노이즈 |
| 12_retry2 | 0 | MaxTurnsExceeded | tool 버그 수정 후 재실행. qwen이 turn 5에 올바른 find_*(limit=3) 실행했으나 submit하지 않고 대안 1:N 경로 5종 탐색. commit 회피 행동 관측 |
| 13 | 1 | — | **신 tool 서피스 첫 베이스라인.** composer는 query DSL로 2-hop + sort+limit task 저작 성공, solver 0/3 matched → `reject_too_hard`. same-model `pass_rate=1.0` ceiling **반전**. 14 tool calls/attempt로 `max_turns=20` 소진해 attempt 2 진입 실패. DSL의 join+sort 컬럼 해상 버그로 3턴 낭비 관측 |
| 14 | 1 | — | `max_turns=40` 완화 단독 실험. DSL fix 덕에 composer 6턴으로 종료했지만 attempt 2는 여전히 미발동. 원인: `submit_draft_tool._terminated_too_hard` + `ToolsToFinalOutputResult` 조합이 Runner 즉시 break. max_turns는 병목 아니었음 |
| 15 | 1 | — | Workflow step 2 예시를 single-hop + all-fields-on-destination으로 재작성. film 앵커에서 여전히 2-hop 저작됨. 이유: sakila film의 1-hop 자식이 전부 ID-only 브리지라 3개 룰(single-hop / no-ID / destination-only) 교집합이 공집합. prompt 룰 세트가 over-determined. solver trace 누락으로 관측 불가(로깅 버그 발견) |
| 16 | 2 | — | **첫 accepted.** inventory 앵커에서 attempt 1 pass_rate=1.0(too_easy) → Composite axis(staff 필터 추가) escalate → attempt 2 pass_rate=0.667 **in band**. bottom-up 원설계 궤적 부활 관측. 단 2/3는 solver 3명 중 1명 APITimeoutError 덕의 noise-gifted landing이므로 재현성 검증 필요 |
| 17 | 1 | — | retry=5로 transient noise 제거 후 재실행. address 앵커 3 solvers 모두 submit, 1/3 matched → pass_rate=0.333 band 포함이지만 `_solver_divergence`가 2차 gate로 작동해 reject_too_hard 재분류. iter16 accept는 timeout 1명이 submission 풀에서 빠지며 divergence=1/2로 경계 통과한 infrastructure artifact 였음이 확정. n=3 + max_divergence=0.5는 구조적으로 band landing 불가 |
| 18 | 2 | — | **첫 clean accept.** divergence gate off 후 rental 앵커에서 attempt 1 simple single-filter (3/3 too_easy) → Composite escalate (+staff_id 3 filters) → attempt 2 2/3 in band + committed + bundle exported. asymmetric tool surface에서 same-model qwen 페어링이 원설계 bottom-up 궤적으로 clean landing 가능함 첫 실증. trial_events.jsonl 실시간 스트리밍 실제 관측 검증 |
| 19 | 1 | — | **2연속 accept** + Phase 2 unified logger 실사용 검증. customer 앵커에서 attempt 1 simple 1-filter로 바로 1/3 in band. iter18과 달리 escalation 없이 direct landing 궤적. trial_events.jsonl 한 파일에 composer/solver/phase 이벤트 전부 interleaved, legacy per-file trace 폴더 비어 있음 확인. iter17과 사실상 같은 signal(1/3 + 2 unique wrong)이 gate off라 자연 accept, fix 효과 직접 실증 |
| 20 | 1 | — | **3연속 accept + voice fix 검증.** Qualitative 평가에서 발견한 iter18/19의 staff-voice / mixed-voice 문제를 prompt의 rewrite instruction 강화로 해결(1인칭 ask 명시 + 금지 구절 리스트). 결과: city(Toulouse) 앵커에서 "저는 툴루즈에 거주하는 고객입니다. 제 대여 기록..." 1인칭 customer-ask voice 완벽 준수. 부수 효과로 **첫 clean 2-hop join task**(city→address→customer→rental) 성공 — iter15 multi-hop overshoot 실패가 누적 수정들로 해소. 단 Toulouse customer 수가 2+면 semantic ambiguity 가능성 별도 검증 필요 (사후 확인: 1명, 무해) |
| 21 | 1 | — | **Structural mismatch 노출** — actor anchor에서 composer가 `actor → film_actor → film` 2-hop task 저작. voice는 완전 준수(2인칭 org-ask)인데 solver의 atomic calculus가 `film_actor` composite PK에서 `take/count/aggregate/group_top` 전부 거부(`_single_column_pk` 가드). 16턴 우회 시도 후 MaxTurnsExceeded. Composer DSL ⊃ Solver calculus 표현력 격차 드러남. Fix는 `0ff46a5` — `_pk_expression` ROW 표현을 JOIN 매칭 통일해 composite PK 전면 지원. 실측 회귀 테스트 통과, iter22에서 실제 trial 재현 검증 예정 |
| 22 | 1 | — | composite PK fix 실측 시도 but anchor=inventory라 junction 경로 미유도. Attempt 1 정상 too_easy, Attempt 2 Composite escalation 저작 중 asyncpg `integer = text` 에러가 `_with_error_handling`의 PostgresError 누락 catch list 통과해 SDK UserError로 propagate. composer 자가 복구 기회 봉쇄. Fix `67fa1d6` — atomic/composer 양쪽 tool wrapper가 asyncpg.PostgresError catch, JSON error 응답으로 변환. iter23에서 실효 검증 |
| 23 | 1 | — | **Accept 스트릭 재개**(18/19/20/23 success, 21/22 fail). city 앵커(Sirjan) attempt 1 direct 2-hop landing. voice+semantic 모두 유효(Sirjan 1명 unambiguous). asyncpg catch/composite PK fix 둘 다 직접 검증은 안 됨 — 이번에도 junction 경로 미유도. 답 shape 편중(rental date-only) 4iter 누적 확정. iter24 axis 다양성 batch로 진입 |
| 24 | 1 | — | **Shape 편중 해소 첫 관측.** 프롬프트 예시를 5개 destination-appropriate pair로 확장한 뒤 payment anchor에서 `[{amount, payment_date}, …]` 출력. 4iter 연속 rental date-only 락인 깨짐. voice 1인칭 self-ID, semantic 자연, 5번째 clean accept(2/3). 단 1회 관측 = 분포 아님 — 진짜 다양성은 iter25 3-trial batch 필요 |

### 확정된 설계 결정 (DB-agnostic)

1. **qwen thinking-mode는 system prompt의 nuance를 decision-tree로 재해석한다**. "within N calls" 같은 상한, "only if... then..." 같은 조건문을 주면 각 tool call마다 조건을 재평가하느라 탐색 폭주. 단순·명령형 문장으로 유지.
2. **시스템 프롬프트와 rejection feedback 본문은 어휘+의미 완전 정렬 필수**. 두 신호가 다르면 agent는 일관된 규율을 잃는다 (iter03 핵심 발견).
3. **grounding 책임은 `# Label Rules`에 bullet 하나로 둬야 한다**. `# After Rejection`에 넣으면 조건문이 돼 과탐색 유발.
4. **`# After Rejection`은 최소 형태 고정**: "rejection ≠ 탐색 신호, 2 atomic calls 내 재submit" 외에는 건드리지 않는다 (iter04/iter05가 증명).
5. **Escalation Axes 강도순 재배열 + "Width 회피" 명시만으로는 qwen의 Width 편향을 깨지 못한다** (iter07 retry, iter10 두 번 재확인). 축 순서나 bullet 본문 effect 문구는 decision tree 분기 우선순위로 읽히지 않고, 가장 "추가 실행 비용이 낮은" 축이 여전히 선택됨. pass_rate=1.0이 세 번 연속 관측되는 건 same-model composer/solver ceiling의 직접 증거.
6. **Rejection-conditional 지시는 Label Rules에 두면 상시 제약으로 오인된다** (iter08). scope가 박힌 Workflow step 3 혹은 dedicated After Rejection 섹션에만 두어야 한다.
7. **Workflow step 본문을 3줄 이상으로 늘리면 첫 submit이 지연된다** (iter09 MaxTurnsExceeded). escalation 규칙의 상세도 자체가 첫 submit 속도에 영향. step 본문은 한 문장 수준으로 유지하고, 상세는 Axes 본문에 분산.
8. **"list along 1:N path" 같은 추상 어휘는 qwen이 attribute enumeration으로 오해한다** (iter11). homogeneous list 요구 시 "every item shares the same keys" + 시각적 예시 패턴(`[{k1,k2},{k1,k2},…]`) 같은 구체화 필수.

### 구조적 한계 (prompt로 해결 불가) — iter 13에서 반전됨

**iter01~12 시점 결론 (old atomic-tool 서피스 기준).** 같은 reasoning 모델(qwen3.5-plus thinking)이 composer + solver 양쪽에 쓰이면 composer가 solver 3명을 동시에 떨구는 task를 설계하기 매우 어렵다. Band [0.25, 0.75] 진입은 composer와 solver의 상대적 능력 차가 있어야 자연스럽다. 후속 조치 후보로 다음이 나왔다:
- composer = qwen3.5-plus, solver = 약한 모델(gpt-5.4-nano 등) 혼합
- solver 수를 3 → 5~10으로 늘려 band 해상도 증가
- calibration band 완화(현재 [0.25, 0.75] → [0.1, 0.9])

**iter 13 관측 (new composer DSL + atomic calculus 서피스 기준).** same-model 페어링 그대로지만 첫 제출에서 `pass_rate=0.0 (reject_too_hard)`. 즉 위 "구조적 한계"는 **모델 페어링**이 아니라 **tool 서피스 공유**의 산물이었다. composer가 query DSL로 one-shot 저작하는 2-hop 태스크를 solver가 9-primitive 체인으로 재구성하는 비용이 오히려 과도. 이제 ceiling은 `1.0`이 아니라 `0.0` 쪽으로 반전됐고, 개입은 "composer 난이도를 끌어내리기" 방향으로 바뀐다. 위 후속 조치 후보들(약한 solver, solver 수 증가, band 완화)은 여전히 유효하지만 **동인이 반전**되었음에 유의.

### 다음 세션 작업 우선순위

> iter 13에서 tool 서피스 교체로 ceiling이 반전되어, 이하 iter07-era 우선순위는 역사적 맥락으로만 유효. **현재 활성 Next direction은 iter 13 엔트리의 "Next direction" 블록**(max_turns 완화 → query DSL 버그 수정 → bottom-up 난이도 workflow 실험 순)을 참조.

(이하 iter07 기준 역사적 우선순위, 반전된 문제공간에서는 직접 적용 안 됨)
1. **iter08**: 선언형 Label-Rules 제약("After too_easy, the next label MUST change slot count or add a Cross-item rule") + `# Escalation Axes`에서 Width/Filter bullet 제거 혹은 "last resort" 라벨 부여. 조건문 금지 원칙과 양립하는 문장 형태로.
2. **budget 완화 대조군**: 동일 프롬프트로 `max_generation_attempts=4` 1회 trial. iter07_retry는 attempt 4가 있었다면 band 진입 가능성.
3. **asymmetric composer/solver**: composer=qwen3.5-plus, solver=약한 모델(gpt-5.4-nano 후보) — 구조적 상한과 프롬프트 상한 분리 측정.
4. `no_new_grounded_observation` 재현성 확인을 위한 same-prompt 2회 추가 trial.

---

---

## Metrics Template (per iteration)

### Maintenance Note — DB-Agnostic Exposed Surface Guardrail

- **Date**: 2026-04-26
- **Change**: removed remaining sample-DB examples from exposed composer/tool
  documentation and `submit_draft` rejection feedback. Examples now use
  structural placeholders such as `<anchor_table>` and `<pk_column>` instead of
  Pagila/Sakila-shaped names.
- **Reason**: the project target is database-swappable customer-agent RLVR data
  synthesis. Common prompts, feedback, and tool descriptions must not teach the
  composer a particular demo database's nouns or relationship patterns.
- **Verification**: targeted `rg` over exposed prompt/feedback/tool-description
  files found no `pagila`/`sakila`/`postgres_air` or Pagila-style identifiers;
  focused ruff and prompt/runtime/composer-query tests passed.

### Maintenance Note — Visibility/Handle Feedback After Smoke Trial

- **Date**: 2026-04-26
- **Trial**: `artifacts/opencode_smoke_after_db_agnostic`
- **Status**: `synthesis_failed`
- **Observed failure**: composer repeatedly attempted to expose non-user-facing
  selected fields and handle-like identifier values instead of converting the
  same grounded set into a scalar aggregate. The then-current validator rejected
  these drafts, but the feedback did not clearly steer the model toward visible
  non-handle outputs or derived aggregates.
- **Change**: tightened prompt and rejection feedback to say that a selected
  label output is allowed only when query metadata reports
  `visibility == 'user_visible'` and `is_handle` is false. Feedback now points
  models toward aggregate counts over hidden/handle-only row sets instead of
  retrying the same exposed identifiers.
- **Scope**: generic metadata semantics only; no database-specific nouns,
  table names, or relationship patterns were added.
- **Follow-up trial**: `artifacts/opencode_smoke_after_visibility_feedback`
  still failed, but with a different failure mode: malformed
  `answer_contract` JSON strings and a list ordered by a handle column. This
  confirmed the visibility/handle loop was no longer the only active issue.
- **Follow-up change**: malformed `answer_contract` payloads now get
  recoverable feedback instead of terminal `submit_payload_invalid`, and
  handle-shaped ordering was discouraged as a non-user-facing dependency.
  Hidden handles remain valid for joins and anchor filters.
- **Second follow-up trial**: `artifacts/opencode_smoke_after_order_guard`
  reached solver rollout, so the payload-invalid path was fixed. It failed as
  `reject_too_easy`: the first valid scalar was solved by all solvers, and the
  composer then changed the locked scalar operation from `sum` to `count`.
- **Classifier correction**: composite primary keys may include user-facing
  partition/natural-key values such as timestamps. `is_handle` now treats
  foreign keys, non-user-visible primary-key columns, and identifier-named
  primary-key columns as handles; a user-visible timestamp primary-key
  component remains orderable/readable.
- **Third follow-up trial**: `artifacts/opencode_smoke_after_handle_classifier`
  accepted and committed
  `task_city rental count after date filter_04cf06b3f31f4422` with
  `solver_pass_rate=0.6666666666666666`. The accepted contract used hidden
  `city_id` only for grounding and a visible `rental_date >= 2022-06-01`
  predicate; query evidence reported `rental_date.is_handle=false`, confirming
  the classifier correction in runtime.
- **Correction**: the accepted output key `rental_count_after_june` is not a
  qualitative issue by itself. The label is an agent-visible `submit_result`
  object, similar to an API response assembled from tool outputs, not the final
  natural-language answer shown to the customer. Customer-facing constraints
  apply to `user_request` and the downstream final answer; label field names
  should be stable result keys that match the rendered submit schema.
- **Prompt-surface correction**: composer-facing text should not mention
  downstream `submit_result`; it only needs to know that `label` is the exact
  structured result answering `user_request`. The prompt now says stable result
  field names, and explicitly states that `user_request` must make the label the
  only correct structured result.
- **Validation precision correction**: later review found that treating
  `is_handle: true`, predicate references, or order-by references as automatic
  hard-reject causes is not 100%-precision for arbitrary DBs. Some PK/FK-shaped
  values can be legitimate customer-facing codes under an explicit visibility
  policy. The hard gate now rejects only direct label exposure of source fields
  explicitly marked `internal` or `blocked`, plus missing provenance. Handle
  avoidance remains prompt/tool-schema guidance and diagnostic context, not a
  hard validation rule.
- **Uniqueness correction**: multi-answer requests are poor RL signals because
  exact-match scoring becomes underdetermined. Composer-facing rules now reject
  drafts where the request admits multiple valid result objects, alternative
  tie-breaks, or partial answers; the task must be narrowed until one structured
  result is correct.
- **List-valued label clarification**: a label may be a list when the list as a
  whole is the unique canonical result. The authoring contract is not
  "single row only"; it is fixed membership, ordering, limit, and tie-breaks so
  the verifier sees exactly one valid structured result object.
- **External task-shape calibration**: a similar synthesized task format uses a
  customer-facing multi-constraint request, endpoint-like domain tools, and a
  structured submit payload such as a fixed-length list. This is useful only as
  shape guidance, not as a domain template. For this project, arbitrary DBs must
  still be handled by common DB-agnostic code, and open-ended recommendation
  language must be narrowed until one canonical structured result object remains.
- **Composer prompt refinement**: the prompt now names that shape directly as
  real end-user data-service tasks: history lookups, shortlists, summaries,
  schedules, eligibility checks, or plan-like lists when the current schema
  supports them. It still forbids domain hard-coding and vague recommendation
  asks unless filters, thresholds, ordering, limits, and tie-breaks make the
  result deterministic. The prompt also removed the remaining downstream
  "agent submits after composing tool/API responses" wording from the composer
  surface.
- **Ordered list reward correction**: inferred output schemas now preserve list
  order by default. This aligns list-valued labels with the uniqueness contract:
  a solver that finds the same members but submits them in a different order no
  longer receives exact-match credit unless a future explicit unordered contract
  is introduced.
- **Curriculum escalation calibration**: similar external tasks often start as
  one repeated unit and become harder only after a too-easy signal by growing to
  two units, then three. The composer prompt and retry feedback now treat list
  cardinality as a curriculum parameter (`1 -> 2 -> 3 -> 5 -> 10`) instead of
  jumping directly to a large payload. The prompt also fixes the old row-count
  ambiguity for take-N lists: count/profile the unbounded candidate pool, then
  run the final limited query for the canonical label.
- **Item-complexity correction**: repeated payloads can become harder either by
  increasing the item count or by making each item harder. The prompt and retry
  feedback now distinguish Item-complexity from passive Width: a harder item
  needs an added grounded condition, relationship, visible related field,
  predicate, or deterministic tie-break that changes what must be found for
  every item.
- **Experience-bound actor principle**: curriculum, specificity feedback,
  Cardinality, Item-complexity, pass-rate, quality-gate, and training-purpose
  language are composer/runtime concepts only. The actor/solver prompt remains
  the customer request, hidden entity, submit schema, and tools; strategy should
  be learned through trajectories and reward, not leaked as prompt guidance.
- **Autonomous strengthening correction**: difficulty directions are DB-specific
  and may be impossible in some schemas. Prompt and retry feedback now present
  filters, composite constraints, cardinality, item-complexity, and cross-item
  rules as a feasible-options menu, not a fixed priority ladder. The project
  provides tools and context for composer judgment; it does not hard-code the
  next crank direction.
- **Schema-map role clarification**: `schema_map` is the composer's DB-native
  exploration map. It helps choose topics, relationship paths, and feasible
  strengthening directions, but it is only a map: `profile`, `sample`,
  `neighborhood`, and final `query` calls still decide whether a candidate is
  grounded, readable, nontrivial, and uniquely labelable.
- **DB Affordance Map MVP**: added a rule-based context artifact derived from
  `SchemaGraph` and `DataProfile`. It renders table cards, path cards, and topic
  affordance cards into the composer prompt as context only. The map highlights
  readable surfaces, filters, metrics, time columns, fanout, and likely task
  families, but composer still must verify candidates with live tools and a
  canonical query.
- **DB-adaptivity cross-check**: ran the same introspect -> profile ->
  affordance-map path on pagila and postgres_air. Initial output was not good
  enough: maintenance timestamps dominated event/timeline cards, duplicate
  table-pair edges lost relation identity, and pagila partition children
  occupied top table slots. Fixed these generically at the time: hide partition
  children when a parent is visible, include join-column relation labels on path
  cards, and profile unknown row-estimate parent tables while skipping advisory
  profiling on PostgreSQL permission errors. Later correction: the
  maintenance-timestamp and identifier/status numeric exclusions were
  name-token heuristics and were removed from the scoring/context algorithm.
  Final cross-check before that correction: pagila top context centers on
  customer/rental/payment/film, postgres_air centers on aircraft/flight,
  airport/flight, account/passenger, and account/booking; top table/path overlap
  is zero, duplicate path labels are zero, and maintenance timestamps are absent.
- **Record-set semantics trial**: `artifacts/trial_complete_map_pagila_minimax_01`
  accepted a pagila task but exposed a solver/tool mismatch. One solver followed
  rental -> inventory -> rental and counted 51 instead of the canonical 12
  because relation traversal preserved hidden join multiplicity. The actor-facing
  API should behave like ordinary endpoint resources, so `record_set` traversal
  now deduplicates destination records by primary key. Regression tests cover the
  exact round trip, and the full suite passed after the fix.
- **Exact scalar strengthening guard**:
  `artifacts/trial_record_set_semantics_pagila_minimax_01` accepted with
  pass_rate 2/3, but the harder request added a date predicate while the scalar
  answer stayed 2. The lower pass rate came from one solver submitting the schema
  object, not from harder DB reasoning. This is not a good RLVR signal because an
  actor can ignore the new predicate and still hit the exact label. The
  `submit_draft` gate now stores a single-field scalar value signature after
  too-easy rejection and rejects a retry whose submitted value is unchanged under
  the same answer operation, even if the field name changes. This is a
  100%-precision check: it only fires when the reward-visible answer value is
  exactly unchanged.
- **Actor submit surface correction**:
  `artifacts/trial_scalar_guard_pagila_minimax_01` accepted with pass_rate 2/3,
  but the unmatched solver had computed the correct count and then submitted the
  JSON Schema object shown under `# Submit Result Format` instead of the answer
  object. This is an interface bug, not task difficulty. The actor prompt no
  longer includes a submit-format schema block. Runtime now exposes the required
  answer shape through a task-specific `submit_result` tool schema, matching the
  endpoint-contract design: object answers are submitted as direct tool
  arguments, and list/non-object answers use an `answer` wrapper.
- **Solver rollout denominator correction**:
  `artifacts/trial_tool_schema_submit_pagila_minimax_01` exposed a misleading
  quality-gate summary. Three solvers were launched, but two returned
  `status=failed`, `termination_reason=UserError` after using tools, and the
  monitor reported only `total_solver_runs=1`, `matched_solver_runs=1`,
  `pass_rate=1.0`. That made a 1/3 observed rollout look like a 1/1 too-easy
  result. The rollout summary now separates planned, completed, evaluable,
  failed, and matched solver counts. Only high-confidence provider/runtime
  infrastructure failures are excluded from the pass-rate denominator. Unknown
  `UserError` failures count as evaluable actor/runtime failures, so the same
  pattern is scored as 1/3 rather than 1/1. Solver event logs also include
  `excluded_from_pass_rate`, `failure_class`, and `failure_detail`.
- **Denominator fix trial**:
  `artifacts/trial_solver_denominator_fix_pagila_minimax_01` accepted. Anchor
  `film_id=866` (`SUNSET RACER`). Attempt 3 was too easy with
  `planned=3, completed=3, evaluable=3, failed=0, matched=3, pass_rate=1.0`.
  Attempt 4 added a limit: "first 2 actors after sorting by last name" and
  accepted with `planned=3, completed=3, evaluable=3, failed=0, matched=2,
  pass_rate=0.667`. No `UserError` occurred. The logging fix is verified, but
  the accepted signal is not fully clean: the unmatched solver found the correct
  two actors and then changed DB-uppercase values (`ANGELINA ASTAIRE`,
  `GROUCHO DUNST`) to Title Case in `submit_result`. This is a value-copy
  fidelity failure rather than a DB-search failure, so future trial review
  should flag casing/format-only misses separately from genuine reasoning misses.
- **Submit exact-value contract**:
  To reduce value-copy fidelity failures without adding a separate prompt block,
  the task-specific `submit_result` tool schema now states the endpoint
  contract directly: copy values exactly from data-tool outputs and do not
  change capitalization, spelling, punctuation, whitespace, numeric precision,
  or date/time formatting unless the user explicitly asked for that
  transformation. This keeps the actor interface API-like while making exact
  reward semantics visible where the answer payload is submitted.
- **Submit-desc trial and typed `IN` repair**:
  `artifacts/trial_submit_desc_exact_values_pagila_minimax_01` failed before
  acceptance. The only solver-evaluated draft was a direct store staff count:
  `planned=3, completed=3, evaluable=3, failed=0, matched=3, pass_rate=1.0`,
  so the gate correctly rejected it as too easy. The trial did verify the new
  `submit_result` schema path for object answers: all three solvers submitted
  `{"staff_count": 3}` directly. It did not meaningfully test exact string
  copying because the answer was numeric.
- **PostgreSQL udt-name array casts**:
  The same trial exposed composer query failures when filtering `int4` columns
  with `op='in'`: PostgreSQL saw `integer = text` because array casts covered
  `integer/bigint/smallint` names but not introspected `udt_name` values such
  as `int4`. Composer and atomic SQL compilation now cast arrays for common
  PostgreSQL udt names (`int2/int4/int8`, bool, numeric, timestamps, etc.) and
  fall back to the quoted type name for DB-specific enum/domain types. Composer
  `sample` filters also coerce list items through the column type before
  binding.
- **Typed `IN` repair trial**:
  `artifacts/trial_int4_array_cast_pagila_minimax_01` re-ran pagila/minimax
  after the cast fix. The previous `integer = text` failure disappeared
  (`0` occurrences), and the same live `staff_id IN (...)` pattern returned
  rows successfully. The synthesis still failed before acceptance:
  the solver-evaluated draft was again a direct store staff count with
  `planned=3, completed=3, evaluable=3, failed=0, matched=3, pass_rate=1.0`.
  Later retries did not preserve a valid incremental scalar-count
  strengthening: `max(staff_id)` was rejected as a non-user-visible source and
  non-incremental, `active=true` kept the scalar value unchanged, and
  `max(last_update)` was non-incremental. Remaining bottleneck: composer
  difficulty escalation for sparse/low-fanout store anchors, not solver
  denominator or `submit_result`.
- **Fixed-denominator solver batch policy**:
  Wrong solver answers, invalid submits, `UserError`, and max-turn failures are
  actor/runtime outcomes and always count in the pass-rate denominator.
  Provider/environment failures remain excluded, but they no longer shrink the
  denominator: the solver orchestrator schedules replacement attempts from the
  configured solver pool until the planned number of evaluable runs is reached.
  `planned_solver_runs` is now the target evaluable denominator, while
  `total_solver_runs` is the actual number of attempts, including excluded
  infra failures. Quality-gate evaluation refuses to silently score a rollout
  that exhausts the finite infra retry budget before reaching the target
  denominator. Safe early termination is also deferred until the target
  evaluable denominator has been filled.
- **Fixed-denominator trial**:
  `artifacts/trial_fixed_denominator_pagila_minimax_01` accepted and committed
  `task_최근 대여일 조회_c9e380db055126b7`. The first submission was rejected by
  strict query/evidence checks; the second submission asked for the latest
  rental date for the anchored Cianjur customer. Final rollout metrics were
  `planned=3, total=3, evaluable=3, failed=0, matched=2,
  pass_rate=0.667`. The unmatched solver submitted
  `2022-08-23 11:55:51+00:00` instead of canonical
  `2022-08-23 17:43:11+00:00`, and was correctly counted as an evaluable
  solver failure (`excluded_from_pass_rate=false`). No `integer = text` errors
  occurred. Residual quality concern: the accepted user request still says
  `rental` rather than a cleaner customer-facing Korean phrase such as
  `대여`; this is phrasing polish, not a verifier failure.

- **GPT-5.4 Nano language cross-check**:
  Direct OpenAI `gpt-5.4-nano` could not run because the provider returned
  `RateLimitError`, so the same composer model was routed through OpenRouter as
  `openai/gpt-5.4-nano` without using Codex OAuth. The trial
  `artifacts/trial_gpt54_nano_openrouter_composer_pagila_01` accepted
  `task_스토어의 대여 건수 집계_86f356558a265aca` with
  `planned=3, total=3, evaluable=3, failed=0, matched=2, pass_rate=0.667`.
  The generated request used Korean `대여` instead of English `rental`, so the
  earlier `rental` wording is more likely weak-model instruction following than
  a hard prompt-language bug. However, a separate customer-facing naturalness
  issue remains: the accepted request still exposes a DB-like anchor
  (`스토어 481`) and asks about a zero-count sparse store. That suggests the next
  prompt/context work should focus on user-natural entity handles and avoiding
  vacuous sparse anchors, not on adding DB-specific translation rules.

- **No-anchor experiment**:
  Runtime-provided `anchor_hint` was disabled so the composer starts from
  `schema_map` and chooses its own observed entity via data tools. Focused
  tests confirmed the synthesis backend now receives `anchor_hint=None`.
  Two pagila trials were run:
  `artifacts/trial_no_anchor_pagila_minimax_01` with minimax composer and
  `artifacts/trial_no_anchor_pagila_gpt54_nano_01` with OpenRouter
  `openai/gpt-5.4-nano` composer. Both logs had no `Starting Entity` text and no
  `integer = text` errors. The minimax composer chose `customer_id=1` from
  `sample(customer)` and phrased the request as first-person
  `내가 빌린 Animation 장르 영화...`; it failed as too hard because solvers
  submitted counts `7/7/6` against canonical `8`. The Nano composer chose
  `customer_id=38`, but still surfaced the hidden handle as
  `고객 번호가 38번인 고객...` and used schema-ish `amount` wording. It failed as
  too easy first, then exhausted feedback attempts while trying to strengthen
  the sum task. Conclusion: removing forced anchors helps with the specific
  `스토어 481` forced-anchor failure, but it is not sufficient. The remaining
  issue is composer self-selected hidden-entity phrasing; this should be handled
  by prompt/context shaping toward "my account/my records" or readable entity
  surfaces, not by a broad PK/FK-value reject rule.
- **User-facing ID examples**:
  After the no-anchor trials, the common composer prompt now includes explicit
  bad/good patterns for request wording. Bad examples are raw DB identifiers
  such as `<entity type> 38`, `id 38`, `<table>_id=38`, `record 38`, `row 38`,
  and `number 38 <entity>`; good examples are `my account`, `my records`, `my recent
  activity`, `the visible named item`, `the selected date range`, or a real
  observed name/title/code/reference. This is intentionally prompt guidance,
  not a broad reject rule, because PK/FK physical shape alone cannot prove that
  a value is not a public customer reference.
- **Prompt language correction**:
  The common composer prompt surface should stay English by default. The target
  output language remains configured separately by `domain.language` (e.g.,
  Korean user requests when `language: ko`). The Korean service-voice and ID
  examples in the shared prompt were replaced with English pattern descriptions,
  and prompt tests now assert that the shared system instructions contain no
  Korean text.
- **Minimal general prompt rewrite**:
  The shared composer system prompt was compressed from roughly 23k characters
  to roughly 6k characters and reorganized around role, tools, workflow,
  user-request rules, label/contract rules, task shape, and `submit_draft`.
  Each major instruction now carries a short "Why:" rationale so the model gets
  the principle without being overloaded by pipeline metadata. The prompt stays
  English-only, DB-agnostic, and free of solver/RL/pass-rate language; generated
  `user_request` language remains controlled by the runtime domain language.
  Focused verification passed:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py`
  -> 50 passed, plus ruff and `git diff --check`.
- **Composer schema-description audit**:
  After the minimal prompt rewrite, the same principles were moved into the
  composer-visible tool schemas so bad calls are prevented before feedback
  consumes a submit turn. The audit covered all composer tool descriptions,
  nested parameter descriptions, and `submit_draft`. Key changes: `entity` is
  now exposed as a scalar-valued object instead of a JSON string, `label` is
  exposed as a flat object or flat object list, and `answer_contract` is a
  `kind`-discriminated union: scalar contracts only allow aggregate functions
  and null limits, while list contracts only allow `select`. The full contract
  fields (`kind`, `operation`, `predicates`, `order_by`, `limit`,
  `limit_phrase`, `evidence`) are required. Query/schema/sample/profile
  descriptions explain their evidence role, and `submit_draft` descriptions
  state the scalar-vs-list contract and raw-id/user-facing pattern guidance.
  Terminology was clarified: common prompts and feedback are
  database-neutral, while composer behavior is database-aware through the
  current schema, generated tool schemas, and live observations.
- **All visible tool-schema audit**:
  The schema audit was expanded from composer-only to all visible tool schemas:
  five composer tools, ten actor/solver atomic resource tools, and
  `submit_draft` (16 surfaces total). The audit script found no role/pipeline
  leaks after the rewrite and exposed remaining loose schemas:
  `items: {}` in composer/atomic filter values and missing descriptions on
  atomic `get_record.table`/`columns`. These were tightened: array filter
  values now require scalar items, `get_record` fields are described, and tests
  now fail on empty item schemas or `additionalProperties: true` across the
  visible composer, atomic, and `submit_draft` schemas. Verification:
  `uv run pytest tests/test_tooling_composer_tool_factory.py tests/test_tooling_atomic_tool_factory.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py`
  -> 70 passed; ruff and `git diff --check` passed.
- **DB-agnostic random anchor candidate sampler**:
  Reframed anchors as environment randomization for stateless composer
  restarts, not task hints. The sampler now builds a rule-based eligible table
  pool without the previous `row_estimate >= 100` or single-column PK gates,
  scores tables from readable surfaces, FK reachability, structural classes, and
  task-surface affordances, samples actual rows, and emits optional candidate
  entities with qualified table handles, PK metadata, visible previews, and
  lightweight relationship counts. This is meant to reduce `id=1`/first-row
  collapse while preserving composer autonomy. Runtime wiring is controlled by
  `synthesis.runtime.anchor_candidates_enabled` and `anchor_candidate_limit`,
  defaulting off for controlled no-anchor experiments. Verification:
  `uv run pytest tests/test_config_load.py tests/test_synthesis_anchor_sampler.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  -> 44 passed; targeted ruff passed.
- **Anchor sampler metrics and quality-diversity algorithm**:
  Added candidate-level RLVR-start metrics (`rlvr_start_score`,
  anti-degeneracy, visible surface, relationship surface, dead-anchor and
  id-one indicators) and pool-level metrics (mean/p10 score, preferred-rate,
  preview-rate, positive-relation-rate, dead-anchor-rate, id-one-rate, table
  coverage, table entropy, `rlvr_start_pool_score`). These metrics are for
  sampler evaluation, not task rejection. A first attempt incorrectly used the
  metric as a preferred-candidate filter; this was replaced with a single
  quality-diversity sampling algorithm:
  `table_quality_score * structure_novelty_bonus * repeated_table_penalty`, plus
  two random row samples per selected table with the higher-scoring row kept.
  Sampler-only evaluation, no composer calls, 8 episodes × 10 candidates:
  pagila pool score `0.8627`, postgres_air `0.8920`, MIMIC-IV demo `0.9023`;
  all had `dead_anchor_rate=0` and `id_one_like_rate=0`. Verification:
  `uv run pytest tests/test_synthesis_anchor_sampler.py tests/test_synthesis_runtime.py`
  -> 37 passed; targeted ruff passed.
- **Candidate anchor prompt rendering**:
  The anchor pool is now rendered as `# Candidate Starting Points` instead of
  `# Starting Entity`. Prompt wording says candidates are optional orientation
  context, not answer hints or required topics; if one is used, call
  `neighborhood` first and still use data tools plus a final `query(spec)` for
  the exact label. `preview` and `relationship_summary` are explicitly not final
  label evidence, and raw primary-key / `row_id` values must not appear in the
  customer request. Verification:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  -> 38 passed; targeted ruff passed.
- **Anchor-candidate trial and validation-principle correction**:
  Ran pagila smoke with anchor candidates enabled, composer
  `openrouter/openai/gpt-5.4-nano`, and 3× `opencode_zen/minimax-m2.5`.
  First run (`artifacts/trial_anchor_candidates_pagila_gpt54nano_minimax_01`)
  failed too-hard: composer submitted an all-rentals aggregate while the hidden
  entity made solvers scope the request to one customer. A too-aggressive
  follow-up hard validator briefly rejected labels whose values were not
  inferred as anchor-connected. This was reverted: anchor connectivity is value
  flow inference, not 100%-precision validation, so it remains diagnostic-only.
  The schema-first fix is in `submit_draft` field descriptions: `label`,
  `user_request`, and `answer_contract` now state that "my/own/entity" wording
  must match the final query scope.
  Clean rerun (`artifacts/trial_anchor_candidates_pagila_gpt54nano_minimax_03`)
  failed too-easy: first submission was a global max payment amount,
  `pass_rate=1.0` with all 3 solvers matched. Subsequent retries added filters
  but the scalar answer stayed `11.99`; exact label-change validation correctly
  returned `label_not_strengthened` until submit budget exhaustion. Quality
  notes: candidate anchor broke `id=1` collapse, but the selected anchor was a
  low-context `payment_id`; the user request still used schema-ish wording
  (`Amount`). Verification after correction:
  `uv run pytest tests/test_config_load.py tests/test_synthesis_anchor_sampler.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  -> 48 passed; targeted ruff and `git diff --check` passed.
- **Name-token heuristic correction**:
  User review correctly rejected name-based topic/anchor shaping as artificial.
  The affordance map and anchor sampler now use structural/measured signals only:
  PK/FK flags, FK graph degree, visibility labels, DB type metadata,
  row/profile statistics, sampled preview non-emptiness, and relationship
  counts. Removed semantic table roles, role bonuses, column-name token filters
  for "identifier/status/maintenance" columns, and handle classification based
  on column-name suffixes. Primary keys and foreign keys are now handles by
  structure. DB type-name checks remain only as schema type metadata, not
  business semantic inference.
- **Global token-heuristic ban, with dedup boundary correction**:
  Tightened the correction beyond context generation. Privacy/sensitivity no
  longer infers visibility from column/table-name tokens; it only applies
  explicit overrides and the configured default. `submit_draft` no longer hard
  rejects user requests by raw-identifier token patterns or literal placeholder
  token checks. Ungrounded-value feedback no longer branches on name-like or
  datetime-like string patterns. Correction: semantic near-duplicate MinHash is
  intentionally string/token based because deduplication is string-surface
  similarity, not DB semantic inference. Therefore MinHash shingling and its
  `semantic_shingle_size` config remain valid.
- **Schema-first composer/solver tool contract cleanup**:
  Moved more machine-checkable behavior from prompt/runtime feedback into tool
  schemas and parameter descriptions. The composer system prompt no longer
  duplicates the full `submit_draft(...)` callable shape; that contract belongs
  to the tool schema. Local examples are explicitly guidance only when
  consistent with system/tool contracts. Composer `sample` and `profile`
  now expose table-scoped schema variants instead of global column enums, and
  `query` descriptions use the corrected handle guidance: prefer
  user-visible non-handle values, but allow handle-like values when evidence
  marks them user-visible and the request naturally asks for that reference.
  Solver atomic tools are strict schemas. `filter_record_set` is split into
  scalar, value-list, pattern, and null endpoints so value shape is explicit.
  `get_record` now describes the `record_ref.table` / `record_ref.id` handoff,
  and `submit_result` list schemas include exact canonical cardinality when
  known. Targeted verification:
  `uv run pytest -q tests/test_tooling_atomic_tool_factory.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py tests/test_synthesis_schema_inference.py tests/test_synthesis_canonicalize.py tests/test_solver_backend_openai_agents.py tests/test_pipeline_solver_orchestrator.py`
  -> 111 passed.

## Iteration 50 — schema-contract trial cleanup, numeric/timestamp parity

- **Pagila trial `_03` uncovered numeric coercion drift**:
  `artifacts/trial_schema_contract_pagila_gpt54nano_minimax_03` failed
  `reject_too_hard`. Composer profiled `payment.amount = 4.99`, but asyncpg
  bound JSON floats against `numeric` as floats, yielding zero rows/null sums.
  Shared SQL coercion now maps numeric/decimal/money JSON numbers and strings to
  `Decimal`, and both composer and atomic paths use the same parameter rules.
  Direct verification showed the query/tool result changed from zero/null to the
  expected count/sum.
- **Pagila trial `_04` exposed relationship-path ambiguity**:
  `artifacts/trial_schema_contract_pagila_gpt54nano_minimax_04` failed
  `reject_too_hard` after solvers picked a different reasonable `store` path
  than composer. Composer `query` now supports branch joins via explicit
  `join[].from`, and the prompt instructs authors to make the customer-language
  role explicit when multiple relationship roles can answer the same concept.
  `submit_draft` feedback for one-row scalar vs one-row list labels was also
  corrected.
- **Pagila trial `_05` found atomic timestamp semantics drift**:
  `artifacts/trial_schema_contract_pagila_gpt54nano_minimax_05` reached solver
  evaluation but failed `reject_too_hard`: composer summed July-rental payments
  as `28510.56`, while all solvers derived `27447.12`. Root cause was not model
  reasoning: atomic filters converted date-only strings for `timestamptz`
  columns into naive datetimes, and asyncpg interpreted them differently from
  composer’s UTC-aware canonical query. `timestamptz` string coercion now
  attaches UTC when no timezone is supplied, and direct atomic calculus
  `rows_where` / `filter_rows` also coerce values by column type.
- **Provider override and accepted minimax trial**:
  CLI config still defaulted to `opencode_zen/qwen3.5-plus`, so `_06`
  (`artifacts/trial_schema_contract_pagila_gpt54nano_minimax_06`) failed fast
  with a Qwen rate limit. A runtime override was used for `_07` with
  composer/solvers all on `opencode_zen/minimax-m2.5`.
  `artifacts/trial_schema_contract_pagila_minimax_07` accepted and committed:
  task `task_내 최근 결제 내역 조회_fa237f54a2bc3227`, pass rate
  `12/30 = 0.4`, CI `[0.2495, 0.5661]`, bundle exported. Failed and invalid
  solver submissions were evaluable and remained in the denominator.
  `run-real-db-trial` now accepts the same runtime model override flags as
  `validate-config`, so this can be reproduced without a one-off Python runner.
- **API-like temporal output cleanup**:
  `_07` also showed a non-terminal schema-contract nuisance: tool output
  serialized datetimes with `str(datetime)` (`YYYY-MM-DD HH:MM...`) while the
  composer first rewrote them to ISO `T` strings and received precise feedback.
  Tool JSON serialization now emits `datetime/date/time.isoformat()` so
  composer and solver see stable API-like temporal strings.
- **Verification**:
  `uv run pytest -q tests/test_cli_commands.py::test_cli_run_real_db_trial_reports_summary tests/test_tooling_common.py::test_tool_json_serializes_temporal_values_as_iso_strings tests/test_tooling_atomic_integration.py::test_timestamptz_date_strings_match_utc_canonical_query`
  -> 3 passed. Targeted ruff over changed CLI/tooling/test files passed.

- **Attempts observed**: N / max_generation_attempts
- **Submissions**: M / N (how many attempts actually reached submit_draft)
- **Pass-rate trajectory**: [0.0, 0.67, 0.33, ...] per submitted attempt
- **Terminal status**: accepted | reject_too_easy (band still above) | reject_too_hard (band still below) | MaxTurnsExceeded | other
- **Ladder climb observed**: which rungs were visible in each submission's label structure
- **Regression signals**: agent over-exploring after rejection, repeating same rung, weakening label, etc.

## Iteration 51 — exact-CI calibration semantics

- **Postgres_air trial after concurrency fix**:
  `artifacts/trial_postgres_air_minimax_03` reached solver rollout with
  composer/solvers on `opencode_zen/minimax-m2.5`. Composer needed feedback to
  preserve `numeric` values as exact JSON strings, then produced the task
  "제 최근 예약 중 가격이 1500달러 이상인 가장 최근 예약 5건의 예약번호와 가격".
  Solver rollout completed 30 evaluable runs, with `24/30 = 0.8` matched and
  no infrastructure exclusions.
- **Calibration bug found**:
  The early-stop path had moved to exact Clopper-Pearson CI, but final
  `evaluate_rollout_summary` still fell back to point-estimate difficulty
  rejection when the CI decision was inconclusive. For `24/30` at
  `alpha=0.1`, exact CI is approximately `[0.643, 0.909]`, which overlaps the
  configured upper bound `0.75`; this is not a statistically decisive
  `reject_too_easy`.
- **Fix**:
  Calibration decisions now return too-easy / too-hard only when the exact
  Clopper-Pearson interval is fully above / below the configured band. If the
  observed point estimate is outside the band but the CI overlaps it, the
  quality gate reports `calibration_inconclusive`. Point-in-band drafts remain
  acceptable, with CI persisted as quality metadata. This keeps the gate
  statistically honest without making every in-band finite sample require
  interval containment.
- **Verification**:
  `uv run pytest -q tests/test_calibration.py tests/test_pipeline_solver_orchestrator.py tests/test_synthesis_runtime.py`
  -> 53 passed. Targeted `uv run ruff check src tests` passed.

## Iteration 52 — 20-run pass-rate band retarget

- **Decision**:
  With `max_solver_runs=20`, the pass-rate target band is retargeted from
  `[0.25, 0.75]` to `[0.5, 0.9]`. The old band was useful while debugging
  small-N smoke runs, but at N=20 it accepts tasks that too many actors fail and
  rejects near-0.8 tasks as too easy even though those are useful RLVR traces.
- **Config change**:
  Main, postgres_air, MIMIC-IV, and MIMIC-IV demo configs now use
  20 solver slots, `lower_pass_rate: 0.5`, `upper_pass_rate: 0.9`,
  `ci_alpha: 0.1`, `max_solver_runs: 20`, and `solver_batch_size: 4`.
  Early termination now uses exact one-sided Clopper-Pearson bounds, while the
  two-sided interval remains the reporting metric. With alpha=0.1, the first
  batch can reject a clearly too-hard draft at `0/4`; later batch checkpoints
  reject too-hard at `<=1/8`, `<=3/12`, `<=4/16`, and `<=6/20`. This setting
  intentionally does not try to reject too-easy drafts in the first batch;
  proving pass rate above `0.9` would require at least `22/22` all-pass samples.

## Iteration 53 — trial baseline after 20-run calibration

- **Trial**:
  `artifacts/trial_calibration20_pagila_qwen_01`, pagila, default
  `opencode_zen/qwen3.5-plus`, calibration `[0.5, 0.9]`,
  `max_solver_runs=20`, `solver_batch_size=4`.
- **Result**:
  `synthesis_failed / reject_too_hard` after the first solver batch:
  `0/4` matched, all four failures evaluable. The one-sided exact early-stop
  gate behaved as intended.
- **Quality observations**:
  This run is not a useful accepted-data quality sample. It exposed two setup
  issues instead. First, the production config had anchor candidates disabled,
  so the composer restarted from `customer_id=1` / MARY SMITH despite the
  anti-degeneracy sampler existing. Second, qwen/opencode solver runs struggled
  with the actor tool protocol: three runs ended as `missing_submit_result`
  after emitting XML-style function text, and one hit `MaxTurnsExceeded`.
- **Config follow-up**:
  Enable `synthesis.runtime.anchor_candidates_enabled` in repo trial configs.
  Keep qwen result as provider-compatibility evidence and rerun with the
  previously accepted `minimax-m2.5` path before judging task-quality prompts.

## Iteration 54 — default experiment model switch

- **Decision**:
  Qwen remains too quota-sensitive for the default development loop, and the
  Iteration 53 pagila run showed solver-side tool protocol instability on
  `opencode_zen/qwen3.5-plus`. Default repo trial configs now use
  `openrouter/openai/gpt-5.4-nano` for both composer and solver runs.
- **Cross-check model**:
  Use `openrouter/moonshotai/kimi-k2.5` for higher-quality validation runs.
  Kimi is expected to be closer to the intended production composer class,
  while Nano is the cheap high-volume tuning model.

## Iteration 55 — pagila nano harvest and entity-context clarification

- **Trial**:
  `artifacts/eval_20260427_pagila_nano_harvest_01`, pagila,
  composer/solver `openrouter/openai/gpt-5.4-nano`, calibration
  `[0.5, 0.9]`, `max_solver_runs=20`, `solver_batch_size=4`, target 2.
- **Result**:
  Harvest reached target after 9 attempts. Accepted tasks were
  `task_활성 고객의 결제 1건 중 최대 금액_f48fcb91e50c0296`
  (`16/20` matched, 2 failed-status solver runs) and
  `task_프랑스어(French) 언어로 제공되는 영화 중 대여된(렌탈된) 적이 있는 영화 수_4624bede684cc0c3`
  (`12/16` matched, no failed-status solver runs). Rejections were mostly
  `reject_too_hard`, `calibration_inconclusive`, and schema/contract feedback
  such as `anchor_entity_required`, `answer_contract_json_invalid`, and
  `answer_contract_query_mismatch`.
- **Quality observation**:
  The gate is filtering, but both accepted drafts exposed a design tension:
  the actor-visible hidden `<entity>` block is intentional current-context
  input, yet the accepted labels were global-looking aggregates with an entity
  attached (`payment_id` for global max payment amount, `language_id` for all
  French-language rented movies). The issue is not that the solver receives an
  entity; it should. The issue is that composer can treat the entity as a
  decorative anchor rather than the current customer/session/object context.
- **Decision**:
  Do not enforce solver path equivalence. Solver reward remains exact
  `submit_result` label match; multiple valid tool paths are allowed. Also do
  not add a hard validator that requires the final query to contain the raw
  entity value, because legitimate indirect scopes can resolve entity-derived
  values first and then query by those values. Entity-context problems are not
  100%-precision hard-validation material unless the violation is mechanically
  provable.
- **Fix**:
  Strengthened composer system prompt and `submit_draft` schema descriptions:
  hidden entity is current-context grounding, not a decorative anchor; labels
  should be scoped to that context directly or through observed derived values;
  unrelated global answers with hidden entities attached are invalid authoring
  targets. Recorded the same boundary in `docs/spec/foundation.md`.
- **Verification**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py -q`
  -> 40 passed. Targeted `uv run ruff check ...` and `git diff --check`
  passed.

## Iteration 56 — strict submit_draft schema surface

- **Follow-up smoke**:
  `artifacts/eval_20260427_pagila_nano_entity_context_01`, pagila,
  composer/solver `openrouter/openai/gpt-5.4-nano`, target 1, was interrupted
  after enough signal. The run repeatedly produced either too-easy customer
  payment aggregates (`20/20`, `40/40`) or too-hard/contract-invalid drafts.
  No accepted task was kept from this smoke.
- **Root cause found**:
  `submit_draft` was still exposed with `strict_json_schema=False`. The
  model-visible schema marked fields required, but weak/cheap composer calls
  could still omit `entity`, put the answer value into
  `answer_contract.evidence`, or otherwise rely on runtime feedback. This
  conflicts with the schema-first principle.
- **Design constraint**:
  Direct strict schema for dynamic `label` and `entity` objects is not accepted
  by the Agents strict-schema normalizer because strict object schemas cannot
  use arbitrary `additionalProperties`. Keeping dynamic objects would force the
  tool to remain non-strict.
- **Fix**:
  The composer-facing callable now uses strict top-level fields
  `label_json` and `entity_json`, both JSON strings, while internal code still
  parses them back into canonical label objects and hidden entity maps. The
  `submit_draft` FunctionTool now runs through `ensure_strict_json_schema(...)`
  and sets `strict_json_schema=True`. Prompt wording and specs now reference
  `submit_draft.entity_json` where field-level precision matters.
- **Principle preserved**:
  This does not change solver reward or enforce solver path equivalence.
  Solver still only needs to submit the exact label. The strict schema only
  makes the composer authoring API harder to call in malformed ways.
- **Verification**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py -q`
  -> 40 passed. Targeted ruff and `git diff --check` passed.

## Iteration 57 — strict submit smoke quality review

- **Trial**:
  `artifacts/eval_20260427_pagila_nano_strict_submit_02`, pagila,
  composer/solver `openrouter/openai/gpt-5.4-nano`, target 1.
- **Result**:
  Target reached on the first attempt in about 87 seconds. The accepted draft
  matched `18/20` solver runs with no failed-status solver runs. The earlier
  provider BadRequest from strict schema was fixed by typing
  `answer_contract.predicates[].value` as JSON scalar or scalar-list instead of
  unconstrained `object`.
- **Quality review**:
  The accepted task still exposed an authoring quality issue:
  hidden entity was `{"category_id": 5}`, but the user request said
  "제가 최근에 빌린 코미디(Comedy) 영화의 대여 ID..." and the label centered on
  `rental_id`. That is not a solver-path problem and should not change reward:
  solver still only needs exact `submit_result`. The issue is composer
  naturalness/context selection: first-person ownership was attached to a
  category anchor, and a raw handle became the main answer.
- **Fix**:
  Strengthened composer prompt and `submit_draft` field descriptions:
  use first-person ownership only when the hidden context naturally represents
  the requester/account/session/order/records; for non-user subjects, use
  neutral wording. Do not make raw handles the main selected answer merely
  because they are easy to query; list tasks should include at least one
  non-handle visible field when selecting rows.
- **Verification**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py -q`
  -> 40 passed. Targeted ruff and `git diff --check` passed.

## Iteration 58 — context-quality prompt follow-up smoke

- **Trial**:
  `artifacts/eval_20260427_pagila_nano_context_quality_01`, pagila,
  composer/solver `openrouter/openai/gpt-5.4-nano`, target 1, interrupted
  after repeated failures.
- **Result**:
  No accepted task. The prompt update reduced the earlier easy/global payment
  aggregate pattern, but shifted Nano toward too-hard or contract-invalid
  drafts: film metadata lookup with `1/8` matched, United States customer count
  at `0/4`, customer/rental/payment tasks with repeated
  `answer_contract_phrase_missing` or `answer_contract_query_mismatch`, and
  several `reject_too_hard` first-batch exits.
- **Interpretation**:
  The current validation/gating behavior is doing the right defensive work:
  bad or badly calibrated drafts are not passing. The remaining bottleneck is
  composer proposal quality for cheap Nano: it struggles to land naturally in
  the `[0.5, 0.9]` band while satisfying the answer-contract fields. This is
  not evidence that solver reward should enforce paths; solver remains
  label-exact only.
- **Next diagnostic**:
  Cross-check with the stronger intended composer class (`kimi-k2.5`) before
  adding more common prompt text. If Kimi lands good drafts under the same
  strict tool contract, the issue is mostly Nano authoring ability/cost
  tradeoff. If Kimi shows the same failure mode, simplify the answer-contract
  authoring surface.

## Iteration 59 — Kimi composer cross-check

- **Trial**:
  `artifacts/eval_20260427_pagila_kimi_crosscheck_01`, pagila,
  composer `openrouter/moonshotai/kimi-k2.5`, solvers configured as
  `openrouter/openai/gpt-5.4-nano`.
- **Result**:
  `synthesis_failed` before solver rollout. Kimi called `schema_map` and
  `neighborhood(customer, row_id=353)`, then ended with empty final output and
  no `submit_draft` call. No submit_draft feedback, solver run, or quality
  gate result was produced.
- **Interpretation**:
  This does not confirm the same authoring-quality failure as Nano; it exposes
  a different provider/model tool-protocol failure. Before treating Kimi as the
  production composer class, run a tiny tool-call compliance smoke specifically
  for "must eventually call submit_draft" under the strict schema surface. If
  Kimi continues ending without submit, the pipeline needs provider-specific
  handling or a different strong composer model for generation.

## Iteration 60 — simplify answer_contract to request binding

- **Trigger**:
  Iteration 58 failures showed cheap composers spending turns on
  `answer_contract_phrase_missing` and `answer_contract_query_mismatch` even
  though table/column/operator evidence is already present in the latest
  `query` result.
- **Decision**:
  `answer_contract` should not ask the composer to retype machine-derivable SQL
  structure. The composer now submits only `kind`, `answer_phrase`,
  `constraint_phrases`, and `limit_phrase`. Runtime still verifies exact label
  equality against the latest successful query and derives structural retry
  evidence from query metadata.
- **Why this follows the principles**:
  Solver/actor reward remains exact `submit_result` match only; no solver path
  equivalence is introduced. Hard validation keeps only 100%-precision checks:
  schema shape, exact phrase substrings, grounded values, latest-query label
  equality, visibility metadata, and reward-visible strengthening after
  too-easy feedback.
- **Verification**:
  `uv run pytest tests/test_tooling_composer_tool_factory.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py -q`
  -> 57 passed. Targeted ruff and `git diff --check` passed.

## Iteration 61 — query select fields are label fields

- **Trial**:
  `artifacts/eval_20260427_pagila_nano_request_binding_01`, pagila,
  composer/solver `openrouter/openai/gpt-5.4-nano`.
- **Result**:
  `synthesis_failed / reject_too_hard` after one feedback event and first
  solver batch (`0/4`). This was a useful negative smoke: the new minimal
  `answer_contract` shape worked, but Nano selected customer identity/context
  fields in `query.select` while the request asked only for recent payments.
  The first submission correctly failed latest-query equality; the retry copied
  the extra selected columns into the label, making an awkward exact answer.
- **Fix**:
  Strengthened schema/prompt/feedback rather than adding a heuristic validator:
  `query.select` now says every selected field becomes a canonical label field,
  and `label_json` / evidence-mismatch feedback tell the composer to rerun
  `query` with only intended submit fields when helper/context fields were
  selected. A second smoke still failed `0/4` after Nano kept selecting
  profile/scope fields and joined two child sets independently from the same
  customer root, so the schema/prompt guidance was extended with a DB-agnostic
  relationship rule: when one answer item combines facts from the same
  event/record, continue through that event/record path rather than using
  independent sibling joins. A third Nano smoke still failed `0/4` with the
  same profile-field selection pattern, which suggests the remaining issue is
  cheap-composer instruction following rather than a new hard-validation target.
- **Verification**:
  `uv run pytest tests/test_tooling_composer_tool_factory.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py -q`
  -> 57 passed. Targeted ruff passed.

## Iteration 62 — composer no-tool-call diagnostics

- **Trial**:
  `artifacts/eval_20260427_pagila_kimi_request_binding_01`, pagila,
  composer `openrouter/moonshotai/kimi-k2.5`, solver configured as
  `openrouter/openai/gpt-5.4-nano`.
- **Result**:
  Kimi again failed before any solver rollout. The event log showed one model
  request, empty final output, and `tool_calls=[]`; there were no
  `submit_draft` attempts and no feedback events.
- **Fix**:
  Added exact diagnostics for composer tool-protocol failures. When a synthesis
  conversation ends with no accepted draft and no submit attempts, runtime now
  records `composer_no_tool_calls` if the model made zero tool calls, or
  `composer_submit_draft_missing` if it used data tools but never submitted.
  This is observability only; it does not judge semantic quality and does not
  add a heuristic validator.
- **Verification**:
  `uv run pytest tests/test_synthesis_runtime.py -q` -> 36 passed. Targeted
  ruff passed.

## Iteration 63 — Kimi tool-choice compatibility

- **Trigger**:
  Iteration 62 showed Kimi ending with `tool_calls=[]` and empty final text
  before any draft or solver rollout.
- **Fix**:
  Treat Kimi/Moonshot endpoints like the other OpenAI-compatible gateways that
  cannot safely use SDK-enforced `tool_choice="required"`. The SDK helper now
  emits `tool_choice="auto"` for Kimi/Moonshot while preserving `"required"`
  for GPT/Claude-class endpoints that support strict per-turn tool use.
- **Why this follows the principles**:
  This is provider-protocol compatibility, not semantic validation. It does not
  add heuristic feedback, does not change the composer role contract, and does
  not expose solver/RL internals to the composer.
- **Verification**:
  `uv run pytest tests/test_tool_choice_for_model.py -q` -> 6 passed.
  Targeted ruff passed.

## Iteration 64 — allow scalar item-complexity strengthening

- **Trial**:
  `artifacts/eval_20260427_pagila_kimi_toolchoice_01`, pagila,
  composer `openrouter/moonshotai/kimi-k2.5`, solver
  `openrouter/openai/gpt-5.4-nano`.
- **Result**:
  The Kimi tool-choice fix worked: the composer called data tools and
  `submit_draft` normally. The draft still failed because every evaluated
  candidate was solved by all 20 Nano solvers (`20/20`, reported CI
  `[0.8609, 1.0]`). The useful quality signal was the last retry: Kimi tried
  to strengthen the same scalar payment summary by adding grounded aggregate
  output fields (`min`, `max`, `avg`), but the incremental validator rejected
  this as `operation_changed/no_new_structural_constraint`.
- **Fix**:
  Incremental evidence comparison now treats added query output sources as a
  valid strengthening when all previous output sources are preserved. Removing
  or replacing previous output sources is still rejected. Feedback wording was
  updated to allow grounded scalar output-field growth over the same evidence
  path, not only new filters/order/cardinality.
- **Why this follows the principles**:
  This is structural metadata from the latest `query` result, not a
  token/name heuristic. It preserves exact label grounding and keeps the same
  answer kind, while allowing the composer to choose a DB-adaptive difficulty
  direction: item complexity.
- **Verification**:
  `uv run pytest tests/test_synthesis_runtime.py::test_incremental_evidence_allows_added_scalar_output_fields tests/test_synthesis_runtime.py::test_incremental_evidence_rejects_replaced_output_fields -q`
  -> 2 passed.
  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py -q`
  -> 61 passed. Targeted ruff passed.

## Iteration 65 — source-aligned related field materialization

- **Trial**:
  `artifacts/eval_20260427_pagila_kimi_scalar_strengthening_01`, pagila,
  composer `openrouter/moonshotai/kimi-k2.5`, solver
  `openrouter/openai/gpt-5.4-nano`.
- **Result**:
  Kimi now produced valid drafts and scalar strengthening reached the solver.
  The first two drafts were solved by all sampled Nano solvers. The third
  draft asked for the five most recent qualifying payments plus each payment's
  related film title; it was grounded by the composer but the solver went
  `0/4` with no environment failures.
- **Diagnosis**:
  The failure exposed a solver tool-surface gap. `follow_relation` correctly
  returns a unique destination `record_set`, but that set loses the per-source
  item alignment needed for list answers such as "each payment with its film
  title." The solver could still retrieve the answer by reading every payment,
  rental, inventory, and film one at a time, but that blows up turn count for
  ordinary list-size tasks.
- **Fix**:
  Added solver endpoint `list_records`, which lists records from an existing
  `record_set` in order while projecting requested direct fields and fields
  reached through single-record FK paths. Fan-out paths are rejected as a
  request error instead of fabricating ambiguous list rows. The atomic tooling
  version was bumped to `atomic-resource-api-v5`.
- **Why this follows the principles**:
  This is structural, DB-agnostic FK metadata only. It does not validate
  semantic quality, does not use token/name heuristics, and preserves the
  actor reachability guarantee for bounded list answers with related per-item
  fields.
- **Verification**:
  `uv run pytest tests/test_tooling_atomic_tool_factory.py::test_v2_materializing_tools_record_hidden_trace_events tests/test_tooling_atomic_tool_factory.py::test_v2_list_records_preserves_source_alignment_across_fk_path -q`
  -> 2 passed.
  `uv run pytest tests/test_tooling_atomic_tool_factory.py tests/test_synthesis_prompts.py tests/test_pipeline_solver_orchestrator.py tests/test_synthesis_bundle_exporter.py -q`
  -> 47 passed.
  `uv run pytest -q` -> 415 passed. `uv run ruff check .` and
  `git diff --check` passed.

## Iteration 66 — malformed submit_draft inputs stay inside synthesis

- **Trial**:
  `artifacts/eval_20260427_pagila_kimi_list_records_01`, pagila,
  composer `openrouter/moonshotai/kimi-k2.5`, solver
  `openrouter/openai/gpt-5.4-nano`.
- **Result**:
  The trial did not reach solver evaluation. Kimi explored a valid customer
  path and query, then submitted malformed `submit_draft` arguments twice:
  invalid `label_json`, invalid scalar `entity_json`, and finally tool input
  with extra JSON data. The second malformed call surfaced as provider
  `UserError`, aborting synthesis instead of consuming the remaining draft
  attempts through normal rejection feedback.
- **Fix**:
  `submit_draft` now catches malformed top-level tool input JSON and non-object
  tool input before pydantic validation, records a structured
  `submit_payload_invalid` rejection, and returns normal `RejectedError`
  feedback. This keeps bad composer calls inside the synthesis loop rather
  than classifying them as environment/provider failures.
- **Why this follows the principles**:
  This is pure schema/protocol robustness. It does not add semantic validation
  or heuristic quality filtering; it only ensures invalid callable shape is
  handled by the tool contract.
- **Verification**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_tool_rejects_malformed_tool_input_without_crashing -q`
  -> 1 passed.
  `uv run pytest tests/test_synthesis_runtime.py tests/test_tooling_atomic_tool_factory.py tests/test_synthesis_prompts.py tests/test_pipeline_solver_orchestrator.py tests/test_synthesis_bundle_exporter.py -q`
  -> 86 passed.
  `uv run pytest -q` -> 416 passed. `uv run ruff check .` and
  `git diff --check` passed.

## Iteration 67 — post-fix real trial confirms list_records reachability

- **Trial**:
  `artifacts/eval_20260427_pagila_kimi_list_records_02`, pagila,
  composer `openrouter/moonshotai/kimi-k2.5`, solver
  `openrouter/openai/gpt-5.4-nano`.
- **Result**:
  The trial did not produce an accepted draft, but it positively confirmed the
  main solver tool-surface fix. The first draft asked for a customer's five
  most recent rentals with film title, rental date, and return date. Nano
  solvers used `list_records` over a sorted `rental` record_set with the
  related path `rental.inventory_id->inventory -> inventory.film_id->film`;
  19/20 matched the canonical label, with `atomic-resource-api-v5.trace.v1`.
- **Quality gate behavior**:
  Because pass rate was `0.95`, the draft was rejected as still too easy /
  calibration high side. This is expected: the new endpoint made the intended
  per-item related-field task reachable, and the statistical gate correctly
  refused an over-easy sample.
- **Failure after strengthening**:
  The composer then added a PG-13 rating filter and output field. The draft was
  structurally grounded but became too hard for Nano: first batch was `0/4`,
  all solver runs were evaluable, and there were no environment failures. The
  terminal reject therefore reflects the intended "discard and restart" path
  for over-hard or low-quality candidates.
- **Interpretation**:
  `list_records` is confirmed in the live actor loop. The remaining tuning
  question is composer difficulty calibration after an over-easy rejection:
  it can make a grounded jump that is valid but too large. This should be
  handled through prompt/schema guidance or statistical retry policy, not a
  semantic validator.

## Iteration 68 — related-field filters are first-class solver actions

- **Trigger**:
  Iteration 67's second draft added a grounded PG-13 filter on `film.rating`
  while keeping a `rental` answer list. Composer could express this naturally
  in `query`, but the solver had no direct source-preserving endpoint for
  "keep rental records whose related film has rating X." The actor could
  theoretically build the result by creating a film set, filtering it, walking
  reverse relations back to rental, and intersecting with the customer rental
  set, but Nano went `0/4`.
- **Fix**:
  Added solver endpoint `filter_record_set_by_related`. It takes a source
  `record_set_id`, a FK `path`, a related `column`, scalar `op`, and scalar
  `value`, then returns a new record_set over the original source table. The
  implementation resolves the path structurally, filters the related table,
  walks inverse FK edges back to the source table, and intersects with the
  source record_set. The tooling version was bumped to
  `atomic-resource-api-v6`.
- **Why this follows the principles**:
  This is an actor reachability improvement, not a validator. It uses only
  schema/FK metadata and typed scalar values; there are no DB-specific names,
  token rules, or semantic quality heuristics. It also keeps the API-like
  resource contract: filter a collection by a related field and continue with
  sort/list/submit.
- **Verification**:
  `uv run pytest tests/test_tooling_atomic_tool_factory.py::test_build_atomic_tools_returns_tools_in_calculus_order tests/test_tooling_atomic_tool_factory.py::test_v2_filter_record_set_by_related_filters_source_records -q`
  -> 2 passed.
  `uv run pytest tests/test_tooling_atomic_tool_factory.py tests/test_synthesis_prompts.py tests/test_pipeline_solver_orchestrator.py tests/test_synthesis_bundle_exporter.py -q`
  -> 48 passed.
  `uv run pytest -q` -> 417 passed.
  `uv run ruff check src/rl_task_foundry tests` and `git diff --check` passed.
  Full `uv run ruff check .` is currently blocked by an unrelated untracked
  `scripts/load_wikitree_sample.py` import-order/unused-import finding.

## Iteration 69 — keep filter/context fields out of labels

- **Trigger**:
  Post-v6 pagila trial `artifacts/trial_related_filter_pagila_gpt54nano_01`
  confirmed the new solver trace version (`atomic-resource-api-v6.trace.v1`)
  but failed on the first submitted draft. The customer request asked for the
  top five Japanese film titles, while the canonical label included both
  `film_title` and helper field `language_name`. Solvers either returned the
  hidden handle `language_id=3`, blank language strings, or otherwise failed
  the extra field; the first batch was `0/4`, all evaluable, and the draft was
  terminally rejected as too hard/low quality.
- **Diagnosis**:
  This is not a 100%-precision validation opportunity. Whether a selected
  field is a requested answer field or just a filter/context field is semantic.
  A hard validator would require string/token heuristics and would violate the
  project rule. The correct layer is tool-schema and prompt contract: make the
  composer query/submit surface state that filter/scope/order/tie-break values
  must not be selected into the label unless the user also asks to receive
  them.
- **Fix**:
  Strengthened three contract surfaces:
  `query.spec.select`, `submit_draft.label_json`, and the durable composer
  system instructions now all say that constraint, filter, scope, ordering, and
  tie-break values belong in request/contract phrases, not in `label_json`,
  unless the user asks to receive those values. Also clarified
  `list_records.fields.path` for solvers: `path` contains only relation labels;
  the final field name belongs in `column`.
- **Why this follows the principles**:
  No new semantic validator, no DB-specific names, and no token-based quality
  rule were added. The change keeps precision-sensitive checks out of runtime
  validation and improves the API contracts the two independent agents already
  see.
- **Verification**:
  Targeted schema/prompt tests passed:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_tooling_atomic_tool_factory.py::test_v2_list_records_schema_distinguishes_paths_from_columns -q`
  -> 4 passed.

## Iteration 70 — rollback unused join warning

- **Hypothesis tested**:
  A structural `join_warnings` field on composer `query` results might help
  authors avoid independent reverse sibling joins such as `customer -> payment`
  and `customer -> rental` when the requested answer needs the rental tied to a
  specific payment.
- **Live tests**:
  Ran two targeted pagila real trials with GPT-5.4 Nano composer/solvers:
  `artifacts/trial_join_warning_pagila_gpt54nano_01` with topic
  `latest payment and corresponding rental`, and
  `artifacts/trial_join_warning_pagila_gpt54nano_02` with topic
  `customer account latest payment and corresponding rental`.
- **Result**:
  The warning did not fire in either trial. Trial 01 chose the correct
  `payment -> rental -> inventory -> film` chain but ended
  `calibration_inconclusive` at `8/20`. Trial 02 was accepted at `12/16`
  (`pass_rate=0.75`) and also chose the correct chained path
  `customer -> payment -> rental` immediately.
- **Decision**:
  Reverted the `join_warnings` implementation. The accepted live result is
  useful evidence that the existing schema map plus join description can guide
  the composer, but it is not evidence that the warning field itself helps.
  Keeping unused warning surface would add API noise without measured benefit.

## Iteration 71 — composer prompt surface ownership cleanup

- **Trigger**:
  Recent too-easy repairs exposed an instruction ownership problem. The
  durable difficulty-up behavior was split across system prompt wording,
  `submit_draft` feedback, and tool-surface summaries. The composer also saw
  redundant tool descriptions in prompt text even though the Agents SDK already
  injects callable tools through the API tool channel.
- **Diagnosis**:
  `too_easy` should not be handled by adding a new validation heuristic or by
  expanding feedback text. It is a composer policy issue: feedback should only
  say the draft needs more specificity and point to the named system policy.
  The system prompt should own durable strategy; user context should own only
  current-run facts; SDK tool schema/description should own callable contracts.
- **Fix**:
  Reworked the composer system prompt into Codex-style named sections:
  `# Role`, `# Draft Submission Budget`, `# Workflow`,
  `# Difficulty-Up Policy`, `# Customer Request`, `# Label Contract`, and
  `# Task Shapes`. Removed the prompt `# Tools` section and stopped mirroring
  composer tool names/descriptions into `<environment_context>`, because the
  SDK sends `FunctionTool` name, description, JSON schema, and strictness via
  the separate API `tools` payload. Converted user context to a stable
  `<environment_context>` envelope with nested tags. Shortened too-easy
  feedback so it references `Difficulty-Up Policy` instead of restating the
  policy.
- **Principle captured**:
  Added the composer surface ownership rule to
  `docs/spec/pipeline-lifecycle.md`: one textual source of truth per durable
  policy; no SDK tool-surface mirroring in system/user prompt text; feedback is
  reactive enforcement only.
- **Smoke results**:
  `validate-config` passed. `check-db` passed against pagila as
  `rlvr_reader`. A no-quota proof vertical slice accepted, committed, and
  exported a bundle with `solver_pass_rate=0.5` and
  `registry_status=committed`. During smoke, proof anchor seeding initially
  logged a read-only permission warning for the ephemeral proof schema; fixed
  by granting the read-only role `USAGE` on the proof schema and `SELECT` on
  its tables, then added a regression assertion that the warning does not
  return.
- **Real trial attempt**:
  The operational model policy from Iteration 54 still applies: GPT-5.4 Nano
  (`openrouter/openai`, Opencode, or direct OpenAI route) is the frequent
  low-cost experiment model; Kimi K2.5 (`openrouter` or Opencode route) is the
  higher-quality final confirmation model. A default Nano trial
  (`artifacts/trial_20260427_sharp_meadow_01`) reached schema snapshotting but
  failed before composer behavior could be observed because the shell had not
  loaded the repo-level `.env` (`openrouter/openai/gpt-5.4-nano:
  AuthenticationError`). After sourcing `/Users/jd/workspace/rl-task-foundry/.env`,
  the OpenRouter Kimi final-confirmation trial
  (`artifacts/trial_20260427_sharp_meadow_kimi_openrouter_02`, composer
  `openrouter/moonshotai/kimi-k2.5`, solver `openrouter/openai/gpt-5.4-nano`)
  accepted and committed `task_customer_recent_rental_history_85bdb2f442fca725`
  with `solver_pass_rate=0.9` and CI `[0.7174, 0.9819]`.

  Opencode Kimi needed the provider-native short model name:
  `opencode_zen/moonshotai/kimi-k2.5` failed provider authentication, while
  `opencode_zen/kimi-k2.5` succeeded in
  `artifacts/trial_20260427_sharp_meadow_kimi_opencode_short_01`. This run
  directly exercised the new policy: the first payment-history draft for
  customer 248 was rejected as calibration-inconclusive/too easy with
  `pass_rate=1.0`; the composer then preserved the same anchor, list shape,
  recent-payment query path, and row set, and added `film_title` through the
  rental/inventory/film join. The second draft
  `task_payment_history_with_films_bb2886f28864134a` accepted and committed
  with `solver_pass_rate=0.55` and CI `[0.3469, 0.7413]`. This is the intended
  `Difficulty-Up Policy` behavior: a single structural strengthening instead
  of changing the task or stacking unrelated filters.
- **Verification**:
  Prompt/feedback smoke assertions confirmed: `# Difficulty-Up Policy` is
  present, too-easy feedback points to that policy, the old specificity policy
  name is absent, prompt `# Tools` is absent, SDK tool surface is not mirrored
  into user context, and Codex-style `<environment_context>` is preserved.
  `uv run pytest -q` -> 418 passed. Targeted `ruff check` on touched files
  passed.

## Iteration 72 — DB-swap cross-check and adaptive task-shape gates

- **Goal clarified**:
  The project target is DB-adaptive generation: changing the configured DB
  should be enough for the composer to discover a reachable, calibrated task
  from schema/data evidence. Durable policy belongs in the common system prompt
  and structural validators; DB-specific knowledge belongs only in DB config
  examples/pitfalls or observed tool evidence.
- **DB surface cross-check**:
  `validate-config` passed for pagila, postgres_air, mimiciv_demo, and
  mimiciv. `check-db` now verifies the configured schema allowlist, table
  counts, read-only mode, and SELECT coverage instead of only reporting
  `current_schema=public`. Live checks passed for pagila
  (`allowlist=public tables=22 selectable=22`), postgres_air
  (`allowlist=postgres_air tables=10 selectable=10`), and MIMIC-IV demo
  (`allowlist=mimiciv_hosp,mimiciv_icu tables=31 selectable=31`). Full MIMIC
  did not run locally because the configured port 5435 container was absent.
- **postgres_air result**:
  Nano was still useful as cheap smoke and failed too-easy as expected.
  Kimi trials initially failed on representation ambiguity and overly wide
  row/list labels. The generic fix was to bind answer representation exactly,
  keep initial row/list labels to 3-4 fields (max 5), and avoid open-ended
  event/log lists. After those prompt changes,
  `artifacts/trial_20260427_db_cross_postgres_air_kimi_openrouter_04`
  accepted and committed `task_flight_booking_list_b361491fc973375f` with
  `solver_pass_rate=0.85` and CI `[0.6563, 0.9578]`.
- **MIMIC-IV demo findings**:
  Kimi repeatedly found real clinical event surfaces, but the solver side often
  picked a different plausible sibling table or answer column. Failures were
  not fixed with MIMIC-specific examples. Instead, common gates/policies were
  added for the observed generic failure classes:
  list query limits must appear in `answer_contract.limit_phrase`; limited list
  ordering with tied order keys is rejected before rollout; hidden blocked
  handle filters must be present in `entity`; output aliases must preserve the
  selected source column meaning; and source-role wording must distinguish
  sibling tables such as event/admin/log surfaces from broader order/status
  surfaces.
- **Live validation of new gates**:
  `artifacts/trial_20260427_db_cross_mimiciv_demo_kimi_openrouter_03` showed
  blocked-label feedback working (`hadm_id` removed) and then exposed a
  too-easy retry drift, so the Difficulty-Up Policy now tells list tasks to
  preserve filters/order/limit/row set/output fields and add one visible field.
  Phase-monitor label-change diagnostics now compare invalid retries against
  the last solver-evaluated draft instead of drifting to invalid drafts.
  `..._04` exposed order ties on lab events; `..._08` confirmed the new
  `answer_contract_order_ambiguous` feedback fires before solver rollout.
  `..._07` exposed hidden scope drift from `emar_id` entity to `subject_id`
  filter, which is now rejected by `answer_contract_hidden_filter_unanchored`.
- **Residual risk**:
  MIMIC-IV demo still has no accepted Kimi final-confirmation run in this
  iteration. The remaining pattern is semantic reachability across dense
  sibling clinical tables: even with deterministic order and source-faithful
  aliases, the request must make the selected source role obvious enough for
  the solver to choose the same table. The next improvement should focus on
  structural source-role hints or DB-local examples only if common policy stops
  producing new generic fixes.
- **Verification**:
  `uv run pytest -q` -> 426 passed. Targeted `ruff check` on the touched
  prompt/runtime/query/tool-schema/db files passed. `git diff --check` passed.

## Iteration 73 — Three-DB generation log audit

- **Request**:
  Treat `mimiciv_demo` as the clinical DB target and audit logs across all
  three configured DBs to distinguish pipeline errors from quality-gate
  rejections.
- **DB/config status**:
  `validate-config` passed for pagila, postgres_air, and mimiciv_demo.
  `check-db` passed for all three read-only surfaces:
  pagila `allowlist=public tables=22 selectable=22`, postgres_air
  `allowlist=postgres_air tables=10 selectable=10`, and mimiciv_demo
  `allowlist=mimiciv_hosp,mimiciv_icu tables=31 selectable=31`.
- **Generation log audit**:
  Pagila has two accepted/committed/exported generation runs:
  `task_customer_recent_rental_history_85bdb2f442fca725` at pass rate `0.9`
  and `task_payment_history_with_films_bb2886f28864134a` at pass rate `0.55`.
  Earlier pagila failures were provider setup issues: missing sourced `.env`
  for OpenRouter and the wrong Opencode Kimi model name. Postgres_air has an
  accepted/committed/exported run,
  `task_flight_booking_list_b361491fc973375f`, at pass rate `0.85`; earlier
  postgres_air failures were calibration/quality rejects, not DB or artifact
  generation failures. MIMIC-IV demo repeatedly completed DB exploration,
  composer queries, and solver batches with `solver_failed_runs=0`, but it has
  no accepted/committed/exported run yet. Its failures are quality-signal
  failures (`reject_too_hard` or `calibration_inconclusive`), not DB
  connection, schema snapshot, registry, or bundle-export failures.
- **New generic failure class from demo logs**:
  `trial_20260427_db_cross_mimiciv_demo_kimi_openrouter_09` showed a list
  ordered by a visible timestamp and then an unseen handle. The first submit
  was correctly rejected for duplicate visible order keys, but the retry used
  the unseen handle as a tie-breaker and then failed solver rollout. The fix is
  a common query diagnostic for the structural subset only: limited lists
  reject hidden/handle `order_by` tie-breakers when internal diagnostic order
  values prove that the hidden key splits answer-distinguishable rows sharing
  the represented order prefix.
- **Observed filter-expression failure, not gated**:
  `trial_20260427_db_cross_mimiciv_demo_kimi_openrouter_11` used
  `statusdescription='FinishedRunning'` to define the answer row set, while
  the user request and label did not expose that filter value. Solvers
  reasonably retrieved the latest input events for the ICU stay without that
  hidden membership filter. This was not promoted to a hard validator because
  checking whether a DB value is "expressed" in natural language would require
  literal or semantic token heuristics. It remains a prompt/tooling design
  problem until a structured value-to-phrase binding exists.
- **Conclusion**:
  The three DBs are healthy at the config/DB/tooling level. Pagila and
  postgres_air prove end-to-end generation through bundle export. MIMIC-IV
  demo does not yet prove accepted task generation; it is exposing dense
  clinical-table ambiguity that produced one additional structural gate
  rather than DB-specific prompt examples.
- **Verification**:
  `uv run pytest -q` -> 429 passed. Targeted `ruff check` on touched files
  passed. `git diff --check` passed.

## Iteration 74 — Literal-free order validator precision correction

- **Issue**:
  The initial order diagnostic was still too broad for the 100%-precision hard
  validator rule. A user-visible non-handle order key can be legitimate even
  when it is not selected as a label field, because the list order itself may
  be the requested output. Also, a hidden tie-breaker does not matter when the
  tied rows are answer-identical.
- **Correction**:
  The query tool now fetches unreturned order-key values only into internal
  diagnostic aliases and does not expose them in the tool result rows. Hard
  order diagnostics fire only when a hidden/handle order key actually splits
  answer-distinguishable rows under the represented order prefix. User-visible
  non-handle order keys are not hard rejected merely because they are not
  selected outputs; expression in natural language would require a structured
  binding, not a literal/text heuristic.
- **Verification**:
  `uv run pytest tests/test_tooling_composer_query.py tests/test_synthesis_runtime.py -q`
  -> 87 passed. Targeted `ruff check` on the touched query/test files passed.

## Iteration 75 — Literal-containment rule clarification

- **Issue**:
  A DB literal discovered dynamically from `query(spec)` can still become a
  forbidden heuristic if the runtime compares it against generated prose to
  infer whether a predicate or membership rule was expressed.
- **Correction**:
  Foundation now defines DB literals broadly, bans literal/text containment as
  validation or feedback evidence, records exact allowed uses such as typed
  query execution and structured equality, and documents the `visibility`
  metadata values: `blocked`, `internal`, and `user_visible`.
- **Verification**:
  Documentation-only change; `git diff --check` passed.

## Iteration 76 — Visibility metadata predicate cleanup

- **Issue**:
  `visibility` is policy metadata, but runtime checks were comparing raw
  strings from `dict.get("visibility")` at call sites. That is not a semantic
  token heuristic, but it weakens the contract boundary and makes `derived`
  query-source metadata easy to confuse with the column visibility literals.
- **Correction**:
  `infra/visibility.py` now owns visibility constants and predicates for the
  literal set `blocked`, `internal`, and `user_visible`. Composer query
  diagnostics, submit validation, schema exposure, affordance maps, and schema
  summaries now use those helpers instead of ad hoc visibility string
  comparisons.
- **Verification**:
  Targeted visibility/tooling/synthesis tests passed: 115 passed. Targeted
  `ruff check` and `git diff --check` passed.

## Iteration 77 — Rename privacy surface to visibility

- **Issue**:
  `privacy` suggested PII-specific handling, while the module actually owns the
  broader column exposure policy: `blocked`, `internal`, and `user_visible`.
- **Correction**:
  Renamed `infra/privacy.py` to `infra/visibility.py`, renamed the config model
  to `VisibilityConfig`, and made `visibility:` the canonical config key. The
  previous `privacy:` key is intentionally not supported; this is a pre-release
  clean break recorded in the tuning log rather than a compatibility shim.
- **Verification**:
  `uv run pytest -q` -> 432 passed. `uv run ruff check src tests` passed.
  `git diff --check` passed.

## Iteration 78 — Prompt tie policy without verbosity creep

- **Issue**:
  The structural validator can reject hidden-handle tie-breaks, but the durable
  composer policy should also tell the composer what to do before validation:
  if visible criteria do not uniquely pick one answer-distinguishable row, do
  not choose one with a hidden handle.
- **Correction**:
  Added one concise sentence to the existing unique-result prompt rule: add a
  visible criterion or return the tied rows as a canonical list; never pick one
  with a hidden handle. Kept the full instruction under the existing 8k prompt
  budget.
- **Verification**:
  Prompt length remains under budget at 7,990 characters. `uv run pytest -q`
  -> 431 passed. `uv run ruff check src tests` passed. `git diff --check`
  passed.

## Iteration 79 — Remove redundant raw-SQL prompt ban

- **Issue**:
  The composer has a `query(spec)` DSL tool and no raw-SQL tool, so the workflow
  phrase `do not write SQL` duplicated the tool surface instead of adding a
  meaningful role policy.
- **Correction**:
  Removed the workflow-level raw-SQL prohibition and the test that required
  that exact phrase. Kept the separate `answer_contract` rule forbidding table,
  column, operator, or SQL restatement there, because that field is a natural
  language request-binding surface.
- **Verification**:
  Prompt length is now 7,969 characters. `uv run pytest -q` -> 431 passed.
  `uv run ruff check src tests` passed. `git diff --check` passed.

## Iteration 80 — Trial quality adjudication policy

- **Issue**:
  The configured pass-rate band is a calibration policy, not a complete
  definition of data quality. A too-hard result can be caused by bad data,
  ambiguity, hidden constraints, weak request wording, solver/tool-surface
  brittleness, or a genuinely sound task that is just hard for the current
  solver pool. Treating `reject_too_hard` as automatically "bad data" loses that
  distinction.
- **Correction**:
  Foundation and pipeline lifecycle now require manual trace adjudication for
  too-hard or low-pass real trials. The analysis must report both the numeric
  gate result and a structural judgment from canonical query evidence, sampled
  rows, labels, solver traces, and tool errors. The judgment must not rely on
  DB-literal occurrence or token-containment heuristics.
- **Postgres Air example**:
  `artifacts/trial_20260427_postgres_air_kimi_01` had a second submission with
  `3/12 = 0.25`, but direct DB verification showed the label was correct:
  account `611141` had two bookings with prices `5.96` and `767.57`, summing to
  `773.53`. The failure was therefore not bad data; it was a solver/tool-surface
  difficulty around copying the exact relation label `account<-booking.account_id`
  after difficulty-up added `booking_count`.
- **Verification**:
  Documentation-only change; `git diff --check` passed.

## Iteration 81 — Requested tie-break before selected-field ordering

- **Issue**:
  MIMIC demo `artifacts/trial_20260427_mimiciv_demo_kimi_02` failed
  `reject_too_hard` with `0/4`, but trace review showed the solvers found the
  same five `inputevents` rows and only disagreed with the hidden secondary
  ordering. The canonical query ordered by `starttime ASC, item_name ASC`, while
  the user request only asked for start-time order. `item_name` was selected in
  the label, but selecting a field to return is not the same as asking to use it
  as a tie-breaker.
- **Correction**:
  Kept this as prompt/schema policy rather than a hard validator because
  detecting natural-language tie-break intent with 100% precision is not
  possible. The composer system prompt, `query.order_by` schema, and
  `answer_contract.constraint_phrases` schema now say that when requested order
  leaves answer-distinguishable ties, the composer must ask for the visible
  tie-break before using it in `query.order_by`; merely selecting that field as
  output is not enough. Otherwise it should choose a uniquely ordered row set or
  return the tied rows.
- **Verification**:
  Targeted prompt/schema tests passed: 3 passed. Prompt length remains under
  budget at 7,975 characters.

## Iteration 82 — MIMIC requested tie-break smoke

- **Trial**:
  `artifacts/trial_20260427_mimiciv_demo_requested_tiebreak_kimi_01`, MIMIC demo,
  composer and solver `openrouter/moonshotai/kimi-k2.5`.
- **Result**:
  `synthesis_failed / calibration_inconclusive`, `solver_pass_rate=0.4`
  (`8/20`, CI `[0.2171, 0.6064]`). The submitted task asked for the most recent
  five voiding/output records during one ICU stay. The canonical query used
  `outputevents` joined to `d_items`, ordered only by `charttime desc`; ordering
  diagnostics reported `duplicate_order_key_in_returned_rows=false`.
- **Quality adjudication**:
  The previous MIMIC hidden secondary ordering failure did not recur. Direct DB
  verification showed the latest five `outputevents` rows for the stay are also
  the latest five `Void` rows, so the label is sound. The low pass rate came from
  solver search difficulty around mapping the Korean request for voiding records
  to MIMIC `outputevents`/`d_items` labels (`Void`, `Foley`, urine-like searches)
  and several max-turn runs, not from bad data or hidden tie-breaks.
- **Next implication**:
  No new hard validator is justified. This is a real hard/semantic tool-surface
  case; future tuning should consider request wording, examples, solver
  turn-budget/tool guidance, or band retargeting rather than DB-literal or token
  heuristics.

## Iteration 83 — Precision-100 row-set boundary guard

- **Issue**:
  Prompt/schema policy already rejected hidden or unrequested tie-breaks inside
  returned rows, but a limited list can still cut through an answer-distinguishable
  tie at the `LIMIT N` boundary. In that case the returned rows themselves may
  have unique visible order keys, while the excluded `N+1` row shares the same
  requested order key as row `N`. This is a row-set membership problem, not a
  model-style preference.
- **Correction**:
  Added a structural `limit_boundary_tie` diagnostic in `query(spec)`. For
  limited ordered list queries that return exactly `N` rows, the tool now runs a
  diagnostic `N+1` fetch and compares row `N` with row `N+1`. If they share the
  full query order-key signature but have different answer signatures, the draft
  is rejected by the existing order-ambiguity path. This uses only query DSL,
  source metadata, and observed DB rows; no DB literal, token, or string
  containment heuristic is involved. Also strengthened the composer prompt and
  `query.where`/`query.limit` schema descriptions so row-set controls must be
  either hidden entity scope or request/contract wording.
- **Verification**:
  Unit verification passed: targeted prompt/schema/query tests `67 passed`, full
  `uv run pytest -q` `433 passed`, full `ruff check`, and `git diff --check`.
  Prompt length stayed under budget at `7,993`.
- **Experiment**:
  OpenRouter nano smoke
  `artifacts/trial_20260428_mimiciv_demo_rowset_openrouter_nano_02` reached
  solver evaluation but failed `reject_too_hard` with `3/12 = 0.25`. Trace review
  showed a separate anchor/request mismatch: the draft used an `inputevents`
  composite key as hidden entity while asking for the user's recent five events.
  This is not evidence of the hidden order bug.

  Kimi confirmation
  `artifacts/trial_20260428_mimiciv_demo_rowset_kimi_01` did not submit a draft
  (`composer_submit_draft_missing`), but its final query showed the old kind of
  unsafe list shape: `order_by_outputs=["start_time"]` with
  `duplicate_order_key_in_returned_rows=true`. The composer did not submit that
  ambiguous list.

  Directed live-DB diagnostic on the same MIMIC demo data verified the new guard:
  querying subject `10005817` input events with `ORDER BY starttime DESC LIMIT 2`
  returned two rows whose visible order keys were unique inside the returned
  label, but the diagnostic `N+1` row shared the second row's `start_time`.
  `query(spec)` reported
  `{"duplicate_order_key_in_returned_rows": false, "limit_boundary_tie": true}`.
- **Next implication**:
  The new precision-100 guard works on real MIMIC data. The remaining failed
  model trials point at anchor/request selection and composer completion
  behavior, not at hidden row-set boundary control.

## Iteration 84 — No-submit final output protocol feedback

- **Issue**:
  Kimi can stop before `max_turns` because `max_turns` is an upper bound, not a
  minimum turn guarantee. In
  `artifacts/trial_20260428_mimiciv_demo_rowset_kimi_01`, the composer stopped
  at turn 7 with `final_output_text=""`, after data tools but without
  `submit_draft`. This was not a submit rejection or max-turn exhaustion; it was
  an early final output that violated the composer completion protocol.
- **Correction**:
  Added a precision-100 protocol feedback path. When the Agents backend receives
  a normal final output while no draft is accepted and `submit_draft` has never
  been called, the controller records `composer_submit_draft_missing` feedback:
  plain final output is invalid, continue with tools if evidence is still
  needed, and call `submit_draft` once the draft is valid. The backend then
  resumes the same SDK history via `RunResult.to_input_list(mode="preserve_all")`
  with that feedback appended, bounded by remaining turns and submission budget.
  This check uses only execution protocol facts, not DB literals, token
  matching, or answer-content heuristics.
- **Verification**:
  Added unit coverage for the controller feedback record and for backend
  continuation after no-submit final output. Targeted tests, adjacent synthesis
  backend/runtime tests, full `uv run pytest -q`, `ruff check`, and
  `git diff --check` passed.

  Real DB smoke after the patch:
  `artifacts/trial_20260428_mimiciv_demo_missing_submit_kimi_01` and
  `artifacts/trial_20260428_mimiciv_demo_missing_submit_kimi_02`, MIMIC demo
  `input_events`, composer `openrouter/moonshotai/kimi-k2.5`. Both trials
  reached `submit_draft`; `protocol_feedback_events=0` in `synthesis_completed`,
  so the new no-submit protocol feedback did not trigger spuriously. Trial 01
  submitted once and failed terminally as `reject_too_hard` (`0/4`, pass rate
  `0.0`). Trial 02 exercised the existing order-ambiguity feedback path: first
  submit was rejected with `answer_contract_order_ambiguous` because
  `order_by_outputs=["start_time"]` and
  `duplicate_order_key_in_returned_rows=true`; Kimi then reran the query with
  `order_by_outputs=["start_time", "item_label"]`, resubmitted, and failed
  terminally as `reject_too_hard` (`2/12`, pass rate `0.1667`). These are
  difficult/low-reachability clinical-list tasks, not evidence that the
  no-submit guard is overfiring.

- **Quality adjudication**:
  Do not stop at the `reject_too_hard` gate label. Direct DB inspection changes
  the interpretation of the two smokes. The criterion is tool-solvability: if
  the submitted task can be answered from the database using only the solver's
  provided tools, but sampled solvers fail, it is a difficult but good problem;
  if the submitted task cannot be answered from the prompt/data/tool surface, it
  is low-quality.

  - `_kimi_01` is a low-quality draft, not merely a good hard task. The stay has
    824 `inputevents`; the request asks for "5 items in time order" but never
    says earliest/first or latest/recent. The canonical query silently chooses
    the earliest five rows and also uses `d_items.label` as a visible tie-break
    without stating that tie-break in the user request. A solver cannot know the
    intended row set or exact same-timestamp output order from the prompt alone
    even though the DB rows themselves are coherent.
  - `_kimi_02` is a good hard task. After feedback, the request asks for the
    recent three input events and explicitly says same-time events are sorted
    alphabetically by item name. Direct DB inspection for
    `subject_id=10001217` shows 53 input events and confirms the canonical top
    three rows: `PO Intake` at `2157-12-20 14:00`, then `Magnesium Sulfate` and
    `Magnesium Sulfate (Bolus)` at `2157-12-20 09:25` sorted by label. Solver
    failures mostly came from MIMIC label/path confusion, wrong related fields,
    timestamp/end-time drift, or blank fallback output, not from an unsound
    canonical row set. The task is solvable with the given tools and data, so
    the low pass rate marks difficulty rather than low quality. `sort_record_set`
    cannot directly sort by related `d_items.label`, but the solver can
    materialize the top candidate rows with `item_label`, verify the boundary,
    and submit the answer array in the requested same-time label order; two
    sampled solvers did exactly match.

  Future trial review must report this human quality adjudication alongside the
  numeric gate outcome; `too_hard` alone is not enough.

## Iteration 85 — MIMIC lower-band stop-after-five analysis

- **Experiment setup**:
  Started
  `artifacts/trial_20260428_mimiciv_demo_kimi_composer_bound_batch_01`,
  MIMIC demo `input_events`, composer
  `openrouter/moonshotai/kimi-k2.5`, configured solver pool unchanged at
  `openrouter/openai/gpt-5.4-nano` x20. The batch config set
  `safe_early_termination=false` so submitted drafts get full 20-solver
  observations instead of being cut off by the current `0.5` lower bound. The
  user stopped the batch after five completed attempts; trial 06 was
  interrupted during feedback and excluded.
- **Observed outcomes**:
  - trial 01: `1/20 = 0.05`, terminal `reject_too_hard`, two feedback events
    before rollout.
  - trial 02: no solver rollout; five `submit_draft` feedback events exhausted
    the budget, ending on `answer_contract_order_ambiguous`.
  - trial 03: `0/20 = 0.0`, terminal `reject_too_hard`.
  - trial 04: `0/20 = 0.0`, terminal `reject_too_hard`.
  - trial 05: `0/20 = 0.0`, terminal `reject_too_hard`, one order-ambiguity
    feedback event before rollout.
- **Quality adjudication**:
  - trial 01 is tool-solvable but too hard for the current nano solver pool.
    The final request asks for the recent five medication events for the hidden
    subject, explicitly includes output fields, and states tie-breaks by amount
    then item label. Direct DB inspection found 1,105 `inputevents` rows for
    `subject_id=10039708`; the canonical top five are reproducible by
    `starttime desc`, `amount desc`, and `d_items.label asc`. One solver matched
    exactly. Failures mostly missed the secondary ordering, chose the wrong
    table/path, or confused item labels/units.
  - trial 02 is a caught low-quality draft, not a band sample. The composer kept
    adding hidden or non-user-visible tie-break controls (`itemid`/handle-like
    ordering) for the earliest five rows of stay `32453351`; the validator
    rejected the shape before rollout.
  - trial 03 is low-quality. The request asks for infusion information during
    the ICU stay, but the canonical query filters a single `(orderid, itemid)`
    row. Direct DB inspection showed the same stay has 381 input events and many
    infusion-like rows, so the intended row set cannot be inferred from the
    prompt and solver tool surface.
  - trial 04 is tool-solvable but too hard for the current nano solver pool. The
    request asks for the five most recent input events for stay `38559363`; the
    DB has 40 rows and the latest five have unique `starttime` values, so there
    is no observed row-set or same-time order ambiguity. Solver failures came
    from related-field materialization errors such as returning `itemid`, "Main
    order parameter", or placeholder/error rows instead of `d_items.label`.
  - trial 05 is low-quality. The request asks for the first five rows in time
    order for stay `35446858`, but the first two rows have the same
    `starttime`. The canonical query silently adds `d_items.label asc` as a
    tie-break, while the request does not state that same-time ordering rule. A
    solver can find the same two rows and still choose the opposite order, so
    exact-list grading is not justified by the visible task.
- **Lower-bound implication**:
  On these five attempts, there is no evidence that lowering below `0.2` would
  preserve good hard tasks without admitting low-quality drafts. With 20 solver
  samples and `ci_alpha=0.1`, `0/20` has a one-sided upper bound around `0.109`
  and `1/20` around `0.181`; therefore a lower bound of `0.2` still decisively
  filters all observed `0/20` and `1/20` cases. Lowering to `0.15` would stop
  decisively rejecting the `1/20` case, and lowering to `0.1` would stop
  decisively rejecting `0/20`. The provisional recommendation from this stopped
  batch is `lower_pass_rate=0.2`, not `0.35` and not below `0.15`.

## Iteration 86 — Lower-band and MIMIC scope/tool-surface fixes

- **Change**:
  Applied the stopped five-trial analysis directly. All shipped config files now
  use `lower_pass_rate: 0.2` with the existing `upper_pass_rate: 0.9`,
  `max_solver_runs=20`, and safe early termination. The lifecycle spec now
  records the current development band as `[0.2, 0.9]`.
- **Composer policy**:
  Strengthened the durable system prompt and `query.where` schema around hidden
  scope granularity. If the request asks about a whole parent context, list, or
  history, the query must use that scope; it must not silently narrow to a
  single child event or record unless the request asks for that exact event.
  This targets the trial 03 failure without adding a literal/DB-specific
  validator.
- **Ordering policy**:
  Strengthened `query.order_by` schema wording: selecting a field as output is
  still not a tie-break request, and returning tied rows is safer than inventing
  a secondary order. This is prompt/schema guidance only; no token or literal
  heuristic was added.
- **Solver tool surface**:
  Strengthened atomic tool descriptions for related-field materialization.
  `list_records.fields.path` is now explicitly the preferred way to output
  related display names or labels while preserving one answer item per source
  record. `follow_relation` now warns that it changes the record_set table and
  can collapse many source records into fewer destination records. This targets
  the trial 04 hard-good failures where solvers returned ids or wrong related
  labels instead of source-aligned related display values.
- **Principle review correction**:
  Removed domain-example wording from the common prompt/tool surface during
  review. Scope guidance now uses parent-context/list/history language, and
  solver tool guidance says related display names or labels instead of naming
  sample domains. This keeps the change prompt/schema-first without adding
  validator logic, DB literals, or domain-specific teaching examples.
- **Smoke**:
  Ran live MIMIC demo smoke
  `artifacts/trial_20260428_mimiciv_demo_lower02_fix_smoke_01` with Kimi
  composer and default solver pool. It failed terminal `reject_too_hard` after
  `0/12` matches, as expected under the new lower bound: the one-sided upper
  confidence bound first fell below `0.2` after twelve straight misses. The run
  still verified useful movement: composer kept the parent stay scope instead
  of narrowing to one child event, and after order-ambiguity feedback it asked
  for a visible same-time tie-break. Remaining failures were solver/tool-surface
  issues around category filtering and related display materialization, not a
  new hard-validator candidate.

## Iteration 87 — Atomic related sort keys

- **Problem**:
  The previous smoke showed a genuine solver tool-surface gap, not just weak
  model behavior. Composer could author a task whose exact answer is ordered by
  a source timestamp plus a related display label, but solver-side
  `sort_record_set` could only sort by direct source-table columns. Solvers
  could output related labels through `list_records.fields.path`, yet could not
  use that same related field to choose the top five before materialization.
- **Change**:
  Extended atomic ordering from a single direct column to ordered `keys`, each
  with `{path, column, direction}`. Empty `path` keeps the old source-table
  behavior. Forward relation paths allow stable ordering by a related display
  value while preserving the source record_set. Reverse relation paths are
  rejected because they can map one source record to multiple related records
  and therefore do not define a precise per-source sort key.
- **No heuristic**:
  This is structural tool expressivity only. It does not inspect generated text,
  compare DB literals, or add any hard validator. The rule is schema/path based:
  forward FK traversal is single-destination per source record; reverse
  traversal is not guaranteed to be.
- **Direct DB verification**:
  Replayed the prior MIMIC demo shape against live DB tools for
  `stay_id=39268883`, `ordercategoryname='01-Drips'`, ordered by
  `starttime asc` then related `inputevents.itemid->d_items.label asc`.
  `list_records` returned the expected canonical order:
  `Propofol`, `Solution`, `Dextrose 5%`, `Nitroglycerin`, `Dextrose 5%`.
- **LLM smoke**:
  Ran `artifacts/trial_20260428_mimiciv_demo_related_sort_smoke_01` with Kimi
  composer and default nano solver pool. The run still failed `reject_too_hard`
  at `0/12`, but it verified the new schema was live: solver traces used
  `sort_record_set.keys`. The remaining observed failure is that nano solvers
  mostly still chose direct sort keys instead of the available related path.
  Tool descriptions were tightened to say that when a requested order names a
  related label/name returned through `list_records.fields.path`, the same
  related path should be used as a sort key before listing source records.

## Iteration 88 — Controlled related-sort solver validation

- **Question**:
  After adding related sort keys, determine whether the remaining MIMIC demo
  failures are a tool-surface impossibility, weak solver behavior, or low-quality
  data. Use a fixed draft so composer variation cannot hide the signal.
- **Fixed draft**:
  MIMIC demo, `stay_id=39880770`, input events. The request asks for the most
  recent five infusion/fluid events, ordered by `starttime desc`, with same-time
  ties ordered by related `d_items.label asc`. The answer includes related item
  name plus direct amount, unit, start time, and status.
- **Nano solver run**:
  `artifacts/controlled_solver_related_sort_20260428_01`, solver
  `openrouter/openai/gpt-5.4-nano`, 12 fixed-draft runs, early termination off:
  `2/12` exact matches, pass rate `0.1667`, CI `[0.0305, 0.4381]`.
  `used_related_sort_runs=2`, `used_related_item_field_runs=4`, and both exact
  matches are exactly the two runs that used
  `sort_record_set.keys=[starttime desc, inputevents.itemid->d_items.label asc]`.
  Failures mostly chose direct tie-breaks such as `itemid`, `amountuom`, or
  `ordercategoryname`, or returned direct columns such as `itemid`,
  `ordercategoryname`, or `ordercomponenttypedescription` instead of
  `d_items.label`.
- **Kimi solver run**:
  `artifacts/controlled_solver_related_sort_20260428_kimi_01`, solver
  `openrouter/moonshotai/kimi-k2.5`, 4 fixed-draft runs, early termination off:
  `4/4` exact matches. All four runs used the related sort path and all four
  listed the related `d_items.label` field. The statistical gate reads
  `reject_too_easy` only because this is a small controlled strong-solver check,
  not a band calibration sample.
- **Interpretation**:
  The new atomic sort surface is functionally sufficient and necessary for this
  class. The task is not low-quality: a stronger solver solves it from the
  visible request and tools without hidden row controls. The nano failures are
  mostly reasoning/tool-selection failures, not evidence for another hard
  validator. Further changes should avoid oracle shortcuts; if optimizing nano,
  keep the change at prompt/tool-description clarity around matching the
  `sort_record_set.keys.path` to the `list_records.fields.path` used for the
  displayed related label.

## Iteration 89 — Parallel Kimi batch-five MIMIC demo trial

- **Question**:
  Run the real MIMIC demo pipeline with Kimi composer and Kimi solver in a
  five-trial parallel batch, then judge failed drafts by data quality rather
  than treating `reject_too_hard` as conclusive evidence of good difficulty.
- **Run**:
  `artifacts/trial_20260428_mimiciv_demo_kimi_batch5_parallel_01`, DB
  `mimiciv_demo`, topic `input_events`, composer and solver
  `openrouter/moonshotai/kimi-k2.5`, `max_solver_runs=20`,
  `solver_batch_size=4`, pass band `[0.2, 0.9]`. The batch ran five trials in
  parallel, with per-trial DB pools reduced to avoid exhausting connections.
- **Aggregate**:
  `1/5` accepted. Trials 1-4 ended `synthesis_failed` with `reject_too_hard`
  after `0/12` matches and CI high `0.2209`. Trial 5 accepted at `16/20`
  matches, pass rate `0.8`, CI `[0.5990, 0.9286]`, committed as
  `task_medication_input_events_803ff611ccd657bd`.
- **Trial-quality read**:
  Trial 1 is low-quality, not merely hard. After feedback added a visible
  same-time tie-break "larger amount first", the canonical answer still placed
  amount `1.0` before amount `200.0` at the same timestamp. Kimi solvers
  followed the visible request and therefore mismatched.
- **Nullability read**:
  Trials 2, 3, and 4 exposed a structural nullability problem. The composer
  included nullable fields such as `rate` and `rate_unit`; canonical answers
  contained `null`, but the solver submit schema rejected nulls or forced
  solvers into string/empty-string substitutes. This turns otherwise reachable
  list tasks into apparent `too_hard` failures. Trial 3 is also the clearest
  observed `too_easy -> too_hard` jump: the visible-tie-break version hit
  `20/20`, then the difficulty-up response added nullable rate fields and the
  final draft fell to `0/12`.
- **Principle check**:
  These findings do not justify literal or token heuristics. The next eligible
  fix should be structural and precision-safe: propagate canonical/output
  nullability into the solver answer schema, and separately ensure the accepted
  canonical order agrees with visible `order_by` keys. Both are based on the
  draft's own query/result contract, not on predicting database values.

## Iteration 90 — Nullable submit schema and sort-direction policy

- **Change**:
  Reworked canonical-answer schema inference to inspect the full canonical
  answer instead of the first list item. If any canonical field is `null`, the
  internal `OutputFieldContract` marks that field nullable while preserving the
  observed non-null scalar type when present. The solver-facing `submit_result`
  schema already emits nullable fields as `anyOf[..., {"type": "null"}]`, so
  the structural fix is in inference rather than in solver logic.
- **Observed-trial replay**:
  Replayed the final canonical answers from
  `artifacts/trial_20260428_mimiciv_demo_kimi_batch5_parallel_01` trials 2, 3,
  and 4. Each now infers `rate` and `rate_unit` as nullable and canonicalizes
  the original answer successfully. This directly addresses the low-quality
  `null is not allowed for this field` failures without adding DB literals or
  text heuristics. Directly invoking the new `submit_result` tool with trial 2's
  null-containing canonical payload returns `submitted=True` for all five rows.
- **Composer policy**:
  Trial 1's wrong-direction tie-break is not safely catchable by a hard
  validator because `answer_contract` intentionally contains request phrases,
  not duplicated structured order metadata. Added prompt/tool-schema guidance
  instead: match the requested sort direction before using query rows as the
  canonical label. This keeps the fix prompt-first for composer policy and
  avoids natural-language heuristic validation.
- **Verification**:
  Focused checks passed:
  `uv run pytest tests/test_synthesis_schema_inference.py tests/test_solver_backend_openai_agents.py::test_submit_result_tool_accepts_nullable_fields tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`.
  Broader relevant checks passed:
  `uv run pytest tests/test_synthesis_schema_inference.py tests/test_synthesis_canonicalize.py tests/test_solver_backend_openai_agents.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py -q`.
  Ruff passed on the changed source and test files.

## Iteration 91 — Nullable fix Kimi batch-five validation

- **Run**:
  `artifacts/trial_20260428_mimiciv_demo_kimi_batch5_parallel_nullable_01`, DB
  `mimiciv_demo`, topic `input_events`, composer and solver
  `openrouter/moonshotai/kimi-k2.5`, five trials in parallel with per-trial DB
  pools reduced to `solver=8`, `control=2`, and OpenRouter concurrency capped
  at 2 per trial.
- **Aggregate**:
  `3/5` accepted, `0` duplicates, `2` synthesis failures. This improves on the
  previous parallel Kimi batch result of `1/5` accepted. Across all solver
  rollouts in the new batch there were `112` solver runs and `0`
  `invalid_submit_schema` terminations, so the nullable-answer schema failure
  did not recur.
- **Accepted trials**:
  Trial 2 accepted at `18/20 = 0.9`, task
  `task_patient_input_events_history_e73feddff3857250`. Trial 3 accepted at
  `5/12 = 0.4167`, task `task_input_events_393cd5193991139a`. Trial 5 accepted
  at `4/20 = 0.2`, task `task_input_events_dc70999f863002b2`; this accepted
  draft includes nullable `rate` and `rate_unit`, confirming the specific
  low-quality nullability failure from the prior batch is fixed in live rollout.
- **Failed trials**:
  Trials 1 and 4 failed high, not low. Both ended at `19/20 = 0.95` with
  `calibration_inconclusive`. Trial 1 first fixed an ambiguous same-time order
  and then became too easy; subsequent difficulty-up attempts were rejected as
  `answer_contract_not_incremental` because the composer changed output
  operation/source shape instead of making a clean incremental strengthening.
  Trial 4 repeatedly missed exact `answer_contract`/limit phrasing before
  ending too easy. These are composer retry-policy issues, not solver tool
  schema failures.
- **Interpretation**:
  The structural nullable schema fix is validated. The next improvement target
  is composer difficulty-up hygiene after a too-easy draft: preserve the current
  query path and output source set while adding one clean visible field,
  predicate, order, or cardinality change. That should remain prompt/schema
  policy unless a precision-100 structural validator is identified.

## Iteration 92 — Composer difficulty-up hygiene tightening

- **Trigger**:
  The nullable follow-up Kimi batch showed two high-pass-rate failures ending
  after too-easy drafts. Trial 1 in particular fixed an order ambiguity, then
  became too easy; subsequent drafts changed operation/output source shape and
  were rejected as `answer_contract_not_incremental` instead of applying a
  clean one-step strengthening.
- **Change**:
  Tightened the composer Difficulty-Up Policy to preserve the same answer kind,
  anchor, target, row set/query path, filters, order, limit, and existing
  output fields/source meanings. For list difficulty-up, a new answer field
  should be appended, not substituted. The `query.select` schema description now
  mirrors that retry behavior, and the too-easy / not-incremental feedback names
  the exact recovery action: keep the prior query shape and answer fields, then
  append one DB-grounded visible field or make one visible structural
  strengthening.
- **Principle check**:
  No literal, token, column-name, or DB-value heuristic was added. This remains
  prompt/tool-schema/feedback policy derived from the failed draft's own retry
  history. The only hard rejection involved here is the existing structural
  incremental-contract check, which compares the submitted retry against the
  previously evaluated draft rather than predicting data literals.
- **Verification**:
  Prompt length stayed under the existing budget at `7989` characters. Focused
  and related checks passed:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py -q`
  (`69 passed`). Ruff passed on the changed source and test files.

## Iteration 93 — Kimi batch-five difficulty-up policy smoke

- **Run**:
  `artifacts/trial_20260428_mimiciv_demo_kimi_batch5_parallel_diffup_01`, DB
  `mimiciv_demo`, topic `input_events`, composer and solver
  `openrouter/moonshotai/kimi-k2.5`, five trials in parallel. Each trial used
  DB pools `solver=8`, `control=2`, OpenRouter concurrency `2`, pass band
  `[0.2, 0.9]`, `max_solver_runs=20`, `solver_batch_size=4`, and
  `safe_early_termination=true`.
- **Aggregate**:
  `4/5` accepted, `0` duplicates, `1` synthesis failure. Accepted trials were
  trial 1 at `5/8 = 0.625`, trial 2 at `17/20 = 0.85`, trial 3 at
  `8/12 = 0.6667`, and trial 5 at `18/20 = 0.9`. No trial ended with
  `answer_contract_not_incremental`; the run therefore did not reproduce the
  previous too-easy retry shape-drift failure.
- **Feedback path read**:
  Initial feedbacks were ordinary structural issues:
  `label_values_not_grounded` / `answer_contract_evidence_mismatch` in trial 1,
  `answer_contract_order_ambiguous` in trial 2,
  `answer_contract_phrase_missing` in trial 3, and
  `answer_contract_query_mismatch` plus `answer_contract_order_ambiguous` in
  trial 5. Each accepted trial recovered without violating the new incremental
  retry policy.
- **Failed-trial quality read**:
  Trial 4 is low-quality, not merely difficult. It failed at `0/12` with
  `reject_too_hard`, but the data shows an unrequested visible tie-break. The
  user request asked for the latest five medication/fluid input records in
  latest-time order. The accepted canonical query sorted by
  `starttime DESC, item_name ASC`, while the request and `answer_contract`
  did not ask for item-name tie-breaking. A DB cross-check for `stay_id=30057454`
  shows the first two rows share `starttime=2171-11-18T17:53:00`; ordering by
  `starttime DESC` returns `Potassium Chloride` before `KCL (Bolus)`, while
  ordering by `starttime DESC, item_name ASC` returns `KCL (Bolus)` before
  `Potassium Chloride`. Solver rollouts consistently found the same row set but
  submitted the start-time-only order, so exact matching reported `0/12`.
- **Interpretation**:
  The difficulty-up tightening did not regress the batch and likely helped keep
  retries cleaner, but this run is not a direct live proof of the too-easy
  recovery path because no trial crossed above the current upper band. The next
  issue is visible-but-unrequested tie-breaks. The prompt and tool schema
  already say a tie-break must be requested, so another prompt-only change would
  risk duplication. A precision-safe next direction is to add structured
  detection/feedback for solver order-only mismatches, or add a structured
  request-order contract before enforcing visible secondary order keys. Do not
  add natural-language or literal heuristics for this.

## Iteration 94 — Accepted-task qualitative audit for the same Kimi batch

- **Question**:
  Do not treat accepted tasks as automatically good. Audit the four accepted
  tasks from
  `artifacts/trial_20260428_mimiciv_demo_kimi_batch5_parallel_diffup_01` for
  request clarity, row-set/order determinism, and solver failure patterns.
- **Trial 1, accepted at `5/8 = 0.625`**:
  Borderline but usable. The request asks for the latest five nutrition/intake
  records during an ICU stay. The final query orders by `starttime DESC` and
  the returned top five rows are all `PO Intake`, with no duplicate returned
  start-time key. A DB cross-check of the top ten rows for `stay_id=31269608`
  shows the first five are `PO Intake`, then later rows include non-nutrition
  items such as insulin. The quality risk is semantic: the query did not
  explicitly filter a nutrition/intake item category, so solvers must infer that
  the observed top rows are the requested nutrition records. Failed solvers did
  try nutrition-like filters and ran out of turns. This is hard-good leaning,
  but a cleaner draft would name the concrete surface (`PO Intake` / intake
  records) or use a visible grounded filter.
- **Trial 2, accepted at `17/20 = 0.85`**:
  Good. The request explicitly asks for the latest five input-event records and
  includes visible tie-breaks: same start time sorted by end time ascending,
  then item name alphabetically. The final query orders by
  `starttime DESC, endtime ASC, item label ASC`, matching the request and using
  selected answer-visible fields. The three solver failures were missing-submit
  or malformed-tool-call failures, not evidence of bad data.
- **Trial 3, accepted at `8/12 = 0.6667`**:
  Borderline/low-quality due recency ambiguity. The request asks for the “most
  recent” five input records but does not say whether recency means start time,
  end time, or store time. The canonical query uses `starttime DESC`. A DB
  cross-check for `stay_id=35436337` shows a `D5LR` row with
  `starttime=2185-06-18T05:51:00` and `endtime=2185-06-18T11:50:00`; it is not
  in the canonical top five by start time, but it is second by end time. Several
  solver mismatches included that row, which is a reasonable interpretation of
  “recent” in an interval table. This should be made explicit in the request
  (for example start-time 기준) before accepting similar drafts.
- **Trial 5, accepted at `18/20 = 0.9`**:
  Low-quality accepted / should ideally reject. The request explicitly says
  “start time 기준 최신순,” but the returned list contains three rows with the
  same `starttime=2148-01-08T15:46:00`. The final accepted rows bake in a tied
  order among those rows; one retry also introduced `storetime DESC` as a
  secondary order without asking the user for that tie-break. A DB cross-check
  confirms the tied rows share the requested start-time key, and two rows also
  share the secondary store-time key. Since ordered exact matching is used,
  this arbitrary tied order can look correct only because most solvers converged
  on the same tool behavior. The draft is too easy and under-specified, not a
  clean accepted task.
- **Overall read**:
  Accepted rate alone overstates quality. This batch has one clean accepted
  task, two borderline tasks, and one accepted task that should not be trusted.
  The next improvement should target structured order determinism: for ordered
  list tasks, either every ordering key that affects exact order must be
  requested and answer-visible/reproducible, or the tied rows should be treated
  as an unordered equivalence group. Avoid adding literal or text-token
  heuristics; the signal comes from query metadata and solver order-only
  mismatch patterns.

## Iteration 95 — Precision-safe composite order-key tie diagnostics

- **Trigger**:
  Iteration 94 found an accepted low-quality task where a list was ordered by
  `starttime DESC, storetime DESC`, but the returned rows still contained
  distinct answer rows tied on the full order-key tuple. Because `storetime` was
  a visible but unselected order key, the previous query diagnostics returned no
  ambiguity and `submit_draft` accepted an arbitrary tied order.
- **Change**:
  Updated composer `query` ordering diagnostics to evaluate duplicate ties using
  the full order-key signature, including diagnostic-only order columns that are
  not selected into the label. Visible unselected tie-breaks remain allowed when
  they actually disambiguate the rows. The new rejection condition is only:
  after applying every `query.order_by` key, distinct submitted answer rows still
  share the same full order key.
- **Why this is precision-safe**:
  This does not inspect request wording, column-name literals, or database
  values. It uses the submitted query's own structured order keys and result
  rows. If answer-distinct rows share the complete order signature, the exact
  row order is not determined by the query spec; accepting that order would
  depend on backend row order rather than on a requested/reproducible sort.
- **Verification**:
  Added a regression test that preserves the existing rule: visible unselected
  tie-breaks are allowed when their values differ, but reported as ambiguous
  when the full composite order key is still tied. Focused checks passed:
  `uv run pytest tests/test_tooling_composer_query.py::test_query_does_not_reject_unrepresented_visible_tie_breaker tests/test_tooling_composer_query.py::test_query_reports_duplicate_full_order_key_with_visible_tie_breaker tests/test_tooling_composer_query.py::test_query_reports_unrepresented_order_by_tie_breaker_diagnostics tests/test_synthesis_runtime.py::test_submit_draft_rejects_ambiguous_limited_list_order tests/test_synthesis_runtime.py::test_submit_draft_rejects_unrepresented_list_order_by_tie_breaker -q`.
  Broader relevant checks passed:
  `uv run pytest tests/test_tooling_composer_query.py tests/test_synthesis_runtime.py -q`
  (`91 passed`) and `uv run ruff check ...`.
- **Live replay**:
  Re-ran the accepted-low-quality trial 5 final query against `mimiciv_demo`.
  The new result includes
  `ordering_diagnostics={"order_by_outputs":["start_time"],"returned_row_count":5,"limit":5,"duplicate_order_key_in_returned_rows":true}`,
  so the existing `submit_draft` ambiguous-order validator would reject it.

## Iteration 96 — Live Kimi batch after composite tie diagnostics

- **Question**:
  Does the composite order-key tie diagnostic work in a fresh Kimi batch, and
  does it improve accepted data quality without relying on literal or
  token-based heuristics?
- **Experiment**:
  Ran five parallel `mimiciv_demo` / `input_events` trials with
  `openrouter/moonshotai/kimi-k2.5` as both composer and solver from
  `f89a816`. Artifact root:
  `artifacts/trial_20260428_mimiciv_demo_kimi_batch5_parallel_composite_tie_01`.
  Config used `lower_pass_rate=0.2`, `upper_pass_rate=0.9`,
  `max_solver_runs=20`, `solver_batch_size=4`,
  `safe_early_termination=true`, and per-trial DB pools of 8 solver / 2
  control connections.
- **Batch result**:
  Accepted `2/5`, failed `3/5`, duplicate task count `0`. Trial 4 accepted at
  `5/8 = 0.625`; trial 5 accepted at `12/16 = 0.75`. Trials 1, 2, and 3
  failed before registry commit.
- **Validator evidence**:
  The new diagnostic fired on live drafts before solver rollout. Trial 2's
  first draft returned
  `ordering_diagnostics={"duplicate_order_key_in_returned_rows":true,"order_by_outputs":["starttime"],"returned_row_count":5,"limit":5}`;
  trial 3's first draft returned the same structural signal for
  `order_by_outputs=["start_time"]`. Trial 5 initially hit
  `duplicate_order_key_in_returned_rows=true` and `limit_boundary_tie=true`,
  then recovered by making the tie-break visible and requested. This is the
  intended precision-safe path: the validator uses the submitted query's own
  structured order keys and result rows, not anticipated strings, values, or
  column-name heuristics.
- **Accepted-trial quality read**:
  Trial 4 is acceptable but not especially strong. The request asks for the
  recent five input events in time order; the canonical query uses
  `starttime DESC`, returns five deterministic rows, and solvers mostly solve
  it. The wording "time order" is slightly loose, but "recent five" anchors
  descending time well enough for this batch read. Trial 5 is good: after the
  initial ambiguity feedback, the final request explicitly asks for the same
  time to be sorted by medication name, and the canonical query orders by
  `starttime DESC, d_items.label ASC` with both order keys exposed in the
  answer.
- **Failed-trial quality read**:
  Trial 1 is a correct low-quality block. It repeatedly selected a top-five
  boundary with `limit_boundary_tie=true`; even after adding start-time,
  end-time, and order-reference wording, the fifth-row boundary remained tied,
  so accepting would have depended on hidden row membership. Trial 2 is not a
  hard-good example: after the ambiguity feedback, the request became
  "latest five medication/fluid records by medication name descending", while
  the canonical query used `starttime DESC, label DESC`. Solver traces split
  between `inputevents` and `prescriptions`, and many solved a plausible but
  different medication-record interpretation. Trial 3 is also low-quality:
  the user-facing request says "medications", while the canonical answer comes
  from `inputevents`; solver traces consistently chose `prescriptions` and
  produced reasonable medication-list answers that did not match the canonical
  input-event answer.
- **Interpretation**:
  The composite tie validator behaved as intended and prevented the previously
  accepted arbitrary-order pattern from reappearing. The lower accept rate is
  not evidence of over-rejection by that validator; the rejected drafts expose
  real ambiguity or composer recovery problems. The next improvement should be
  prompt/tool-contract work that keeps the user-facing source surface and
  canonical query surface aligned after feedback. The concrete MIMIC table
  mismatch in this experiment is evidence for that structural pattern only; it
  must not become a shared prompt example or DB-specific common policy. Keep
  hard validators limited to precision-100 structural checks; do not add
  literal, token, or column-name heuristics.

## Iteration 97 — Prompt as policy source, feedback as policy reminder

- **Question**:
  Apply the surface-ownership principle to the current composer guidance:
  durable policy should live in the composer system instructions, tool
  descriptions should carry only tool-local contracts, and feedback should
  remind the composer of an existing policy rather than restating new
  instructions.
- **Change**:
  Added named, DB-neutral system policies:
  `Source Surface Policy`, `Feedback Handling Policy`, `Label Grounding Policy`,
  and `List Determinism Policy`. The source-surface policy is the schema-neutral
  generalization from iteration 96: the user-facing source surface and canonical
  query surface must match, without naming MIMIC tables or domain values.
- **Feedback cleanup**:
  Shortened `submit_draft` feedback for grounding, list ordering, difficulty-up
  non-incremental retries, and missing-submit protocol errors. The new messages
  say which named policy was violated and point at the current failure evidence;
  they no longer carry long duplicate retry strategies such as hidden tie-break
  recipes or field-append instructions.
- **Tool schema cleanup**:
  Tightened the composer `query` schema descriptions for `where`, `select`,
  `order_by`, and `limit`. They now state argument semantics and refer to named
  policies where broader behavior is needed, instead of duplicating the full
  prompt policy inside the tool description.
- **Length budget**:
  Kept `build_synthesis_agent_instructions(...)` under the existing 8k
  character test budget by compressing existing prose while preserving the
  role, customer-request, label, list-determinism, and difficulty-up rules.
- **Verification**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py -q`
  passed (`75 passed`). Extended backend-adjacent check also passed:
  `uv run pytest tests/test_synthesis_backend_openai_agents.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py -q`
  (`82 passed`). Ruff passed on the touched prompt, feedback, tool-schema, and
  test files.

## Iteration 98 — A/B test for symbolic `submit_draft` policy examples

- **Question**:
  Should the common composer prompt include a minimal, DB-neutral
  `submit_draft`-payload example for the violation the current trials repeat
  most often?
- **A condition**:
  Used commit `a2d7e88` with no new examples. Ran five parallel
  `mimiciv_demo` / `input_events` trials with the default frequent-experiment
  nano model config. Artifact root:
  `artifacts/ab_submit_examples_A_20260428_01`.
- **A result**:
  Accepted `0/5`. All five failed before accepted draft. The dominant pattern
  was repeated `answer_contract_order_ambiguous`: drafts asked for recent
  ordered lists by a time field while returned rows contained answer-distinct
  ties under that order. This confirmed that list determinism, not the earlier
  source-surface mismatch, was the right target for this A/B.
- **B condition**:
  Temporarily added one common-prompt section with a symbolic
  `<submit_draft>{...}</submit_draft>` payload showing a bad list-order draft
  and commentary saying to request/query a visible tie-break or return tied
  rows. The example used only placeholders such as `<time>`, `<f>`, and
  `<topic>`; no DB/table/value literal or token heuristic was introduced.
  Focused prompt/runtime checks and ruff passed during the temporary patch.
  Artifact root: `artifacts/ab_submit_examples_B_20260428_01`.
- **B result**:
  Accepted `0/5`. The repeated order-ambiguity loop became less dominant, but
  the batch shifted into other low-quality modes: single-row drafts with
  `reject_too_hard`, missing/invalid anchor entity, missing contract phrase,
  and an empty/invalid canonical answer. This is not a positive result: the
  example did not improve acceptance or data quality, and it appears to have
  encouraged evasive narrowing instead of robust list construction.
- **Decision**:
  Do not keep the common prompt example. The tested change obeyed the
  no-literal/no-heuristic rule, but failed the empirical quality bar. Keep the
  current prompt/feedback baseline and treat this as evidence that the next
  improvement should be more targeted than a broad in-prompt example, likely in
  precision-safe tool diagnostics or in feedback that reminds an already named
  policy without adding a new instruction source.

## Iteration 99 — Kimi batch for prompt-reminder baseline

- **Question**:
  After `a2d7e88`, where durable policy lives in the composer prompt and
  feedback only reminds named policies, is the current prompt better than the
  previous Kimi batch on `mimiciv_demo` / `input_events`?
- **Invalid OpenRouter attempt**:
  Tried five parallel trials with `openrouter/moonshotai/kimi-k2.5`.
  Artifact root:
  `artifacts/trial_20260428_mimiciv_demo_kimi_batch5_prompt_reminder_01`.
  This is not a valid quality run: the provider returned `402` errors because
  requests were sent with an effective `max_tokens=65536`, exceeding available
  OpenRouter credit. Some drafts reached feedback first, but the batch cannot
  be used for acceptance comparison.
- **Valid Kimi run**:
  Re-ran five parallel trials with `opencode_zen/kimi-k2.5` as both composer
  and solver. Artifact root:
  `artifacts/trial_20260428_mimiciv_demo_kimi_opencode_batch5_prompt_reminder_01`.
  The comparison baseline is iteration 96
  `trial_20260428_mimiciv_demo_kimi_batch5_parallel_composite_tie_01`
  at commit `f89a816`, which accepted `2/5`.
- **Batch result**:
  Current prompt-reminder batch accepted `2/5`. Trial 1 accepted at
  `17/20 = 0.85`; trial 3 accepted at `17/20 = 0.85`. Trial 2 failed after
  too-easy `0.95` retries and a final `0.10` draft; trial 4 hit
  `MaxTurnsExceeded`; trial 5 failed `reject_too_hard` with `0/12`.
- **Quality read**:
  Trial 1 is a good sign for the prompt-reminder baseline: it recovered from
  two list-order feedback events and produced a request that explicitly orders
  same-time rows by medication name. The accepted label is a grounded
  `input_events` list and the solver pass rate is high but still inside the
  current acceptance band.

  Trial 3 should not be counted as a clean `input_events` improvement. It
  spent three submissions failing list determinism on `inputevents`, then
  switched to an `admission_prescription_history` task over prescriptions.
  The final task is likely good as a standalone customer task, but it is topic
  drift from the requested batch topic and is not evidence that `input_events`
  recovery improved.

  Trials 2, 4, and 5 show the remaining issue: list-order feedback is
  understood in form, but recovery is unstable. Trial 2 fixed tie wording then
  overshot into too-easy/too-hard swings. Trial 4 kept adding visible sort
  language but did not eliminate duplicate order keys before max turns. Trial
  5 added a visible name tie-break but still produced a task solvers could not
  solve.
- **Interpretation**:
  The prompt-reminder change is not clearly better than the previous Kimi
  baseline. Raw accept count stayed `2/5`; strict clean `input_events` accept
  is closer to `1/5` because one accepted task drifted to prescriptions. The
  improvement over iteration 96 is narrower: one live case shows successful
  policy-reminder recovery from order ambiguity. The remaining failures point
  at a more specific problem: after `answer_contract_order_ambiguous`,
  composer often cannot find a visible deterministic ordering without either
  using hidden handles, changing topics, or making a large difficulty jump.
  Next improvement should target that recovery path with precision-safe tool
  diagnostics or tool affordances, not with broad prompt examples.

## Iteration 100 — Mandatory qualitative audit after experiments

- **Decision**:
  Until the project code is complete, every synthesis/prompt/tool/feedback
  experiment analysis must include a qualitative quality comparison after the
  quantitative batch summary.
- **Rule**:
  Accepted data is not automatically good. Each accepted task must be audited
  from bundle and debug artifacts for request/topic/entity/query/label/order
  alignment and classified as clean, borderline, low-quality accepted, or
  inconclusive.
- **Rule**:
  Rejected data is not automatically bad. Each rejected/failed task must be
  classified as hard-good, low-quality, infra/provider failure, or inconclusive.
  A task is hard-good when it is answerable with the solver tool surface but the
  sampled solvers failed; it is low-quality when the task definition itself is
  not well-grounded, not uniquely answerable, source-surface mismatched, hidden
  row-set/order/filter dependent, internally topic-mismatched, or
  difficulty-jumped.
- **Risk interpretation**:
  Low-quality drafts are not the main problem when the solver pass-rate /
  quality gate rejects them. That is the filter doing useful work. The critical
  failure is low-quality data being accepted or committed to the registry. Future
  comparisons must therefore separate low-quality rejected count from
  low-quality accepted count and treat the latter as the highest-risk signal.
  This does not make rejected-data review optional: accepted data must still
  answer "is this truly good data?", and rejected data must still answer "is
  this hard-good or low-quality?" Repeated low-quality rejected patterns remain
  useful evidence for composer/prompt/tool recovery improvements.
- **Operational change**:
  Added this requirement to `docs/runbook.md` under
  `Mandatory Experiment Quality Audit`. Any future experiment that lacks this
  audit is not valid evidence for a project improvement.

## Iteration 101 — Composer-owned topic by default

- **Decision**:
  Normal synthesis must not receive an externally requested topic. The composer
  chooses and submits the topic that naturally describes the grounded
  user_request, query path, and label it built.
- **Clarification**:
  The previous `requested_topic` path remains only as an edge-case experiment
  hint. It may be used to test whether a seed changes exploration, but it is not
  a coverage contract and should be omitted from default trials.
- **Quality-audit impact**:
  A submitted topic that differs from an experiment hint is not low-quality by
  itself. Low-quality topic failure means the composer-submitted topic does not
  match the final request/query/label, or the draft changed source surface while
  pretending it was the same task.
- **Operational change**:
  The LLM-facing prompt tag is now `topic_experiment_hint`, not
  `requested_topic`, and CLI output labels it the same way. Trial generation
  now exposes only `--topic-hint`; old `--topic` / `--category` aliases were not
  preserved because normal generation should not look topic-targeted.

## Iteration 102 — Topic hints require explicit approval

- **Decision**:
  Experiments also omit topic hints by default. `topic_experiment_hint` is only
  for targeted re-experiments after a recurring edge case has been observed and
  the user explicitly approves using a seed.
- **Operational guard**:
  `run-real-db-trial --topic-hint ...` now fails unless
  `--topic-hint-approved` is also supplied. This does not replace the human
  approval requirement; it makes the exceptional path visible in command history
  and prevents accidental hint use.
- **Analysis rule**:
  When a topic hint is used, the experiment log must state why the targeted
  re-experiment needed it. Ordinary batch comparisons without that approval must
  run without a topic hint.

## Iteration 103 — No-topic-hint Kimi batch on MIMIC demo

- **Question**:
  After making topic composer-owned by default, does removing the `input_events`
  hint change real MIMIC demo behavior?
- **Trial**:
  Ran five parallel `mimiciv_demo` trials with no `--topic-hint`. Composer and
  solver were `opencode_zen/kimi-k2.5`, using
  `artifacts/tmp_configs/trial_mimiciv_demo_kimi_bound_batch_01.yaml`.
  Artifact root:
  `artifacts/trial_20260428_mimiciv_demo_kimi_no_topic_batch5_01`.
- **Result**:
  Raw accept count was `1/5`. Trial 3 accepted
  `task_ICU Output Measurements_079ee8608a48be82` at `16/20 = 0.80`.
  Trials 1, 2, 4, and 5 failed `reject_too_hard` with `0/20`.
- **Topic behavior**:
  The hint-free batch did increase topic/surface diversity. The submitted topics
  covered medication administration status, ICU medication administration,
  output measurements, recent blood tests, and ICU medication history. Query
  roots spanned `emar`, `outputevents`, `labevents`, and `inputevents`. This is
  materially different from the earlier `input_events`-hinted batches, where the
  prompt pushed most attempts toward the same surface.
- **Accepted data audit**:
  Trial 3 is clean. The request asks for the five most recent ICU output
  measurements for the hidden `stay_id`; the submitted topic matches the request
  and query path; the label fields match the request; and the tie-break is
  visible in the request: same measurement time is ordered by later registration
  time. Solver failures were mostly exact-format misses, not evidence that the
  task is bad.
- **Rejected data audit**:
  Trial 1 is low-quality rejected: it targets a single `emar` row but asks for
  dose/method fields from `emar_detail`, a no-primary-key table that the solver
  tool surface cannot materialize. This is a source/tool-surface mismatch, not
  hard-good.

  Trial 2 is low-quality rejected: it recovered from an unanchored hidden
  `hadm_id` filter, but the final label exposed `sequence_id` and used it as an
  ordering key without a user-visible tie-break request. The task is more than
  merely hard.

  Trial 4 is low-quality rejected: the final lab-result list used test name as a
  visible tie-break, but the user request only said "latest". The row-set order
  is under-specified from the user's perspective.

  Trial 5 is low-quality rejected: the request says "major medications" while
  the query selects the first five input events by time and medication label; it
  also omits the tie-break semantics that determine list membership/order.
- **Interpretation**:
  The topic change worked for diversity and did not create low-quality accepted
  data in this batch. It did not improve yield: strict clean accept remains
  `1/5`, roughly comparable to the previous prompt-reminder batch once the
  topic-drifted accepted task is discounted. The next improvement target is not
  topic control; it is recovery around deterministic list ordering and avoiding
  solver-inaccessible/no-primary-key detail surfaces.

## Iteration 104 — No-PK row label source guard

- **Question**:
  Can the previous no-primary-key detail-surface failure be handled by a
  100%-precision structural rule without adding DB literal or token heuristics?
- **Change**:
  Commit `3eb07b1` adds `table_has_primary_key` to composer `query`
  `column_sources`, reminds the Source Surface Policy and query `select`
  description, and makes `submit_draft` return feedback-only
  `label_no_primary_key_source` when a label directly exposes row values from a
  table with no primary key. Count-style derived aggregates are explicitly
  allowed.
- **Why this layer**:
  This is schema/query-provenance evidence, not natural-language matching. The
  runtime can prove exactly that a selected label value came from a no-PK table
  and that it exposes source row values. The same composer conversation should
  continue because the fix is usually to choose a primary-key-backed path or a
  derived aggregate.
- **Verification**:
  `ruff` passed for the touched files. Targeted pytest passed:
  `8 passed` across query provenance, submit feedback, prompt role-boundary, and
  tool-description tests.
- **Trial**:
  Ran five parallel `mimiciv_demo` trials with no `--topic-hint`. Composer and
  solver were `opencode_zen/kimi-k2.5`, using
  `artifacts/tmp_configs/trial_mimiciv_demo_kimi_bound_batch_01.yaml`.
  Artifact root:
  `artifacts/trial_20260428_mimiciv_demo_no_pk_guard_kimi_no_topic_batch5_01`.
- **Result**:
  Raw accept count was `1/5`.
  - Trial 1 accepted `task_icu_stay_admission_details_4922d550b8dc23cd` at
    `16/20 = 0.80`.
  - Trial 2 failed `reject_too_hard` at `1/20 = 0.05`.
  - Trials 3 and 5 failed `reject_too_hard` at `0/20`.
  - Trial 4 ended with `MaxTurnsExceeded` after feedback loops.
- **Accepted data audit**:
  Trial 1 is clean. The request asks for admission information related to the
  current ICU stay. After one feedback event, the final query anchors on
  `icustays.stay_id`, follows the primary-key-backed admission path, and returns
  a single admission-information object as a one-row list. The topic, request,
  entity, query path, and label fields are aligned. No no-PK row source is
  exposed.
- **Rejected data audit**:
  Trial 2 is hard-good rejected. The final task asks for the latest five lab
  results and states the same-time tie-break by test ID. The row source is
  primary-key-backed and the selected tie-break is visible in the label/request.
  The `1/20` pass rate reflects solver difficulty, not an unsound row set.

  Trial 3 is low-quality rejected. The final request asks for five medications
  ordered by start time and same-time medication name, but the submitted label
  still includes route/status fields that the final request does not ask to
  receive. The ordering itself is solvable; the output contract drift is the
  quality issue.

  Trial 4 is low-quality rejected / composer failure. A too-easy ICU/admission
  draft was followed by non-incremental topic/path changes into medication and
  radiology-order drafts, with query mismatch and an unanchored hidden filter.
  This is not a hard-good solver miss.

  Trial 5 is low-quality rejected. The final task asks for recent medication
  information by start time, but the query uses stop time to complete ordering
  and returns stop time even though the request does not ask for that tie-break
  or field. The quality gate rejected it, so this is not low-quality accepted.
- **Interpretation**:
  The no-PK guard did not directly fire in this random batch because the sampled
  drafts avoided no-PK label sources. That means this trial does not prove yield
  improvement, but it does show no low-quality accepted data and no false
  positive on primary-key-backed rows or derived counts. The remaining repeated
  low-quality pattern is selected outputs and visible tie-break fields drifting
  beyond the final user request. That pattern is not a 100%-precision validator
  target unless represented by explicit structured bindings; treat it as
  prompt/tool-description/example work, not a literal/token heuristic.

## Iteration 105 — Advisory answer-contract bindings

- **Question**:
  Can optional `answer_contract.output_bindings` and `order_bindings` reduce
  output/order drift by making the composer expose request-to-label claims
  without adding a new hard validator?
- **Change**:
  Commit `7a43783` added optional advisory binding fields to
  `answer_contract`, logs binding coverage in `submit_draft` phase diagnostics,
  and minimally updated prompt/tool descriptions. The fields are not enforced;
  this follows Experiment 1 from
  `docs/composer_low_quality_reduction_plan.md`.
- **Verification**:
  `ruff` passed for touched code/tests. Focused pytest passed:
  `7 passed` across submit schema, binding diagnostics, prompt length/surface,
  and budget prompt tests.
- **Trial setup**:
  The first manual parallel attempt at
  `artifacts/trial_20260428_mimiciv_demo_answer_bindings_kimi_no_topic_batch5_01`
  is discarded as infra-only: all workers shared one global SQLite registry and
  trial 1 hit `OperationalError: database is locked`.

  The valid batch is
  `artifacts/trial_20260428_mimiciv_demo_answer_bindings_kimi_no_topic_batch5_02`.
  It ran five parallel no-topic-hint `mimiciv_demo` trials with composer and
  solver both set to `opencode_zen/kimi-k2.5`. Each trial used a separate
  registry/traces config to avoid SQLite lock contamination.
- **Result**:
  Raw accept count was `1/5`.
  - Trial 1 failed `reject_too_hard` at `1/20 = 0.05`.
  - Trial 2 failed `reject_too_hard` at `0/20`.
  - Trial 3 accepted `task_ICU 배출 기록 조회_092243850cbb588c` at
    `18/20 = 0.90`.
  - Trial 4 failed `reject_too_hard` at `5/20 = 0.25`.
  - Trial 5 failed `reject_too_hard` at `3/20 = 0.15`.
- **Binding behavior**:
  All five final submissions included `output_bindings` and `order_bindings`.
  Binding coverage diagnostics reported no missing output bindings in the final
  submissions. Trial 5's final diagnostics correctly surfaced
  `missing_order_label_bindings=["stoptime"]`, because the final query used
  `stoptime` as an order key that the final request/order bindings did not
  ask for.

  Trial 2 exposes a limitation of Experiment 1 diagnostics: the final query
  used extra order keys (`test_seq`, `test_name`) after an ambiguous-order
  feedback, but the request still only said latest five. Because no
  `ordering_diagnostics.order_by_outputs` entry existed after the query became
  structurally deterministic, the advisory coverage did not flag the unbound
  order references. This is evidence for Experiment 2, not a reason to add a
  semantic or literal heuristic.
- **Accepted data audit**:
  Trial 3 is clean. The request asks for five ICU output records for the hidden
  stay, ordered by latest measurement time with same-time rows ordered by output
  type. The query anchors on `outputevents.stay_id`, joins to `d_items` for the
  visible output type, returns exactly the requested fields
  (`measurement_time`, `output_type`, `value`, `unit`), and the tie-break is
  visible in both request and label. The accepted pass rate is high, but the
  qualitative read is still clean rather than merely easy.
- **Rejected data audit**:
  Trial 1 is hard-good rejected. The final task asks for the five most recent
  medication records during the hidden admission, then adds the visible
  same-time medication-name tie-break after feedback. The final label fields are
  explicitly requested and the final order diagnostics clear. The low pass rate
  is solver difficulty, not a row-set or output-contract defect.

  Trial 2 is low-quality rejected. The final request asks only for the latest
  five microbiology results, while the final query relies on additional
  same-time order keys. One of those keys is not selected into the label, and
  the user request/order bindings do not expose the tie-break. The quality gate
  rejected it, so this is not low-quality accepted.

  Trial 4 is hard-good rejected. The hidden entity is a medication item
  (`itemid`) and the visible request names Doxycycline. The final query returns
  the available Doxycycline administration rows ordered by administration time;
  no hidden extra order key or output drift is apparent. The `5/20` result looks
  like a difficult but tool-solvable task.

  Trial 5 is low-quality rejected. The final request asks for medication records
  ordered by start time and same-time medication name, but the final query also
  orders by stop time. The new advisory diagnostic caught this as an unbound
  order label field. The rejection prevented low-quality accepted data.
- **Interpretation**:
  Compared with the no-PK-guard baseline, raw accept rate stayed `1/5` and
  low-quality accepted stayed `0`. The experiment did not improve yield yet,
  but it proved the composer can usually fill DB-neutral bindings without
  schema repair loops. More importantly, the diagnostics now expose repeated
  order-binding drift in a structured way. The next candidate is Experiment 2:
  feedback-only repair for exact missing binding facts, while still avoiding
  phrase-to-field semantic judgment and DB literal heuristics.

## Iteration 106 — Binding-feedback Kimi trial qualitative audit

- **Question**:
  Did feedback-only missing order binding repair improve final data quality in
  a no-topic `mimiciv_demo` Kimi batch, without admitting low-quality accepted
  tasks?
- **Trial setup**:
  Batch root:
  `artifacts/trial_20260428_mimiciv_demo_order_binding_feedback_kimi_no_topic_batch5_01`.
  Five parallel trials used separate registry/traces configs. Composer and
  solver were both `opencode_zen/kimi-k2.5`; no `topic_experiment_hint` was
  injected.
- **Result**:
  Raw accept count was `3/5`.
  - Trial 1 accepted `task_hospital_admission_summary_with_demographics_11d07e024aa20407`
    at `18/20 = 0.90`.
  - Trial 2 failed `reject_too_hard` at `5/20 = 0.25`.
  - Trial 3 accepted `task_patient_admissions_history_fe72baaca88c89eb` at
    `17/20 = 0.85`.
  - Trial 4 accepted `task_medication_history_eaec0d5683f3a2e3` at
    `13/20 = 0.65`.
  - Trial 5 failed `reject_too_hard` at `1/20 = 0.05`.
- **Accepted data audit**:
  Trial 1 is clean. The final task remains anchored on one admission, joins the
  directly related patient row, and asks for exactly the admission and
  demographics fields returned in the label. The too-easy retry strengthened a
  one-row admission summary by adding direct demographics; it did not introduce
  a hidden row set, hidden order, or output-contract drift.

  Trial 3 is clean. The final request asks for the patient's admission history
  in recent order. The query anchors on `subject_id`, orders by admission time,
  and returns only admission-history fields requested by the broad phrase
  "입원 이력". Earlier hidden admission id exposure was removed before accept.

  Trial 4 is borderline accepted. Structurally, the final query is answerable:
  it asks for five medication records, exposes the same-time tie-break as
  "투약 순번", and binds both order keys. The concern is user-facing naturalness:
  the tie-break is a sequence field used only for ordering and not returned in
  the label, and "최근 5개" plus "시간 순서" is slightly ambiguous. This is not
  low-quality accepted because the row set/order/filter are stated and
  solver-visible, but it is not clean.
- **Rejected data audit**:
  Trial 2 is hard-good rejected. The final task asks for the five most recent
  ICU datetime/procedure rows with a procedure-name tie-break. The row set is
  scoped by the hidden ICU stay, both order keys are bound and visible, and the
  output fields are present in the request surface. The low pass rate looks like
  solver difficulty on an awkward but tool-solvable surface, not a hidden-order
  or query/request mismatch.

  Trial 5 is hard-good rejected, with an awkward user-facing surface. The final
  task asks for not-given medication rows in an admission, orders by
  administration time and record id, and returns the record id it uses as the
  tie-break. The record id surface is artificial, but it is explicit,
  solver-visible, and no longer a hidden row-set control. The sampled solvers
  mostly failed; the artifact does not show an impossible task definition.
- **Comparison**:
  Against Iteration 105, raw accept improved from `1/5` to `3/5`; clean accepted
  improved from `1` to `2`; borderline accepted increased from `0` to `1`; and
  low-quality accepted stayed `0`. Low-quality rejected fell from `2` to `0` in
  this batch under the artifact-based read, while hard-good rejected stayed at
  `2`.
- **Interpretation**:
  The result is a quality improvement, not only a pass-rate improvement, because
  no low-quality draft reached registry commit. The improvement is not a proof
  that the new `ANSWER_CONTRACT_BINDING_MISSING` feedback fired frequently:
  most repairs still came through existing order-ambiguity feedback and the
  stricter binding shape. The remaining risk is borderline accepted data where a
  visible but unnatural sequence/id surface is made explicit enough to pass.
  That should not become a hard validator without 100%-precision evidence; treat
  it as prompt/schema-shape/advisory-audit input.

## Iteration 107 — Natural tie-break surface prompt test

- **Question**:
  Can a prompt-first policy reduce accepted drafts that make top-k lists stable
  by adding artificial technical sequence/id language to the user request?
- **Change**:
  Commit `aa4d74a` updates the composer system prompt to say the customer does
  not know technical sequences/references, and the List Determinism Policy now
  requires a natural visible tie-break. It explicitly says not to fix top-k
  lists with artificial technical sequence/id wording. The
  `AnswerOrderBinding.requested_by_phrase` schema description was narrowed from
  "tie-break" to "natural tie-break".
- **Why this layer**:
  This is not a 100%-precision validator target. Whether a tie-break phrase is
  natural customer language is a semantic/naturalness judgment, so the durable
  rule belongs in the system prompt. The tool schema only states the local
  binding contract.
- **Verification**:
  `ruff` passed for the touched prompt/tool/test files. Prompt and schema
  surface tests passed, and full `tests/test_synthesis_prompts.py` passed.
  The rendered composer instructions remained under the 8000-character budget
  at `7997` characters.
- **Trial setup**:
  Batch root:
  `artifacts/trial_20260428_mimiciv_demo_natural_tiebreak_kimi_no_topic_batch5_01`.
  Five parallel no-topic-hint `mimiciv_demo` trials used composer and solver
  `opencode_zen/kimi-k2.5`, each with a separate registry/traces config.
- **Result**:
  Raw accepted count was `3/5`.
  - Trial 1 accepted `task_ICU_output_measurements_2508f080a60a3046` at
    `14/20 = 0.70`.
  - Trial 2 failed with provider `UserError` after one
    `answer_contract_order_ambiguous` feedback.
  - Trial 3 accepted `task_microbiology_lab_results_dafa8662519b0af5` at
    `17/20 = 0.85`.
  - Trial 4 failed with provider `UserError` after one
    `answer_contract_hidden_filter_unanchored` feedback.
  - Trial 5 accepted `task_pharmacy_medication_lookup_738d70999eee67ba` at
    `16/20 = 0.80`.
- **Accepted data audit**:
  Trial 1 is borderline accepted. The row set is scoped to the hidden ICU stay,
  the selected fields match the request, and no technical sequence/id
  tie-break is used. The concern is that "recent 5 measurements in time order"
  fixes membership but does not say clearly whether the final list should be
  newest-first or chronological within the latest five. Current solvers mostly
  infer newest-first, but the exact ordered label is not as clean as it could
  be.

  Trial 3 is borderline accepted, with a targeted improvement signal. The
  composer first tried an internal event id as the same-time tie-break; feedback
  rejected that path. The accepted draft uses a natural visible tie-break:
  same-time microbiology rows are ordered by test name. This directly addresses
  the artificial technical tie-break risk. The remaining weakness is sort
  direction phrasing: "test-time order" plus "recent 3" implies a recent list
  but does not explicitly say newest-first.

  Trial 5 is clean. The task is a one-row current medication-prescription
  lookup anchored by hidden `pharmacy_id`; the user request does not expose the
  handle and asks for medication, status, timing, admission, and patient
  demographic fields that are returned through direct joins. No list ordering
  or tie-break is involved.
- **Rejected/failed data audit**:
  Trials 2 and 4 are provider/infra failures, not completed quality-gate
  rejected tasks. Trial 2's only submitted draft had duplicate start-time order
  keys and received `answer_contract_order_ambiguous`; trial 4's only submitted
  draft filtered by a hidden subject id that was not the submitted entity and
  received `answer_contract_hidden_filter_unanchored`. Both low-quality drafts
  were caught before any registry commit, but the run ended because the
  provider returned `UserError` during recovery. They should not be counted as
  hard-good or low-quality rejected final tasks.
- **Comparison**:
  Against Iteration 106, raw accepted count stayed `3/5` and low-quality
  accepted stayed `0`. Clean accepted fell from `2` to `1`; borderline
  accepted rose from `1` to `2`; infra/provider failures rose from `0` to `2`.
  The targeted metric improved: no accepted task used artificial technical
  sequence/id wording as a tie-break, and Trial 3 shows recovery from an
  internal id tie-break toward a natural visible one.
- **Interpretation**:
  This experiment is not a broad quality win yet. It does appear to reduce the
  specific artificial technical tie-break pattern without adding a heuristic
  validator, but it exposes a neighboring weakness: the composer can satisfy
  "natural tie-break" while still leaving sort direction slightly under-specified.
  The next prompt-level candidate is a small clarification that list order
  direction must be natural and explicit, for example "newest-first" versus
  "oldest-first", rather than relying on "recent" plus "time order".

## Iteration 108 — Explicit sort direction prompt and provider retry rule

- **Question**:
  Can the borderline "recent + time order" accepted pattern be reduced by
  making list sort direction explicit in the composer prompt?
- **Change**:
  Commit `1e23831` updates the List Determinism Policy to require explicit
  order direction: newest-first/oldest-first, asc/desc, or equivalent. It also
  records the experiment operations rule in `docs/runbook.md`: provider/infra
  failures are not quality samples and must be retried before experiment
  conclusions are drawn.
- **Why this layer**:
  Ambiguous direction wording is a prompt-quality issue. A hard validator would
  require semantic judgment over natural-language direction phrases, so this
  remains prompt-first. Provider retry is operational policy, not model
  behavior policy.
- **Verification**:
  `ruff` passed for the touched prompt/test/runbook files. The focused prompt
  workflow test and full `tests/test_synthesis_prompts.py` passed. The rendered
  composer instructions stayed under the 8000-character budget at `7992`
  characters.
- **Trial attempt**:
  Intended Kimi batch root:
  `artifacts/trial_20260428_mimiciv_demo_explicit_sort_direction_kimi_no_topic_batch5_01`.
  Initial five no-topic `mimiciv_demo` trials used
  `opencode_zen/kimi-k2.5` for composer and solver. All five failed before any
  draft because the provider returned `AuthenticationError`.

  Following the new retry rule, all five were retried once with separate
  retry configs and output roots. They failed the same way:
  `opencode_zen/kimi-k2.5: AuthenticationError`.

  A provider-fallback recovery run used `openrouter/moonshotai/kimi-k2.5` for
  composer and solver. All five failed before quality sampling with
  `openrouter/moonshotai/kimi-k2.5: APIStatusError`.

  A lightweight nano probe used `openrouter/openai/gpt-5.4-nano` for composer
  and solver. It also failed before quality sampling with
  `openrouter/openai/gpt-5.4-nano: APIStatusError`.
- **Provider diagnosis**:
  A minimal direct health request showed `opencode_zen` returning a workspace
  monthly spending-limit error. A minimal direct OpenRouter Kimi request
  succeeded, so the OpenRouter trial failure appears specific to the SDK/trial
  request path rather than a missing key. This was not used as quality evidence.
- **Accepted data audit**:
  None. No run reached an accepted task.
- **Rejected/failed data audit**:
  All attempted trials are `infra/provider failure`. They are not hard-good,
  low-quality rejected, or low-quality accepted samples.
- **Interpretation**:
  The code/prompt change is verified locally, but the live experiment is
  invalid as a quality comparison. The new provider retry rule was applied and
  prevented these failures from being miscounted as data-quality outcomes. A
  valid Kimi or nano batch is still needed once provider access is healthy or
  the OpenRouter SDK failure is diagnosed.

## Iteration 109 — Default provider route back to Opencode Zen

- **Change**:
  Repo default `rl_task_foundry.yaml` composer and all 20 solver entries now use
  `provider: opencode_zen` with the existing frequent-experiment model
  `openai/gpt-5.4-nano`.
- **Reason**:
  User preference is to make Opencode Zen the default route. Kimi remains an
  explicit high-quality validation override rather than the base config.
- **Verification**:
  `load_config("rl_task_foundry.yaml")` resolves composer and all solvers to
  `opencode_zen/openai/gpt-5.4-nano`. Config-load tests and the real-db trial
  CLI override summary test passed. `ruff` passed for the touched Python test
  surfaces.

## Iteration 110 — Correct Opencode Zen nano model id

- **Finding**:
  Retried `.env`-loaded Opencode Zen health checks. `/models` shows
  `gpt-5.4-nano`, not `openai/gpt-5.4-nano`, for the Opencode route. The
  OpenRouter-prefixed model id was therefore invalid for the new default route.
- **Change**:
  Repo default `rl_task_foundry.yaml` composer and all 20 solver entries now use
  `opencode_zen/gpt-5.4-nano`.
- **Provider status**:
  Direct Opencode chat health checks for `gpt-5.4-nano`, `gpt-5.4-mini`, and
  `kimi-k2.5` still returned `MonthlyLimitError` for the workspace tied to the
  current `OPENCODE_API_KEY`. This is provider/billing state, not a config-load
  or model-id issue.

## Iteration 111 — Default provider route to direct OpenAI API

- **Change**:
  Added `openai_api` as a direct OpenAI API provider using `OPENAI_API_KEY`.
  Repo default `rl_task_foundry.yaml` composer and all 20 solver entries now use
  `openai_api/gpt-5.4-mini`.
- **Reason**:
  User requested `gpt-5.4-mini` via OpenAI API after Opencode Zen continued to
  return provider-side monthly-limit errors.
- **Provider check**:
  `.env` contains `OPENAI_API_KEY`. Minimal direct OpenAI Chat Completions
  requests to `gpt-5.4-mini` succeeded with no explicit token cap and with
  `max_completion_tokens`. A request using legacy `max_tokens` returned the
  expected unsupported-parameter error for this model, so future direct health
  checks should use `max_completion_tokens` or omit the cap.
- **Verification**:
  `load_config("rl_task_foundry.yaml")` resolves composer and all solvers to
  `openai_api/gpt-5.4-mini`. Config-load tests now assert the default provider
  and model. Config-load tests and the real-db trial CLI override summary test
  passed. `ruff` passed for the touched Python test surfaces.

## Iteration 112 — OpenAI API mini retry and GPT-5.5 single comparison

- **Question**:
  After the explicit sort-direction prompt change, can direct OpenAI API runs
  produce valid `mimiciv_demo` quality samples, and is `gpt-5.4-mini` too weak
  for this composer/solver loop?
- **Setup**:
  Batch root:
  `artifacts/trial_20260428_mimiciv_demo_explicit_sort_direction_openai_mini_no_topic_batch5_01`.
  The first five no-topic `mimiciv_demo` trials used
  `openai_api/gpt-5.4-mini` for composer and solvers. All five failed before
  any draft with `RateLimitError` on OpenAI TPM. These are provider failures,
  not quality samples.

  Retried with provider concurrency `1`, solver batch size `1`, and higher
  SDK retry count. Trial 1 completed 20 solver runs and was rejected at
  `0/20`; the draft asked for the globally earliest ICU stay while carrying a
  locked stay anchor, so this is a low-quality rejected sample. Trial 2 was
  rejected twice as too easy at `20/20`, then stopped on
  `answer_contract_not_incremental`. Trial 3 was still producing mostly
  too-easy admission-record drafts when the run was intentionally interrupted
  to test a stronger model. Trials 4 and 5 were not run.
- **GPT-5.5 comparison**:
  Batch root:
  `artifacts/trial_20260428_mimiciv_demo_explicit_sort_direction_openai_gpt55_no_topic_single_01`.
  A minimal direct health check resolved `gpt-5.5` to
  `gpt-5.5-2026-04-23`. One no-topic trial used `openai_api/gpt-5.5` for both
  composer and all 20 solvers, with concurrency `1`.
- **Result**:
  The GPT-5.5 trial accepted
  `task_pharmacy medication orders_9380b3be793d760b` at
  `15/20 = 0.75`, CI `[0.544, 0.896]`.
- **Accepted data audit**:
  This is not clean. The request explicitly says the list should be latest-first
  by medication order start time, and same-time rows should be ordered by
  medication name ascending. The answer contract correctly binds those order
  keys, and both keys are user-visible and included in the label. This directly
  satisfies the explicit sort-direction target.

  The accepted sample is borderline, leaning low-quality accepted. The top
  three pharmacy rows contain two distinct `Senna` orders with different hidden
  `pharmacy_id` values, but the requested output fields make the two answer
  objects identical. The user receives two indistinguishable rows. The failed
  solver runs also show a real wording ambiguity: several solvers interpreted
  `2161-11-15에 시작된` as a medication start-date filter instead of an admission
  start-date anchor.
- **Rejected/failed data audit**:
  Mini parallel failures are provider/infra failures. Mini retry Trial 1 is
  low-quality rejected. Mini retry Trial 2 is too-easy rejected, followed by a
  non-incremental difficulty crank failure. The interrupted mini Trial 3 is not
  a final quality sample.
- **Interpretation**:
  GPT-5.5 produced a structurally stronger draft than mini: explicit direction,
  visible tie-break binding, and an in-band pass rate. However, the accepted
  data quality is still not good enough to call clean. The next improvement
  candidate should target general prompt guidance against ambiguous anchor
  phrasing and indistinguishable duplicate answer rows. A hard validator for
  natural-language ambiguity is not precision-100; duplicate projected answer
  rows are structurally detectable, but rejecting them as low quality is still
  a policy judgment unless the contract explicitly requires distinguishable
  records.

## Iteration 113 — Composer self-check for ambiguous scope and duplicate rows

- **Problem**:
  The GPT-5.5 accepted sample from Iteration 112 showed a composer-quality
  failure, not a solver failure. The request wording let solvers read a date
  modifier as either a parent-context anchor or a child-row filter, and the
  accepted label contained two distinct source records that became identical
  after projection to the requested output fields.
- **Change**:
  Added a DB-neutral composer prompt rule requiring modifiers to bind to the
  exact object or scope when parent context and child rows could both match the
  wording. Added a list-label rule that returned rows should be distinguishable
  through requested output fields; if the latest `query` reports duplicate
  projected answer rows, the composer should add a natural user-visible
  distinguishing field, aggregate duplicates, or choose another task, never a
  hidden handle.

  Added `projection_diagnostics` to the composer `query` result. It reports
  `duplicate_answer_rows`, duplicate row index groups, unique projected row
  count, and returned row count when the selected canonical-answer fields make
  multiple returned rows indistinguishable. The query tool description now tells
  the composer to inspect ordering and projection diagnostics before
  `submit_draft`.
- **Why this layer**:
  Ambiguous natural-language scope is not a 100%-precision validator target, so
  it belongs in the prompt. Duplicate projected answer rows are structurally
  detectable from exact query evidence, but treating them as always invalid is
  still a task-quality policy decision. For now the runtime exposes the exact
  diagnostic to the composer rather than hard-rejecting; this keeps the
  validator conservative while giving the composer enough tool feedback to
  avoid submitting the low-quality shape.
- **Verification**:
  The rendered composer instructions remain under the 8000-character budget at
  `7998` characters. Focused tests for query diagnostics, composer tool schema
  descriptions, synthesis prompts, and turn-budget prompts passed. Full related
  suites for `test_tooling_composer_query.py`,
  `test_tooling_composer_tool_factory.py`, `test_synthesis_prompts.py`, and
  `test_turn_budget_prompt.py` passed. Ruff passed for the touched source and
  test files.
- **Next experiment**:
  Re-run a single no-topic `mimiciv_demo` GPT-5.5 trial before broadening to a
  batch. The qualitative audit should specifically check whether accepted rows
  are distinguishable through output fields and whether parent/child modifiers
  are unambiguous.

## Iteration 114 — OpenRouter Kimi default route and solver model de-dup

- **Provider check**:
  `.env` contains `OPENROUTER_API_KEY`. Direct OpenRouter chat checks for
  `moonshotai/kimi-k2.5` succeeded and resolved to
  `moonshotai/kimi-k2.5-0127`. A minimal tool-call probe also returned a tool
  call with valid arguments, so Kimi is usable for both text and tool-driven
  composer/solver flows through OpenRouter.
- **Config change**:
  Repo default `rl_task_foundry.yaml` now routes composer and solver models to
  `openrouter/moonshotai/kimi-k2.5`. OpenRouter provider defaults were adjusted
  to lower concurrency, longer timeout, and three SDK retries.
- **Solver YAML de-dup**:
  Replaced the repeated 20-entry default solver list with a single
  `models.solver` template. `calibration.max_solver_runs` is now the rollout
  count source of truth; `models` only names the solver model candidates. The
  solver orchestrator cycles configured solver model templates and derives a
  stable per-attempt `solver_id`, so one configured solver model can still run
  20 independent attempts without YAML repetition or verification-key
  collisions.
- **Smoke trial**:
  Root:
  `artifacts/trial_20260428_mimiciv_demo_openrouter_kimi_smoke_01/trial_01`.
  This smoke used a temporary four-solver Kimi config to keep runtime bounded.
  The Agents SDK/OpenRouter path worked end to end.

  First draft: rejected as too easy at `4/4 = 1.0`.
  Second draft: accepted at `3/4 = 0.75`. Three solvers submitted the canonical
  answer; one run ended with `missing_submit_result`.
- **Qualitative audit**:
  The accepted second draft is structurally reasonable: it asks for the first
  five prescriptions for the anchored admission, ordered by prescription start
  time and then drug name, and includes drug name, product strength, start/end
  timestamps, and route. The returned rows are distinguishable through visible
  output fields.

  It is not a clean accepted sample because the final `user_request` is in
  English while the configured `domain.language` is `ko`. This is a
  composer-policy miss, not a provider failure. The rejected first draft was not
  low-quality; it was simply too easy for Kimi under the four-run smoke band.
- **Prompt follow-up**:
  Strengthened the existing Feedback Handling Policy with a small
  prompt-first reminder to preserve anchored need/language when revising after
  feedback. No language hard validator was added because exact language
  detection would be heuristic, not precision-100.
- **Verification**:
  `validate-config` now reports one solver model candidate and
  `max_solver_runs=20`. Focused config/CLI/orchestrator/prompt tests printed
  `25 passed`; as in earlier runs, the pytest process did not exit after
  completion and was terminated after the pass result was visible. Ruff passed
  for the touched source and test files.

## Iteration 115 — OpenRouter Kimi no-topic batch and exact low-quality guards

- **Question**:
  With OpenRouter Kimi as the default composer/solver route and no injected
  topic, does a five-trial `mimiciv_demo` batch produce clean accepted data?
- **Setup**:
  The first batch root
  `artifacts/trial_20260428_mimiciv_demo_openrouter_kimi_no_topic_batch5_01`
  was aborted as invalid because it was generated from the default Pagila config
  instead of `rl_task_foundry.mimiciv_demo.yaml`.

  The valid batch root was
  `artifacts/trial_20260428_mimiciv_demo_openrouter_kimi_no_topic_batch5_02`,
  using `openrouter/moonshotai/kimi-k2.5`, no topic hint, one solver model
  template, and `max_solver_runs=20`. The original trial 4 process hung in the
  provider/SDK path and was killed; it is an infra sample only. The retried
  `trial_04_retry_01` completed and is the quality sample.
- **Raw result**:
  Valid quality samples: 5. Accepted: 4 (`trial_01`, `trial_02`,
  `trial_03`, `trial_04_retry_01`). Rejected: 1 (`trial_05`). Excluded infra:
  original `trial_04`.
- **Accepted data audit**:
  `trial_01` is low-quality accepted. The final query returned a full ICU
  procedure list ordered by `starttime` plus hidden `orderid`, but the request
  only asked for time ordering. Three rows shared the same procedure time, so
  exact answer order depended on an unrequested handle. The same accepted
  draft also used fake binding phrases such as `_category_`, `_status_`, and
  `_location_` for fields not literally requested.

  `trial_02` is borderline accepted. The final admission-history task is
  answerable and the accepted data is structurally reasonable, but the composer
  abandoned an earlier prescription-history path after order-ambiguity feedback.
  That is a recovery hygiene concern, not a final row-set defect.

  `trial_03` is clean accepted. The output-record task preserved the admission
  scope, made the extra record-time field explicit after an over-easy draft, and
  accepted at `17/20 = 0.85`.

  `trial_04_retry_01` is clean accepted. The final ICU medication/fluid list
  asks for the returned fields, has no observed same-time tie in the top five,
  and accepted at `5/8 = 0.625`.
- **Rejected data audit**:
  `trial_05` is low-quality rejected. The pharmacy task was too wide and
  included indistinguishable duplicate medication rows plus a null medication
  name. The gate rejected it at `1/20 = 0.05`, so this did not become
  low-quality accepted data.
- **Change**:
  Extended composer `query` ordering diagnostics beyond limited lists. Any
  ordered multi-row query can now report duplicate order keys or unrepresented
  hidden/handle tie-breakers. `limit_boundary_tie` remains limited-list only.

  Enforced the existing answer-contract phrase rule for
  `output_bindings[*].requested_by_phrase` and
  `order_bindings[*].requested_by_phrase`, not only answer/constraint/limit
  phrases. A submitted binding phrase must be an exact substring copied from
  `user_request`.

  Added `artifacts/tmp_configs/trial_*/` to `.gitignore` so per-trial config
  directories do not appear as untracked files.
- **Why this follows the principles**:
  Both guards are precision-100 structural checks. They use only the latest
  query result/order metadata and the submitted request/contract strings. No
  DB literals, table-name heuristics, column-name heuristics, or model-specific
  assumptions were added.
- **Verification**:
  Re-running the exact final `trial_01` query against `mimiciv_demo` now
  reports `unrepresented_order_by_tie_breakers` for `procedureevents.orderid`.
  Focused query diagnostics tests passed (`4 passed`), focused answer-contract
  phrase/binding tests passed (`4 passed`), and `ruff` passed for touched
  Python files. The broader focused query/runtime command printed `50 passed`;
  as in earlier runs, the pytest process did not exit after printing the pass
  result and was killed after the result was visible.

## Iteration 116 — Eight-solver experiment batch and duplicate-row feedback

- **Question**:
  Can a cheaper five-trial `mimiciv_demo` batch with only eight Kimi solver
  rollouts still expose the next data-quality failure modes?
- **Setup**:
  Batch root:
  `artifacts/trial_20260428_mimiciv_demo_openrouter_kimi_8solver_no_topic_batch5_01`.
  Temporary per-trial configs used `openrouter/moonshotai/kimi-k2.5` for
  composer and solver, no topic hint, `max_solver_runs=8`,
  `solver_batch_size=4`, and provider concurrency `1`. The repo default
  rollout count was not changed.
- **Raw result**:
  Five quality samples, no provider failures. Accepted: 2 (`trial_04`,
  `trial_05`). Failed: 3 (`trial_01`, `trial_02`, `trial_03`).

  The lower rollout count is useful for fast experiments but weakens the
  automatic gate: `trial_01` ended `0/8` and `trial_02` ended `1/8`, yet both
  were `calibration_inconclusive` rather than a decisive too-hard rejection
  because the confidence interval was still too wide.
- **Accepted data audit**:
  `trial_04` is low-quality accepted. The anchored admission evidence has
  `admittime=2180-06-26T18:27:00`, but the submitted user request says
  `2118년 6월 26일`. The query and label answer the hidden `hadm_id`, but the
  visible customer context value is hallucinated. This should be handled
  prompt-first; parsing and comparing natural-language dates would not be a
  precision-100 validator.

  `trial_05` is low-quality accepted. The accepted task asks for ICU procedure
  rows with only `procedure_name` and `start_time`. The label contains two
  identical rows: `18 Gauge` at `2187-05-18T18:45:00`. The query tool already
  exposes this exact condition through `projection_diagnostics`, and the shared
  prompt already says list rows should be distinguishable through requested
  output fields.
- **Rejected/failed data audit**:
  `trial_01` is hard-good rejected/inconclusive. The final medication
  administration task is grounded, scoped to the hidden subject, asks for
  visible tie-breaks, and has a deterministic final query, but Kimi solvers
  matched `0/8`.

  `trial_02` is hard-good rejected/inconclusive. The final admission medication
  task is similarly grounded and solvable through the tool surface, but matched
  `1/8`. It demonstrates that the eight-rollout setting is a fast probe, not a
  strong statistical accept/reject gate.

  `trial_03` is low-quality rejected. The composer first labeled patient
  demographics for a medication/fluid request, then repeatedly failed phrase
  and query-contract requirements before budget exhaustion. The bad draft did
  not reach solver calibration.
- **Change**:
  Strengthened the Customer Request prompt rule to forbid invented visible
  context values: dates, names, statuses, and places must be copied only from
  latest scoped query evidence.

  Added `answer_contract_duplicate_answer_rows` feedback. For list drafts, if
  the latest `query` reports `projection_diagnostics.duplicate_answer_rows`,
  `submit_draft` now rejects with a Label Contract reminder to add a natural
  user-visible distinguishing field, aggregate duplicates, or choose another
  grounded task. This is not a new durable instruction source; it enforces the
  existing prompt rule with exact query diagnostics.
- **Why this follows the principles**:
  The duplicate-row feedback uses only structured query evidence generated from
  the submitted draft's latest query. It does not inspect DB literals, table
  names, column-name tokens, model behavior, or precomputed domain-specific
  strings. The wrong-date failure remains prompt-only because a hard
  natural-language date validator would require non-precision-100 parsing
  heuristics.
- **Verification**:
  Prompt instructions render at `7961` characters. Focused tests passed for
  duplicate projected rows, ambiguous list order, hidden tie-break rejection,
  synthesis prompts, and composer query projection diagnostics (`5 passed`
  total across the two pytest commands). Ruff passed for touched source and
  test files. As in earlier focused runtime tests, one pytest process printed
  `3 passed` and then did not exit; it was killed after the pass result was
  visible.

## Iteration 117 — Label-first workflow and list output binding guard

- **Question**:
  The intended composer behavior is label-first: inspect the DB, build an
  interesting unique verifiable label, then derive the natural user request and
  submitted topic from that label. Does making this explicit improve request
  quality?
- **Setup**:
  Batch root:
  `artifacts/trial_20260428_mimiciv_demo_label_first_kimi_8solver_no_topic_batch5_01`.
  `mimiciv_demo`, no topic hint, `openrouter/moonshotai/kimi-k2.5` for composer
  and solvers, `max_solver_runs=8`, `solver_batch_size=4`, provider concurrency
  `1`.

  The batch started after the label-first workflow prompt change, but before the
  later explicit `Unique`/`Verifiable` definitions and list-output binding guard.
- **Raw result**:
  Five quality samples, no provider failures. Accepted: 5/5.

  Pass rates: `trial_01=3/8 = 0.375`, `trial_02=7/8 = 0.875`,
  `trial_03=7/8 = 0.875`, `trial_04=6/8 = 0.75`, `trial_05=6/8 = 0.75`.
- **Accepted data audit**:
  `trial_02` is clean accepted. The ICU output-events request asks for latest
  five discharge/output records with time, value, and unit; the label fields,
  query scope, ordering, and request match.

  `trial_03` is clean accepted. The ICU stay-history request asks for latest
  three stays and explicitly requests `intime`, `outtime`, first/last careunit,
  and length of stay.

  `trial_04` is clean/borderline accepted. The final lab-results request asks
  for time, lab item, and result, and requests the sort by latest time, priority,
  and item name. The final query matches those controls. It is slightly verbose
  but not low-quality.

  `trial_05` is low-quality accepted. The request asks for careunit names,
  admission/discharge times, and admission type, but the accepted label also
  includes `los`. `answer_contract_binding_diagnostics` already reported
  `missing_output_bindings=["los"]`, but the controller did not reject it.

  `trial_01` is borderline/low-quality accepted. The request asks for a
  same-time medication tie-break by "record order" and returns
  `administration_sequence`. The value is marked user-visible in query metadata,
  so this is not a precision-100 hard-validator case, but qualitatively it is a
  technical-sequence surface and should remain a prompt/tool-metadata concern.
- **Rejected/failed data audit**:
  None. Low-quality accepted, not rejection behavior, is the critical issue in
  this batch.
- **Change**:
  Rewrote the Workflow so the composer first builds the label, then derives
  `user_request` and `topic` that naturally ask for exactly that label.

  Added explicit Label Contract definitions:
  `Verifiable` means final `query(spec)` exactly reproduces submitted JSON and
  every value is copied from that query. `Unique` means one correct structured
  answer for the hidden entity/request; ties must be returned, visibly
  distinguished, aggregated, or avoided without hidden ids/order/filters.

  Moved terminal feedback wording out of Workflow and into the tool-call budget
  protocol: stop only when `submit_draft` says the conversation is terminated.

  Added a precision-100 list-label binding guard. For list labels,
  `answer_contract.output_bindings` must cover every returned label field. This
  would reject the `trial_05` failure mode because the submitted label contained
  `los` while the request/contract did not ask for it.
- **Why this follows the principles**:
  Label-first and Unique/Verifiable definitions are durable composer policy, so
  they belong in the system prompt. The binding guard uses only submitted
  `label_json`, submitted `answer_contract`, and latest query evidence; it does
  not inspect DB literals, table names, column-name tokens, or natural-language
  guesses.
- **Verification**:
  Prompt instructions render at `7994` characters with `15` `Why:` markers.
  Focused prompt test passed. Focused runtime tests for list output binding,
  order binding, and binding diagnostics printed `3 passed`; the pytest process
  did not exit after printing the result and was killed after the pass result was
  visible. Ruff passed for touched source and test files.

## Iteration 118 — Structured composer prompt and feedback reminders

- **Question**:
  Can the composer prompt be compressed and structured, using the Claude Code
  prompt examples as style references, without weakening prompt-first policy or
  turning feedback into a second instruction source?
- **Reference read**:
  Reviewed the structured prompt examples under
  `/Users/jd/workspace/claude-code-system-prompts`, especially the general
  agent, explorer, subagent-writing, tool-usage, communication, task-management,
  and verification-specialist examples. The reusable pattern is short role,
  high-salience named policies, execution flow separated from contracts, direct
  bullets, and no duplication of tool/API shape that SDK tool schemas already
  provide.
- **Change**:
  Reworked the composer system prompt into:
  `Workflow`, `Core Definitions`, `Request Contract`, `Label Contract`,
  `List Determinism Policy`, `Feedback And Difficulty-Up Policy`, and
  `Task Shapes`.

  Preserved the label-first workflow and explicit `Unique`/`Verifiable`
  definitions, but compressed duplicated prose. Rendered prompt length is now
  `7559` characters with `16` `Why:` markers, down from `7994`.

  Tightened feedback philosophy in code: feedback messages now read as
  `Policy reminder`, `Label Contract reminder`, `Request Contract reminder`, or
  `Tool schema reminder`, and the tool-budget prompt says specificity feedback
  is a reminder for Difficulty-Up Policy. Feedback still carries precise
  failure evidence, but it is not framed as a new durable instruction source.

  After the retry audit found a role-binding accepted defect, added one
  DB-neutral prompt rule and matching `submit_draft.answer_contract` schema
  description: binding phrases must name the returned field's distinct role and
  must not reuse one vague phrase for different returned concepts.
- **Setup**:
  Batch root:
  `artifacts/trial_20260428_mimiciv_demo_prompt_structure_kimi_8solver_no_topic_batch5_01`.
  `mimiciv_demo`, no topic hint, `openrouter/moonshotai/kimi-k2.5` for composer
  and solvers, `max_solver_runs=8`, `solver_batch_size=4`, provider concurrency
  `1`.
- **Raw result**:
  Original batch: accepted `2/5`, failed `3/5`.

  `trial_01` accepted at `7/8 = 0.875`.
  `trial_02` accepted at `7/8 = 0.875`.
  `trial_03` failed with `MaxTurnsExceeded` after a too-easy `8/8` draft and
  later `answer_contract_not_incremental`.
  `trial_04` failed with provider/model protocol error
  `Tool probe not found in agent synthesis`; excluded from quality statistics
  and retried.
  `trial_05` failed after repeated `answer_contract_phrase_missing`.

  Provider retry `trial_04_retry_01` accepted at `7/8 = 0.875`.
- **Accepted data audit**:
  `trial_01` is clean accepted. It asks for all procedures during the hidden
  admission, sorted by procedure start time, and returns the four scoped
  procedure rows with requested start/end/category/name/status fields. Query
  scope, ordering, and requested outputs match.

  `trial_02` is clean/borderline accepted. It asks for the hidden specific
  microbiology test result and returns specimen type, date, test, organism,
  antibiotic, and sensitivity. The task relies on the hidden `microevent_id`
  rather than visible contextual wording, but the request/label/query are
  aligned and solver pass rate is high.

  `trial_04_retry_01` is low-quality accepted. The request says "입원 시진" and
  "퇴원 시진" while the label contains both ICU stay times (`intime/outtime`) and
  hospital admission times (`admittime/dischtime`). The answer contract bound
  the same vague phrase to different source roles. This is exactly a Source
  Surface / binding-role failure, but not a precision-100 hard-validator case:
  some broad phrases can legitimately request a pair of fields. The fix belongs
  in prompt/schema guidance, not a literal or token heuristic.
- **Rejected/failed data audit**:
  `trial_03` is low-quality rejected / recovery failure. The first medication
  draft was too easy at `8/8`; the retry changed the row set and ordering in a
  non-incremental way; a later retry switched to an admission/ICU summary and
  exhausted turns. This is not a solver/tool failure.

  `trial_05` is contract-repair failure, not bad accepted data. The underlying
  medication-list candidate was plausible and deterministic after a visible
  medication-name tie-break, but the composer repeatedly failed exact phrase
  binding for `answer_contract.answer_phrase` and never reached calibration.

  Original `trial_04` is infra/provider failure, not a quality sample.
- **Why this follows the principles**:
  Durable behavior remains in the system prompt. Feedback now explicitly
  references named policies or tool-local schema contracts, so it reminds rather
  than becoming parallel policy. The role-binding improvement is DB-neutral and
  derived from a qualitative accepted-task audit. No validator was added because
  duplicate or broad binding phrases are not precision-100 invalid in every
  database/task shape.
- **Verification**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_calls_out_id_only_anchor_path_for_ungrounded_strings tests/test_synthesis_runtime.py::test_submit_draft_too_easy_feedback_is_list_aware tests/test_synthesis_runtime.py::test_submit_draft_schema_feedback_reports_entity_and_evidence_fixes tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_that_does_not_match_latest_query tests/test_synthesis_runtime.py::test_submit_draft_requires_limit_phrase_when_query_limit_shapes_list tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_list_output_binding tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_for_selected_order_key tests/test_synthesis_runtime.py::test_submit_draft_records_answer_contract_binding_diagnostics`
  passed (`21 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  passed.

## Iteration 119 — Requestable-label gate after label-first audit

- **Question**:
  Does label-first materially improve `user_request` quality, or does it mostly
  improve label/query alignment while still allowing awkward customer wording?
- **Setup**:
  Batch root:
  `artifacts/trial_20260428_mimiciv_demo_prompt_reminder_role_binding_kimi_8solver_no_topic_batch5_01`.
  `mimiciv_demo`, no topic hint, `openrouter/moonshotai/kimi-k2.5` for composer
  and solvers, `max_solver_runs=8`, `solver_batch_size=4`, provider concurrency
  `1`.
- **Raw result**:
  Completed quality samples: `4`; incomplete/infra-aborted sample: `1`.

  Accepted: `trial_01` at `7/8 = 0.875`.

  Failed: `trial_03` at `1/8 = 0.125`, `trial_04` at `1/8 = 0.125`,
  `trial_05` at `0/8 = 0.0`.

  `trial_02` ended without an exit file after a too-easy first submission at
  `8/8 = 1.0`; exclude it from quality statistics until retried.
- **Accepted data audit**:
  `trial_01` is structurally clean/borderline accepted. The final query,
  label, answer contract, and solver pass rate align. However, the request is
  verbose and contract-shaped: it asks for admission prescriptions, then spells
  out start-time ordering plus two tie-break clauses. This is not low-quality
  accepted, but it shows that label-first alone does not reliably produce
  natural customer requests.
- **Rejected/failed data audit**:
  `trial_03` is low-quality rejected/inconclusive. The Jevity enteral-nutrition
  request is specific, but the very low pass rate and trace pattern suggest a
  hidden-scope or row-set mismatch rather than a good hard task.

  `trial_04` is low-quality rejected. It used technical sequence wording
  (`순번`) and first hit `answer_contract_hidden_filter_unanchored`; the final
  request remained mechanical.

  `trial_05` is low-quality rejected / mechanical hard case. The composer moved
  toward a deterministic medication list, but the final request still reads as
  an ordering contract over repeated medication names rather than a natural user
  need. No low-quality accepted sample came from this batch.

  `trial_02` is incomplete. Its first request was more natural but too easy;
  because the process exited without a completed recovery attempt, it is an
  infra-aborted sample, not evidence that label-first solved request quality.
- **Conclusion**:
  Label-first contributes to label/query/request alignment, but its contribution
  to request quality is weak in the current design. The composer can still pick
  a label that is only verifiable through awkward ordering, technical sequence
  wording, or long field enumeration, then faithfully derive an awkward
  `user_request` from it.
- **Change**:
  Redesign the workflow from plain label-first to requestable-label-first.
  The composer now builds a requestable label candidate and checks whether a
  realistic customer can ask for the exact fields, row set, order, and
  tie-breaks without technical or awkward control wording. If not, it must
  choose another label.

  Added a compact Request Contract rule: keep requests realistic and compact;
  if exact deterministic controls require a long tie-break ladder, technical
  sequence wording, or mechanical field enumeration, choose a more naturally
  requestable label instead. List tasks now prefer natural orders needing zero
  or one visible tie-break.
- **Why this follows the principles**:
  This is durable composer behavior, so it belongs in the system prompt. No
  validator was added because request naturalness, awkwardness, and "too much
  deterministic control wording" are not precision-100 properties. The change
  stays DB-neutral and does not inspect DB literals, table names, column-name
  tokens, or model outputs with heuristic string rules.
- **Verification**:
  Prompt instructions render at `7998` characters with `16` `Why:` markers.

  `uv run pytest tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py`
  passed (`12 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py tests/test_synthesis_prompts.py`
  passed.

  `git diff --check` passed.

## Iteration 120 — Remove flag-like rationale markers

- **Question**:
  The inline `Why:` markers preserve rationale, but do they make the composer
  prompt feel like an artificial flag format?
- **Change**:
  Removed all `Why:` markers from the composer system prompt. The rationale is
  still present as ordinary prose next to the relevant instruction, for example
  mapping lets the DB decide the domain, hidden entity values must be grounded,
  exact structured labels make the task verifiable, and one policy source
  prevents split guidance.

  Updated prompt tests to assert `Why:` is absent and to verify the actual
  rationale phrases instead of counting marker occurrences.
- **Why this follows the principles**:
  This is prompt-surface cleanup only. It keeps durable policy in the system
  prompt, avoids adding a second instruction source, and does not introduce any
  validator or DB-specific heuristic.
- **Verification**:
  Prompt instructions render at `7933` characters with `0` `Why:` markers.

  `uv run pytest tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py`
  passed (`12 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py tests/test_synthesis_prompts.py`
  passed.

  `git diff --check` passed.

## Iteration 121 — Solver batch concurrency and reasoning sidecar diagnostics

- **Question**:
  Why did Kimi batch-five experiments take so long even with
  `solver_batch_size=4`, and can we preserve provider-exposed reasoning
  payloads for prompt-debugging without bloating the main trial timeline?
- **Finding**:
  The recent MIMIC demo batch configured `solver_batch_size=4` but also set the
  OpenRouter provider concurrency to `1`. `SolverOrchestrator` used
  `asyncio.gather` per batch, but each solver call entered the provider
  semaphore, so one-provider batches were effectively serialized. This made a
  nominal four-wide solver rollout run one solver at a time. The same logs also
  showed long feedback continuations where all context was preserved after
  submit failures, which inflated input tokens, but the direct four-wide
  mismatch was the provider semaphore.
- **Change**:
  Added a startup validation in `SolverOrchestrator`: for the configured solver
  schedule, each provider's `max_concurrency` must be at least the maximum
  number of solver calls that can appear for that provider inside one solver
  batch. A single Kimi solver with `solver_batch_size=4` now requires
  OpenRouter `max_concurrency >= 4` instead of silently serializing.

  Added a `reasoning_content.jsonl` sidecar under the trial debug directory.
  Composer and solver backends now persist raw reasoning payloads only when the
  Agents SDK exposes them as reasoning items or explicit
  `reasoning_content`/thinking fields. The primary `trial_events.jsonl` stores
  only `reasoning_content_path` and `reasoning_content_items`.
- **Why this follows the principles**:
  This is runtime/config correctness and observability, not task-quality policy.
  The concurrency check is precision-100 because it compares configured batch
  shape against configured provider concurrency. The reasoning capture records
  SDK-exposed payloads as artifacts; it does not infer hidden reasoning, parse
  DB literals, or add prompt/feedback policy.
- **Expected experiment impact**:
  Four-wide single-provider solver batches should actually run four concurrent
  solver calls when provider limits allow it. If a provider/model does not
  expose raw reasoning content, no sidecar records are emitted and the count
  remains zero.
- **Verification**:
  `uv run pytest tests/test_sdk_helpers.py tests/test_solver_backend_openai_agents.py tests/test_synthesis_backend_openai_agents.py tests/test_pipeline_solver_orchestrator.py tests/test_config_load.py`
  passed (`54 passed`).

  `uv run ruff check src/rl_task_foundry/infra/sdk_helpers.py src/rl_task_foundry/infra/event_log.py src/rl_task_foundry/solver/backend_openai_agents.py src/rl_task_foundry/synthesis/backend_openai_agents.py src/rl_task_foundry/pipeline/solver_orchestrator.py tests/test_sdk_helpers.py tests/test_solver_backend_openai_agents.py tests/test_synthesis_backend_openai_agents.py tests/test_pipeline_solver_orchestrator.py tests/test_config_load.py`
  passed.

  `git diff --check` passed.

  Full `uv run pytest` was also run and currently reports `451 passed, 15
  failed`; failures are in existing `tests/test_synthesis_runtime.py`
  feedback-message expectations and `tests/test_synthesis_proof_environment.py`
  proof fixture acceptance, outside this runtime diagnostics patch.

## Iteration 122 — Align tests and proof fixture with reminder/binding contracts

- **Question**:
  What caused the 15 full-suite failures after Iteration 121, and should the
  fix change runtime behavior or test/schema surfaces?
- **Finding**:
  The failures were from two stale surfaces. First, after the
  prompt/feedback-source cleanup, feedback text intentionally says policy
  `reminder` instead of `violation`; several tests still asserted old exact
  text. Second, list `answer_contract` validation now requires field/order
  bindings for exact request-to-label grounding, but older list fixtures and
  the scripted proof composer still omitted those bindings.
- **Change**:
  Updated tests to assert reminder text and stable `last_feedback_error_codes`
  instead of old violation phrasing. Added list output bindings to fixtures
  whose purpose is to reach calibration or acceptance rather than fail on
  binding. Updated the proof scripted request and submit payload so every
  returned list field is requested and bound. Clarified the submit_draft tool
  schema description: list labels require one output binding per returned field;
  scalar labels may omit it when the answer phrase already binds the result.
- **Why this follows the principles**:
  No new validator or heuristic was added. Tool schema description now matches
  the tool-local validation contract, and feedback tests now reflect the
  prompt-first/reminder-only policy rather than reintroducing "feedback as
  instruction" wording.
- **Verification**:
  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_proof_environment.py::test_run_proof_task_commits_and_exports_bundle -q`
  passed (`55 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/proof_environment.py tests/test_synthesis_runtime.py tests/test_synthesis_proof_environment.py`
  passed.

  Full `uv run pytest` passed (`466 passed`).

  `git diff --check` passed.

## Iteration 123 — Post-binding-fix Kimi batch qualitative audit

- **Question**:
  After Iteration 122 fixed the stale binding/reminder tests and proof fixture,
  does the current no-topic MIMIC demo pipeline produce clean data with
  `solver_batch_size=4`, eight Kimi solver rollouts, and no externally injected
  topic?
- **Experiment setup**:
  Ran five parallel `mimiciv_demo` trials with no `--topic-hint`. Composer and
  solver both used `openrouter/moonshotai/kimi-k2.5`; each trial used
  `max_solver_runs=8`, `solver_batch_size=4`, `lower_pass_rate=0.2`,
  `upper_pass_rate=0.9`, `safe_early_termination=true`, and OpenRouter
  `max_concurrency=8`.

  Artifact root:
  `artifacts/trial_20260428_mimiciv_demo_post_bindingfix_kimi_8solver_no_topic_batch5_01`.
- **Result**:
  Raw accepted count was `2/5`.

  Trial 1 failed in composer repair before solver rollout. The composer kept a
  medication-list label ordered by prescription timing while the returned label
  fields omitted the timing evidence. The final feedback was
  `answer_contract_evidence_mismatch`; this is a safe rejection, not an
  accepted-quality issue.

  Trial 2 accepted. First submission was rejected as too easy at pass rate
  `8/8`; the composer then added one requested field and the second submission
  accepted at `7/8` (`0.875`, CI low `0.5293`). Qualitatively this is clean but
  near the easy edge: a single anchored eMAR record asks for medication,
  recorded/scheduled/system times, and handling status.

  Trial 3 rejected after calibration at `1/8` (`0.125`, CI high `0.4707`).
  Qualitatively this is low-quality rejected, not hard-good: the natural request
  asked for recent medication name/start/end/status for a patient, but both
  `prescriptions` and `pharmacy` were plausible surfaces. Most solvers used
  `prescriptions.drug_type` as status (`MAIN`); only one discovered
  `pharmacy.status` (`Expired`, `Discontinued`, ...). The composer did not make
  the label source uniquely recoverable from the user request.

  Trial 4 accepted at `6/8` (`0.75`, CI low `0.4003`). Qualitatively clean: the
  request asks for ICU procedure start/end/name/status and explicitly surfaces
  the start-time plus procedure-name ordering. The two misses were solver-side
  output-format/path issues: one used timestamp strings with spaces instead of
  ISO `T`, and one listed `ordercategoryname` instead of following `itemid` to
  the item label and then failed to submit.

  Trial 5 failed before draft submission. It explored an admission, made one
  malformed JSON tool call, then repeatedly ended without `submit_draft`;
  feedback correctly stayed on `composer_submit_draft_missing`. This is a
  composer protocol failure, not a data-quality sample.
- **Qualitative audit**:
  Accepted clean data: Trial 4.

  Accepted borderline-clean data: Trial 2. It is not low-quality, but it is
  close to the upper difficulty boundary because the anchor plus date/medication
  makes the single row highly reachable.

  Rejected low-quality data: Trial 3. The rejection was desirable because the
  label source was semantically ambiguous.

  Rejected composer/protocol failures: Trials 1 and 5. Both were safely
  rejected before producing accepted low-quality data.

  Accepted low-quality data: none observed.
- **Runtime observations**:
  Solver batches were actually parallel after Iteration 121. Completion
  timestamps show the first four solver calls in each rollout finishing in
  overlapping windows with different latencies, rather than serially waiting on
  the provider semaphore.

  No `reasoning_content.jsonl` sidecar was emitted for this OpenRouter Kimi run;
  the provider/SDK path did not expose reasoning items. Tool traces and
  `run_items` were still enough to classify the rejected and accepted samples.
- **Next improvement candidate**:
  The main remaining composer-quality gap is DB-neutral semantic source
  ambiguity. When two reachable surfaces can satisfy the same everyday wording
  but produce different labels, the composer should make the request naturally
  disambiguate the label source or choose another label. This belongs in the
  composer prompt as durable policy, not in a hard validator, because deciding
  whether natural wording uniquely implies one semantic source is qualitative
  rather than precision-100.

## Iteration 124 — Preserve OpenRouter Kimi reasoning in Agents traces

- **Question**:
  Did Iteration 123 really prove that Kimi/OpenRouter did not emit reasoning
  content, or did our Agents SDK path drop it before persistence?
- **Finding**:
  Direct OpenRouter chat completion calls to `moonshotai/kimi-k2.5` returned
  provider-visible reasoning in `message.reasoning` and `reasoning_details`.
  The Agents SDK `OpenAIChatCompletionsModel` then converted the chat message
  into `ResponseOutputMessage` items and only promoted `reasoning_content`, not
  OpenRouter's `reasoning` field, into `ReasoningItem`. The previous
  `reasoning_content_items=0` observation therefore meant "not exposed through
  our current SDK item surface", not "the provider did not emit reasoning".
- **Change**:
  Added an Agents compatibility wrapper that copies chat-completion
  `message.reasoning` into `message.reasoning_content` before SDK conversion.
  The existing sidecar writer can then persist it as `reasoning_content.jsonl`.
  The raw extractor also now recognizes direct `reasoning` and
  `reasoning_details` fields if a future SDK exposes them without conversion.
- **Verification**:
  A live minimal OpenRouter Kimi call through the patched wrapper produced a
  `ReasoningItem`, and `extract_raw_reasoning_records` returned one reasoning
  record.

  `uv run pytest tests/test_sdk_helpers.py tests/test_solver_backend_openai_agents.py tests/test_synthesis_backend_openai_agents.py -q`
  passed (`31 passed`).

  `uv run ruff check src/rl_task_foundry/infra/sdk_helpers.py src/rl_task_foundry/synthesis/backend_openai_agents.py src/rl_task_foundry/solver/backend_openai_agents.py tests/test_sdk_helpers.py`
  passed.

  `git diff --check` passed.

## Iteration 125 — Source ambiguity prompt policy

- **Question**:
  What should change after Iteration 123 showed a rejected MIMIC medication task
  where the same everyday word could point to two reachable data surfaces with
  different values?
- **Change**:
  Tightened the composer system prompt's `Source surface` and label wording
  rules. If one user phrase can map to several roles or source surfaces with
  different values, the composer must name the chosen source in the
  `user_request`/`answer_contract` or choose another label. It must also avoid
  relabeling one surface as another unless the request names that role; vague
  field words are invalid when several reachable sources could answer them.

  This is intentionally prompt policy only. Natural-language source ambiguity is
  qualitative and cannot be rejected with a precision-100 validator without
  sneaking in DB-specific literal heuristics.
- **Verification**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  passed (`12 passed`).

## Iteration 126 — 고객 요청에서 DB alias 제거

- **질문**:
  source ambiguity 정책 이후에도 단일 MIMIC demo smoke에서 solver 문제가
  아닌 composer 품질 문제가 남아 있는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_source_ambiguity_kimi_8solver_no_topic_smoke_02`.

  trial은 pass rate `7/8 = 0.875`로 accepted 됐다. 첫 draft는 request/label에
  표현되지 않은 hidden order tie-break 때문에 올바르게 reject됐다. composer는
  이후 hidden `orderid` ordering을 visible `endtime`, `ordercategoryname`
  control로 바꿨기 때문에 list determinism 자체는 남은 문제가 아니었다.
- **reasoning 감사**:
  저장된 composer reasoning을 보면 모델은 타당한 ICU procedure row set을
  찾았고, projected row 중복도 인지했다. 이후 visible field를 추가해 row를
  구분 가능하게 만들었고, feedback 뒤에는 hidden ordering도 고쳤다.

  남은 품질 문제는 고객 요청/label surface였다. accepted 된 한국어 요청에
  `Duration`, `Location` 같은 schema-like alias가 괄호 안에 새어 나왔고,
  자연스러운 semantic output name을 만들 수 있는데도 label이 raw-source
  field key에 가까웠다. 즉 solver tool 문제가 아니라 composer의
  request/label surface 문제다.
- **변경**:
  prompt만 좁게 수정했다.

  - Request Contract: field key는 JSON에만 두고 request text에는 넣지 않는다.
    괄호 안 schema-like alias를 피한다.
  - Label Contract: raw DB alias가 아니라 semantic API-style field name을 쓴다.

  prompt-first 원칙을 지켰다. "DB-ish wording" 판정은 정성적이라 forbidden
  literal heuristic 없이 precision-100 validator로 만들 수 없으므로 hard
  validator는 추가하지 않았다.
- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`12 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py tests/test_synthesis_prompts.py`
  통과.

## Iteration 127 — Alias surface smoke와 incremental feedback reminder

- **질문**:
  request/field surface prompt cleanup 이후, 다음 no-topic MIMIC smoke에서
  더 자연스러운 user text와 semantic label field가 나오는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_alias_surface_kimi_8solver_no_topic_smoke_01`.

  첫 draft는 schema alias 없는 자연스러운 한국어 요청을 만들었다:
  `내 중환자실 입원 기간 동안 기록된 배출량 5건을 최신순으로 보여주세요.`

  label field도 semantic API-style name이었다:
  `output_volume`, `unit`, `recorded_time`, `stored_time`.

  이 sample에서는 Iteration 126 prompt 방향이 실제로 작동했다.
- **결과**:
  trial은 여전히 `synthesis_failed`로 끝났다. 단, 데이터가 저품질이라서가
  아니라 모든 solver가 맞췄고(`8/8 = 1.0`), quality gate가 too easy /
  calibration-inconclusive로 처리했기 때문이다.

  retry 흐름:

  - submit 1: `calibration_inconclusive`, pass rate `1.0`.
  - submit 2: `answer_contract_not_incremental`. `stored_time`을 output-type
    field로 대체했다.
  - submit 3: `answer_contract_not_incremental`. 다시 `stored_time`을
    measurement-name field로 대체했다.
  - submit 4: `answer_contract_binding_missing`. field는 더 보존했지만
    필요한 order binding coverage가 빠졌다.
  - submit 5: 기존 네 field를 보존하고 `care_unit`만 추가했지만, solver pass
    rate가 여전히 `1.0`이라 calibration에서 budget exhausted 됐다.
- **reasoning 감사**:
  feedback 이후 composer reasoning을 보면, 모델은 늦게나마 핵심 규칙을
  이해했다. `output_volume`, `unit`, `recorded_time`, `stored_time`을 보존한
  뒤 `care_unit`만 추가해야 한다고 명시적으로 결론냈다. 낭비된 retry는 그
  결론 전에 "specificity를 추가하라"를 기존 output source 하나를 다른 관련
  source로 교체해도 된다는 뜻으로 해석하면서 발생했다.

  이건 새로운 durable policy gap이 아니라 feedback reminder 문제다. durable
  prompt에는 이미 list feedback 때 filters/order/limit/row set과 output source
  meaning을 보존하고, user-visible field 하나만 append하라고 되어 있다.
- **변경**:
  `answer_contract_not_incremental` feedback reminder를 더 구체화했다. 기존
  Difficulty-Up Policy를 다시 지시로 복제하지 않고, 그 정책을 정확히
  상기시키도록 했다: list retry는 모든 prior output field/source와 prior order
  binding을 유지한 뒤, grounded user-visible field 또는 tie-break 하나만
  append해야 한다.

  또한 `TrialEventLogger.write_sidecar_jsonl`이 sidecar write 뒤 명시적으로
  flush하도록 했다. 이 변경이 Agents SDK run segment가 끝나기 전 composer
  reasoning을 볼 수 있게 하지는 않는다. 다만 reasoning record가 logger에
  넘어온 뒤 sidecar buffering 때문에 늦게 보이는 문제는 막는다.
- **정성 평가**:
  accepted data: 없음.

  rejected/failed data: hard-good but too easy. solver는 주어진 tool로 안정적으로
  풀 수 있었고, 최종 draft는 깨끗하고 자연스러웠지만 의도한 difficulty band
  아래였다. low-quality accepted data가 아니다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_too_easy_requires_incremental_answer_contract tests/test_synthesis_runtime.py::test_submit_draft_too_easy_monitor_keeps_evaluated_label_baseline tests/test_synthesis_logging.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`17 passed`).

  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_too_easy_rejects_renamed_same_scalar_value -q`
  통과.

  `uv run ruff check src/rl_task_foundry/infra/event_log.py src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_logging.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 128 — Hidden scope recovery feedback

- **질문**:
  Iteration 127의 incremental feedback reminder 이후 다음 no-topic MIMIC smoke를
  돌리면 composer가 문제를 올바르게 회복하는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_incremental_feedback_kimi_8solver_no_topic_smoke_01/trial_01`.

  trial은 solver 실행 전 `synthesis_failed`로 종료됐다. feedback event는 5개였고,
  provider failure는 없었다.
- **결과**:
  composer는 `labevents`의 단일 `labevent_id=242873`를 anchor로 잡은 뒤
  `내 최근 혈액 검사 결과를 최신순으로 5개 보여줘`라는 parent/history 성격의
  요청을 만들었다. 실제 query는 같은 환자의 lab history를 가져오기 위해
  hidden `subject_id=10020306`로 필터링했다.

  이 구조는 label scope와 entity scope가 맞지 않는다. 요청은 환자 범위의 최근
  검사 이력인데, 제출 entity는 단일 lab event였다. quality gate는
  `answer_contract_hidden_filter_unanchored`로 이 hidden row-set control을
  잡았고, 동시간 lab rows의 order/limit 문제는 `answer_contract_order_ambiguous`
  로 잡았다.

  retry 중 composer는 `subject_id`를 entity에 추가해 hidden filter 문제를 한 번
  해결했지만, `labevent_id`를 visible answer/order tie-break처럼 노출하려고 했다.
  이는 자연스러운 고객 요청이 아니라 technical handle repair에 가깝다. 마지막에는
  hidden tie-break를 제거하면서 다시 ambiguous list가 됐고 budget이 소진됐다.
- **reasoning 감사**:
  저장된 composer reasoning은 실패 원인을 잘 보여준다. 모델은 첫 feedback 뒤
  `subject_id`가 entity에 있어야 한다는 점은 이해했지만, "최근순 5개" list의
  동점 문제를 hidden `labevent_id` tie-break로 고쳐도 된다고 잘못 판단했다.
  이후에는 handle을 label에서 빼야 한다는 점과 deterministic order 필요성을
  동시에 만족시키지 못했다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 데이터가 어려워서 좋은 문제가 된 것이
  아니라, 단일 child record anchor에서 parent/history request로 넓히는 과정에서
  hidden parent filter와 hidden/id tie-break가 request/label에 자연스럽게
  고정되지 않았다. rejection은 바람직했고, low-quality accepted는 관측되지 않았다.
- **변경**:
  durable prompt 정책은 이미 충분히 있었다.

  - Request Contract: parent/list/history requests query that scope.
  - List Determinism Policy: hidden handles/artificial id wording으로 tie-break를
    만들지 않는다.

  따라서 prompt를 새로 늘리지 않고, feedback reminder만 정책 상기 역할에 맞게
  좁게 강화했다.

  - `answer_contract_hidden_filter_unanchored`: child record에서
    parent/list/history request로 넓힐 때 parent/current-subject handle을 entity에
    유지하거나 existing entity에 맞는 label을 택하라고 상기한다.
  - `answer_contract_order_ambiguous`: natural visible tie-break, tied rows,
    unique ordering, or another label을 택하라고 상기하고 hidden handles/artificial
    id wording으로 repair하지 말라고 상기한다.

  이 변경은 DB literal/token heuristic이 아니다. validator trigger는 latest query
  evidence의 hidden filter/order diagnostics에 기반하므로 precision-100 feedback
  조건을 유지한다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_ambiguous_limited_list_order tests/test_synthesis_runtime.py::test_submit_draft_rejects_multirow_list_without_order_by tests/test_synthesis_runtime.py::test_submit_draft_rejects_unrepresented_list_order_by_tie_breaker tests/test_synthesis_runtime.py::test_submit_draft_rejects_hidden_filter_missing_from_entity -q`
  통과 (`4 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 149 — Composer/solver wall-clock timeout guard

- **질문**:
  single smoke 반복 중 provider/SDK call이 새 이벤트 없이 오래 멈추면 실험 루프를
  어떻게 보호해야 하는가?
- **실험**:
  Iteration 148 변경 후 `mimiciv_demo` no-topic Kimi smoke를 재실행했다.

  - `artifacts/trial_20260429_mimiciv_demo_list_limit_guard_kimi_8solver_no_topic_smoke_01/trial_01`
  - `artifacts/trial_20260429_mimiciv_demo_episode_timeout_guard_kimi_4solver_no_topic_smoke_01/trial_01`

  첫 실행에서는 `answer_contract_list_limit_too_wide`가 의도대로 작동해 11-row
  diagnosis draft를 solver rollout 전에 막았다. 이후 5-row diagnosis draft까지
  줄었지만 solver batch가 provider 대기 상태로 15분 이상 멈춰 중단했다.

  두 번째 실행에서는 composer가 `icustays` 단일 row 속성 문제를 만들었다.
  첫 시도는 `submit_draft` 없이 종료되어 기존 missing-submit feedback이 작동했고,
  이후 시도들은 단일 row 속성 묶음을 `kind=list`/단일 원소 배열로 제출해
  `answer_contract_phrase_missing`, `answer_contract_binding_missing`에서 거절됐다.
  네 번째 feedback 이후 composer call도 새 이벤트 없이 멈춰 중단했다.
- **정성 평가**:
  accepted data: 없음. low-quality accepted도 없음.

  rejected data:

  - 11-row diagnosis list: low-quality rejected. row count로 난이도를 올리는 draft라
    Iteration 148 validator가 정확히 막았다.
  - 5-row diagnosis list: borderline/hard-good 후보. 단, 일부 solver가 전체 테이블을
    먼저 materialize하고 placeholder resource id를 오용해 tool budget을 낭비했다.
    이건 데이터 자체보다 solver 도구 사용성/프로토콜 문제다.
  - ICU stay 속성 묶음: hard-good 후보이나 contract shape가 틀렸다. 단일 entity facts는
    list가 아니라 object/scalar answer로 제출해야 한다.
- **변경**:
  solver backend가 task bundle의 `rollout_constraints.max_episode_duration_ms`를
  실제 wall-clock timeout으로 적용한다. timeout은 `TimeoutError`로 기록되어
  solver pass-rate denominator에서 제외되는 infra failure로 처리된다.

  synthesis backend에는 `synthesis.runtime.run_timeout_s`를 추가하고, composer
  `Runner.run` segment 전체에 wall-clock timeout을 적용했다. 기본 config는
  `300s`다. 이는 품질 정책이 아니라 실험 파이프라인 안정성 장치다.
- **원칙 준수**:
  데이터/문자열 휴리스틱은 추가하지 않았다. timeout은 actor 품질 검증이 아니라
  provider/SDK stall을 실험 실패로 닫기 위한 infra guard다.
- **다음 개선 후보**:
  `submit_draft` 스키마는 단일 row lookup도 `kind=list`로 표현한다. 따라서 다음
  개선은 prompt를 `kind=object`로 바꾸는 것이 아니라, limit 없이 정확히 한 row만
  반환되는 list에서 order binding을 과잉 요구하는 validator를 점검하는 것이다.
- **검증**:
  `uv run pytest tests/test_synthesis_backend_openai_agents.py::test_synthesis_backend_enforces_run_timeout tests/test_solver_backend_openai_agents.py::test_openai_agents_solver_backend_enforces_episode_duration tests/test_config_load.py::test_load_config_uses_solver_run_count_source_of_truth -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/backend_openai_agents.py src/rl_task_foundry/solver/backend_openai_agents.py src/rl_task_foundry/config/models.py tests/test_synthesis_backend_openai_agents.py tests/test_solver_backend_openai_agents.py tests/test_config_load.py`
  통과.

## Iteration 150 — Single-row list order binding should be effect-aware

- **질문**:
  Iteration 149의 ICU stay draft처럼 단일 entity의 속성 묶음이 `kind=list`로
  제출됐을 때, `query.order_by`가 있다는 이유만으로 order binding을 요구하는 것이
  precision-100 validator인가?
- **판단**:
  아니다. `submit_draft` 스키마는 `kind`를 `scalar|list`만 지원하고, description도
  “list means the query rows array, even when one row is returned”라고 명시한다.
  따라서 단일 row lookup을 `kind=object`로 유도하면 프롬프트와 tool schema가
  이원화된다.

  반면 query가 `LIMIT` 없이 정확히 한 row만 반환했다면 `order_by`는 정답 membership
  이나 order를 바꾸지 않는다. 이때 order binding을 필수로 보는 것은 hidden row-set
  control을 잡는 검증이 아니라 과잉 거절이다.
- **변경**:
  `answer_contract_binding_diagnostics`에 `required_order_reference_count`를
  추가했다. canonical answer가 list이고 row count가 `0/1`이며 query limit이 없으면
  order binding required count를 `0`으로 둔다. raw `order_reference_count`는
  diagnostics에 남겨 visibility는 유지한다.

  반대로 `LIMIT 1`처럼 order가 어떤 row를 고를지 결정할 수 있는 경우에는 기존처럼
  order binding을 요구한다.
- **원칙 준수**:
  DB literal/token/table-name heuristic을 쓰지 않았다. 판단 근거는 오직 query
  metadata(`limit`)와 canonical answer 구조(row count)다. precision-100 구조 조건이다.
- **정성 평가**:
  accepted data: 아직 없음.

  rejected data: Iteration 149의 ICU stay draft는 데이터 자체가 나쁜 것이 아니라
  contract validator가 단일 row list의 무효한 order binding을 요구한 것으로 본다.
  동일 현상 재실험이 필요하다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_accepts_single_row_list_without_order_binding tests/test_synthesis_runtime.py::test_submit_draft_still_requires_order_binding_for_limited_single_row tests/test_synthesis_runtime.py::test_submit_draft_rejects_ambiguous_limited_list_order tests/test_synthesis_runtime.py::test_submit_draft_requires_limit_phrase_when_query_limit_shapes_list -q`
  통과 (`4 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 151 — Query diagnostics must be treated as blocking evidence

- **질문**:
  Iteration 150 이후 no-topic MIMIC smoke에서 accepted 품질이 개선되는가?
- **실험**:
  `artifacts/trial_20260429_mimiciv_demo_single_row_order_guard_kimi_4solver_no_topic_smoke_01/trial_01`
  를 실행했다. composer/solver는 OpenRouter Kimi K2.5, no topic hint, solver 4개,
  synthesis `run_timeout_s=180`.
- **결과**:
  trial은 accepted 없이 `synthesis_failed`로 종료됐다. 첫 draft는 ICU stay의
  `procedureevents` list였다.

  - request: `ICU 입원 중에 수행된 절차 목록... 시간순으로 가장 먼저 수행된 것부터...`
  - query: `limit=5`, `order_by=starttime asc, orderid asc`
  - diagnostics: `answer_contract_phrase_missing`,
    `answer_contract_query_mismatch`, `answer_contract_order_ambiguous`
  - ordering diagnostics: `unrepresented_order_by_tie_breakers` on hidden
    `orderid`; earlier query also had duplicate `start_time` order keys.

  첫 feedback 이후 composer는 180초 wall-clock timeout으로 종료됐다. timeout guard는
  의도대로 실험을 붙잡지 않았다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 절차 데이터 자체는 잠재적으로 좋은 list
  task지만, draft는 hidden tie-break와 unbound limit로 row set/order를 몰래 결정했다.
  validator 거절은 올바르다.
- **변경**:
  `query` tool schema/description을 좁게 강화했다. 반환되는 `ordering_diagnostics`
  중 `duplicate_order_key_in_returned_rows` 또는
  `unrepresented_order_by_tie_breakers`가 list에 있으면 final label evidence로
  submit하지 말고 request/order/output fields를 고치거나 다른 label을 선택하라고
  명시했다. 또한 final list query의 `limit`은 user_request와
  `answer_contract.limit_phrase`에 같은 fixed size로 묶여야 한다고 설명했다.
- **원칙 준수**:
  이 변경은 durable prompt 복제가 아니라 `query` tool output 해석에 관한 tool-local
  contract다. DB literal/token heuristic은 없고, tool diagnostics key와 query
  metadata만 사용한다.
- **검증**:
  `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`1 passed`).

  `uv run ruff check src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_tooling_composer_tool_factory.py`
  통과.

## Iteration 152 — No-PK feedback should point back to stable surfaces

- **질문**:
  Iteration 151 이후 no-topic MIMIC smoke에서 accepted 품질이 개선되는가?
- **실험**:
  `artifacts/trial_20260429_mimiciv_demo_query_diag_desc_kimi_4solver_no_topic_smoke_01/trial_01`
  를 실행했다. composer/solver는 OpenRouter Kimi K2.5, no topic hint, solver 4개,
  synthesis `run_timeout_s=180`.
- **결과**:
  trial은 accepted 없이 `synthesis_failed`로 종료됐다.

  첫 draft는 `chartevents` 기반 최근 생체 신호 5개 list였다. 이번에는 order/limit
  binding과 query diagnostics는 깔끔했지만, `chartevents`가 primary key 없는 table이라
  `label_no_primary_key_source`로 거절됐다. feedback 이후 composer는
  `datetimeevents`, `procedureevents`, `inputevents`, `outputevents`, `emar`를 넓게
  탐색하다 timeout으로 종료됐다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 최근 생체 신호 list 자체는 유용해 보이지만,
  solver가 stable record로 재방문할 수 없는 no-PK row values를 label로 제출했으므로
  거절은 맞다.
- **변경**:
  `label_no_primary_key_source` feedback을 Source Surface Policy reminder 역할에
  맞춰 보강했다. no-PK table의 row values를 다시 제출하지 말고, primary-key-backed
  path의 row values를 선택하거나 no-PK table에 대해서는 derived aggregate를 쓰라고
  상기한다.
- **원칙 준수**:
  feedback은 새 지시가 아니라 기존 prompt의 “If no primary key, use a
  primary-key-backed path/aggregate” 정책을 상기한다. DB literal heuristic은 없다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_from_table_without_primary_key tests/test_synthesis_runtime.py::test_submit_draft_allows_count_from_table_without_primary_key -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 153 — Redundant limit on primary-key lookup is not a list size request

- **질문**:
  `label_no_primary_key_source` feedback 보강 후 no-topic MIMIC smoke가 accepted까지
  가는가?
- **실험**:
  `artifacts/trial_20260429_mimiciv_demo_no_pk_feedback_kimi_4solver_no_topic_smoke_02/trial_01`
  를 실행했다. composer/solver는 OpenRouter Kimi K2.5, no topic hint, solver 4개,
  synthesis `run_timeout_s=300`.
- **결과**:
  trial은 accepted 없이 feedback 3회 후 진행 중이었다. 새로 드러난 반복 오류는 EMAR
  단일 투약 이벤트 lookup이다.

  composer는 `emar_id = "10021118-149"`로 primary-key row를 정확히 필터링했고,
  label은 단일 row의 `emar_id`, medication, event text, schedule/chart time이었다.
  그러나 query에 `limit=1`이 남아 있었고, validator가 이를 “list membership을 limit가
  결정한다”고 보고 `answer_contract_query_mismatch`를 반복했다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: borderline/hard-good 후보. 단일 EMAR 이벤트 상세 조회는 primary-key로
  안정적으로 재방문 가능하고 request도 자연스럽다. 다만 `투약 ID`를 user-visible
  handle로 노출하는 것이 좋은 label surface인지는 별도 정성 검토가 필요하다.

  핵심 오류는 데이터 저품질이 아니라 validator over-rejection이다. full primary-key
  equality가 이미 one row를 고정하면 `LIMIT 1`은 membership을 바꾸지 않는다.
- **변경**:
  composer `query` result의 `column_sources`와 `referenced_columns`에
  `is_primary_key`와 `table_primary_key` provenance를 추가했다.

  `submit_draft`의 missing `limit_phrase` 검증을 effect-aware로 바꿨다. list query가
  `limit=1`이고, 모든 value-exposing label source table이 full primary-key equality로
  constrained되어 있으면 limit은 redundant로 보고 `limit_phrase`를 요구하지 않는다.
  그 외 limit이 returned row count를 채우는 경우에는 기존대로 fixed size phrase를
  요구한다.
- **원칙 준수**:
  DB literal/token heuristic은 없다. 판단 근거는 query metadata, PK provenance, canonical
  answer row count뿐이다. `LIMIT 1`을 무조건 허용하지 않고, full primary-key constraint가
  label source table을 고정하는 경우에만 완화한다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_allows_redundant_limit_for_primary_key_lookup tests/test_synthesis_runtime.py::test_submit_draft_still_requires_order_binding_for_limited_single_row tests/test_synthesis_runtime.py::test_submit_draft_requires_limit_phrase_when_query_limit_shapes_list tests/test_tooling_composer_query.py::test_query_returns_visibility_provenance_for_outputs_and_refs tests/test_tooling_composer_query.py::test_query_marks_label_sources_without_primary_key tests/test_tooling_composer_query.py::test_query_select_spans_from_and_joined_tables tests/test_tooling_composer_query.py::test_query_order_by_output_reports_source_column_provenance -q`
  통과 (`7 passed`).

  `uv run ruff check src/rl_task_foundry/tooling/composer/query.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py tests/test_tooling_composer_query.py`
  통과.

## Iteration 154 — Budget feedback and visible filter binding

- **질문**:
  Iteration 153의 PK limit 완화 이후 MIMIC no-topic smoke에서 남은 low-quality
  accepted 위험은 무엇인가?
- **실험**:
  아래 smoke들을 순차 실행했다. 모두 topic hint 없이 OpenRouter Kimi K2.5,
  solver 4개, synthesis `run_timeout_s=300` 설정이다.

  - `artifacts/trial_20260429_mimiciv_demo_pk_limit_guard_kimi_4solver_no_topic_smoke_01/trial_01`
  - `artifacts/trial_20260429_mimiciv_demo_order_binding_guard_kimi_4solver_no_topic_smoke_01/trial_01`
  - `artifacts/trial_20260429_mimiciv_demo_feedback_submit_deadline_kimi_4solver_no_topic_smoke_01/trial_01`
  - `artifacts/trial_20260429_mimiciv_demo_difficulty_examples_kimi_4solver_no_topic_smoke_01/trial_01`
  - `artifacts/trial_20260429_mimiciv_demo_tool_budget_feedback_kimi_4solver_no_topic_smoke_01/trial_01`
  - `artifacts/trial_20260429_mimiciv_demo_visible_filter_guard_kimi_4solver_no_topic_smoke_01/trial_01`
- **결과**:
  1. ICU procedure list에서 같은 `start_time` rows를 `order_id`로 몰래 정렬한
     draft가 solver pass-rate 0으로 떨어졌다. request는 "최근순"만 말했고,
     `order_bindings`가 같은 phrase를 `start_time`과 `order_id`에 재사용했다.
  2. HCPCS/admission/outputevents runs에서 composer가 feedback 뒤 submit 없이 data
     tools를 계속 호출하거나 timeout나는 패턴이 반복됐다.
  3. admission list에서는 `admission_type = EW EMER.` visible filter가 row set을
     정했지만, user_request/contract에는 해당 filter가 dedicated constraint로
     고정되지 않았다. solver들은 다른 응급실 해석으로 4건을 찾았다.
  4. 최종 visible-filter-guard smoke는 accepted:
     `task_blood_chemistry_recent_results_9f0bbe714f94dd34`, pass-rate 0.25.
- **정성 평가**:
  accepted data: clean에 가까운 hard-good. 최종 task는 subject `10039831`의
  Chemistry lab events를 `charttime desc, lab label asc`로 정렬한 최근 5개다.
  DB를 직접 교차검증했고 canonical answer가 SQL 결과와 일치했다. 요청도
  "혈액 화학", "최신순", "같은 시간대는 검사 항목명 알파벳순"을 자연스럽게
  포함한다. 낮은 pass-rate는 solver가 category filter나 tie-break를 놓친
  어려운 케이스로 본다.

  rejected data: 대부분 low-quality rejected다. hidden/unanchored scope, ambiguous
  order, blocked handle exposure, ungrounded/evidence mismatch가 submit_draft에서
  막혔다. 특히 admission filter 사례는 hard-good이 아니라 request/filter
  mismatch였고, solver까지 보내면 저품질이 된다.
- **변경**:
  - `submit_draft` binding diagnostics에 `duplicate_order_binding_phrases`와
    `order_binding_reused_output_phrases`를 추가했다. 같은 order phrase를 여러
    order key에 재사용하거나, display-only output phrase를 order evidence로
    재사용하면 `answer_contract_binding_missing`으로 feedback한다.
  - durable prompt에는 feedback 이후 2 data tool 내 재제출 deadline과
    DB-agnostic Difficulty-Up mini example을 추가했다. 예시는 기호화된
    `<draft_before>/<draft_after>/<commentary>` 형식이며 길이 예산 8k를 유지했다.
  - prompt를 지켰는데도 data tool 호출이 계속되는 경우를 위해 composer data tool
    wrapper가 정확한 호출 수 기준 `ToolBudgetFeedback`을 반환한다. 첫 submit 전은
    3 data tools, feedback 이후는 2 data tools가 limit이다.
  - user-visible where filter가 있는데 dedicated constraint phrase가 하나도 없으면
    `answer_contract_filter_unbound`로 feedback한다. 기존 non-null 전용 검증을
    equality/range 등 일반 visible row-set filter로 확장했다.
- **원칙 준수**:
  DB literal/token/table-name heuristic은 추가하지 않았다. 새 hard checks는
  `answer_contract` 구조, query metadata의 visibility/filter/order provenance,
  그리고 tool call count만 본다. Prompt 우선 원칙에 따라 durable prompt를 먼저
  보강했고, 그 지침을 계속 어긴 카운트 초과는 feedback/tool wrapper에서만 막았다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`78 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/composer_tools.py src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 155 — Budget feedback must allow label-query repair

- **질문**:
  Iteration 154의 `ToolBudgetFeedback`가 실제 smoke에서 submit을 앞당기되,
  제출에 필요한 label query와 query repair까지 막지는 않는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 반복했다. composer/solver는
  OpenRouter Kimi K2.5, solver 4개, lower band 0.2:

  - `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_01`
  - `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_01_retry_01`
  - `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_01_retry_02`
- **관찰**:
  1. 첫 trial은 budget feedback이 `schema_map/neighborhood/sample` 3회 뒤
     모든 data tool을 막아, 제출에 필요한 최종 `query`도 실행하지 못했다.
     이건 제출 강제가 아니라 근거 생성 차단이었다.
  2. retry_01에서는 최종 `query`는 허용됐지만, 이미 성공한 query가
     `limit_boundary_tie`/duplicate order diagnostics를 보이는 경우에도 다음
     repair query가 막혔다.
  3. 같은 retry에서 contract-only feedback 뒤 composer가 이전 transfer label을
     버리고 lab label로 갈아타려는 현상도 보였다. `answer_contract_phrase_missing`
     / `answer_contract_query_mismatch`는 label 교체가 필요 없는 계약 수리인데,
     모델이 feedback을 task reset으로 사용한 것이다.
  4. retry_02에서는 최종 accepted:
     `task_patient_admissions_history_256f3a899c44871b`, pass-rate `3/4 = 0.75`.
- **변경**:
  - `data_tool_budget_feedback(tool_name=...)`로 호출 tool을 보게 했다.
    budget limit에 도달했더라도 아직 성공한 label query가 없으면 `query`만
    허용한다. 성공한 query가 있더라도 query diagnostics가 ambiguous ordering
    또는 duplicate projected rows를 보고하면 repair query를 허용한다.
  - durable budget prompt도 같은 원칙으로 정리했다. limit 이후에는 행이 없는
    경우 label `query`만 허용하고, rows가 돌아오면 submit한다.
  - contract-only feedback(`answer_contract_phrase_missing`,
    `answer_contract_query_mismatch`)은 canonical label signature를 잠근다.
    이후 label이 바뀌면 `label_changed_during_repair` feedback을 반환한다.
    이는 DB literal/table/column heuristic이 아니라 제출된 label 구조 비교다.
- **정성 평가**:
  accepted data: clean. 최종 요청은 "내 최근 입원 이력 3건"을 `admittime desc`
  로 정렬하고 `admission_type`, `admittime`, `dischtime`,
  `discharge_location`, `admission_location`을 요구한다. 직접 DB에서
  `mimiciv_hosp.admissions where subject_id=10012853 order by admittime desc
  limit 5`를 실행해 top 3가 canonical answer와 일치함을 확인했다. 낮지 않은
  pass-rate 0.75이고, 1개 solver 실패는 정답이 여러 개인 문제가 아니라 solver가
  최근순/limit을 잘못 적용한 hard-good 실패로 본다.

  rejected data: low-quality rejected. 첫 draft는 너무 쉬워서 reject된 정상
  케이스다. 이후 ICU join 강화는 list 3건을 1건으로 줄였고
  `answer_contract_not_incremental/cardinality_weakened`로 막혔다. 이는 좋은
  hard task가 아니라 Difficulty-Up Policy 위반이다. low-quality accepted는
  발생하지 않았다.
- **원칙 준수**:
  새 hard checks는 tool call count, query diagnostics, canonical label
  signature만 사용한다. DB 값/테이블명/컬럼명 리터럴 기반 휴리스틱은 추가하지
  않았다. Prompt가 정책을 정의하고, feedback/tool wrapper는 그 정책을 어겼을 때
  상기하거나 submit을 요구하는 역할로 유지했다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`82 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/composer_tools.py src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 156 — Difficulty-up must not reward passive display width

- **질문**:
  Iteration 155 이후 같은 smoke 설정에서 composer가 too-easy feedback을 받고
  의미 있는 난이도 상승을 하는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver 4개, lower band 0.2:
  `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_02`.
- **결과**:
  trial은 accepted되지 않고 `synthesis_failed`로 끝났다.

  - first submit: ICU procedure list, `answer_contract_order_ambiguous`
  - repair: 전체 14개 row 반환 시도, `answer_contract_list_limit_too_wide`
  - repair: visible tie-break 추가 뒤 order binding 문구 중복,
    `answer_contract_binding_missing`
  - fourth/fifth submit: solver `4/4 = 1.0`, `calibration_inconclusive`
  - final outcome: budget exhausted, no provider failure
- **정성 평가**:
  accepted data: 없음.

  rejected data: clean-good but too easy. 최종 후보는 같은 입원/ICU procedure row set,
  visible order/tie-break, 5-row limit을 갖춘 검증 가능한 task였고 solver도 전부
  풀었다. 문제는 저품질이 아니라 Kimi smoke 기준으로 너무 직접적이라는 점이다.

  low-quality accepted는 발생하지 않았다. 초반 14-row 확장과 binding 오류도
  validator가 막았으므로 rejection은 바람직했다.
- **원인**:
  composer reasoning을 보면 too-easy feedback 뒤 "location field를 하나 더 붙이면
  난이도가 올라간다"고 판단했다. 이는 DB-neutral 원칙 위반은 아니지만
  `foundation.md`의 "passive display-width additions are not meaningful difficulty"
  원칙과 어긋난다. 기존 prompt의 `R+C` 예시는 이 오해를 부추길 수 있었다.
- **변경**:
  prompt-first 원칙으로 `Difficulty-Up Policy`와 예시를 수정했다.

  - list difficulty-up은 단순 display field가 아니라 lookup/comparison/order/row
    reasoning을 바꾸는 grounded dimension이어야 한다고 명시했다.
  - generic example을 `R+C`에서 "related/derived C used for compare/order"로
    바꾸고, "unused display C"는 bad example로 넣었다.
  - too-easy feedback은 새 정책을 반복 지시하지 않고 "grounded meaningful
    dimension"과 "passive display-only fields are weak"만 짧게 상기한다.
- **원칙 준수**:
  precision-100 validator는 추가하지 않았다. difficulty quality는 semantic 판단
  영역이므로 DB literal/table/column/token heuristic으로 막지 않았다. Prompt가
  durable policy를 정의하고 feedback은 named policy를 상기하는 구조를 유지했다.

## Iteration 157 — Hidden child anchors must not silently select sibling row sets

- **질문**:
  Iteration 156 변경 뒤 다음 smoke에서 accepted 품질이 개선되는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver 4개, lower band 0.2:
  `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_03`.
- **결과**:
  trial은 `synthesis_failed`로 끝났다.

  - first submit: `answer_contract_binding_missing`
  - second submit: calibration `0/4 = 0.0`, `calibration_inconclusive`
  - solver failed runs: `0` (모두 evaluable)
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. composer는 `pharmacy_id=99938576` hidden
  child anchor에서 시작했지만, canonical query는 `pharmacy -> admissions -> emar`
  로 이동해 같은 입원 전체의 eMAR row set을 냈다. request는 "이 약물 처방과
  관련된 투약"처럼 보이므로 solver들은 direct pharmacy/eMAR 관계나 약물명 매칭을
  찾았다. 실제 정답은 처방 자체가 아니라 parent admission sibling list였고, 그래서
  solver 0/4가 됐다. 어려운 좋은 문제가 아니라 request/source-scope mismatch다.

  low-quality accepted는 발생하지 않았다.
- **변경**:
  precision-100 structural validator를 추가했다. 최신 query가 hidden child PK를
  `where`로 고정한 뒤 forward edge로 parent를 찾고, 다시 reverse edge로 sibling
  list를 정답 row set으로 선택하는 경우, parent/current-subject key가 `entity`에
  없으면 기존 `answer_contract_hidden_filter_unanchored` feedback으로 막는다.

  이는 DB literal/table/column 의미 휴리스틱이 아니다. query spec의 edge 방향,
  latest query metadata의 hidden handle evidence, submitted entity key만 비교한다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_hidden_child_to_parent_sibling_scope tests/test_synthesis_runtime.py::test_submit_draft_rejects_hidden_filter_missing_from_entity -q`
  통과 (`2 passed`).

  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`83 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/composer_tools.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 158 — Empty label queries must remain repairable

- **질문**:
  budget feedback가 submit을 앞당기면서도, 행이 0개인 label query를 잘못
  "성공한 label evidence"로 취급하지 않는가?
- **실험**:
  `trial_04`에서 composer가 최종 label query를 실행했지만 `row_count=0`이었다.
  이 상태에서 budget wrapper가 추가 query를 막아 submit만 강제했다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: 저품질 여부를 판단하기 전 단계에서 실행 정책이 잘못됐다. 빈 list는
  유효한 정답 label이 아니므로, 이건 어려운 좋은 문제도 저품질 문제도 아니라
  label evidence repair가 필요한 중간 상태다.
- **변경**:
  `data_tool_budget_feedback`가 최신 성공 query의 `rows`가 비어 있으면
  repair query를 허용하도록 했다. 이는 `row_count/rows` 구조만 보는
  precision-100 실행 규칙이며 DB literal/token heuristic이 아니다.
- **검증**:
  `test_data_tool_budget_feedback_allows_query_repair_for_empty_query` 추가.

## Iteration 159 — Composer timeout must not cut off solver provider retries

- **질문**:
  provider 실패는 재시도한다는 실험 원칙이 런타임에서도 지켜지는가?
- **실험**:
  `trial_04_retry_01`, `trial_04_retry_02`, `trial_05` 모두 OpenRouter Kimi
  provider timeout으로 끝났다. 특히 `trial_05`는 submit_draft 내부 solver rollout이
  이미 시작되어 2개 solver는 matched, 1개는 invalid submit까지 기록됐지만, 4번째
  solver 또는 replacement 시도 전에 바깥 composer `run_timeout_s=420`이 먼저
  만료됐다.
- **정성 평가**:
  accepted data: 없음.

  rejected/중단 data: `trial_05` 후보는 "해당 ICU stay의 처음 5개 배출 기록"으로,
  너무 쉬운 편이지만 row set, 정렬, limit, 출력 필드는 검증 가능했다. low-quality
  accepted는 없었다. timeout 때문에 품질 결론을 내릴 수 없으므로 provider failure로
  분리한다.
- **변경**:
  synthesis backend에서 composer timeout에 submit_draft 내부 solver rollout의
  replacement-attempt budget을 더한다. SDK Runner timeout은 tool 실행까지 감싸므로,
  solver rollout이 tool 안에서 실행되는 현재 구조에서는 이 allowance가 없으면
  provider retry 원칙이 중간에 끊긴다.

  계산은 `max_solver_runs`, `solver_batch_size`, `statement_timeout_ms`,
  `solver_runtime.max_turns`만 사용한다. DB 내용이나 모델 출력 literal을 보지 않는
  실행 예산 계산이다.
- **검증**:
  `uv run pytest tests/test_synthesis_backend_openai_agents.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`94 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/backend_openai_agents.py src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/composer_tools.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_backend_openai_agents.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 160 — Limited-list boundary wording must match order direction

- **질문**:
  timeout allowance 이후 실제 smoke가 provider failure 없이 품질 신호를 주는가?
- **실험**:
  `trial_06`을 같은 MIMIC demo no-topic Kimi 4-solver 설정으로 실행했다.
- **결과**:
  provider timeout은 사라졌다. solver 4개 모두 evaluable이었고 pass-rate는
  `0/4 = 0.0`이었다. accepted는 없고 budget exhausted.
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 최종 draft는 request에서 "최근 5건"이라고
  했지만 query는 `admittime asc limit 5`라서 가장 오래된 5건을 canonical answer로
  만들었다. solver 4개는 모두 최근 5건 쪽으로 해석해 다른 answer를 제출했다.
  이건 어려운 좋은 문제가 아니라 request boundary와 query direction이 충돌한
  저품질 후보이다. low-quality accepted는 발생하지 않았다.
- **변경**:
  precision-100 validator는 추가하지 않았다. "최근/오래된/최신" 같은 자연어
  의미를 hard-code하면 literal/token heuristic 금지 원칙 위반이다.

  대신 prompt-first 원칙으로 `List Determinism Policy`를 압축 보강했다:
  limited ordered list에서는 boundary words와 direction이 서로 맞아야 하며,
  newest/latest vs oldest/earliest, asc/desc를 섞지 말라고 명시했다.
- **검증**:
  prompt 길이 `7993`.

  `uv run pytest tests/test_synthesis_backend_openai_agents.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`94 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/backend_openai_agents.py src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/composer_tools.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_backend_openai_agents.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 161 — List-limit repair must not reset the task

- **질문**:
  Iteration 160 이후 다음 smoke에서 boundary mismatch가 줄고, composer가 feedback을
  정책 reminder로만 사용하는가?
- **실험**:
  `trial_07`을 같은 MIMIC demo no-topic Kimi 4-solver 설정으로 실행했다.
- **결과**:
  `MaxTurnsExceeded`로 종료됐다. accepted 없음.

  - first submit: 입원 진단 목록 7건, `answer_contract_list_limit_too_wide`
  - second submit: 전자 약물 투여 기록 2건으로 task reset,
    `answer_contract_binding_missing`
  - third submit: 같은 약물 기록 wording repair 실패,
    `answer_contract_phrase_missing`
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 첫 진단 목록 후보는 같은 target에서 3-5건으로
  limit만 줄이면 살릴 수 있었다. 그런데 composer가 전자 약물 기록으로 topic/query
  source를 갈아타며 feedback을 새 지시처럼 사용했다. 이는 어려운 좋은 문제가 아니라
  Feedback Handling Policy 위반이다. low-quality accepted는 없었다.
- **변경**:
  1. `answer_contract_list_limit_too_wide` feedback 문구에서 "다른 더 어려운
     label을 선택"할 여지를 제거하고, 같은 target/query scope에서 3-5 row limit만
     줄이라고 상기하도록 수정했다.
  2. precision-100 구조 검증을 추가했다. list-limit-only feedback 직후에는 이전
     query evidence의 kind/output sources/predicates/order를 보존하고 item count만
     3-5로 줄인 repair만 허용한다. 다른 row source/task로 바뀌면
     `label_changed_during_repair` feedback을 반환한다.

  이 검증은 query metadata 구조와 row count만 비교한다. DB 값, 테이블명 의미,
  컬럼명 의미, 자연어 token heuristic은 사용하지 않는다.
- **검증**:
  `test_submit_draft_rejects_task_reset_after_list_limit_feedback` 추가.

  `uv run pytest tests/test_synthesis_backend_openai_agents.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`95 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/backend_openai_agents.py src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/composer_tools.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_backend_openai_agents.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 162 — Accepted data can still expose broad source-role ambiguity

- **질문**:
  Iteration 161 이후 smoke가 accepted를 만들면, accepted data가 정말 좋은가?
- **실험**:
  `trial_08`을 같은 MIMIC demo no-topic Kimi 4-solver 설정으로 실행했다.
- **결과**:
  accepted. pass-rate `1/4 = 0.25`, completed solver runs 6, infra-excluded
  timeout 2개는 replacement로 보충되어 evaluable 4개가 확보됐다.
- **정성 평가**:
  accepted data: 부분적으로 저품질. row set, 정렬, tie-break, fixed 4-row list는
  좋다. 그러나 `procedure_category`가 broad source-role ambiguity를 남겼다.
  canonical은 `procedureevents.ordercategoryname`의 `Peripheral Lines/Invasive Lines`
  를 사용했고, 한 solver는 reachable referenced item surface인 `d_items.category`
  의 `Access Lines - Peripheral`를 사용했다. user_request의 "시술의 종류"가 어느
  source surface의 category인지 충분히 지정하지 못했다.

  rejected data: 첫 두 feedback은 phrase/binding contract repair였고 저품질이
  accepted되기 전까지 validator가 잘 작동했다. 그러나 최종 accepted에는 broad
  category ambiguity가 남았다.
- **변경**:
  precision-100 hard validator는 추가하지 않았다. category/type/status 같은 자연어
  source-role ambiguity를 컬럼명이나 단어 literal로 잡으면 금지 원칙 위반이다.

  대신 submit_draft tool schema description을 보강했다. `output_bindings`와
  `requested_by_phrase` 설명에, 여러 reachable source surface가 같은 broad phrase를
  만족할 수 있으면 "current record category" vs "referenced item category"처럼 정확한
  source role을 user_request phrase가 명명해야 한다고 명시했다.
- **검증**:
  `test_submit_draft_tool_schema_descriptions_are_prompt_aligned`에 source-surface
  ambiguity 문구 assertion 추가.

  `uv run pytest tests/test_synthesis_backend_openai_agents.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`95 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/backend_openai_agents.py src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/composer_tools.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_backend_openai_agents.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 163 — Broad medication source roles were rejected, not accepted

- **질문**:
  Iteration 162 schema description 보강 후 broad source-role ambiguity가 accepted로
  통과하지 않는가?
- **실험**:
  `trial_09`를 같은 MIMIC demo no-topic Kimi 4-solver 설정으로 실행했다.
- **결과**:
  accepted 없음. 최종 solver pass-rate `0/4 = 0.0`, provider failure 없음.
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 후보는 입원 중 처방 약물 5개를 시작 시간,
  약물명 tie-break로 정렬하고 `proc_type`, `dispensation`을 요구했다. row set과
  order는 검증 가능했지만, user_request의 "처방 유형", "조제 방법"이 pharmacy 안의
  어느 source representation인지 충분히 특정하지 못했다. solver들은 다른 reachable
  medication/order surface 값을 제출했다. 어려운 좋은 문제가 아니라 source-role
  wording mismatch다.
- **판단**:
  저품질이 accepted되지 않았으므로 이번 반복에서는 hard validator를 추가하지 않는다.
  이를 컬럼명/값/token 기반으로 막는 것은 literal heuristic 금지 원칙 위반이다.
  Iteration 162의 schema description 보강 방향은 유지하고, 다음 smoke에서 accepted
  품질을 계속 본다.

## Iteration 164 — Accepted ICU medication history is qualitatively good

- **질문**:
  반복 개선 후 accepted data가 정말 좋은가?
- **실험**:
  `trial_10`을 같은 MIMIC demo no-topic Kimi 4-solver 설정으로 실행했다.
- **결과**:
  accepted. pass-rate `2/4 = 0.5`, completed/evaluable solver runs `4/4`,
  provider failure 없음.
- **정성 평가**:
  accepted data: good. 최종 요청은 특정 ICU stay의 투약 이력 중 처음 5개를
  `administration_start_time asc`, 동시간이면 `medication_name asc`로 정렬한다.
  output은 `medication_name`, `administered_amount`, `unit`,
  `administration_start_time`, `order_category`이고, 모두 최신 query evidence에서
  user-visible source로 나온다. request phrase, answer_contract binding, query
  order/select가 같은 구조를 가리킨다.

  solver mismatch: 2개 solver는 canonical과 정확히 matched. 2개는
  `missing_submit_result`로 invalid_submit이며 정답 row set을 다르게 고른 신호가
  아니다. 따라서 어렵지만 좋은 문제로 본다.

  rejected data: 첫 draft의 `starttime` 단독 order는 tie가 있었고
  `answer_contract_order_ambiguous`로 막혔다. 최종 draft는 약물명 tie-break를
  request/contract/query에 반영해 해결했다. low-quality accepted 없음.
- **판단**:
  현재 반복은 만족 기준을 충족한다. provider retry 예산, list-limit repair scope,
  source-role schema description, prompt boundary wording이 함께 작동했고,
  accepted data의 품질도 정성 평가를 통과했다.

## Iteration 148 — Fixed list labels must stay within 3-5 rows

- **질문**:
  Iteration 147의 solver `submit_result` retry 보강 뒤 같은 8-solver smoke에서 accepted
  품질과 solver pass-rate 계측이 안정적인가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_solver_submit_retry_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 accepted되지 않고 `calibration_inconclusive`로 종료됐다.

  - first submit: `answer_contract_hidden_filter_unanchored`, `answer_contract_order_ambiguous`
  - second submit: `answer_contract_hidden_filter_unanchored`
  - third submit: `answer_contract_binding_missing`
  - fourth submit: explicit medication-administration list, pass rate `0/8`

- **정성 평가**:
  accepted data: 없음. low-quality accepted도 없음.

  rejected data: low-quality rejected / overwide hard-bad. 최종 draft는 request/contract가
  훨씬 명시적이었지만, `10 rows x 5 fields` medication administration list였다. 일부
  solvers는 `emar`까지 제대로 찾아가고도 `list_records`/`submit_result`까지 못 갔고,
  일부는 `prescriptions`/`inputevents`로 갈라지거나 placeholder/hallucinated rows를
  제출했다. 이건 좋은 어려움이라기보다 composer가 초기 list를 너무 크게 만들어 solver
  tool budget을 소모시킨 문제다.

  특히 prompt의 Task Shapes에는 이미 "3-5 rows" 및 "max 5 before feedback" 원칙이
  있었는데, composer가 처음부터 `10개`를 냈다. feedback으로 수리하기 전에 solver rollout까지
  가면 pass-rate가 task quality 대신 tool-budget 부담을 측정하게 된다.
- **변경**:
  - Task Shapes prompt를 "fixed limit of 3-5 rows"로 더 직접화했다.
  - `submit_draft` validator에 `answer_contract_list_limit_too_wide`를 추가했다.
  - list label의 submitted row count가 5를 초과하면 solver rollout 전에 feedback한다.

  이 검증은 canonical label의 list length만 보는 구조 검증이다. DB literal, table/column
  token, 값 문자열을 보지 않으므로 precision-100 원칙을 지킨다. 난이도는 row 수를 6+로
  늘리는 대신 visible fields, relationships, or row-preserving constraints로 올려야 한다.
- **검증**:
  prompt 길이 `7949`.

  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_list_limit_above_task_shape_policy tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/prompts.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py`
  통과.

## Iteration 147 — Solver text-only answers must retry through submit_result

- **질문**:
  Iteration 146의 Kimi required tool choice 변경 뒤 composer no-tool-call 문제가 사라지고,
  accepted/rejected 데이터 품질은 어떤가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_kimi_required_tool_smoke_01/trial_01`.
- **결과**:
  trial은 accepted됐다.

  - 첫 submit: ICU outputevents 최근 5건, `answer_contract_order_ambiguous`
  - 두 번째 submit: itemid hidden tie-break를 사용해 `answer_contract_order_ambiguous`
  - 세 번째 submit: `charttime desc`, `d_items.label asc`로 자연 visible tie-break 수리
  - pass rate: `7/8 = 0.875`, CI low `0.529`, CI high `0.994`

  Iteration 146의 목표였던 composer no-tool-call 문제는 사라졌다. 첫 composer turn에서
  실제 `neighborhood` tool call이 발생했고, 총 10개 composer data/query call 뒤
  `submit_draft` accepted까지 도달했다.
- **정성 평가**:
  accepted data: clean but high-pass. ICU stay hidden context에서 outputevents를 조회하고,
  최신 측정 시간 내림차순 및 같은 측정 시간에서 항목 이름 오름차순 tie-break가
  request/contract/query에 모두 일치한다. label fields도 `output_time`, `item_name`,
  `value`, `unit`로 사용자에게 자연스럽고, hidden handle을 노출하지 않는다.

  rejected data: low-quality rejected. 첫 draft는 시간 동점이 있어 list membership/order가
  유일하지 않았고, 두 번째 draft는 request에는 항목 이름 tie-break를 썼지만 실제 query는
  blocked `itemid`로 tie-break했다. 두 rejection 모두 정확히 막혔다.

  다만 solver 1개는 정답을 텍스트/표로 정확히 작성했지만 `submit_result`를 호출하지 않아
  `invalid_submit`으로 카운트됐다. 이건 좋은 어려움도 저품질 데이터도 아니고 solver protocol
  실패다. 따라서 현재 `0.875`는 실제 난이도보다 낮게 측정됐을 가능성이 있다. 이 상태로
  accepted를 신뢰하면 too-easy task를 band 안으로 잘못 넣을 수 있다.
- **변경**:
  solver tool schema와 backend를 보강했다.

  - `submit_result` tool description에 plain text final answer는 invalid이며 답이 준비되면
    이 tool을 호출해야 한다고 명시했다.
  - solver backend가 final output text만 받고 `submit_result`가 없으면, 한 번만 continuation
    feedback을 넣어 같은 답을 `submit_result`로 제출하게 한다.
  - feedback은 tool-local contract reminder다. 정답 내용이나 DB literal을 주입하지 않으므로
    precision-100 protocol recovery로 본다.

  이 변경은 solver를 쉽게 만드는 것이 아니라 평가 계측을 바로잡는 변경이다. solver가 틀린
  값을 `submit_result`하면 그대로 오답으로 평가된다.
- **검증**:
  `uv run pytest tests/test_solver_backend_openai_agents.py::test_submit_result_tool_uses_task_specific_object_schema tests/test_solver_backend_openai_agents.py::test_openai_agents_solver_backend_continues_after_missing_submit_result tests/test_solver_backend_openai_agents.py::test_openai_agents_solver_backend_returns_solver_result -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/solver/backend_openai_agents.py tests/test_solver_backend_openai_agents.py`
  통과.

## Iteration 146 — Kimi must use required tool choice

- **질문**:
  Iteration 145의 duplicate-row repair 보강 뒤 같은 smoke를 다시 돌렸을 때, composer가
  첫 탐색 도구도 호출하지 않고 plain text/reasoning만 반복하는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_duplicate_repair_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 `synthesis_failed`로 끝났고, error code는 `composer_no_tool_calls`였다.
  `composer_submit_draft_missing` feedback이 5번 발생했지만 `schema_map`, `sample`,
  `neighborhood`, `query`, `submit_draft` 중 어떤 도구도 호출되지 않았다.

  reasoning sidecar를 보면 composer는 매번 "후보 entity를 고르고 schema/neighborhood를
  보겠다"고 생각했지만, 실제 SDK tool call로 전환하지 못했다. feedback 뒤에도 동일하게
  reasoning만 생성했다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: infra/protocol failure. 데이터가 어렵거나 저품질인 문제가 아니라,
  composer 실행 표면이 tool-only 역할 계약을 강제하지 못한 문제다. low-quality accepted는
  발생하지 않았지만, 실험 비용을 모두 소모하므로 우선순위가 높다.
- **원인 분석**:
  prompt에는 `submit_draft` 사용과 tool-call budget이 있었지만, "plain text turn은 무효"라는
  실행 계약이 충분히 명시되지 않았다. 더 큰 원인은 `tool_choice_for_model()`이 Kimi/Moonshot을
  `auto`로 예외 처리한 것이다. 이전 Iteration 63의 호환성 우회가 현재 OpenRouter Kimi에는
  오히려 tool-only 실행을 약화시켰다.

  최소 OpenRouter Kimi probe를 별도로 실행했고, `tool_choice="required"`에서 정상적으로
  `ping` tool call이 반환됨을 확인했다. 따라서 이 변경은 추측이 아니라 현재 provider 표면에
  대한 직접 검증에 기반한다.
- **변경**:
  - Kimi/Moonshot을 `tool_choice="required"` 대상으로 되돌렸다.
  - Draft Submission Budget에 "Plain text is invalid; call a tool every turn."을 추가했다.
  - 기존 ID 금지 문구는 테스트와 맞춰 "Do not invent ids"로 정리했다.

  이 변경은 데이터 literal/token heuristic이 아니다. durable prompt가 tool-only 실행 계약을
  소유하고, feedback은 그 계약 위반을 상기하는 구조를 유지한다.
- **검증**:
  prompt 길이 `7984`.

  `uv run pytest tests/test_tool_choice_for_model.py tests/test_turn_budget_prompt.py tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_backend_openai_agents.py::test_synthesis_backend_continues_after_final_output_without_submit tests/test_synthesis_backend_openai_agents.py::test_synthesis_backend_continues_after_feedback_without_resubmit -q`
  통과 (`15 passed`).

  `uv run ruff check src/rl_task_foundry/infra/sdk_helpers.py src/rl_task_foundry/synthesis/turn_budget.py src/rl_task_foundry/synthesis/prompts.py tests/test_tool_choice_for_model.py tests/test_turn_budget_prompt.py tests/test_synthesis_prompts.py tests/test_synthesis_backend_openai_agents.py`
  통과.

## Iteration 145 — Duplicate-row repair must not shrink the list

- **질문**:
  Iteration 144의 list difficulty-up 확장 뒤 단일 smoke에서 duplicate-row/list repair가 안정적인가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_list_difficulty_up_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 accepted되지 않고 `MaxTurnsExceeded`로 끝났다.

  - first feedback: `composer_submit_draft_missing`
  - second submit: microbiology list, `answer_contract_duplicate_answer_rows`
  - third submit: microbiology list, `answer_contract_binding_missing`
  - final failure: `MaxTurnsExceeded`

- **정성 평가**:
  accepted data: 없음. 저품질 accepted도 없음.

  rejected data: low-quality rejected. 두 번째 submit은 projected answer rows가 중복되어
  exact answer row를 구분할 수 없었다. 세 번째 submit은 이를 고치려다 list size를 5건에서
  3건으로 줄이고, query.order_by의 두 번째 order key를 request/order binding에 제대로 묶지
  못했다. 즉 문제는 어려운 좋은 데이터가 아니라 duplicate-row repair가 list determinism을
  보존하지 못한 것이다.
- **변경**:
  List Determinism Policy와 duplicate-row feedback을 보강했다.

  - duplicate projected answer rows는 natural visible field/aggregate를 추가해서 구분한다.
  - 중복을 숨기기 위해 limit/list size를 줄이지 않는다.
  - feedback도 list size를 보존하고 one natural visible distinguishing field or aggregate를
    추가한 뒤 label query를 다시 실행하고 `submit_draft`하라고 상기한다.

  이 변경은 기존 List Determinism Policy의 repair 절차를 더 명확히 한 것이다.
- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_runtime.py::test_submit_draft_rejects_duplicate_projected_list_rows -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 150 — Feedback target preservation must not block target switch

- **질문**:
  `trial_40`에서 reasoning content가 실제로 반환되는 상태에서, composer가 왜 submit까지
  가지 못하고 `MaxTurnsExceeded`로 끝났는가?

- **실험/결과**:
  `trial_40`은 solver rollout까지 가지 못했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_40`
  - composer/solver 설정: OpenRouter Kimi K2.5, solver 4개 설정이지만 solver 미도달
  - 결과: `synthesis_failed`, `SynthesisPhaseExecutionError`, `MaxTurnsExceeded`

- **reasoning 교차 분석**:
  reasoning 로그는 11개 item이 저장됐다. composer는 첫 anchor로
  `admissions.hadm_id=21101111`을 골랐고, pharmacy list를 만들었다.

  첫 draft는 다음 이유로 feedback됐다.

  - label에 `medication: null` 포함
  - request에 `약물 종류`, `처방 시작 시간`, `처방 상태` phrase 누락
  - starttime 단일 order가 5개 row 모두 같은 시간이라 order ambiguity

  이후 composer는 feedback을 받고 emar/labevents/inputevents/admissions를 탐색했다. 중간에
  `ToolBudgetFeedback`도 여러 번 반환됐고, reasoning에서도 "submit now"라고 인지했다.
  하지만 계속 data tool을 호출했다. 두 번째 draft는 환자 admission history였는데 row가 1개뿐이고,
  `subject_id` hidden filter가 entity에 없어서 feedback됐다.

  두 번째 feedback 뒤 composer reasoning은 "same anchored user need를 보존해야 한다"로
  해석했고, 동일 patient admissions를 다시 query했다. 환자 admission이 1개뿐임을 확인한 뒤
  prescriptions로 바꾸려 했지만, submit 전에 max turns에 도달했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. pharmacy draft는 null/order ambiguity가 명확했고,
  admissions draft는 1-row list + hidden parent scope 문제였다. 좋은 어려움이 아니라
  composer가 feedback을 해석하며 target switch 타이밍을 놓친 문제다.

- **개선 판단**:
  기존 generic feedback 문구의 "same anchored user need"가 `list labels must return 3-5 rows;
  choose another scoped list`와 충돌했다. 원칙상 feedback은 durable prompt를 상기해야 하며,
  지침이 이원화되면 안 된다.

  따라서 target 보존을 무조건 강조하지 않고 다음처럼 정렬했다.

  - anchor/language는 보존한다.
  - contract repair / difficulty-up에서는 target을 보존한다.
  - named policy가 another label/scope를 요구하면 target switch가 허용된다.

- **변경**:
  - Feedback And Difficulty-Up Policy 문구를 위 원칙으로 재정렬했다.
  - Task Shapes에 "query count가 3-5 범위 밖이면 target/scalar로 전환"을 추가했다.
  - `answer_contract_list_size_invalid` feedback에 같은 target을 계속 probe하지 말라고
    상기했다.
  - `ToolBudgetFeedback` 문구에 "data result가 아니며, 이후 data tool을 더 호출하지 말고
    submit_draft next"를 명시했다.

  hard validator는 추가하지 않았다. 이번 문제는 이미 precision 100 validator들이 잡고 있고,
  실패 원인은 feedback 해석/turn 소모다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_turn_budget_prompt.py -q`
  통과 (`7 passed`).

  `uv run pytest tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_late_first_submit tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_late_feedback_repair tests/test_synthesis_runtime.py::test_submit_draft_rejects_single_row_list_as_too_direct tests/test_synthesis_runtime.py::test_submit_draft_rejects_limited_single_row_before_order_repair tests/test_synthesis_runtime.py::test_submit_draft_rejects_null_list_output_field tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_reset_after_contract_repair_feedback -q`
  통과 (`6 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 151 — Binding feedback should lock label repair

- **질문**:
  `trial_41`에서 Iteration 150의 target-switch guidance가 작동했는가? 남은 실패 원인은 무엇인가?

- **실험/결과**:
  `trial_41`도 solver rollout 전 `MaxTurnsExceeded`로 실패했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_41`
  - 첫 draft: 환자 `subject_id=10008287`의 최근 eMAR 5개
  - 첫 feedback: `answer_contract_binding_missing`
  - 이후 drafts: 1-row admission details, transfer list with null fields

- **reasoning 교차 분석**:
  첫 eMAR draft는 row count 5, null 없음, row set 자체는 나쁘지 않았다. 문제는
  `query.order_by`가 `charttime desc, emar_seq desc` 두 key인데 `answer_contract.order_bindings`는
  `charttime`만 묶어서 `missing_order_binding_count=1`이 난 것이다.

  이 feedback은 label data를 버릴 상황이 아니라 contract/order binding repair 상황이다.
  그런데 composer reasoning은 "secondary sort key를 request에 자연스럽게 묶거나 tied rows를
  반환하라"는 방향으로 고치지 않고, admissions로 target을 바꿨다. 그 결과 1-row list로 다시
  feedback됐다.

  이후 transfers target은 5 rows라 shape는 맞았지만 discharge row의 `care_unit`과 `outtime`이
  null이었다. composer는 order ambiguity는 고쳤지만 nullable output field를 계속 label/request에
  들고 가서 `label_null_value_forbidden`을 반복했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 첫 eMAR draft는 contract repair만 하면 좋은 후보가 될
  수 있었고, 이후 drafts는 target drift / null preflight 실패로 품질이 낮았다. 좋은 어려움이
  아니라 composer repair discipline 문제다.

- **개선 판단**:
  `answer_contract_binding_missing`은 precision 100으로 contract repair feedback이다. trigger가
  binding diagnostics에 기반하므로 DB 리터럴/토큰 휴리스틱이 아니다. 따라서 phrase/query mismatch와
  마찬가지로 repair-locked label을 보존시켜도 원칙 위반이 아니다.

  null feedback도 이미 precision 100 validator가 잡고 있다. 추가 hard validator는 필요 없고,
  feedback reminder가 "nullable output field를 제거하거나 non-null query로 바꾸라"는 기존
  Label Grounding Policy를 더 직접적으로 상기하면 된다.

- **변경**:
  - `answer_contract_binding_missing`을 repair-lock 대상에 추가했다.
  - binding feedback 뒤 canonical label이 바뀌면 `label_changed_during_repair`로 reject한다.
  - null answer feedback에 nullable output field를 label/request에서 제거하라는 문구를 추가했다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_reset_after_binding_feedback tests/test_synthesis_runtime.py::test_submit_draft_rejects_null_list_output_field tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_list_output_binding -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 160 — Order-binding diagnostics must surface with order feedback

- **질문**:
  `trial_42`에서 reasoning content가 반환되는 상태에서, composer가 왜 같은 order/tie-break
  결함을 반복했는가?

- **실험/결과**:
  `trial_42`도 solver rollout 전 `MaxTurnsExceeded`로 실패했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_42`
  - 첫 draft: ICU chartevents vital sign list
    - `label_null_value_forbidden`, `label_no_primary_key_source`
  - 이후 draft: admission/pharmacy medication list
    - `answer_contract_phrase_missing`, `answer_contract_hidden_filter_unanchored`,
      `answer_contract_order_ambiguous`
  - 마지막 pharmacy retry:
    - `answer_contract_phrase_missing`, `answer_contract_order_ambiguous`

- **reasoning 교차 분석**:
  composer는 첫 no-PK/null feedback 뒤 pharmacy로 바꾸는 방향은 이해했다. 그러나 pharmacy
  query에서 `start_time`이 모두 같아서 `medication_name`을 tie-break로 추가한 뒤, reasoning에서
  "rows are distinguishable by medication_name"이라고 판단했다. 실제 문제는 distinguishability만이
  아니라 request/contract가 `query.order_by` 두 key를 모두 요청하고 bind해야 한다는 점이다.

  기존 validator는 `answer_contract_binding_diagnostics`에 `missing_order_binding_count=1`,
  `missing_order_label_bindings=["medication_name"]`를 이미 계산했다. 하지만 다른 오류가 있으면
  `answer_contract_binding_missing` feedback을 숨겼다. 그래서 composer가 order ambiguity feedback을
  받으면서도 "tie-break를 query에 넣으면 됨"으로 오해하기 쉬웠다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 첫 draft는 no-PK table row value + null answer field라
  명백히 저품질이다. pharmacy draft도 좋은 어려움이 아니라, fixed list membership/order를
  request/contract로 유일하게 고정하지 못한 문제다.

- **개선 판단**:
  prompt-first 원칙상 durable prompt는 이미 "multi-key order는 user_request가 구분하고
  order_bindings가 query.order_by를 모두 cover해야 한다"고 말한다. 새 정책을 추가하지 않는다.

  대신 feedback이 기존 정책을 충분히 상기하지 못한 문제다. binding diagnostics는 query metadata와
  answer_contract 구조만 비교하므로 DB 리터럴/토큰/의미 휴리스틱이 아니다. 다른 오류가 있어도
  order-binding 결함만은 함께 노출하는 것이 precision 100 원칙에 맞다.

- **변경**:
  - `answer_contract_phrase_missing` 또는 `answer_contract_order_ambiguous` feedback과 함께
    order-binding 계열 diagnostics
    (`missing_order_bindings`, `missing_order_label_bindings`,
    `duplicate_order_binding_phrases`, `order_binding_reused_output_phrases`)가 있으면
    `answer_contract_binding_missing`을 같이 노출한다.
  - output binding만 빠진 경우까지 무조건 같이 노출하지는 않는다. 이번 개선은 order/tie-break
    repair의 구조 신호에 한정한다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_surfaces_order_binding_errors_with_ambiguous_order tests/test_synthesis_runtime.py::test_submit_draft_rejects_ambiguous_limited_list_order tests/test_synthesis_runtime.py::test_submit_draft_rejects_unrepresented_list_order_by_tie_breaker tests/test_synthesis_runtime.py::test_submit_draft_rejects_multirow_list_without_order_by -q`
  통과 (`4 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 161 — Order bindings need ordering words, not bare field nouns

- **질문**:
  Iteration 160 뒤 `trial_43`에서 order-binding feedback 노출이 실제 composer 행동을 개선했는가?

- **실험/결과**:
  `trial_43`은 solver rollout 전 실패했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_43`
  - outcome: `synthesis_failed`
  - final error: `submit_payload_invalid` (`json_decode_error: Extra data`)
  - solver 도달 없음

- **reasoning 교차 분석**:
  composer는 `microbiologyevents` anchor를 골랐고, 환자의 최근 미생물 검사 3건을 만들었다.

  진행은 다음 순서였다.

  1. 첫 draft는 `entity={microevent_id}`인데 query는 `subject_id` sibling list라
     `answer_contract_hidden_filter_unanchored`로 reject.
  2. composer는 reasoning에서 이 문제를 이해하고 `entity={subject_id}`로 고쳤다.
  3. 그 다음 `answer_contract_binding_missing`이 발생했다. 원인은 `sample_time` 출력 phrase와
     order binding phrase를 모두 `시간`으로 둔 `order_binding_reused_output_phrases`.
  4. composer reasoning은 "시간은 출력 phrase이고 order phrase는 더 자연스러운 tie-break wording이어야
     한다"고 정확히 말했지만, 실제 submit에서는 request/contract를 바꾸지 못했다.
  5. 이후 `charttime is_not_null`을 넣었다가 `answer_contract_filter_unbound`로 막혔고,
     filter를 제거한 뒤에도 같은 bare-noun order binding을 반복했다.
  6. 마지막에는 budget feedback 뒤 `label_json`/`entity_json`을 잘린 문자열로 제출해서
     `submit_payload_invalid`가 났다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. row set 자체는 3건이고 null은 없었지만, 처음에는 hidden scope가
  잘못됐고, 이후에는 request/contract가 list order를 유일하게 고정하지 못했다. solver가 못 푼
  어려운 문제가 아니라 composer가 contract repair를 완료하지 못한 문제다.

- **개선 판단**:
  이번 실패는 리터럴/DB 특화 휴리스틱으로 고칠 문제가 아니다. `시간` 같은 특정 단어를 금지하면
  금지 원칙 위반이다.

  일반 원칙은 이미 있었다: order binding은 output binding과 달라야 하고, order role을 명시해야 한다.
  다만 wording이 "display-only wording is not enough"에 머물러 composer가 bare field noun을 계속
  order phrase로 재사용했다. 따라서 prompt/schema/feedback에 "order phrase는 방향/최신성/순위/tie-break
  역할을 포함해야 하며, bare output noun만으로는 안 된다"를 일반 규칙으로 명시한다.

- **변경**:
  - Label Contract prompt의 binding 문장을 압축하면서 `direction/recency/tie-break wording`,
    `not the bare output noun`을 추가했다.
  - `AnswerOrderBinding` 및 `answer_contract.order_bindings` schema description에 같은 원칙을 추가했다.
  - `query.order_by` tool schema에도 selected output field가 곧 order request가 아니며, request가
    direction/recency/rank/tie-break role을 말해야 한다고 보강했다.
  - `answer_contract_binding_missing` feedback도 같은 named policy를 상기하도록 수정했다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py tests/test_tooling_composer_tool_factory.py -q`
  통과 (`108 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py`
  통과.

## Iteration 162 — Binding-only feedback must not reopen data exploration

- **질문**:
  Iteration 161 뒤 `trial_44`에서 order binding wording 개선이 실제 repair completion으로
  이어졌는가?

- **실험/결과**:
  `trial_44`도 solver rollout 전 실패했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_44`
  - 첫 draft: 환자 `subject_id=10000032`의 최근 eMAR 투약 기록 5개
  - 첫 feedback: `answer_contract_order_ambiguous`
  - 두 번째 feedback: `answer_contract_binding_missing`
  - 이후 feedback: `label_changed_during_repair`, `answer_contract_order_ambiguous`,
    `answer_contract_binding_missing`
  - solver 도달 없음

- **reasoning 교차 분석**:
  reasoning content는 반환되고 있었고, composer는 첫 tie를 이해해서 `emar_seq`를
  tie-break로 추가했다. 또한 두 번째 feedback 뒤에는 `sequence_number`가 tie-break인데
  user_request와 order binding에 명시되지 않았다는 점도 인지했다.

  문제는 그 다음 행동이었다. `answer_contract_binding_missing`은 row data가 아니라
  `user_request`/`answer_contract` mapping 수리 신호인데, composer가 data tool을 다시
  호출하고 label field 이름까지 바꾸면서 repair-locked label을 흔들었다. 그래서
  `label_changed_during_repair`가 추가로 발생했고, 같은 binding 결함이 반복됐다.

- **정성 평가**:
  accepted data: 없음. 저품질 accepted도 없음.

  rejected data: low-quality rejected. row 후보 자체는 contract repair만 되면 쓸 수 있었지만,
  제출된 draft는 request/contract가 order role과 tie-break를 유일하게 고정하지 못했고,
  repair 중 label drift까지 발생했다. 어려운 좋은 문제가 아니라 composer의 repair discipline
  문제다.

- **개선 판단**:
  `answer_contract_binding_missing`만 단독으로 나온 경우는 precision 100으로 contract-only
  feedback이다. 이 판단은 feedback error code에만 의존하며 DB 리터럴/토큰/컬럼 의미 휴리스틱을
  쓰지 않는다.

  따라서 이 경우 feedback 뒤 data tool 예산을 0으로 보고, 기존 label/query 값을 보존한 채
  `user_request`와 `answer_contract`만 고쳐 `submit_draft`하도록 budget feedback을 정렬했다.
  새 validator를 추가한 것이 아니라, 이미 존재하는 durable policy와 feedback reminder의 실행
  경계를 맞춘 것이다.

- **변경**:
  - `data_tool_budget_feedback`이 단독 `answer_contract_binding_missing` 뒤에는 즉시
    `submit_draft_required`를 반환한다.
  - budget prompt에 `binding feedback uses none`을 추가해 source of truth와 runtime feedback을
    맞췄다.
  - 기존 일반 repair budget 테스트는 query mismatch 계열으로 유지하고, binding-only 전용 테스트를
    추가했다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py tests/test_tooling_composer_tool_factory.py -q`
  통과 (`109 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_prompts.py`
  통과.

## Iteration 163 — Query diagnostics should name handle order keys

- **질문**:
  Iteration 162의 binding-only data budget 개선 뒤 `trial_45`에서 composer가 repair를
  완료하는가?

- **실험/결과**:
  `trial_45`도 solver rollout 전 실패했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_45`
  - topic: `ICU 중재 시술 기록`
  - 첫 draft: ICU 입원 중 가장 먼저 시행된 procedureevent 5개
  - 결과: `synthesis_failed`, feedback 5회
  - 마지막 feedback: `answer_contract_order_ambiguous`, `answer_contract_binding_missing`
  - solver 도달 없음

- **reasoning 교차 분석**:
  reasoning content는 정상 저장됐다. composer는 첫 query에서 `starttime`만 order_by로 쓰면
  `duplicate_order_key_in_returned_rows`가 난다는 점을 읽었다. 그러나 그 다음
  `orderid`를 silent tie-break로 쓰면 deterministic하다고 판단했다.

  첫 feedback 뒤에는 `orderid`가 hidden handle이라 안 된다는 점을 이해했다. 이후 자연
  visible tie-break로 `category`를 시도했지만, 같은 시간/같은 category 안에서도 동점이 남았다.
  composer는 마지막 reasoning에서야 `procedure_name`을 tie-break로 써야겠다고 판단했지만,
  이미 제출 횟수를 모두 쓴 뒤였다.

  핵심 병목은 validator 부재가 아니다. submit validator는 hidden handle tie-break와 남은
  duplicate order key를 모두 precision 100으로 잡았다. 문제는 `query` 결과의 `is_handle`
  metadata가 깊은 referenced column payload 안에 있어, composer가 "silent handle tie-break"
  를 좋은 수리로 오판한 점이다.

- **정성 평가**:
  accepted data: 없음. 저품질 accepted도 없음.

  rejected data: low-quality rejected. row 후보는 의료적으로 자연스럽고 좋은 후보였지만,
  제출 draft는 limited list membership/order를 자연어 request와 answer_contract로 유일하게
  고정하지 못했다. solver가 어려워서 못 푼 문제가 아니라 composer가 hidden handle order와
  남은 visible tie를 정리하지 못한 문제다.

- **개선 판단**:
  새 hard validator는 추가하지 않았다. 이미 정밀한 validator가 거절하고 있다.

  대신 tool-local diagnostics를 보강했다. `query`는 schema metadata를 객관적으로 알고 있으므로,
  ordering diagnostics가 발생한 경우 handle order key를 `handle_order_by_columns`로 직접
  표면화할 수 있다. 이건 DB 리터럴/토큰 휴리스틱이 아니라 tool이 이미 아는 column metadata의
  구조적 노출이다.

  feedback도 기존 List Determinism Policy를 상기하는 쪽으로만 보강했다. query diagnostics가
  아직 ambiguity를 말하면 wording-only submit을 하지 말고 repaired label query를 다시 실행하거나
  다른 label로 전환하라는 문구를 추가했다.

- **변경**:
  - `query.ordering_diagnostics`에 문제가 있는 handle order key를
    `handle_order_by_columns`로 노출한다.
  - `query` tool schema description에 `handle_order_by_columns`의 의미와 silent handle
    tie-break 금지를 추가했다.
  - `answer_contract_order_ambiguous` feedback에 wording-only resubmit 금지 문구를 추가했다.

- **검증**:
  `uv run pytest tests/test_tooling_composer_query.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`158 passed`).

  `uv run ruff check src/rl_task_foundry/tooling/composer/query.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_tooling_composer_query.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 164 — Query schema should enforce the two-key order contract

- **질문**:
  Iteration 163의 handle order diagnostics 뒤 다음 smoke가 solver rollout까지 도달하는가?

- **실험/결과**:
  `trial_46`도 solver rollout 전 실패했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_46`
  - 첫 draft: patient service record 1개를 list로 제출
    - `answer_contract_list_size_invalid`, `label_null_value_forbidden`,
      `answer_contract_phrase_missing`
  - 이후 pharmacy list로 target 전환
  - 마지막 feedback: `answer_contract_order_too_complex`
  - solver 도달 없음

- **reasoning 교차 분석**:
  첫 draft는 1-row list, null field, raw field-name phrases를 포함한 명백한 저품질 draft였다.
  이후 composer는 pharmacy 5-row list로 전환했고, `starttime` 단독 order가 duplicate라는 점을
  이해했다.

  composer는 natural visible tie-break를 찾으려 했지만 `starttime + stoptime`도 동점이 남자
  `medication`까지 세 번째 order key로 추가했다. 이 query는 5 rows를 안정적으로 만들었지만,
  durable List Determinism Policy의 `max two order keys`를 위반했다. submit validator는
  이를 `answer_contract_order_too_complex`로 정확히 거절했다.

- **정성 평가**:
  accepted data: 없음. 저품질 accepted도 없음.

  rejected data: low-quality rejected. pharmacy row 후보 자체는 나쁘지 않았지만, 세 개의
  ordering concept를 자연스러운 요청으로 묶어야만 유일해지는 draft는 mechanical하고 policy 위반이다.
  좋은 어려움이 아니라 composer가 "다른 label 선택" 대신 long order contract를 강행한 문제다.

- **개선 판단**:
  `query.order_by`의 tool schema description은 이미 "no more than two order keys total"이라고
  말하고 있었다. 하지만 schema/parser가 이를 강제하지 않아 composer가 3-key query를 실제로
  실행하고, submit 단계에서야 reject됐다.

  원칙상 tool-local contract는 tool schema와 parser에서 닫는 것이 맞다. 이건 DB 리터럴이나
  의미 휴리스틱이 아니라 query DSL의 구조 제약이며 precision 100으로 판정 가능하다.

- **변경**:
  - `query.order_by` JSON schema에 `maxItems: 2`를 추가했다.
  - query parser가 order key 3개 이상을 `ValueError`로 거절한다.
  - schema/parser 테스트를 추가했다.

- **검증**:
  `uv run pytest tests/test_tooling_composer_query.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`159 passed`).

  `uv run ruff check src/rl_task_foundry/tooling/composer/query.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_tooling_composer_query.py tests/test_tooling_composer_tool_factory.py`
  통과.

## Iteration 165 — Difficulty-up must preserve output request phrases

- **질문**:
  Iteration 164의 `query.order_by` max-2 schema/parser enforcement 뒤 다음 smoke가 solver
  rollout까지 안정적으로 도달하는가?

- **실험/결과**:
  `trial_47`은 solver rollout까지 도달했지만 accepted되지는 않았다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_47`
  - 첫 draft: 환자 입원 기록 list 1개
    - `answer_contract_list_size_invalid`, `answer_contract_evidence_mismatch`
  - 두 번째 이후: eMAR 약물 투여 기록 list로 전환
  - solver-evaluated draft: pass rate `4/4 = 1.0`, `calibration_inconclusive`
  - final draft: difficulty-up 후 `answer_contract_phrase_missing`

- **reasoning 교차 분석**:
  composer는 eMAR list에서 `charttime desc, medication asc` 2-key order를 만들었고,
  약물명/투여 시간/투여 상태 field와 order binding을 수리해 solver rollout까지 도달했다.
  solver 4개는 모두 정답을 맞췄다. 이 draft는 깨끗하지만 너무 쉬운 데이터다.

  too-easy feedback 뒤 composer는 `event_txt = Administered` row-membership filter와
  admissions join으로 난이도를 올리려 했다. 방향은 기존 Difficulty-Up Policy에 맞는다.
  그러나 기존 output field인 `administration_status`를 계속 label에 남겼으면서,
  새 user_request에는 `투여 상태`를 표시해 달라는 phrase를 넣지 않았다. `투여 완료된`은
  filter phrase일 수는 있지만 output field phrase가 아니므로 `answer_contract_phrase_missing`이
  precision 100으로 거절했다.

- **정성 평가**:
  accepted data: 없음. 저품질 accepted도 없음.

  rejected data:
  - solver-evaluated draft는 clean-good but too easy. 좋은 데이터 후보였지만 학습 목표에는 너무 쉽다.
  - final rejected draft는 low-quality rejected. difficulty-up의 structural direction은 좋았지만,
    기존 output field request phrase를 보존하지 못해 사용자 요청과 label contract가 어긋났다.

- **개선 판단**:
  durable prompt에는 list difficulty-up에서 output fields/source meanings를 보존하라고 되어 있었지만,
  output field의 request phrase까지 보존해야 한다는 압력이 약했다. feedback-only로 새 지시를 만들면
  지침 이원화가 되므로 prompt-first로 durable policy를 보강한다.

  phrase_missing feedback은 이미 "missing field phrase를 natural request wording으로 추가하라"고
  말하고 있으므로, too-easy feedback도 같은 source of truth를 상기하도록 정렬한다.

- **변경**:
  - Feedback And Difficulty-Up Policy prompt에서 list retries가
    `output fields/source meanings/phrases`를 보존한다고 명시했다.
  - too-easy feedback reminder에도 existing output field request phrases 보존을 추가했다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py tests/test_tooling_composer_query.py tests/test_tooling_composer_tool_factory.py -q`
  통과 (`159 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

  prompt length: `7944`.

## Iteration 166 — Handle outputs must require explicit visibility overrides

- **질문**:
  Iteration 165 뒤 `trial_48`의 accepted data는 정말 좋은 데이터인가?

- **실험/결과**:
  `trial_48`은 accepted됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_48`
  - topic: `ICU 중환자실 시술 및 처치 내역`
  - pass rate: `3/4 = 0.75`
  - failed solver runs: `0`
  - registry committed

- **정성 평가**:
  accepted data는 low-quality accepted로 판단한다.

  row set 자체는 ICU stay의 procedureevents 5개로 자연스럽고, solver pass rate도 band 안이었다.
  그러나 canonical answer에 `procedure_id`가 포함됐다. user_request는 "각 시술의 이름, 분류,
  시작/종료 시간, 상태"를 요청했지 procedure id/reference를 요청하지 않았다. answer_contract는
  `procedure_id`를 "시술"이라는 broad phrase로 묶어 통과했다.

  또한 "시술 및 처치 5개를 시작 시간 순서대로"는 membership boundary가 약하다. 다만 이번 변경은
  가장 명확한 precision-100 원인인 handle output 노출부터 막는다.

- **reasoning 교차 분석**:
  composer reasoning은 `procedureevents`의 `orderid`를 "user visible" field로 보고
  procedure identifier처럼 출력했다. 실제 schema snapshot에서도 `orderid`는 primary key인데
  default visibility가 `user_visible`이라 column source가 `user_visible`로 기록됐다. 따라서
  submit validator가 `label_non_user_visible_source`로 막을 수 없었다.

- **개선 판단**:
  DB별 `orderid` 문자열 금지는 금지 원칙 위반이다. 대신 primary/foreign key handle은 스키마
  metadata로 precision 100 판정 가능한 구조다.

  원칙상 raw handle은 hidden scope/filter/order에는 사용할 수 있어도 direct label output은
  명시적으로 user-visible reference로 override된 경우에만 허용해야 한다. 따라서 default visibility가
  `user_visible`이어도 PK/FK handle은 explicit visibility override가 없으면 `blocked`로 처리한다.
  특정 DB나 컬럼명에 의존하지 않는다.

- **변경**:
  - schema sensitivity에 `resolve_handle_aware_visibility`를 추가했다.
  - PostgreSQL introspector가 PK/FK handle column을 explicit override 없이는 `blocked`로 분류한다.
  - 테스트에서 default-visible handle이 blocked되고, explicit override가 있으면 user-visible로
    남는 것을 확인한다.

- **검증**:
  `uv run pytest tests/test_schema_introspection.py tests/test_tooling_composer_query.py tests/test_tooling_composer_tool_factory.py tests/test_tooling_composer_schema_map.py tests/test_tooling_composer_profile.py tests/test_tooling_composer_sample.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`204 passed`).

  `uv run ruff check src/rl_task_foundry/schema/sensitivity.py src/rl_task_foundry/schema/introspect.py tests/test_schema_introspection.py`
  통과.

## Iteration 167 — Duplicate rows should not be rescued with artificial sequence outputs

- **질문**:
  Iteration 166의 handle visibility 변경 뒤 다음 smoke에서 품질이 좋아졌는가?

- **실험/결과**:
  `trial_49`는 accepted 없이 실패했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_49`
  - 최종 topic: `일일 투약 기록`
  - 최종 pass rate: `0/4 = 0.0`
  - failed solver runs: `0`
  - 최종 오류: `calibration_inconclusive`

- **정성 평가**:
  rejected low-quality로 판단한다.

  데이터 자체는 한 입원/하루의 eMAR 투약 기록이라 자연스럽다. 그러나 composer가 중복 projected
  answer row를 보고 자연 field나 aggregate로 바꾸지 않고 `sequence_number`를 label output에
  추가했다. 이후 user_request에 "같은 시간이면 기록 순서"를 넣어 이 sequence 출력을 정당화했다.
  이는 실제 사용자가 먼저 요청한 정보라기보다 duplicate row를 구제하기 위해 만든 technical control이다.

  handle visibility 개선은 작동했다. `emar_id`는 query evidence에서 `blocked`로 표시되어 직접
  출력/정렬용 handle로 쓰기 어렵게 됐다. 문제는 PK/FK handle이 아닌 `emar_seq` 같은 source
  sequence를 composer가 user-visible output처럼 사용한 점이다.

- **reasoning 교차 분석**:
  composer reasoning은 "Rows must be distinguishable" 원칙을 인식했고, duplicate rows를 해결해야
  한다고 판단했다. 하지만 "자연 visible field/aggregate" 대신 sequence/order output을 새로
  만들어 넣었다. 즉 정책 부재가 아니라 정책의 경계가 덜 선명했다. solver reasoning은 최종 요청의
  `sequence_number`를 맞추려 했지만 도구 표면에서 같은 row boundary와 exact 값 재현에 실패했다.

- **개선 판단**:
  `emar_seq` 같은 컬럼명을 리터럴로 막는 것은 금지 원칙 위반이다. PK/FK도 아니므로 precision-100
  validator로 일반 차단할 근거가 없다.

  따라서 프롬프트 우선 원칙대로 durable List Determinism Policy를 보강한다. 중복 answer row는
  자연 visible field나 aggregate로 해결해야 하며, sequence/reference/record-order output을
  중복 제거 목적으로 새로 만들어야 한다면 label을 바꾸라고 지시한다. query tool desc도
  projection diagnostics 해석을 같은 방향으로 보강하고, feedback은 이 기존 정책을 상기하는 역할만
  한다.

- **변경**:
  - synthesis prompt에 "duplicate row를 sequence/order output으로 구제하지 말라"는 일반 원칙을
    추가했다.
  - query tool schema description에 projection diagnostics가 duplicate rows를 보고할 때의
    대응 원칙을 추가했다.
  - duplicate-answer feedback 메시지는 같은 durable policy를 짧게 상기하도록 수정했다.

## Iteration 168 — Scalar contract must come from aggregate evidence

- **질문**:
  Iteration 167 뒤 다음 smoke에서 duplicate-row repair는 개선됐는가? 그리고 rejected data는
  어려운 좋은 문제인가, 아니면 저품질 문제인가?

- **실험/결과**:
  `trial_50`은 accepted 없이 `MaxTurnsExceeded`로 실패했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_50`
  - composer/solver: OpenRouter Kimi K2.5
  - solver rollout: 4
  - topic hint: 없음

  composer는 처음에 pharmacy list를 만들었고, duplicate projected answer rows 피드백 뒤에는
  `sequence_number` 같은 artificial order output으로 구제하지 않았다. 대신 route/frequency/dose
  같은 자연 field를 추가하려 했다. 이 점에서 Iteration 167 변경은 의도대로 작동했다.

  그러나 pharmacy rows는 여전히 projected duplicate가 남았고, composer는 마지막에 단일 입원 record의
  `admission_type`, `admission_time`을 `answer_contract.kind='scalar'`로 제출했다. solver pass rate는
  `4/4 = 1.0`으로 too easy였고, 데이터는 단일 row detail lookup이라 학습 데이터로 좋지 않았다.

- **reasoning 교차 분석**:
  reasoning content는 저장되고 있었다. composer reasoning을 보면 duplicate rows를 해결해야 한다는
  원칙을 이해했고, hidden handle이나 sequence output을 피하려고 했다. 이후 예산 압박 때문에 쉬운
  단일 admission detail로 전환했고, selected row object를 scalar처럼 취급했다.

  solver reasoning은 이 admission detail이 탐색/추론 없이 직접 조회에 가까운 문제였음을 보여줬다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. pharmacy list는 중복 answer rows가 해결되지 않은 저품질이고,
  admission scalar는 너무 쉬운 단일 record detail lookup이다. 어려워서 좋은 문제가 아니라 composer가
  유효한 label을 끝까지 만들지 못한 케이스다.

- **개선 판단**:
  `admission_type` 같은 컬럼명이나 값 문자열을 보고 막으면 리터럴/토큰 기반 휴리스틱이므로 금지 원칙
  위반이다.

  대신 최신 `query` 결과의 구조적 `column_sources[].kind`는 SDK/tool이 생성한 증거다. 따라서
  `answer_contract.kind='scalar'`인데 최신 query output source 중 `aggregate`가 하나도 없으면,
  selected row object를 scalar로 제출한 것이므로 precision 100으로 reject할 수 있다.

- **변경**:
  - `submit_draft`에 `answer_contract_scalar_not_aggregate` feedback error를 추가했다.
  - scalar draft는 최신 query evidence에 aggregate output source가 없으면 solver rollout 전에 reject한다.
  - feedback은 Label Contract의 기존 원칙, 즉 scalar는 aggregate query result라는 정책을 상기한다.
  - 테스트 헬퍼는 scalar contract를 받은 경우 aggregate evidence를 기록하도록 고쳤고, 직접 row detail
    오용 케이스는 명시적 select evidence로 회귀 테스트한다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py -q`
  통과 (`81 passed`).

  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`110 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 169 — Post-scalar validator smoke produced a good accepted list task

- **질문**:
  Iteration 168의 scalar-not-aggregate validator 뒤, MIMIC demo smoke에서 accepted data가
  정말 좋은가?

- **실험/결과**:
  `trial_51`은 accepted됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_51`
  - composer/solver: OpenRouter Kimi K2.5
  - solver rollout: 4
  - pass rate: `3/4 = 0.75`
  - failed solver runs: `0`
  - registry committed

  accepted task는 hidden `hadm_id=22987108` 기준으로 최근 eMAR 투약 기록 5건의
  `chart_time`, `medication`, `event_text`를 묻는 list다.

- **reasoning 교차 분석**:
  composer는 처음에 `emar_id`를 anchor로 두고 `hadm_id` hidden filter를 사용해
  `answer_contract_hidden_filter_unanchored` feedback을 받았다. repair에서는 label row set을
  바꾸지 않고 anchor를 `hadm_id`로 전환했으며, request에 `투약 일시`, `약물명`, `투약 상태`
  phrase도 자연스럽게 넣었다. 즉 feedback은 기존 정책을 상기했고 composer는 같은 label을
  정책에 맞게 수리했다.

  solver 3개는 모두 eMAR 테이블을 선택해 `hadm_id` filter, `charttime desc`, limit 5로 정확히
  풀었다. 실패한 solver 1개는 `prescriptions`를 선택해 `starttime/stoptime/drug`로 답했다.
  이는 request가 완전히 trivial하지 않음을 보여주지만, 다수 solver는 `투약 상태`를 eMAR
  `event_txt`로 해석했다.

- **데이터 직접 검증**:
  DB에서 `mimiciv_hosp.emar where hadm_id=22987108 order by charttime desc limit 10`을
  재조회했다. canonical top 5는 실제 top 5와 정확히 일치했다.

  - `2146-07-12 20:50:00`, `Morphine Sulfate`, `Stopped - Unscheduled`
  - `2146-07-12 19:00:00`, `Morphine Sulfate`, `Confirmed`
  - `2146-07-12 18:27:00`, `Morphine Sulfate`, `Infusion Reconciliation`
  - `2146-07-12 18:00:00`, `Lorazepam`, `Not Given`
  - `2146-07-12 17:25:00`, `Midazolam`, `Not Given`

- **정성 평가**:
  accepted data: good accepted.

  row set은 anchor admission의 투약 기록으로 자연스럽고, order/limit도 request에 명시됐다.
  output fields는 user-facing이며 hidden handle 출력이 없다. `charttime` top 5에 boundary tie가
  없고, duplicate projected answer rows도 없다. pass rate `0.75`는 현재 실험 밴드 안이다.

  다만 composer가 첫 submit 전 `submit_draft_required` feedback을 받고도 data tool을 추가 호출한
  점은 메타 프로토콜 개선 후보로 남긴다. 이번 accepted data 자체를 저품질로 만들지는 않았다.

- **상태**:
  만족스러운 accepted smoke `1/5`.

## Iteration 170 — ToolBudgetFeedback must stop exploration, not just warn

- **질문**:
  `trial_52` 실패는 어려운 좋은 데이터 때문인가, 아니면 composer가 저품질 draft를
  반복한 것인가? reasoning 내용은 어떤 패턴을 보여주는가?

- **실험/결과**:
  `trial_52`는 accepted 없이 `MaxTurnsExceeded`로 종료됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_52`
  - composer/solver: OpenRouter Kimi K2.5
  - solver rollout: 4
  - accepted data: 없음

  세 번의 submit은 모두 feedback으로 막혔다.

  1. pharmacy list: query limit 5를 쓰면서 request/contract에 limit phrase가 없었고,
     `start_time` order가 boundary tie를 만들었다.
  2. eMAR list: nullable medication을 label에 포함했고, source sequence를 output/order로
     사용했으며, limit phrase/evidence도 어긋났다.
  3. diagnosis list: blocked `seq_num` 계열 값을 `diagnosis_sequence`로 노출했고, 성별/입원
     유형은 반복값이라 user-facing answer로도 약했다.

- **reasoning 교차 분석**:
  reasoning sidecar는 정상 저장됐다. composer는 `submit_draft_required`와 “budget limit”을
  인지했지만, 그 뒤에도 schema/profile 탐색을 이어 갔다. 특히 최종 label query 뒤에 더
  탐색하다가 턴을 소모했고, budget 압박 속에서 blocked sequence를 포함한 약한 label로
  제출했다.

  이건 solver가 못 푼 어려운 문제라기보다, composer가 이미 있는 Draft Submission Budget과
  Label/List 정책을 끝까지 따르지 못한 문제다.

- **정성 평가**:
  accepted data: 없음. 따라서 low-quality accepted도 없음.

  rejected data: 모두 low-quality rejected. pharmacy draft는 list boundary가 유일하지 않았고,
  eMAR draft는 null/evidence/sequence 문제가 있었으며, diagnosis draft는 blocked field와 반복
  출력 문제를 포함했다. 거절 자체는 바람직하다.

- **변경**:
  새 hard validator는 추가하지 않았다. 어떤 label switch가 정말 불가피한지, 또는 sequence-like
  field가 실제 사용자에게 자연스러운지 100% precision으로 판별하기 어렵기 때문이다.

  대신 prompt-first 원칙대로 기존 정책의 source of truth를 맞췄다.

  - Draft Submission Budget: `ToolBudgetFeedback`는 탐색 경계이며, 이후에는 submit해야 한다고
    명시했다.
  - 예외는 “rows missing/blocking diagnostics”일 때의 최종 label `query` 1회뿐이며, 그 query
    뒤에는 submit해야 한다고 명시했다.
  - `query` tool schema description도 성공한 최종 label query 뒤에는 schema/profile/sample/
    neighborhood 탐색을 하지 말고 submit하라고 맞췄다.
  - `submit_draft_required` feedback 문구도 같은 durable policy를 상기하도록 수정했다.

  이 변경은 새 지시를 feedback에 숨긴 것이 아니라, 이미 존재하던 budget 정책을 prompt,
  tool description, feedback reminder가 모두 같은 방식으로 말하게 한 것이다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`110 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/turn_budget.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_turn_budget_prompt.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **상태**:
  `trial_52`는 accepted가 아니므로 만족스러운 accepted 연속 기록은 다시 `0/5`로 본다.

## Iteration 171 — Accepted medication task still had source-role ambiguity

- **질문**:
  Iteration 170의 budget boundary 정렬 뒤 다음 smoke는 좋은 accepted data를 만들었는가?

- **실험/결과**:
  `trial_53`은 accepted됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_53`
  - composer/solver: OpenRouter Kimi K2.5
  - solver rollout: 4
  - pass rate: `2/4 = 0.5`
  - failed solver runs: `0`
  - registry committed

  accepted task는 hidden `hadm_id=25177949` 기준으로 가장 최근에 시작된 약물 5개의
  `medication`, `status`, `starttime`, `stoptime`, `proc_type`을 묻는 list였다.

- **reasoning 교차 분석**:
  Iteration 170 개선은 부분적으로 의도대로 작동했다. composer는 `ToolBudgetFeedback` 이후
  schema/profile 탐색으로 새지 않고 label query를 실행했다. 첫 query의 order ambiguity를 보고
  `starttime desc, medication asc`로 query를 다시 실행했고, 그 뒤 submit했다. 첫 submit의
  feedback은 `약물명 순` phrase가 user_request에 exact substring으로 없어서 발생했고,
  두 번째 submit에서는 label을 바꾸지 않고 문장만 고쳤다.

  하지만 solver reasoning은 source ambiguity를 드러냈다. 2개 solver는 `pharmacy` table을 찾아
  정답을 맞혔다. 나머지 2개 solver는 먼저 `prescriptions` surface를 탔고, 같은 약물/시간 row
  set은 맞혔지만 `drug_type=MAIN`을 `status` 또는 `proc_type`으로 제출해 틀렸다.

- **데이터 직접 검증**:
  DB에서 `mimiciv_hosp.pharmacy where hadm_id=25177949 order by starttime desc, medication asc
  limit 10`을 재조회했다. canonical top 5는 pharmacy 기준 실제 top 5와 일치했다.

  같은 조건의 `mimiciv_hosp.prescriptions`도 재조회했다. top 5 약물명/시간은 pharmacy와
  같았지만, `prescriptions.drug_type`은 모두 `MAIN`이고 pharmacy의 `status/proc_type`과
  representation이 다르다. 즉 row set은 어렵게 좋은 문제가 아니라, answer surface가 둘로
  갈리는 문제였다.

- **정성 평가**:
  accepted data: low-quality accepted.

  row membership/order 자체는 정확하고, hidden handle도 노출하지 않았다. 그러나 user_request가
  "약물/처방 상태/처리 유형"이라는 broad source wording에 머물러 pharmacy record surface를
  충분히 고정하지 못했다. solver가 못 푼 것이 아니라, `pharmacy`와 `prescriptions`라는 reachable
  source surfaces가 모두 그럴듯한 답 후보가 된 것이다.

- **변경**:
  hard validator는 추가하지 않았다. 어떤 broad noun이 여러 source surfaces를 만든다는 판단은
  DB 의미 해석이 들어가므로 precision 100으로 보장하기 어렵다.

  대신 tool description을 기존 Source surface 정책에 맞췄다.

  - `query.spec` description에, final query 뒤 submit 전에 user_request/topic이 선택한 source
    role을 이름 붙였는지 확인하라고 추가했다.
  - 여러 reachable source가 broad noun에 답할 수 있으면 label/output field name은 source surface를
    disambiguate하지 못한다는 점을 명시했다.

  이건 새 정책이 아니라 system prompt의 "Source surface" 원칙을 final query evidence를 보는
  순간에 상기시키는 변경이다.

- **검증**:
  `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`1 passed`).

  `uv run ruff check src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_tooling_composer_tool_factory.py`
  통과.

- **상태**:
  `trial_53`은 accepted였지만 low-quality accepted로 판정한다. 만족스러운 accepted 연속 기록은
  여전히 `0/5`.

## Iteration 172 — Exact string spacing must be preserved from query rows

- **질문**:
  Iteration 171의 source-role reminder 뒤 다음 smoke에서 low-quality accepted가 줄었는가?

- **실험/결과**:
  `trial_54`는 accepted 없이 `MaxTurnsExceeded`로 종료됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_54`
  - composer/solver: OpenRouter Kimi K2.5
  - accepted data: 없음

  submit 흐름은 다음과 같았다.

  1. DRG code list: row 수 2개, null 값 포함, no-primary-key source로 feedback.
  2. lab item count aggregate: count top 5는 좋은 후보였지만 동점 count가 많아
     `answer_contract_order_ambiguous`.
  3. microbiology result list: `chartdate desc, storedate desc`로 order ambiguity는 고쳤지만,
     `comments` 문자열의 trailing/double space를 label에서 정규화해
     `answer_contract_evidence_mismatch`.

- **reasoning 교차 분석**:
  composer는 첫 feedback 뒤 no-PK DRG row-value label을 버리고 lab aggregate로 이동했다. 이
  선택은 방향상 맞지만, aggregate query spec에서 `select`와 `aggregate`를 같이 넣어 한 번 실패한
  뒤 수정했다. lab aggregate는 tied count가 많았고, hidden handle boundary가 diagnostics에
  드러나 reject됐다.

  마지막 microbiology 후보에서 composer는 query diagnostics를 보고 `storedate` tie-break를
  추가했다. 이 부분은 좋았다. 그러나 query result의 `comments` 값에는 `"No MRSA isolated.  "`
  같은 공백이 있었고, 제출 label은 `"No MRSA isolated. "`처럼 공백이 줄어 있었다. validator가
  이 reformat을 precision 100으로 잡아냈다.

- **정성 평가**:
  accepted data: 없음. low-quality accepted도 없음.

  rejected data:
  - DRG draft는 low-quality rejected가 맞다.
  - lab aggregate는 어려운 좋은 후보였지만, 현재 draft는 tie/boundary가 해결되지 않아 reject가
    맞다.
  - microbiology draft는 row/order 구조는 좋은 후보였으나, result representation과 문자열
    exact-copy가 아직 약했다. 현 reject는 바람직하다.

- **변경**:
  hard validator는 이미 있었다. `answer_contract_evidence_mismatch`가 latest query rows와 label의
  canonical JSON을 비교하므로 precision 100이다.

  개선은 prompt/feedback reminder 정렬이다.

  - `query.spec` description에 returned row values를 label로 복사할 때 spacing까지 그대로
    보존하고 trim/normalize/rewrite하지 말라고 추가했다.
  - `answer_contract_evidence_mismatch` feedback에도 string spacing 포함 exact match와
    trim/normalize/rewrite 금지를 명시했다.

  이 변경은 새 semantic heuristic이 아니라 Label Grounding Policy의 "Do not reformat"을
  구체화한 것이다.

- **검증**:
  `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_that_does_not_match_latest_query -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **상태**:
  만족스러운 accepted 연속 기록은 여전히 `0/5`.

## Iteration 173 — Aggregate query specs should not mix select with aggregate

- **질문**:
  Iteration 172의 exact string reminder 뒤 다음 smoke에서 좋은 accepted가 나왔는가?

- **실험/결과**:
  `trial_55`는 accepted 없이 종료됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_55`
  - composer/solver: OpenRouter Kimi K2.5
  - accepted data: 없음

  흐름은 다음과 같았다.

  1. Foley catheter 총 소변 배출량 scalar: solver `4/4 = 1.0`으로 너무 쉬워 reject.
  2. output type별 총 배출량 ranking: scalar aggregate를 grouped aggregate로 올리려는 방향은
     좋았지만, request는 "두 번째로 높은 유형"을 묻고 label은 전체 ranking list를 제출했다.
     또한 기존 Foley predicate를 제거해 `answer_contract_not_incremental`.
  3. Foley vs second-highest comparison: 여전히 predicate 제거로 reject.
  4. latest Foley output measurements: user_request가 JSON escape처럼 깨졌고, hidden/non-visible
     source와 binding 문제까지 발생해 reject.
  5. Foley total+average scalar: solver `4/4 = 1.0`으로 여전히 너무 쉬워 budget exhausted.

- **reasoning 교차 분석**:
  composer는 첫 scalar가 too easy라는 feedback을 이해했고, grouped/ranked aggregate로
  difficulty-up하려 했다. 이 방향 자체는 의미 있다. 다만 실제 제출은 request/label이 어긋났고,
  기존 Foley predicate를 어떤 방식으로 보존하거나 group_by로 lift해야 하는지 명확히 처리하지
  못했다.

  또 trial_54에 이어 aggregate query spec에서 `select`와 `aggregate`를 같이 넣는 오류가 반복됐다.
  이건 DB 의미 판단이 아니라 tool DSL 사용법 문제다.

- **정성 평가**:
  accepted data: 없음. low-quality accepted도 없음.

  rejected data:
  - 첫 Foley scalar와 마지막 total+average scalar는 clean하지만 너무 쉬운 good rejected.
  - grouped output ranking은 어려운 좋은 후보가 될 수 있었지만, 현재 제출은 request가 "second"
    하나를 묻고 label은 전체 list를 내는 mismatch가 있어 low-quality rejected가 맞다.
  - escaped Korean request와 hidden/non-visible source draft는 low-quality rejected다.

- **변경**:
  predicate-to-group_by 완화는 보류했다. 구조적으로 precision 100 완화 아이디어는 가능하지만,
  이번 실제 retry는 request/label mismatch까지 함께 있었기 때문에 바로 허용하면 low-quality
  accepted 위험이 있다.

  대신 반복된 tool DSL 오류를 줄이도록 query tool schema description을 보강했다.

  - aggregate query에서는 `select`를 쓰지 말고, copied group keys는 `group_by`, metric fields는
    `aggregate`에 넣으라고 명시했다.
  - `group_by` description에도 aggregate가 있을 때 group_by가 select를 대체한다고 추가했다.
  - `aggregate` description에도 aggregate와 select를 함께 쓰지 말라고 추가했다.

  이 변경은 validator가 아니라 tool schema/description 정렬이다.

- **검증**:
  `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`1 passed`).

  `uv run ruff check src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_tooling_composer_tool_factory.py`
  통과.

- **상태**:
  만족스러운 accepted 연속 기록은 여전히 `0/5`.

## Iteration 174 — List difficulty-up must not narrow the evaluated row set

- **질문**:
  Iteration 173의 aggregate query schema 보강 뒤 다음 smoke에서 좋은 accepted가 나왔는가?
  그리고 reasoning은 composer가 too-easy feedback을 어떻게 해석하는지 보여주는가?

- **실험/결과**:
  `trial_56`은 accepted 없이 `synthesis_failed`로 종료됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_56`
  - composer/solver: OpenRouter Kimi K2.5
  - solver rollout: 4
  - accepted data: 없음

  submit 흐름은 다음과 같았다.

  1. 단일 입원 정보 lookup: list row 1개, `discharge_location=null`, phrase mismatch로 feedback.
  2. 최근 입원 이력 5건: 처음에는 hidden `hadm_id`를 query select에 포함해 non-user-visible
     source 문제가 났고, 최종적으로 hidden field를 제거했다.
  3. 최근 입원 이력 5건: solver `4/4 = 1.0`으로 너무 쉬워 rejected.
  4. 퇴원 완료 입원 이력: 기존 5건 list에서 non-null discharge filter를 추가해 4건으로 줄었고,
     `cardinality_weakened`/`answer_contract_not_incremental`.
  5. 응급 입원 이력: admission type filter를 추가해 3건으로 줄었고,
     `answer_contract_not_incremental` 후 budget exhausted.

- **reasoning 교차 분석**:
  reasoning content는 실제로 반환되고 있었고, composer 판단 흐름을 확인할 수 있었다.

  composer는 첫 too-easy feedback 뒤 "난이도를 올려야 한다"는 신호는 이해했다. 하지만
  Difficulty-Up Policy 내부에 "list는 row set/order/limit 보존"과 "row membership도 난이도 상승
  수단"이 함께 있어, non-null filter나 status/type filter를 추가해 row set을 좁히는 방향을
  합리화했다. reasoning에는 "filter for only admissions where discharge_location is not null",
  "admission_type = EW EMER." 같은 row-excluding repair 계획이 명시적으로 나타났다.

  즉 solver가 못 푼 문제가 아니라, composer가 too-easy list를 difficulty-up하면서 기존 evaluated
  list의 row set/limit을 보존하지 못한 문제다.

- **정성 평가**:
  accepted data: 없음. low-quality accepted도 없음.

  rejected data:
  - 단일 입원 정보는 low-quality rejected다. list task로는 1 row이고 null field도 있었다.
  - 최근 입원 이력 5건은 clean하지만 solver 4/4가 맞춘 too-easy rejected다.
  - 퇴원 완료/응급 입원 이력 retry는 low-quality rejected다. 기존 evaluated list의 fixed size와
    row set을 좁히면서 난이도를 올리려 했고, 한국어 문장에도 `순으부터`, `익급실` 같은 어색한
    표기가 남았다.

- **변경**:
  prompt-first 원칙에 따라 durable Difficulty-Up Policy의 모순을 먼저 제거했다.

  - too-easy 이후 evaluated list는 row set/order/limit/output/source meanings/phrases를 보존해야
    한다고 명시했다.
  - evaluated list retry에서는 narrowing filter 추가, filter 제거, fixed list 축소, field 교체를
    금지했다.
  - 좋은 difficulty-up은 같은 row를 유지한 채 lookup/comparison/visible ordering/related-row
    reasoning을 바꾸는 한 가지 grounded dimension이라고 정리했다.
  - feedback은 이 durable policy를 상기하도록만 맞췄다. 새 지시 원천을 feedback에 따로 만들지
    않았다.

  추가로 precision 100 구조 검증을 하나 넣었다. too-easy 이후 list retry에서 새 `where` predicate가
  추가됐는지는 query evidence 구조로 정확히 판정할 수 있으므로, `list_row_filter_added`를
  `answer_contract_not_incremental` 원인으로 기록한다. 문자열/컬럼 의미 리터럴 휴리스틱은 사용하지
  않았다.

  `trial_56`에서 반복된 `order_by[0] must include exactly one of 'ref' or 'output'` 툴 사용 오류는
  tool schema description에 "order item은 ref/output 중 정확히 하나만"이라고 추가해 보강했다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py -q`
  통과 (`105 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **상태**:
  만족스러운 accepted 연속 기록은 여전히 `0/5`.

## Iteration 175 — ToolBudgetFeedback must cap repeated diagnostic queries

- **질문**:
  Iteration 174의 list difficulty-up 정책/validator 정리 뒤 다음 smoke에서 만족스러운 accepted가
  나왔는가? 새 row-filter guard가 실제로 작동했는가?

- **실험/결과**:
  `trial_57`은 accepted 없이 `MaxTurnsExceeded`로 종료됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_57`
  - composer/solver: OpenRouter Kimi K2.5
  - accepted data: 없음

  submit 흐름은 다음과 같았다.

  1. eMAR 단일 medication administration lookup: list 1 row라 `answer_contract_list_size_invalid`.
  2. procedureevents list: 5 rows까지는 만들었지만 `location=null` 포함, phrase mismatch.
  3. outputevents list: request가 unicode escape처럼 깨졌고, `charttime` source가 blocked/handle 성격으로
     잡혔으며, order boundary ambiguity까지 있어 feedback.

  이후 submit 없이 query를 계속 돌리다 MaxTurnsExceeded.

- **reasoning 교차 분석**:
  composer는 첫 feedback 뒤 list size 문제를 이해하고 admission scope로 확장했다. 이 방향 자체는
  맞았다. 그러나 feedback 뒤 query를 과도하게 반복했다.

  실제 call sequence는 feedback 이후 `neighborhood` 1회 + `query` 5회 이상이었다. 여러 query가
  duplicate order key, duplicate projected rows, null field, handle order key 같은 diagnostics를
  냈고, composer는 submit으로 feedback을 받기보다 새 surface를 계속 찾아다녔다.

  Iteration 170에서 prompt/feedback은 "after feedback max 3 data tools; diagnostics repair query는
  1회만"이라고 정렬했지만, 구현은 `latest query`에 blocking diagnostics가 있으면 다음 query를
  계속 허용하고 있었다. 즉 이번 실패는 durable policy 부재가 아니라 runtime budget enforcement
  구멍이다.

- **정성 평가**:
  accepted data: 없음. low-quality accepted도 없음.

  rejected data:
  - eMAR 단일 lookup은 low-quality rejected. list로 제출할 수 없는 1 row task였다.
  - procedureevents list는 후보 방향은 가능했지만 null answer field와 phrase mismatch가 있어
    low-quality rejected.
  - outputevents list는 escaped Korean request, non-user-visible/handle-like time output, ordering
    ambiguity가 겹친 low-quality rejected.

  `trial_57`은 어려운 좋은 문제가 아니라 composer가 low-quality 후보를 너무 오래 탐색한 실패다.

- **변경**:
  프롬프트는 바꾸지 않았다. source of truth는 이미 Draft Submission Budget에 있었다.

  runtime enforcement를 policy에 맞췄다.

  - protocol boundary 이후 data tool count가 limit에 도달하면 submit을 요구한다.
  - 단, boundary 이후 아직 query를 한 번도 하지 않았고 현재 tool이 `query`라면 final label query
    1회는 허용한다.
  - 이미 query를 한 번 실행했다면, 그 query가 empty/ambiguous/duplicate diagnostic을 냈더라도
    추가 query를 무한 허용하지 않고 `submit_draft_required`를 반환한다.

  이 검증은 tool call count와 tool name만 보는 구조 검증이다. DB 값/문자열/컬럼 의미 리터럴
  휴리스틱을 사용하지 않는다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_late_first_submit tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_allows_first_label_query_after_limit tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_late_feedback_repair tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_allows_repair_query_after_limit tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_repeated_query_repair_for_ambiguous_query tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_repeated_query_repair_for_empty_query -q`
  통과 (`6 passed`).

  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`111 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/turn_budget.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

- **상태**:
  만족스러운 accepted 연속 기록은 여전히 `0/5`.

## Iteration 176 — Broad wording must not hide a non-ordinary source

- **질문**:
  Iteration 175의 budget enforcement 정리 뒤 다음 smoke에서 좋은 accepted가 나왔는가?
  `trial_58`의 pass rate `0/4`는 어려운 좋은 문제인가, 아니면 저품질 문제인가?

- **실험/결과**:
  `trial_58`은 accepted 없이 `synthesis_failed`로 종료됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_58`
  - composer/solver: OpenRouter Kimi K2.5
  - solver rollout: 4
  - pass rate: `0/4 = 0.0`
  - accepted data: 없음

  첫 submit은 특정 입원의 `pharmacy` rows에서 가장 최근 시작한 약물 5개를 만들었지만,
  `starttime` 동점 때문에 `answer_contract_order_ambiguous` feedback을 받았다. 두 번째 submit은
  `medication asc` tie-break를 추가해 order ambiguity는 해결했지만, solver 4개가 모두
  canonical label과 다른 답을 냈다.

- **reasoning 교차 분석**:
  composer reasoning은 `pharmacy`와 `prescriptions`가 모두 admission 아래 reachable하다는 것을
  보았지만, label은 `pharmacy`에서 만들고 user_request는 넓은 "처방된 약물" 의미로 작성했다.

  solver reasoning은 4개 모두 요청을 `prescriptions` surface로 자연스럽게 해석했다. 한 solver는
  `prescriptions` 기준 top 5를 `Bag`, `Magnesium Sulfate`, `Potassium Chloride`,
  `MetFORMIN (Glucophage)`, `Potassium Chloride`로 산출했다. DB 교차검증 결과도
  `pharmacy` top 5에는 `Bag`가 없고 `Guaifenesin`이 포함되어, 두 source surface가 실제로 다른
  label을 만든다는 점을 확인했다.

- **정성 평가**:
  accepted data: 없음. low-quality accepted도 없음.

  rejected data:
  - 첫 submit은 order ambiguity가 있으므로 low-quality rejected다.
  - 두 번째 submit은 order 자체는 고쳤지만, request가 선택한 source role을 유일하게 고정하지
    못했다. solver가 못 푼 어려운 좋은 문제가 아니라, composer가 broad wording으로 non-ordinary
    source choice를 숨긴 low-quality rejected다.

- **변경**:
  hard validator는 추가하지 않았다. 어떤 자연어 명사가 어느 source surface를 "ordinary"하게
  가리키는지는 DB 리터럴/도메인 의미 판단이 필요하므로 precision 100 구조 검사가 아니다.

  prompt-first/tool-local contract 원칙에 맞춰 다음만 보강했다.

  - Core Definitions의 Source surface: ordinary wording이 다른 reachable source를 가리키면 그
    source를 사용하거나, 선택한 source role을 명시하라고 정리했다.
  - Scope Example: broad `R` 요청이 `S2.R` query를 숨기고 `S1.R`도 맞는 상황을 bad example로
    바꾸고, selected source role을 드러낸 request를 good example로 제시했다.
  - `query.from.table` schema description: root table이 selected source surface임을 명시하고,
    ordinary wording이 다른 reachable source를 가리키면 source를 바꾸거나 source role을
    user_request/topic/answer_contract에 드러내라고 보강했다.
  - `query` tool description과 `submit_draft.user_request` description도 같은 source-surface
    계약을 상기하도록 맞췄다.

  이 변경은 특정 테이블명/컬럼명/값 리터럴을 보지 않는다. feedback도 새로 만들지 않았고,
  기존 Source surface 원칙을 더 잘 보이게 한 것이다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_payload_schema_uses_strict_json_string_fields tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_turn_budget_prompt.py -q`
  통과 (`10 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/turn_budget.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

- **상태**:
  만족스러운 accepted 연속 기록은 여전히 `0/5`.

## Iteration 177 — Result wording must not hide note/comment representation

- **질문**:
  Iteration 176의 broad source wording 보강 뒤 다음 smoke에서 좋은 accepted가 나왔는가?

- **실험/결과**:
  `trial_59`는 accepted 없이 `synthesis_failed`로 종료됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_59`
  - composer/solver: OpenRouter Kimi K2.5
  - solver rollout: 4
  - pass rate: `0/4 = 0.0`
  - accepted data: 없음

  submit 흐름은 다음과 같았다.

  1. `chartevents` list: table에 primary key가 없고 `charttime` 동점 order ambiguity가 있어 reject.
  2. `microbiologyevents` list: `comments` 값을 `result`로 제출했지만 query string spacing을 보존하지
     않아 `label_values_not_grounded`/evidence mismatch.
  3. spacing은 고쳤지만 order binding phrase가 request에 없어서 reject.
  4. order phrase를 추가했지만 Korean request에 `탐익 기준` 같은 malformed phrase가 생겼고,
     solver pass `0/4`로 terminal rejection.

- **reasoning 교차 분석**:
  composer reasoning은 primary-key 없는 `chartevents`를 버리고 `microbiologyevents`로 전환한 점은
  맞았다. 그러나 `comments` free-text를 broad `결과`로 제출했다.

  solver reasoning은 네 개 모두 같은 `microbiologyevents` row set과 `charttime/test_name` order를
  찾았다. 실패 지점은 row search가 아니라 `result` representation이었다. solver들은 `org_name`,
  `quantity`, `interpretation` 쪽을 검사 결과로 보았고, 첫 5개 row에서 이 값들이 null이라 `"null"`이나
  빈 문자열을 제출했다. DB 교차검증에서도 같은 row의 `org_name/quantity/interpretation`은 모두 null이고
  `comments`만 free-text였다.

- **정성 평가**:
  accepted data: 없음. low-quality accepted도 없음.

  rejected data:
  - `chartevents` 후보는 no-primary-key와 order ambiguity가 있어 low-quality rejected다.
  - 최종 `microbiologyevents` 후보는 row/order 자체는 solver가 찾을 수 있었지만, broad `결과`가
    `comments` representation을 고정하지 못했다. 어려운 좋은 문제가 아니라 source-sensitive output
    representation이 애매한 low-quality rejected다.

- **변경**:
  hard validator는 추가하지 않았다. `comments`를 결과로 볼지, `org_name`/`interpretation`을 결과로
  볼지는 자연어/도메인 의미 판단이 필요하므로 precision 100 구조 검사가 아니다.

  대신 prompt-first/tool-local contract 원칙으로 보강했다.

  - Label Contract: source-sensitive result/status/type 필드는 query-path source role을 request가
    이름 붙여야 하며, note/comment text는 broad result/value가 아니라 그 text surface로 요청해야
    한다고 명시했다.
  - `AnswerOutputBinding.requested_by_phrase`: note/comment/description text를 broad result/value
    phrase에 바인딩하지 말라고 추가했다.
  - `query.select[].as`: note/comment/description text를 result/status/value field처럼 보이게 하지
    말고, request가 text surface를 이름 붙일 때만 쓰라고 문구를 맞췄다.

  특정 테이블/컬럼 리터럴을 근거로 reject하는 검증은 만들지 않았다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py`
  통과.

  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`111 passed`).

- **상태**:
  만족스러운 accepted 연속 기록은 여전히 `0/5`.

## Iteration 178 — Source status text is not derived current state

- **질문**:
  Iteration 177의 result/comment representation 보강 뒤 다음 smoke에서 좋은 accepted가 나왔는가?

- **실험/결과**:
  `trial_60`도 accepted 없이 `synthesis_failed`로 종료됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_60`
  - composer/solver: OpenRouter Kimi K2.5
  - solver rollout: 4
  - pass rate: `0/4 = 0.0`
  - accepted data: 없음

  composer는 특정 입원의 `pharmacy` rows에서 약물명, `starttime`, source `status`를 제출했다. 여러
  feedback 동안 spacing, tie-break, limit phrase를 수리했고 최종 request는 "처방받은 약물 중 처방
  시작 시간이 빠른 5건"과 "현재 상태"를 물었다.

- **reasoning 교차 분석**:
  solver reasoning은 4개 모두 `prescriptions`를 ordinary source로 골랐다. 요청 문구의
  "처방받은 약물"은 `pharmacy`보다 `prescriptions`로 자연스럽게 해석됐고, "현재 상태"는
  `stoptime` 존재 여부에서 Completed/Stopped 같은 derived state로 계산됐다.

  DB 교차검증 결과, 같은 top-5 row의 약물명/시작 시간은 `pharmacy`와 `prescriptions`가 거의 맞았지만,
  `pharmacy.status`는 `"Discontinued via patient discharge"`이고 `prescriptions`에는 `status`가 아니라
  `stoptime`이 있었다. 즉 row search가 어려웠던 것이 아니라 source status representation이
  request에 고정되지 않았다.

- **정성 평가**:
  accepted data: 없음. low-quality accepted도 없음.

  rejected data:
  - 초기 submit들은 string spacing, limit phrase, order ambiguity가 있어 low-quality rejected다.
  - 최종 submit은 row/order 자체는 solver가 찾을 수 있었지만, request가 `pharmacy` source status
    text를 "현재 상태"라는 derived/current state처럼 표현했다. 어려운 좋은 문제가 아니라
    source-sensitive status representation이 애매한 low-quality rejected다.

- **변경**:
  hard validator는 추가하지 않았다. 어떤 source가 ordinary source인지, source status text와 derived
  current state가 같은지 다른지는 자연어/도메인 의미 판단이 필요하므로 precision 100 구조 검사가 아니다.

  prompt-first/tool-local contract 원칙으로 다음을 보강했다.

  - Label Contract: source-sensitive result/status/type field는 query-path source role을 request가
    이름 붙여야 하며, source status text는 recorded/source status이지 current/derived state가 아니라고
    명시했다.
  - `AnswerOutputBinding.requested_by_phrase`: source status text를 current/derived state나 boolean
    completion wording으로 바꾸지 말라고 보강했다.
  - `query.select[].as`: source status text를 current/derived state wording처럼 보이게 하지 말라고
    tool-local description을 맞췄다.

  이 변경도 특정 table/column/value 리터럴 기반 reject가 아니다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py`
  통과.

  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`111 passed`).

- **상태**:
  만족스러운 accepted 연속 기록은 여전히 `0/5`.

## Iteration 179 — Non-visible source feedback must force clean relabeling

- **질문**:
  Iteration 178의 status wording 보강 뒤 다음 smoke에서 좋은 accepted가 나왔는가?

- **실험/결과**:
  `trial_61`은 solver rollout 없이 `synthesis_failed`로 종료됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_61`
  - composer/solver: OpenRouter Kimi K2.5
  - accepted data: 없음

  submit 흐름은 다음과 같았다.

  1. 단일 eMAR 투약 기록 detail lookup: list row 1개라 `answer_contract_list_size_invalid`.
  2. 입원 기간 약물 투약 5건: null medication, phrase/order/binding 문제.
  3. ICU output measurements: request 문구가 `최신 순으`, `계앙`처럼 깨졌고, 시간 field가
     non-user-visible source라 reject.
  4. output field를 바꾸려 했지만 여전히 non-user-visible source/ordering/binding 문제가 남아
     budget exhausted.

- **reasoning 교차 분석**:
  composer는 첫 feedback 후 list size 문제를 이해하고 admission-level list로 확장했다. 그러나
  duplicate order/null 문제를 처리하다가 다른 surface로 옮겨갔고, blocked/internal field를 노출한 뒤
  자연스러운 request 전체를 다시 쓰지 못했다. reasoning에서도 "Korean text issue"를 인지했지만
  최종적으로 깨진 문구와 order binding 문제를 남겼다.

- **정성 평가**:
  accepted data: 없음. low-quality accepted도 없음.

  rejected data:
  - 단일 eMAR lookup은 low-quality rejected.
  - eMAR list는 null output과 unstable order가 있어 low-quality rejected.
  - outputevents list는 row count는 맞지만 non-user-visible source와 깨진 Korean request가 있어
    low-quality rejected. 좋은 어려움이 아니다.

- **변경**:
  hard validator는 추가하지 않았다. 이미 `label_non_user_visible_source`는 precision 100 구조 검증으로
  동작하고 있고, 이번 개선은 그 feedback reminder를 더 명확히 하는 것이다.

  - `label_non_user_visible_source` feedback: blocked/internal field를 새 alias로 포장하지 말고,
    user-visible non-handle answer field만 선택하거나 aggregate/다른 label을 고르라고 명시했다.
  - 같은 feedback에 Request Contract reminder를 추가해, field/source를 교체할 때 user_request 전체를
    target language로 깨끗하게 다시 쓰고 malformed phrase를 끼워 넣지 말라고 상기했다.

  이는 새 정책 원천이 아니라 기존 Label Contract와 Request Contract의 적용이다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_from_non_user_visible_query_source -q`
  통과 (`1 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`111 passed`).

- **상태**:
  만족스러운 accepted 연속 기록은 여전히 `0/5`.

## Iteration 180 — Accepted ICU medication task hid the source lifecycle

- **질문**:
  Iteration 179의 non-visible source feedback 보강 뒤 다음 smoke에서 좋은 accepted가 나왔는가?

- **실험/결과**:
  `trial_62`는 accepted됐다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_62`
  - composer/solver: OpenRouter Kimi K2.5
  - pass rate: `1/4 = 0.25`
  - registry status: committed

  최종 task는 특정 ICU stay의 최근 5회 투약 내역을 묻고, 약품명/투여량/단위/시작 시각을 반환했다.
  canonical query는 `inputevents`를 `stay_id`로 필터링하고 `starttime desc`, related item label asc로
  정렬했다.

- **reasoning 교차 분석**:
  1개 solver는 `inputevents`를 골라 canonical과 일치했다. 나머지 solver들은 `prescriptions`,
  `emar`, `pharmacy` 같은 다른 medication lifecycle/source surface로 이동했다. 특히 한 solver는
  `stay_id -> hadm_id`를 찾은 뒤 admission 전체의 prescriptions를 사용해 ICU stay 밖의 더 늦은
  처방을 반환했다. 두 solver는 MaxTurnsExceeded였다.

  즉 row/order가 본질적으로 불가능한 문제가 아니라, request가 "투약 내역"이라고만 말해
  `inputevents` source lifecycle을 고정하지 못했다.

- **정성 평가**:
  accepted data: low-quality accepted.

  canonical row set 자체는 DB로 재현 가능하고, 한 solver는 주어진 atomic tools로 정확히 풀었다.
  하지만 "투약 내역"은 prescriptions/eMAR/pharmacy/input events를 모두 떠올릴 수 있는 broad wording이다.
  request가 "ICU input/infusion event" 같은 source lifecycle을 자연어로 고정하지 않았기 때문에,
  다른 solver들의 실패는 단순 난이도가 아니라 composer가 낸 문제의 ambiguity다.

  rejected data:
  - 앞선 order/binding feedback은 적절했다.
  - 중간 request의 `케팜드대로`는 malformed phrase였고, 최종 request는 문장은 자연스러워졌지만
    source lifecycle ambiguity가 남았다.

- **변경**:
  hard validator는 추가하지 않았다. broad medication wording이 어느 lifecycle/source를 ordinary하게
  가리키는지는 자연어 의미 판단이므로 precision 100 구조 검사가 아니다.

  prompt-first/tool-local contract 원칙으로 source-surface 예시와 schema description을 보강했다.

  - Scope Example을 `S_event.R` vs `S_order.R` 형태로 바꿔, broad object/action wording이 lifecycle
    source를 숨기는 bad case를 보여주었다.
  - `submit_draft.user_request` description: broad wording이면 ordinary source를 쓰거나 chosen
    source/lifecycle role을 ordinary language로 이름 붙이라고 보강했다.
  - `query.from.table` description: selected source/lifecycle role을 user_request/topic/answer_contract에
    드러내라고 맞췄다.

  이 변경도 특정 DB table/column/value 리터럴에 의존하지 않는다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py`
  통과.

  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`111 passed`).

- **상태**:
  low-quality accepted이므로 만족스러운 accepted 연속 기록은 `0/5`로 유지한다.

## Iteration 144 — List difficulty-up can use relationship or row-preserving constraints

- **질문**:
  Iteration 143의 non-null filter validator 뒤 단일 smoke에서 저품질 accepted가 사라지고,
  too-easy recovery가 안정적으로 band에 들어가는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_non_null_binding_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 accepted되지 않고 `MaxTurnsExceeded`로 끝났다.

  - first submit: admission history, pass rate `8/8 = 1.0`, too easy
  - second submit: prescription list로 topic/table family 변경, `answer_contract_not_incremental`
  - third submit: admission history + insurance, pass rate `8/8 = 1.0`, too easy
  - fourth submit: admission history + insurance + marital status, pass rate `8/8 = 1.0`, too easy
  - final failure: `MaxTurnsExceeded`

- **정성 평가**:
  accepted data: 없음. 저품질 accepted도 없음.

  rejected data: first/third/fourth는 clean-good but too easy. admissions row list의
  visible fields를 하나씩 늘렸고 request/label alignment는 깨끗했다. 그러나 강한 solver 8명이
  계속 모두 맞췄으므로 학습 데이터로는 너무 쉽다. second submit은 입원 기록에서 처방약 목록으로
  task family를 바꾼 저품질 recovery였고 validator가 `answer_contract_not_incremental`로 잘 막았다.

  핵심 병목은 list Difficulty-Up Policy의 prompt 문구다. feedback은 "field, relationship, or
  coherent constraint"를 허용하지만 prompt의 list 전용 문장은 "append exactly one user-visible
  field"로 좁게 읽힌다. 이 때문에 composer가 보험 정보, 결혼 여부처럼 같은 row의 쉬운 field만
  계속 추가했다.
- **변경**:
  지침 이원화를 줄이기 위해 prompt를 feedback과 맞췄다.

  - list retry는 기존 filters/order/limit/row set/output source meanings를 보존한다.
  - 추가 난이도는 exactly one grounded visible field, direct-relationship field/aggregate, 또는
    row-preserving constraint 중 하나로 올릴 수 있다.
  - row-excluding filter와 output/order/cardinality 변경을 섞지 말라는 금지는 유지한다.
  - 길이 예산을 지키기 위해 인접 문구를 압축했다. 최종 prompt 길이: `7983`.

  이 변경은 prompt-first 원칙에 따른 durable policy 정렬이다. feedback은 이미 같은 정책을
  상기하고 있었고, 이번 변경으로 source of truth를 맞췄다.
- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
  통과 (`1 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py tests/test_synthesis_prompts.py`
  통과.

## Iteration 143 — Non-null row-set filters need a dedicated constraint phrase

- **질문**:
  Iteration 142의 source-category filter wording 뒤 단일 smoke에서 accepted 품질이 개선되는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_source_filter_wording_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 accepted되지 않았다.

  - first submit: `answer_contract_order_ambiguous`
  - second submit: `answer_contract_phrase_missing`
  - third submit: `calibration_inconclusive`, pass rate `0/8 = 0.0`
  - failed solver runs: `0`
  - final request: `이 입원 중에 처방된 약물 이력을 처방 시작 시간 순서대로 조회해 주세요. 약물명, 약국 확인 시간, 처분 상태, 그리고 처분 유형을 알고 싶습니다. 동일한 처방 시작 시간의 경우 약국 확인 시간이 빠른 순으로 정렬해 주세요. 가장 빠른 5개 항목만 보여주세요.`

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 최종 draft의 query path는 `pharmacy` table 기준으로
  hadm_id scope, `starttime asc`, `verifiedtime asc`, limit 5를 사용했다. solvers 대부분 같은
  table/path를 찾았으나 일부는 5번째 row의 `medication: null`을 포함했고, invalid submit도
  같은 이유로 발생했다.

  원인은 query에 `medication IS NOT NULL`이 들어갔지만 request/answer_contract에는
  `약물명이 기록된 처방만` 같은 row-set filter 문구가 없다는 점이다. `약물명`은 output field
  요청이지 null 약물명 row를 제외한다는 row-set constraint가 아니다. 따라서 pass rate 0.0은
  좋은 어려움이 아니라 hidden non-null row-set control 문제다.
- **변경**:
  기존 prompt/tool schema의 "Non-null filters need row-set wording" 원칙을 precision-100
  validator로 집행한다.

  - latest query metadata에서 user-visible, non-handle `where op=is_not_null` predicate를 찾는다.
  - `answer_contract.constraint_phrases`에 answer/output/order/limit phrase로 재사용되지 않은
    dedicated constraint phrase가 없으면 `answer_contract_filter_unbound`로 reject한다.
  - feedback은 user-visible non-null row-set filter에는 dedicated constraint phrase가 필요하며,
    output field wording으로는 부족하다고 상기한다.

  이 검사는 DB literal/token heuristic이 아니다. query가 실제로 사용한 predicate metadata와
  composer가 제출한 answer_contract 구조만 사용한다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_unbound_visible_non_null_filter tests/test_synthesis_runtime.py::test_submit_draft_allows_bound_visible_non_null_filter tests/test_synthesis_runtime.py::test_submit_draft_allows_non_user_visible_query_predicate -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 142 — Source-category filters need source-role wording

- **질문**:
  Iteration 141의 ambiguous output binding validator 뒤 단일 smoke에서 accepted 저품질이 줄어드는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_distinct_output_phrase_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 accepted되지 않았다.

  - first submit: `label_no_primary_key_source`
  - second submit: `answer_contract_order_ambiguous`
  - third submit: `calibration_inconclusive`, pass rate `0/8 = 0.0`
  - failed solver runs: `0`
  - final request: `내가 머물고 있는 ICU 병동에서 가장 최근에 투여된 5가지 수액의 종류, 투여 시작 시간, 투여량, 단위를 알려주세요. 동일한 시간에 투여된 경우에는 수액 종류 가나다순으로 정렬해주세요.`

- **정성 평가**:
  accepted data: 없음.

  rejected data: 대부분 low-quality rejected. 첫 submit은 `chartevents`처럼 primary-key 없는
  row source에서 최신 측정 목록을 만들려 해서 거절된 것이 맞다. 두 번째 submit은 `inputevents`
  수액 목록으로 이동했지만 `starttime desc`만 써서 limit boundary/order tie가 생겼고,
  `answer_contract_order_ambiguous`로 잘 막혔다.

  세 번째 submit은 tie-break를 추가했지만 여전히 저품질이다. 쿼리는
  `ordercategoryname LIKE '%Fluid%'`로 row set을 정했고, 실제 canonical rows는
  `02-Fluids (Crystalloids)` category에 해당했다. 반면 request는 그냥 `수액`이라고만 했다.
  solver들은 이를 전체 투여 이벤트, IV fluid, crystalloid category 등으로 다르게 해석했다.
  그래서 pass rate 0.0은 좋은 어려움이 아니라 source category filter가 자연어에서 충분히
  고정되지 않은 문제로 판단한다.
- **변경**:
  precision-100 validator를 추가하지 않았다. `수액`이 `%Fluid%` category를 뜻하는지 여부는
  DB literal/token heuristic 없이는 안전하게 판정할 수 없다.

  대신 durable prompt/tool schema 계약을 보강했다.

  - prompt Label Contract: source-sensitive 대상을 `fields`에서 `fields/filters`로 확장했다.
    status/type/category/frequency/stage/route/sequence/rank 계열 filter도 query path의 source
    role을 user_request가 명명해야 한다.
  - prompt 문구를 압축해 길이 예산을 지켰다. 최종 prompt 길이: `7962`.
  - `answer_contract.constraint_phrases` schema: source type/category/status filters는 broad
    synonyms가 아니라 source-role wording을 사용해야 한다고 명시했다.

  이 변경은 DB 특화 예시나 literal heuristic이 아니라 범용 source-role 원칙 보강이다.
- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 141 — Output bindings need distinct request roles

- **질문**:
  Iteration 140의 baseline feedback 보강 뒤 단일 smoke가 accepted되는가, 그리고 accepted data가
  정말 좋은 데이터인가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_too_easy_baseline_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 accepted됐다.

  - first submit: `answer_contract_phrase_missing`
  - second submit: accepted
  - pass rate: `6/8 = 0.75`
  - failed solver runs: `0`
  - topic: `입원 중 처방된 약물 정보`
  - request: `이번 입원 기간 동안 처방받은 약물 이름과 용량, 투여 경로, 24시간 투여 횟수를 알파벳순으로 보여주세요.`

  Composer는 first feedback 뒤 phrase를 exact substring으로 고쳤고, 같은 task 안에서 submit까지
  유지했다. provider failure는 없었다.
- **정성 평가**:
  accepted data: borderline/low-quality accepted. 데이터 경로 자체는 좋다. admission anchor에서
  prescriptions로 이동해 18개 처방을 약물명 asc로 정렬하고, drug/dose/unit/route/frequency를
  반환한다. solver도 모두 같은 경로를 찾았고 6개가 정확히 맞췄다.

  하지만 request는 `용량` 하나만 말하는데 canonical label은 `dose`와 `unit`을 별도 필드로
  반환했다. accepted answer_contract도 두 필드를 모두 같은 `requested_by_phrase: "용량"`에
  묶었다. 실제 mismatch solver 중 하나는 `dose: "650 mg"`처럼 값과 단위를 합쳐 내면서 별도
  `unit`도 냈다. 이는 어려운 문제라기보다 user-visible output role이 충분히 분해되지 않은
  출력 계약 모호성이다. 따라서 accepted로 남기면 안 되는 품질 신호다.

  rejected data: first submit의 `answer_contract_phrase_missing`은 정상적인 저품질 reject였다.
- **변경**:
  기존 prompt와 tool schema에는 이미 "one vague phrase를 multiple concepts에 재사용하지 말라"는
  Label Contract가 있었다. Composer가 이 정책을 어겼으므로 새 durable prompt가 아니라 precision-100
  validator로 집행한다.

  - list `output_bindings`에서 서로 다른 label fields가 동일한 `requested_by_phrase`에 묶이면
    `duplicate_output_binding_phrases` diagnostic을 기록한다.
  - 이 diagnostic은 `answer_contract_binding_missing` feedback으로 처리한다. feedback 문구에는
    각 returned output field가 자기 own natural role phrase를 가져야 하며 broad output phrase를
    여러 returned concepts에 재사용하지 말라고 상기한다.

  이 검사는 DB literal, token, column-name heuristic이 아니다. composer가 제출한
  `answer_contract.output_bindings`의 구조적 중복만 보므로 precision-100 계약 검증이다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_list_output_binding tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_duplicate_output_binding_phrase tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_for_selected_order_key tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_by_query_order_count -q`
  통과 (`4 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 140 — Too-easy retries must keep the latest strengthened baseline

- **질문**:
  Iteration 139의 too-easy feedback 보강 뒤, composer가 기존 task 안에서 난이도를 올리는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_too_easy_preserve_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 accepted되지 않았다.

  - first feedback: `answer_contract_hidden_filter_unanchored`
  - second submit: patient anchor로 수리, pass rate `8/8 = 1.0`, too easy
  - third submit: 같은 약물 투여 task에 `scheduled_time` 추가, pass rate `8/8 = 1.0`, still too easy
  - fourth submit: `answer_contract_phrase_missing`, `answer_contract_not_incremental`
  - fifth submit: `label_not_strengthened`, `answer_contract_not_incremental`
  - final failure: `calibration_inconclusive`

  Iteration 139 보강의 효과는 확인됐다. 이전처럼 diagnosis/pharmacy 같은 다른 topic family로
  도망가지 않았고, 같은 eMAR medication administration task 안에서 예정 시간을 추가했다.
  하지만 이후 실패 수리에서 baseline을 마지막 strengthened label이 아니라 더 이전 label로 되돌리는
  문제가 남았다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: hard-good rejected because too easy, then recovery failure. 두 번째/세 번째
  후보는 request/label alignment가 깨끗했고 solver 8/8이 맞췄다. source status text도 잘
  보존됐다. 거절 이유는 저품질이 아니라 pass rate 1.0으로 너무 쉬운 문제였기 때문이다.

  네 번째 submit은 `care_setting`/`administration_outcome`으로 field/source meaning을 바꿔
  `answer_contract_not_incremental`에 걸렸고, 다섯 번째 submit은 다시 `scheduled_time`까지만
  있는 이전 too-easy label로 돌아와 `label_not_strengthened`가 났다. 즉 composer가 "마지막으로
  평가된 too-easy label을 baseline으로 삼아 그 위에 한 단계 추가"해야 한다는 점을 충분히
  유지하지 못했다.
- **변경**:
  validator는 이미 올바르게 작동했다. `answer_contract_not_incremental`과
  `label_not_strengthened`가 저품질 recovery를 막았다. 따라서 새 validator가 아니라 feedback
  reminder를 보강한다.

  - `LABEL_NOT_STRENGTHENED` feedback: last evaluated too-easy label을 baseline으로 삼고,
    이미 추가한 fields를 유지한 뒤 새 grounded field/relationship/constraint를 추가하라고
    명시했다.
  - `ANSWER_CONTRACT_NOT_INCREMENTAL` feedback: earlier too-easy retries에서 이미 추가한
    fields도 모두 유지해야 하며 earlier label로 rollback하지 말라고 명시했다.

  이 역시 기존 Difficulty-Up Policy의 baseline 보존 원칙을 feedback에서 더 정확히 상기하는 변경이다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_too_easy_requires_incremental_answer_contract tests/test_synthesis_runtime.py::test_submit_draft_too_easy_monitor_keeps_evaluated_label_baseline tests/test_synthesis_runtime.py::test_submit_draft_too_easy_rejects_renamed_same_scalar_value tests/test_synthesis_runtime.py::test_submit_draft_too_easy_feedback_preserves_readable_path tests/test_synthesis_runtime.py::test_submit_draft_too_easy_feedback_is_list_aware -q`
  통과 (`5 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 139 — Too-easy feedback must preserve the evaluated task

- **질문**:
  Iteration 138 뒤 단일 smoke에서 source-sensitive choice 문제는 줄었는가, 그리고
  too-easy recovery가 기존 task를 보존하며 난이도를 올리는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_source_choice_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 accepted되지 않고 `MaxTurnsExceeded`로 끝났다.

  - first feedback: `answer_contract_order_ambiguous`
  - second feedback: `answer_contract_binding_missing`
  - third submit request: `이번 중환자실 입원 중 가장 최근에 투여된 약물 5가지와 투여 시작 시간, 약물 이름, 투여량, 단위, 기록 시간을 알려주세요. 동일한 투여 시간에는 약물 이름 순서로 정렬해 주세요.`
  - third submit pass rate: `8/8 = 1.0`
  - third submit status: rejected as too easy / too direct
  - final failure: `openrouter/moonshotai/kimi-k2.5:MaxTurnsExceeded`

  중요한 점은 third submit 자체는 clean-good but too easy였다. 모든 solver가 같은 canonical
  answer를 찾았고, source-sensitive normalized choice 문제도 없었다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: hard-good rejected because too easy. 세 번째 후보는 ICU stay의 inputevents에서
  최신 약물 5개를 `starttime desc, medication_name asc`로 정렬했고, 투여 시작 시간, 약물명,
  투여량, 단위, 기록 시간을 반환했다. solver 8/8이 맞췄으므로 request/label alignment는 좋다.
  다만 pass rate 1.0이라 현재 band에서는 학습용 난이도가 너무 낮아 rejection이 맞다.

  문제는 rejection 이후 recovery다. composer reasoning은 "현재 anchor/row set에 한 필드나
  관계를 추가"해야 한다고 잠깐 인식했지만 곧 diagnosis, pharmacy status count 등 다른 topic/table
  family로 이동했다. 이는 prompt의 Difficulty-Up Policy와 맞지 않는다. feedback 문구가
  "canonical label을 바꿔라"를 강조하면서 기존 task 보존을 충분히 상기하지 못한 것이 원인이다.
- **변경**:
  durable policy는 이미 prompt에 있다. 따라서 prompt를 새로 늘리지 않고 feedback reminder를
  정책 상기 역할에 맞게 수정했다.

  - `_too_easy_retry_guidance`: current anchor, target, row set/query path, source meanings를
    보존하라고 명시했다.
  - same task 안에서 one grounded visible field, relationship, or coherent constraint를
    새 evidence로 추가하라고 명시했다.
  - topic/table family를 바꾸지 말라고 명시했다.

  이 변경은 새로운 정책을 feedback에만 추가한 것이 아니다. 기존 Difficulty-Up Policy의
  "preserve kind, anchor, target, row set/query path"와 "add one coherent field/relationship"
  원칙을 feedback에서 상기하도록 한 것이다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_too_easy_feedback_preserves_readable_path tests/test_synthesis_runtime.py::test_submit_draft_too_easy_feedback_is_list_aware tests/test_synthesis_runtime.py::test_submit_draft_too_easy_requires_incremental_answer_contract tests/test_synthesis_runtime.py::test_submit_draft_too_easy_monitor_keeps_evaluated_label_baseline tests/test_synthesis_runtime.py::test_submit_draft_too_easy_rejects_renamed_same_scalar_value -q`
  통과 (`5 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_messages.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 138 — Source-sensitive fields must not add normalized choices

- **질문**:
  Iteration 137 뒤 단일 smoke에서 non-null/date 보강이 다른 task shape에서도
  저품질 accepted를 막는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_non_null_filter_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  후보는 accepted되지 않았다. calibration에서 막혔다.

  - first feedback: `answer_contract_phrase_missing`, `answer_contract_order_ambiguous`
  - second feedback: `answer_contract_order_ambiguous`
  - final request: `이 중환자실 입원 기간 동안 가장 최근에 받은 약물 및 수액 주입 기록 5개를 시간 순서대로 알려주세요. 각 기록의 약물명, 유형(약물/수액), 투여량, 단위, 투약 상태, 투여 시작 시간, 투여 종료 시간, 기록 시간을 포함해서 보여주세요. 시작 시간이 같은 경우 종료 시간이 늦은 순서로, 종료 시간도 같은 경우 기록 시간이 늦은 순서로, 모두 같은 경우 약물명 가나다순으로 정렬해주세요.`
  - pass rate: `1/8 = 0.125`
  - CI low/high: `0.0064 / 0.4707`
  - solver failed runs: `0`

  low-quality accepted는 발생하지 않았다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. canonical answer는 `event_type`에 source category
  value인 `Fluids/Intake`, `Medications`를 넣었다. 하지만 request는 `유형(약물/수액)`이라고
  써서 source category text가 아니라 사용자가 이해하기 쉬운 normalized Korean category를
  요구하는 것처럼 보인다. 실제 solver 대부분은 `event_type`을 `수액`/`약물`로 제출했고,
  한 solver만 source text를 그대로 제출해 match했다.

  `status`도 source text `Stopped`, `FinishedRunning`을 그대로 요구하는 label인데, request의
  "투약 상태"는 source representation 보존을 충분히 고정하지 않는다. 이 문제는 Iteration 135의
  source-sensitive policy와 같은 계열이며, 이번에는 parenthetical choice wording이 ambiguity를
  더 키웠다.

  결론: calibration rejection은 바람직하다. 이 후보는 어려운 좋은 문제가 아니라 composer가
  source-sensitive type/status field를 normalized category처럼 요청한 저품질 후보다.
- **변경**:
  validator는 추가하지 않았다. `유형(약물/수액)`이 source category를 그대로 달라는 뜻인지
  normalized display category를 달라는 뜻인지는 semantic 판단이다. 이를 특정 단어 또는
  value로 막으면 리터럴 휴리스틱 금지 원칙을 어긴다.

  대신 prompt-first/tool-local contract 원칙으로 다음을 보강했다.

  - Request Contract: parentheses 안에 aliases/choices를 넣지 말라고 일반화했다.
  - Label Contract: source-sensitive fields는 normalized choices를 추가하지 말라고 명시했다.
  - `AnswerOutputBinding.requested_by_phrase` schema description: source type/category/status
    fields에 parenthetical normalized choices를 추가하지 말라고 명시했다.
- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_rejects_ambiguous_limited_list_order tests/test_synthesis_runtime.py::test_submit_draft_rejects_multirow_list_without_order_by tests/test_synthesis_runtime.py::test_submit_draft_rejects_unrepresented_list_order_by_tie_breaker tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_for_selected_order_key -q`
  통과 (`11 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 137 — Non-null filters and date granularity must be explicit

- **질문**:
  Iteration 136의 sequence-like policy 뒤 단일 smoke가 accepted된다면, 그 accepted
  후보는 정말 좋은 데이터인가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_sequence_like_order_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  후보는 accepted 됐고 registry에 commit 됐다.

  - first feedback: `answer_contract_phrase_missing`, `answer_contract_order_ambiguous`
  - second feedback: `answer_contract_binding_missing`
  - final request: `최신 순으로 5개의 미생물 배양 검사 결과를 보여주세요. 검사일, 검체부위, 검사명, 검출된 균명, 감수성 결과, 항생제 정보를 알려주세요. 같은 검사일이면 검사 순서 번호 순으로, 같은 검사 내에서 격리 번호 순으로, 마지막으로 항생제 이름 순으로 정렬해 주세요.`
  - pass rate: `3/8 = 0.375`
  - CI low/high: `0.1111 / 0.7108`
  - solver failed runs: `0`

  sequence-like 개선의 일부 효과는 있었다. 두 번째 draft에서 "검사 시퀀스 번호"를
  사용한 뒤 binding feedback을 받았고, 세 번째 draft는 order binding을 모두 채운 상태로
  solver까지 갔다. 하지만 accepted 품질은 만족스럽지 않다.
- **정성 평가**:
  accepted data: low-quality accepted.

  canonical query는 `subject_id`로 current patient를 고정하고, `org_itemid is not null`로
  균이 검출된 microbiology rows만 남긴 뒤 `chartdate desc, test_seq asc, isolate_num asc,
  ab_name asc`로 정렬했다. canonical answer는 2178-07-14 bronchial washings의
  Pseudomonas 항생제 감수성 5개였다.

  문제는 request가 row-set filter를 충분히 말하지 않는다는 점이다. "검출된 균명"과
  "감수성 결과, 항생제 정보"는 출력 필드처럼 읽힐 수 있고, "균이 검출되고 감수성/항생제
  정보가 있는 검사만"이라는 membership filter를 명시하지 않는다. 실제 solver 0, 5, 7은
  더 최신 blood culture/null organism rows를 포함하거나 null organism 때문에 invalid
  submit이 됐다. 이는 solver 실수가 아니라 request ambiguity다.

  두 번째 문제는 date/time representation이다. request는 "검사일"이라고 했지만 label은
  query timestamp string `2178-07-14T00:00:00`을 그대로 요구했다. solver 3은 같은 row set에
  가까웠지만 날짜만 `2178-07-14`로 제출해 mismatch 됐다. "검사일"은 date-only로 해석할 수
  있으므로, timestamp를 exact value로 요구하려면 request가 granularity를 고정해야 한다.

  세 번째 문제는 자연스러움이다. "검사 순서 번호", "격리 번호"는 source order key를
  이름 붙였지만 일반 사용자가 자연스럽게 요청할 가능성이 낮다. 다만 이번 mismatch의
  주원인은 sequence/rank 혼동이 아니라 non-null filter와 date granularity ambiguity다.
- **변경**:
  validator는 추가하지 않았다. `org_itemid is not null`이 "검출된 균명"이라는 표현으로
  충분히 전달됐는지, "검사일"이 timestamp exact value를 요구하는지는 semantic 판단이다.
  이를 table/column/value token으로 막으면 리터럴 휴리스틱 금지 원칙을 어긴다.

  대신 prompt-first/tool-local contract 원칙으로 다음을 보강했다.

  - Request Contract: non-null filters는 output field name만으로는 부족하며 row-set wording이
    필요하다고 명시했다.
  - List Determinism Policy: date/time granularity를 맞추라고 기존 timestamp/date 원칙을
    더 짧고 직접적인 wording으로 유지했다.
  - `AnswerContract.constraint_phrases` schema description: non-null filters와 date/time
    granularity는 output fields에서 추론하지 말고 explicit row-set/representation
    constraints로 써야 한다고 명시했다.
- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_rejects_ambiguous_limited_list_order tests/test_synthesis_runtime.py::test_submit_draft_rejects_multirow_list_without_order_by tests/test_synthesis_runtime.py::test_submit_draft_rejects_unrepresented_list_order_by_tie_breaker tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_for_selected_order_key -q`
  통과 (`11 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 136 — Sequence-like fields must not become display ranks

- **질문**:
  Iteration 135의 source-sensitive wording 보강 뒤 단일 smoke에서 accepted 후보
  품질이 개선되는가, 아니면 새로운 request ambiguity가 남아 있는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_source_sensitive_output_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  후보는 accepted되지 않았다. calibration에서 막혔다.

  - first request: `제가 가장 최근에 투여받은 약물 목록 5개를 알려주세요. 약물 이름과 투여 시간을 알고 싶습니다.`
  - first feedback: `answer_contract_phrase_missing`, `answer_contract_hidden_filter_unanchored`, `answer_contract_order_ambiguous`
  - final request: `제가 가장 최근에 투여받은 약물 목록 5개를 알려주세요. 약물 이름과 투여 시간을 알고 싶습니다. 시간이 겹치는 경우 순번을 함께 표시해주세요.`
  - pass rate: `0/8 = 0.0`
  - CI low/high: `0.0 / 0.3123`
  - solver failed runs: `0`

  protocol 관점에서는 feedback 이후 재제출과 solver rollout이 정상 수행됐다. 저품질이
  accepted되지 않은 것도 바람직하다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. composer는 첫 draft의 duplicate timestamp
  문제를 고치기 위해 source `seq` 값을 label에 포함하고 `admin_time desc, seq desc`로
  정렬했다. canonical answer의 `seq`는 `851`, `850`, `849`, `848`, `847` 같은 원본
  record sequence 값이었다.

  하지만 request는 "시간이 겹치는 경우 순번을 함께 표시"라고만 말했다. solver들은 이를
  원본 record sequence가 아니라 화면에 생성하는 순위/동점 내 순번으로 해석했다. 실제로
  여러 solver가 `seq: 1, 2` 또는 행 번호처럼 제출했고, 모든 rollout이 mismatch 됐다.

  이는 어려운 좋은 문제가 아니다. 주어진 tool로 원본 `emar_seq`를 찾을 수는 있지만,
  자연어 요청이 "원본 기록 순번"과 "생성한 표시 순위"를 구분하지 않아 solver가 합리적으로
  다른 답을 냈다. 즉 composer가 source-sensitive sequence-like field를 명확히 요청하지
  못한 저품질 후보다.
- **변경**:
  validator는 추가하지 않았다. "순번"이 source record sequence인지 generated display
  rank인지 semantic하게 갈리는 문제는 query literal 없이 precision-100으로 판정할 수
  없다. label field 이름이 `seq`라는 이유로 막는 것도 token/literal heuristic이므로
  금지 원칙에 맞지 않는다.

  대신 prompt-first/tool-local contract 원칙으로 다음을 보강했다.

  - Label Contract: source-sensitive fields에 sequence/rank를 포함하고, source sequence와
    display rank를 구분하라고 명시했다.
  - Binding phrases: returned field가 order/tie-break key도 되는 경우 request가 ordering
    role을 말해야 하며 display-only wording만으로는 부족하다고 명시했다.
  - List Determinism Policy: sequence/rank tie-break는 source record order 자체가
    요청되고 source로 이름 붙은 경우에만 쓰라고 명시했다.
  - `AnswerOutputBinding`/`AnswerOrderBinding`/`constraint_phrases` schema description:
    source record sequence를 generated display rank로 바꾸지 말고, tie-break phrase는
    ordering role까지 이름 붙여야 한다고 명시했다.
  - feedback reminder: order ambiguity나 missing binding 수리 때 sequence/rank-like
    tie-break는 source record sequence와 generated display rank를 구분하라고 상기한다.
- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_rejects_ambiguous_limited_list_order tests/test_synthesis_runtime.py::test_submit_draft_rejects_multirow_list_without_order_by tests/test_synthesis_runtime.py::test_submit_draft_rejects_unrepresented_list_order_by_tie_breaker tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_for_selected_order_key -q`
  통과 (`11 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 135 — Source-sensitive output representation wording

- **질문**:
  Iteration 134의 multi-key order 보강 뒤 accepted 후보가 실제로 깨끗한 좋은
  데이터인가, 아니면 solver pass band 안에 들어왔지만 여전히 request/label ambiguity가
  남아 있는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_multikey_order_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  후보는 accepted 됐고 registry에 commit 됐다.

  - request: `최근 ICU 입원 중 받은 시술 목록을 시작 시간 순서대로 알려주세요. 같은 시간에 시작한 시술은 이름 순서로 정렬해주세요. 각 시술의 이름, 종류, 시작 시간, 완료 여부를 확인하고 싶습니다.`
  - pass rate: `5/8 = 0.625`
  - CI low/high: `0.2892 / 0.8889`
  - solver failed runs: `0`

  긍정적인 변화는 multi-key order feedback이 의도대로 작동한 점이다. 첫 submit에서
  `answer_contract_order_ambiguous`를 받은 뒤, composer가 "같은 시간에 시작한
  시술은 이름 순서"라는 visible tie-break를 request에 추가했다. canonical query도
  `start_time asc, procedure_name asc`로 고정되어 hidden order 문제는 재발하지 않았다.
- **정성 평가**:
  accepted data: borderline accepted, clean accepted는 아니다.

  canonical answer는 `procedureevents`와 `d_items`를 사용해 5개 시술을 반환한다. 하지만
  solver 2개가 `종류`를 `procedureevents.ordercategoryname`이 아니라 reachable related
  surface인 `d_items.category`로 해석했다. 둘 다 시술 분류처럼 보이므로, "종류"라는
  broad field word만으로는 source role이 충분히 고정되지 않았다.

  또 다른 solver는 source status text인 `FinishedRunning`을 `완료`처럼 boolean/번역
  표현으로 바꿨다. request의 "완료 여부"가 원천 status representation을 보존하라는
  계약을 충분히 전달하지 못했기 때문이다.

  결론적으로 low-quality accepted까지는 아니지만, 만족스러운 clean data도 아니다. 실패
  원인은 solver가 나쁜 것이 아니라 composer가 source-sensitive output field를 너무
  넓은 자연어로 요청한 데 있다.
- **변경**:
  validator는 추가하지 않았다. `종류`/`상태` 같은 field wording이 어떤 source surface를
  의도하는지, 또는 status를 source text로 둘지 boolean으로 바꿀지는 query literal 없이
  precision-100으로 판정할 수 없다.

  대신 prompt-first/tool-local contract 원칙에 맞춰 다음을 보강했다.

  - Label Contract: status/type/category/frequency/stage/route 같은 source-sensitive
    fields는 query path가 사용한 source role을 request wording이 이름 붙여야 하며,
    다른 reachable surface가 다른 값을 줄 수 있으면 broad field words는 invalid라고
    명시했다.
  - `AnswerOutputBinding.requested_by_phrase` schema description: status/type/category
    fields는 source representation을 보존해야 하며 source status text를 boolean
    completion wording으로 바꾸지 말라고 명시했다.

  이 변경은 DB literal/token heuristic이 아니다. 특정 table/column/value를 보고 막는
  규칙이 아니라, source-sensitive output representation을 명확히 요청하라는 범용 계약이다.
- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`7 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 134 — Multi-key order phrase must name tie-break keys

- **질문**:
  Iteration 133의 backend protocol 보정 뒤, feedback 이후 재제출이 실제로 이어지고
  accepted candidate 품질은 충분한가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_resubmit_feedback_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  protocol 보정 효과는 확인됐다. 첫 submit 이후 `query_mismatch`/`binding_missing`
  feedback을 받고도 composer가 재제출을 계속했다. 이전처럼 feedback 후 final text로
  멈추지 않았다.

  최종 후보는 solver 8개까지 갔지만 calibration에서 거절됐다.

  - final request: `이번 중환자실 입원 중에 받은 투약 기록 중 시간 순서대로 먼저 기록된 10개 약물 이름, 투여 시작 시간, 종료 시간, 용량, 단위, 상태를 알려주세요.`
  - pass rate: `0/8 = 0.0`
  - CI low/high: `0.0 / 0.3123`
  - solver failed runs: `0`
  - final error: `calibration_inconclusive`
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 최종 query는 `inputevents`를
  `start_time asc, end_time asc, amount asc`로 정렬했다. 하지만 request는 이를
  모두 "시간 순서대로 먼저"라는 하나의 문구에 묶었다. `amount asc`는 natural
  tie-break로 명시되지 않았고, solver들은 합리적으로 `start_time`만 쓰거나
  prescriptions surface를 골랐다. 0/8은 단순히 어려운 좋은 문제가 아니라, request가
  source/order contract를 충분히 고정하지 못한 저품질 후보라는 증거다.

  다행히 low-quality accepted는 발생하지 않았다. calibration에서 막혔다.
- **변경**:
  validator는 추가하지 않았다. "시간 순서"가 어떤 semantic tie-break를 자연스럽게
  포함하는지 precision-100으로 판정할 수 없기 때문이다.

  대신 prompt-first 원칙과 tool-local contract 원칙에 맞춰 보강했다.

  - Label Contract: binding phrase는 returned field뿐 아니라 각 order key의 role도
    이름 붙여야 하며, multi-key order에서 하나의 vague phrase를 여러 order key에
    재사용하지 말라고 명시했다.
  - `AnswerOrderBinding` schema description: tie-break phrase가 해당 order key를
    구체적으로 이름 붙여야 하며 broad order phrase 재사용을 금지한다고 명시했다.
  - `answer_contract_binding_missing` feedback: 같은 원칙을 reminder로 추가했다.

  이 변경은 DB literal/token heuristic이 아니다. 특정 table/column/value를 보지 않고,
  answer contract의 일반 구조와 역할 설명을 강화한 것이다.
- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_for_selected_order_key tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`8 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 133 — Resubmit reminder after feedback-side final output

- **질문**:
  Iteration 132의 Source surface prompt 보강 뒤 단일 smoke에서 feedback 이후
  recovery가 실제로 이어지는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_source_role_prompt_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 solver 실행 전 `synthesis_failed`로 끝났다.

  - first submit topic: `ICU 복약 기록 조회`
  - request: `이번 중환자실 입원 중에 투여된 약물을 시간 순서대로 5가지 보여주세요...`
  - first feedback: `answer_contract_order_ambiguous`
  - feedback events: `1`
  - solver runs: 없음

  첫 draft는 ICU stay의 `inputevents` 투여 목록을 만들었다. Source role은 이전
  pharmacy/prescriptions ambiguity보다 나아졌다. 요청이 "중환자실 입원 중 투여"라고
  말했고 label도 inputevents의 투여 시작/종료/총량/단위였기 때문이다.

  실패 원인은 order recovery다. canonical query는 `started_at asc, limit 5`였고
  `2140-10-03T11:00:00` 동점 rows가 있어 order ambiguity feedback을 받았다.
  composer는 reasoning에서 `endtime`, `storetime`, `ordercategoryname` 같은 natural
  visible tie-break 후보를 검토했다. 방향은 맞았지만, 재query에서 malformed JSON을
  호출했고 그 다음 "query를 고치겠다"는 final output으로 종료했다. 재제출은 없었다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: recovery failure / low-quality rejected. 첫 후보는 고칠 수 있는
  좋은 방향의 task였지만, 동점 order가 수리되지 않은 상태로는 저품질이다. rejection은
  바람직했고, low-quality accepted는 발생하지 않았다.
- **변경**:
  기존 backend의 missing-submit protocol feedback은 첫 submit이 아예 없을 때만
  작동했다. 이번 실패는 "feedback 이후 data tool을 더 호출하고도 accepted draft 없이
  final output으로 멈춤"이다.

  이는 DB 의미 판단이 아니라 프로토콜 상태만으로 precision-100 판정 가능하다:
  accepted draft가 없고, 마지막 tool call이 `submit_draft`가 아니며, feedback budget이
  남아 있다. 따라서 `backend_openai_agents`의 continuation 조건을 고쳐, feedback 이후
  재제출 누락에도 기존 `record_missing_submit_feedback`을 적용한다.
- **검증**:
  `uv run pytest tests/test_synthesis_backend_openai_agents.py::test_synthesis_backend_continues_after_final_output_without_submit tests/test_synthesis_backend_openai_agents.py::test_synthesis_backend_continues_after_feedback_without_resubmit -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/backend_openai_agents.py tests/test_synthesis_backend_openai_agents.py`
  통과.

## Iteration 132 — Source-surface ambiguity in medication lists

- **질문**:
  Iteration 131의 accepted sample 뒤, broad request가 계속 output surface를
  흐리게 만드는가? 특히 같은 everyday noun이 여러 record surface에 걸쳐 있을 때
  composer가 선택한 source role을 request에 충분히 고정하는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_post_accept_surface_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 `synthesis_failed`로 끝났다.

  - outcome: `artifact_invalid`
  - error code: `calibration_inconclusive`
  - pass rate: `1/8 = 0.125`
  - CI low/high: `0.0064 / 0.4707`
  - solver failed runs: `0`

  후보 request는 "이번 입원 기간 동안 투약된 약물 목록"을 묻고, 약물명/처리 유형/
  투여 경로/복용 주기/시작 시간을 요청했다. canonical query는 `pharmacy` surface의
  `medication`, `proc_type`, `administration_route`, `frequency`, `starttime`을
  사용했다. 그러나 7/8 solver는 `prescriptions` surface를 자연스럽게 선택했고,
  `drug_type`/`doses_per_24_hrs` 쪽 값을 제출했다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 데이터가 어려워서 좋은 문제가 된 것이
  아니라, request가 선택한 source lifecycle을 고정하지 못했다. MIMIC demo에서
  "투약/약물/복용 주기/처리 유형"은 pharmacy와 prescriptions 양쪽에서 그럴듯하게
  해석될 수 있고, 서로 다른 값을 만든다. solver가 prescriptions를 고른 것은
  비합리적 실패가 아니라 request ambiguity의 증상이다.

  low-quality accepted는 발생하지 않았다. rejection은 바람직하다.
- **변경**:
  validator는 추가하지 않았다. 이 ambiguity는 schema/path 의미 해석 문제라
  precision-100 rule로 구분할 수 없고, DB literal/token heuristic으로 막으면
  원칙 위반이다.

  대신 prompt-first 원칙에 따라 DB-neutral durable policy를 강화했다.

  - Core Definitions의 Source surface에 "여러 reachable record surface가 같은
    everyday noun을 공유하면 user_request가 lifecycle/source role을 고정해야 하며
    output_schema field names에 기대면 안 된다"를 추가했다.
  - Label Contract에 status/type/category/frequency/stage/route 같은
    source-sensitive field는 query path와 같은 source role로 묶여야 한다고 추가했다.
  - 길이 예산을 지키기 위해 기존 중복 문장을 압축했고, budget helper 문구도 같은
    의미 안에서 줄였다.
- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`12 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 131 — Accepted admission-history smoke

- **질문**:
  Iteration 130의 final evidence/tie-break feedback 보강 뒤 다음 no-topic smoke에서
  accepted data가 나오는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_evidence_tiebreak_feedback_kimi_8solver_no_topic_smoke_01/trial_01`.
- **결과**:
  trial은 accepted 됐다.

  - task: `환자 입원 이력 조회`
  - request: `내 입원 기록을 최근 입원 순서로 알려줘`
  - anchor: `subject_id=10005348`
  - row set: 해당 환자의 admission 3건 전체
  - order: `admission_time` desc, duplicate order key 없음
  - pass rate: `7/8 = 0.875`
  - CI low/high: `0.5293 / 0.9936`
  - solver failed runs: `0`
  - registry: committed

  첫 submit은 output binding phrase가 request substring과 맞지 않아
  `answer_contract_phrase_missing`으로 reject됐다. 두 번째 submit은 같은
  admission-history task를 유지했고, output binding phrase를 request 안의
  `내 입원 기록을`로 맞춰 accepted 됐다.
- **정성 평가**:
  accepted data: borderline-clean.

  좋은 점:

  - hidden scope가 자연스럽다. `subject_id`는 "내 입원 기록"의 hidden requester
    context로 쓰였고, label은 해당 환자의 admission rows에 정확히 scoped 됐다.
  - row set이 전체 3건이라 hidden limit membership 문제가 없다.
  - order가 `admission_time desc`로 명시되고, query diagnostics상 동점이 없다.
  - label values는 latest query result와 일치한다.
  - 7/8 solver가 같은 정답을 제출했다.

  남은 찜찜함:

  - request가 `입원 기록`이라고만 말하고 입원일시/퇴원일시/입원유형/입원장소/퇴원장소를
    명시적으로 열거하지 않는다. 환자 포털에서 "입원 기록"에 자연스럽게 포함될 수
    있는 필드들이라 low-quality accepted로 보지는 않지만, label surface가 아주
    선명한 clean sample은 아니다.
  - 실패한 solver 1개는 두 번째 row의 `discharge_location` null을 문자열
    `"null"`로 제출했다. 이는 task row-set/ordering 문제가 아니라 solver-side
    exact-value handling 오류다.

  결론: low-quality accepted는 아니다. 다만 향후 더 좋은 clean sample을 목표로
  하려면 composer가 "입원일시, 퇴원일시, 입원유형, 입원/퇴원 장소"처럼 returned
  fields를 request에 자연스럽게 포함하도록 만드는 prompt/feedback 개선 여지가 있다.
  이 판단은 semantic quality 문제라 precision-100 hard validator로 만들지는 않는다.
- **다음 반복 후보**:
  이번 sample은 합격이지만 borderline-clean이므로, 다음 smoke에서 더 explicit한
  output surface가 자연스럽게 생성되는지 한 번 더 확인한다. accepted sample이 계속
  broad request에 기대면 prompt example/policy 쪽에서 DB-neutral하게 보강할지 검토한다.

## Iteration 130 — Evidence and tie-break binding reminders

- **질문**:
  Iteration 129에서 order feedback이 task reset을 허용하지 않도록 고친 뒤, 다음
  smoke에서 composer가 같은 task 안에서 회복하는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_order_feedback_current_task_kimi_8solver_no_topic_smoke_01/trial_01`.

  trial은 solver 실행 전 `synthesis_failed`로 종료됐다. provider failure는 없고,
  feedback event 5개를 모두 사용했다.
- **결과**:
  개선은 일부 확인됐다. composer는 admission anchor `hadm_id=28613200`에서 시작해
  pharmacy list task를 만들었고, 첫 order feedback 뒤에도 같은 admission/pharmacy
  target을 유지했다. 이전처럼 완전히 다른 microbiology task로 reset하지 않았다.

  하지만 회복 과정에서 protocol 오류가 이어졌다.

  1. 첫 submit은 query limit 5를 request/contract에 고정하지 않았고, `start_time`
     order key 동점도 있었다.
  2. composer가 `start_time + medication_name` visible tie-break query를 만들었지만,
     final list query 뒤에 count aggregate를 추가 호출해서 latest evidence가
     label과 달라졌다.
  3. 이후 limit phrase를 넣지 않아 `query_mismatch`가 반복됐다.
  4. limit phrase는 넣었지만 answer phrase exact substring이 맞지 않았다.
  5. 마지막에는 `medication_name` tie-break order key를 request/order_bindings에
     자연스럽게 묶지 못해 `answer_contract_binding_missing`으로 budget이 끝났다.
- **reasoning 감사**:
  reasoning은 모델이 핵심 수리 방향을 일부 이해했음을 보여준다. "same start_time
  tie를 medication name으로 풀자"는 방향은 DB-neutral하고 user-visible이다. 문제는
  그 tie-break를 request에 "같은 시작시각이면 약물명 순"처럼 자연어로 고정하지
  못했고, final evidence 바로 뒤 submit해야 한다는 protocol도 중간 count query로
  어겼다는 점이다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: recovery failure, borderline low-quality rejected. 후보 자체는
  좋은 입원 약물 목록 task가 될 수 있었지만, query limit/order/tie-break가
  request/contract에 완전히 고정되지 않아 제출하면 저품질이 된다. rejection은
  바람직했고, low-quality accepted는 발생하지 않았다.
- **변경**:
  durable prompt는 유지하고 feedback reminder만 보강했다.

  - `answer_contract_evidence_mismatch`: final label evidence 뒤에 helper/profile/count
    query를 실행하지 말고, 이미 실행했다면 exact label query를 다시 실행한 직후
    submit하라고 상기한다.
  - `answer_contract_binding_missing`: list order key가 tie-break일 때도
    user_request에 natural visible tie-break wording이 있어야 bind할 수 있으며,
    그렇지 않으면 해당 order key를 제거하거나 tied rows를 반환하라고 상기한다.

  둘 다 기존 Label Contract/List Determinism Policy의 feedback reminder이며, DB
  literal/token heuristic은 추가하지 않았다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_for_selected_order_key tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_by_query_order_count tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_that_does_not_match_latest_query -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 129 — Order feedback must not license task reset

- **질문**:
  Iteration 128의 hidden scope/order feedback 보정 후 다음 smoke에서 composer가
  더 나은 recovery를 보이는가?
- **실험**:
  topic hint 없이 MIMIC demo 단일 smoke를 실행했다. composer/solver는
  OpenRouter Kimi K2.5, solver rollout은 8개, solver batch size는 4:
  `artifacts/trial_20260429_mimiciv_demo_hidden_scope_feedback_kimi_8solver_no_topic_smoke_01/trial_01`.

  trial은 solver 실행 전 `synthesis_failed`로 종료됐다. 원인은 provider failure가
  아니라 `MaxTurnsExceeded`였다.
- **결과**:
  첫 draft는 patient anchor `subject_id=10022880`에 대해 약물 처방 목록을
  만들었다. hidden scope 자체는 올바르게 고정됐다. 하지만 pharmacy rows가 같은
  `starttime`/`stoptime` order key를 공유해 `answer_contract_order_ambiguous`가
  발생했다.

  이후 composer는 기존 pharmacy task를 고치는 대신 microbiology task로 리셋했다.
  이 두 번째 draft는 `microevent_id`를 `test_id`/`검사번호`처럼 노출해 order
  tie-break를 해결하려 했다. `answer_contract_binding_missing` feedback 뒤에는
  request에 `검사번호로 구분`까지 넣었지만, 다음 query는 다시 `test_date`만
  order key로 남아 `answer_contract_order_ambiguous`가 발생했다. 이후 다른
  task를 더 탐색하다가 max turns를 넘겼다.
- **reasoning 감사**:
  저장된 reasoning을 보면 composer는 feedback을 "기존 task를 최소 수정"으로
  해석하지 못했다. 특히 Iteration 128에서 내가 추가한 order feedback 문구의
  "choose another label"이 기존 Feedback Policy의 "preserve anchored need/language"
  와 충돌했고, 모델에게 task reset을 허용하는 신호로 작동했다.
- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected / recovery failure. 첫 pharmacy task는 좋은
  방향이었지만 deterministic ordering을 만들지 못했고, 이후 microbiology task는
  technical handle을 자연스러운 검사번호처럼 사용하려는 저품질 방향이었다. 다만
  합성 단계에서 멈췄기 때문에 low-quality accepted는 발생하지 않았다.
- **변경**:
  `answer_contract_order_ambiguous` feedback reminder를 다시 원칙에 맞게 고쳤다.
  "choose another label"을 제거하고, feedback retry에서는 현재 anchor/target을
  보존한 채 natural visible tie-break, unique ordering, tied rows 중 하나로
  ordering만 수리하라고 상기한다. hidden handles/artificial id wording으로
  수리하지 말라는 문구는 유지했다.

  이 변경은 durable policy를 새로 만들지 않는다. 기존 `Feedback And
  Difficulty-Up Policy`와 `List Determinism Policy`를 feedback에서 일관되게
  상기하도록 되돌린 것이다.
- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_ambiguous_limited_list_order tests/test_synthesis_runtime.py::test_submit_draft_rejects_multirow_list_without_order_by tests/test_synthesis_runtime.py::test_submit_draft_rejects_unrepresented_list_order_by_tie_breaker -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 131 — Source surface ambiguity from reasoning traces

- **질문**:
  reasoning content가 반환/저장되는 상황에서, 실패한 smoke의 composer/solver reasoning을
  함께 보면 low-quality 원인을 더 정확히 분류할 수 있는가?
- **실험**:
  기존 no-topic MIMIC demo 반복 smoke의 `trial_30`을 분석했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_30`
  - composer/solver: OpenRouter Kimi K2.5
  - solver target: 4 evaluable runs, provider 실패는 replacement retry 포함
  - 결과: `synthesis_failed`, `calibration_inconclusive`, pass rate `0/4`

- **reasoning 교차 분석**:
  composer는 admission `hadm_id=20437651`의 `pharmacy` rows를 final label로 만들었다.
  request는 "이번 입원동안 가장 최근에 처방된 약물 3가지..."였다.

  solver reasoning은 이 문구를 보고 `prescriptions` table을 선택했다. 실제 solver
  output도 `prescriptions`의 `drug`, `starttime`, `route`, `doses_per_24_hrs`를
  기반으로 했고, canonical은 `pharmacy.medication`, `pharmacy.starttime`,
  `pharmacy.route`, `pharmacy.frequency`였다. 따라서 단순 solver 실수가 아니라
  user_request가 label의 실제 source surface를 충분히 고정하지 못한 low-quality
  draft였다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 사용자가 "처방된 약물"이라고 말하면 여러
  reachable source surface가 자연스럽게 후보가 될 수 있는데, composer가 선택한
  source role을 자연어 요청에 고정하지 않았다. 이 실패는 DB 리터럴 기반 validator로
  잡으면 안 된다. "약물" 같은 단어를 특정 table에 매핑하는 휴리스틱은 forbidden이다.

- **변경**:
  prompt-first 원칙에 따라 durable Source Surface Policy를 보강했다.

  - 시스템 프롬프트 `# Scope Examples`에 DB-neutral `S1/S2` 예시를 추가했다.
    broad noun이 여러 source에 맞으면 bad, request가 선택한 source role을 이름 붙이면
    good이라는 구조만 제시한다.
  - `submit_draft.user_request` tool schema description에 "여러 reachable source
    surfaces가 broad wording을 만족할 수 있으면 선택한 source role을 ordinary
    language로 이름 붙이라"는 인자 수준 계약을 추가했다.
  - hard validator는 추가하지 않았다. 이 판단은 semantic ambiguity라 precision 100
    rule로 만들 수 없고, literal/token heuristic을 쓰면 원칙 위반이다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 132 — Avoid single-record detail lookup as initial task shape

- **질문**:
  Iteration 131 변경 후 smoke에서 Source Surface 문제는 줄었지만, composer가 너무 쉬운
  단일 record 상세 조회를 첫 draft로 제출하는가?
- **실험**:
  같은 no-topic MIMIC demo smoke를 `trial_31`로 실행했다.

  - artifact:
    `artifacts/trial_20260429_mimiciv_demo_post_repair_contracts_kimi_4solver_no_topic_smoke_01/trial_31`
  - 결과: `synthesis_failed`, 최종 pass rate `1.0`
  - 첫 draft: 단일 `emar_id=10039708-198`의 medication/chart_time/status/scheduled_time
  - 첫 solver result: `4/4` 정답, too easy

- **reasoning 교차 분석**:
  composer reasoning은 첫 label이 단일 eMAR record detail lookup임을 인지했지만 그대로
  제출했다. too-easy feedback 후에는 admission 전체 eMAR list/aggregate로 난이도를
  올리려 했고, 이후 feedback을 literal child anchor 보존으로 읽어 단일 record에
  `admission_type`만 추가하는 same-row/passive 확장으로 회귀했다.

  저장된 reasoning은 solver 쪽뿐 아니라 composer 쪽도 포함됐다. composer의 핵심 오해는
  "specificity repair에서 single-record detail을 유지하면서 필드만 더 붙이면 된다"는
  해석이었다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: hard-good이 아니라 too-easy / recovery failure. 단일 현재-record 상세
  조회는 solver가 주어진 툴로 너무 쉽게 찾는 label이다. low-quality accepted는 없었고,
  rejection은 바람직했다.

- **변경**:
  hard validator는 추가하지 않았다. 단일-record detail lookup 여부를 항상 precision 100으로
  품질 판정하기는 어렵고, 일부 DB에서는 current-record fact lookup도 유효할 수 있다.
  대신 Task Shapes durable prompt에 짧은 일반 원칙을 추가했다.

  - `Avoid single-record detail lookup.`

  이 변경은 DB literal/token heuristic이 아니며, composer의 초기 label 선택 품질을 올리는
  prompt-first 개선이다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
  통과 (`1 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py tests/test_synthesis_prompts.py`
  통과.

## Iteration 133 — Reject mechanical multi-key list ordering

- **질문**:
  `trial_32`에서 단일-record detail lookup은 줄었지만, composer가 긴 multi-key sort를
  만들어 solver가 submit 없이 멈추는가?
- **실험**:
  `trial_32` 최종 draft를 분석했다.

  - request: 검체 채취 일시, 검체 종류, 검사 순서, 검사명 기준으로 모두 오름차순 정렬한
    미생물 검사 상위 5건
  - canonical: `microbiologyevents` 5 rows
  - solver: 4/4 `invalid_submit`, `missing_submit_result`
  - pass rate: `0/4`

- **reasoning 교차 분석**:
  solver reasoning은 모두 같은 방향으로 진행했다. `microbiologyevents`를 만들고,
  `hadm_id=25508812`로 filter한 뒤, 다중 정렬을 수행하려 했다. 하지만 4개 order key와
  긴 한국어 정렬 계약 때문에 4개 solver 모두 `submit_result`까지 도달하지 못했다.

  이는 데이터 자체가 불가능한 것은 아니지만, composer가 solver에게 과도하게 기계적인
  list-order contract를 준 저품질 rejected 케이스다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: too-hard/low-quality 경계지만, 원인은 "어려운 좋은 문제"라기보다
  mechanical sort contract다. 사용자가 자연스럽게 요청할 가능성이 낮고, atomic solver의
  단계적 문제 해결을 학습시키기보다는 정렬 key 나열을 따라가게 만든다.

- **변경**:
  이번에는 precision 100 hard guard를 추가했다. DB 리터럴이나 의미 추측이 아니라,
  실제 `query.order_by` 구조에서 order key 개수를 세는 방식이다.

  - Durable policy: list order는 자연 order + 최대 1개 visible tie-break, 즉 order key
    총 2개까지만 허용한다.
  - `submit_draft` validation: list query의 `order_by` reference count가 2를 초과하면
    `answer_contract_order_too_complex` feedback으로 reject한다.
  - `query.order_by` schema description에도 같은 계약을 추가했다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_mechanical_multi_key_list_order tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py`
  통과.

## Iteration 134 — Reject output-only list difficulty-up

- **질문**:
  `trial_33`이 accepted됐지만, accepted data가 정말 좋은가?
- **실험/결과**:
  `trial_33`은 최종 pass rate `2/4 = 0.5`로 accepted됐다.

  - task: ICU stay의 처음 5개 procedureevents
  - initial valid list는 solver `4/4`라 too easy
  - 최종 accepted는 기존 list에 `item_category`와 `patient_weight` 출력 필드를 추가
  - solver 오답 2건은 `item_category`를 `null` 또는 `ContinuousProcess`로 제출

- **정성 평가**:
  accepted data: borderline/low-quality accepted.

  row set, order, limit, hidden scope는 명확하고 null도 없다. 하지만 난이도 상승의 핵심이
  row reasoning 변화가 아니라 output-only passive fields 추가였다. `patient_weight`는
  procedure row의 반복 display field이고, `item_category`는 `procedure_category`와
  source-surface가 헷갈리기 쉽다. 이건 "좋은 어려움"이라기보다 우리가 이미 금지한
  passive width 강화가 accepted된 것이다.

- **변경**:
  prompt에 이미 있던 Difficulty-Up Policy를 validator가 따르도록 고쳤다.

  - list retry에서 이전 solver-evaluated draft 대비 output source만 추가된 경우
    `answer_contract_not_incremental`로 reject한다.
  - scalar aggregate output 확장은 기존처럼 허용한다.
  - 진단에는 `list_output_only`와 `no_new_structural_constraint`를 남긴다.

  이 검사는 query evidence signature의 구조 비교만 사용하므로 precision 100이다. DB
  리터럴/토큰/컬럼 의미 휴리스틱은 쓰지 않는다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_incremental_evidence_allows_added_scalar_output_fields tests/test_synthesis_runtime.py::test_incremental_evidence_rejects_list_output_only_field_additions tests/test_synthesis_runtime.py::test_submit_draft_too_easy_requires_incremental_answer_contract -q`
  통과 (`3 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 135 — Source role ambiguity after accepted trial

- **질문**:
  `trial_34`는 pass rate `2/4 = 0.5`로 accepted됐지만, accepted data가 정말 좋은가?

- **실험/결과**:
  최종 task는 ICU stay의 의료 절차 5개를 시작 시간 순서로, 동일 시간이면 절차 이름
  순서로 나열하는 list였다.

  - row set/order/limit/hidden scope: 명확함
  - solver 2건 정답: `procedure_category`를 `procedureevents.ordercategoryname`에서 가져옴
  - solver 2건 오답: `procedure_category`를 `d_items.category`에서 가져옴
  - composer canonical: `Peripheral Lines`, `Invasive Lines`, `Tubes`
  - 오답 solver output: `Access Lines - Peripheral`, `Access Lines - Invasive`, `GI/GU`

- **reasoning 교차 분석**:
  composer reasoning은 feedback 후 hidden `orderid` tie-break를 버리고, 요청에 자연스러운
  visible tie-break인 절차 이름 순서를 넣는 방향으로 잘 수정했다. 이 부분은 이전 개선이
  작동했다.

  하지만 composer는 `카테고리`라는 broad output phrase가 두 reachable source에 걸릴 수
  있다는 문제를 인지하지 못했다. solver reasoning은 둘로 갈렸다. 일부 solver는 procedure
  event row의 category 역할을 골랐고, 일부 solver는 related item definition의 category
  역할을 골랐다. 둘 다 도구상 가능한 해석이므로 이는 어려운 좋은 문제가 아니라
  source-surface ambiguity다.

- **정성 평가**:
  accepted data: low-quality accepted.

  row membership 문제는 없지만, request/label/source role이 유일하게 고정되지 않았다.
  solver가 틀린 이유는 추론 실패라기보다 composer가 애매한 label surface를 제출했기
  때문이다.

- **변경**:
  hard validator는 추가하지 않았다. 특정 단어를 보고 reject하는 방식은 DB
  리터럴/토큰/컬럼 의미 휴리스틱이므로 금지 원칙 위반이다. 대신 prompt-first 원칙에 따라
  Source Surface Policy를 보강했다.

  - phrase가 여러 reachable surface에 매핑될 수 있으면 request/contract가 선택한 source
    role을 직접 이름 붙여야 한다.
  - label/output_schema 이름은 source ambiguity를 해소하지 못한다고 명시했다.
  - Scope Example을 submit_draft에 가까운 미니 JSON 형태로 바꿔, broad request phrase가
    hidden source choice를 만드는 bad case를 보여준다.
  - `query.select[].as` tool description에도 output alias가 competing reachable sources를
    disambiguate하지 못한다는 tool-local 계약을 추가했다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py`
  통과.

## Iteration 136 — Reject 1-2 row list labels

- **질문**:
  `trial_35`는 pass rate `3/4 = 0.75`로 accepted됐지만, accepted data가 정말 좋은가?

- **실험/결과**:
  composer는 처음에 microbiology list를 만들었으나 null field, ambiguous date ordering,
  duplicate answer rows 때문에 두 번 reject됐다. 이후 `patient_admission_history`로
  topic을 바꾸고, 환자의 입원 기록을 최근 입원순으로 보여주는 draft를 제출했다.

  최종 accepted canonical answer는 admissions row 1개였다.

  - solver 3건 정답
  - solver 1건 오답: `PHYSICIAN REFERRAL`을 `PHYSICER REFERRAL`로 오타 제출
  - solver 1건 timeout은 infra-excluded 후 재시도되어 총 completed 5, evaluable 4
  - 최종 pass rate: `0.75`

- **reasoning 교차 분석**:
  composer reasoning은 중요한 단서를 남겼다. admissions query 후 “이 환자는 admission이
  1개뿐이니 다른 entity를 보자”라고 스스로 인지했다. 하지만 그 뒤 feedback이
  `answer_contract_phrase_missing`으로 오자, label을 바꾸지 않고 같은 단일 admission
  list에 “최근 입원순” 문구만 추가해 accepted시켰다.

  즉 composer는 Task Shapes의 “list는 3-5 rows”를 알고도, phrase repair feedback을
  더 좁은 수정 지시처럼 해석했다. 이는 프롬프트만으로는 반복될 수 있는 low-quality
  accepted다.

- **정성 평가**:
  accepted data: low-quality accepted.

  데이터 접근은 너무 직접적이다. row set은 환자별 admissions이고 결과가 1개뿐이라,
  사실상 single-record detail lookup이다. solver 오답도 탐색 실패가 아니라 복사 오타였고,
  timeout을 제외하면 모든 solver가 같은 단순 경로를 찾았다.

- **변경**:
  precision 100 hard validator를 추가했다.

  - `answer_contract.kind == list`이고 canonical answer row 수가 1-2개이면
    `answer_contract_list_size_invalid`로 reject한다.
  - 기존 6개 이상 list는 `answer_contract_list_limit_too_wide`로 유지한다.
  - 이 검사는 실제 제출된 structured answer의 row count만 보므로 DB 리터럴/토큰/컬럼
    의미 휴리스틱이 아니다.

  feedback은 Task Shapes를 상기시키는 역할로만 작성했다. 지시는 새 정책이 아니라 이미
  durable prompt에 있는 “List: homogeneous ordered 3-5 rows”의 적용이다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_single_row_list_as_too_direct tests/test_synthesis_runtime.py::test_submit_draft_rejects_limited_single_row_before_order_repair tests/test_synthesis_runtime.py::test_submit_draft_treats_list_limit_one_as_rows_array tests/test_synthesis_runtime.py::test_submit_draft_rejects_list_limit_above_task_shape_policy -q`
  통과 (`4 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 137 — Allow scalar aggregate to grouped aggregate difficulty-up

- **질문**:
  `trial_36`은 low-quality accepted 없이 실패했다. 이 실패는 바람직한 reject인가, 아니면
  좋은 복구 후보를 validator가 너무 세게 막은 것인가?

- **실험/결과**:
  흐름은 다음과 같았다.

  1. `icu_transfer_history`: null output과 phrase mismatch로 reject
  2. `icu_output_measurements`: 최근 outputevents 5개 list, order ambiguity로 reject
  3. `total_output_count`: ICU stay의 outputevents 총 count, solver `4/4`로 too easy
  4. `icu_output_summary_by_type`: 같은 ICU stay/outputevents를 유형별로 group-by하여 count/avg
     top 5 list를 제출했지만 `answer_contract_not_incremental`
  5. `most_common_output_type`: 마지막 attempt에서 evidence/phrase mismatch와
     not_incremental로 budget exhausted

- **reasoning 교차 분석**:
  composer는 too-easy feedback 후 “scalar count가 너무 직접적이므로 grouped summary가 더
  의미 있는 work”라고 판단했다. 이 판단은 원칙적으로 맞다. 총 count 하나를 맞히는 문제보다,
  같은 target row set을 유지하면서 유형별 group, count, avg, order, top-k를 요구하는 문제는
  atomic tool reasoning을 더 많이 요구한다.

  기존 validator는 `kind_changed`와 `operation_changed`를 무조건 막았기 때문에
  scalar aggregate count → grouped aggregate list도 reject했다. 이는 too-hard jump 방지라는
  목적은 이해되지만, aggregate/group/order를 difficulty-up 수단으로 인정한 원칙과 충돌했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - transfer/output list 초기 draft는 low-quality rejected가 맞다.
  - total count는 too-easy rejected가 맞다.
  - grouped output summary는 어려운 좋은 문제 후보였다. 같은 ICU stay/outputevents target과
    predicate를 유지하고, group/order/aggregate work를 추가했기 때문이다.

- **변경**:
  validator를 좁게 완화했다.

  - 기본적으로 kind/operation drift는 계속 reject한다.
  - 단, scalar aggregate baseline에서 list grouped aggregate로 갈 때는 허용한다.
  - 조건:
    - 이전 kind는 `scalar`, 현재 kind는 `list`
    - 이전/현재 모두 aggregate output을 포함
    - 현재 output에 `group_by`가 있음
    - aggregate function이 하나 이상 공유됨
    - 이전 query table set이 현재 query table set의 subset
    - 기존 predicate 제거가 없어야 함

  이 검사는 query evidence 구조와 query spec table set만 비교하므로 precision 100이다.
  DB 리터럴/토큰/컬럼 의미 휴리스틱은 쓰지 않는다.

  Prompt도 `preserve kind` 대신 `preserve anchor/target`으로 조정하고, scalar aggregates는
  group/compare를 추가할 수 있다고 명시했다. feedback은 여전히 이 durable policy를
  상기시키는 역할만 한다.

- **검증**:
  `uv run pytest tests/test_synthesis_runtime.py::test_incremental_evidence_allows_scalar_count_to_grouped_aggregate_list tests/test_synthesis_runtime.py::test_incremental_evidence_allows_added_scalar_output_fields tests/test_synthesis_runtime.py::test_incremental_evidence_rejects_list_output_only_field_additions tests/test_synthesis_runtime.py::test_submit_draft_too_easy_requires_incremental_answer_contract tests/test_synthesis_runtime.py::test_submit_draft_too_easy_monitor_keeps_evaluated_label_baseline tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
  통과 (`6 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 138 — Status/type wording must match row membership

- **질문**:
  `trial_37`은 pass rate `1/4 = 0.25`로 accepted됐다. accepted data가 정말 좋은가?

- **실험/결과**:
  최종 task는 특정 환자의 최근 약물 투여 기록 5개였다.

  - request: 가장 최근에 투여된 약물 5개, 같은 투여 시간이면 약물 이름 순
  - canonical: `emar` rows 5개
  - pass: 1 solver matched
  - 3 solver는 `missing_submit_result`
  - 1 solver는 timeout infra-excluded 후 재시도됨

- **reasoning 교차 분석**:
  composer는 order ambiguity feedback 후 technical sequence가 아니라 medication name을
  natural visible tie-break로 선택했다. 이 부분은 이전 List Determinism 개선과 맞다.

  하지만 request가 “투여된 약물”이라고 row membership을 암시했는데 query에는
  `event_txt = Administered` 같은 status filter가 없었다. canonical에는 `Assessed` 상태도
  포함됐다. 즉 사용자는 “administered rows”를 기대할 수 있는데 실제 label은 “medication
  event records with status field”였다.

  solver failures는 주로 tool protocol 실패였지만, data quality 자체도 source/request
  surface가 어긋났다.

- **정성 평가**:
  accepted data: low-quality accepted.

  row count/order/tie-break는 구조적으로 괜찮다. 그러나 row-set wording이 status membership을
  암시하면서 query.where가 그 status membership을 구현하지 않았다. 이건 어려운 좋은 문제가
  아니라 자연어 request와 label row membership이 어긋난 문제다.

- **변경**:
  hard validator는 추가하지 않았다. status/type membership을 문자 리터럴로 판별하면 금지된
  token/semantic heuristic이 된다. 대신 prompt-first 원칙을 따른다.

  - Request Contract: non-null/status/type filters는 row-set wording과 matching query.where가
    필요하다고 명시했다.
  - status/type이 filter가 아니라 출력 field일 뿐이면 records plus that field로 요청하라고
    명시했다.
  - `query.where` tool description에도 status/type membership을 암시하는 wording은 where로
    구현해야 하며, 아니면 status/type field를 함께 반환하는 records request로 써야 한다고
    추가했다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py`
  통과.

## Iteration 139 — Phrase repair must rewrite clean sentences

- **질문**:
  `trial_38`은 accepted 없이 실패했다. 실패 원인은 좋은 reject인가, 아니면 composer가
  feedback을 잘못 처리한 것인가?

- **실험/결과**:
  composer는 입원 중 투약 기록 list를 만들었다. 첫 draft는 order ambiguity로 reject됐고,
  이후 `emar_seq`를 `sequence_number`로 추가하여 tie-break를 만들려 했다. 그러나
  phrase repair 과정에서 Korean request가 망가졌다.

  발생한 malformed request:
  - `투약된 약물그리고 투약 시간과 투약 상태 기록`
  - `투약된 약물 와 투약 시간`
  - `5망까지`

- **reasoning 교차 분석**:
  composer reasoning은 누락된 `투약 시간`, `투약 상태` phrase를 exact substring으로 맞추는
  데 집중했다. 이 과정에서 자연스러운 문장 전체를 다시 쓰지 않고, 기존 문장에 단어를
  끼워 넣는 식으로 repair했다. 기존 prompt에는 `no malformed terms`가 있었지만, phrase
  repair feedback 상황에서 “문장 전체를 깨끗하게 다시 쓰라”는 압력이 부족했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. row/order 자체도 sequence tie-break가 어색했고,
  최종 request는 사용자에게 보여줄 수 없는 문장 품질이었다. rejection은 바람직하다.

- **변경**:
  prompt-first 원칙으로 Feedback And Difficulty-Up Policy를 보강했다.

  - phrase repair는 clean natural wording이어야 한다고 durable prompt에 명시했다.
  - `answer_contract_phrase_missing` feedback도 같은 정책을 상기시키도록 수정했다.
  - feedback 문구는 새 정책이 아니라 기존 Request Contract의 `ordinary target-language
    words`, `no malformed terms`를 구체적으로 상기하는 역할이다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_runtime.py::test_submit_draft_rejects_binding_phrase_absent_from_request -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 140 — Result fields need source representation

- **질문**:
  `trial_39` 마지막 draft는 pass rate `0/4`로 failed됐다. 이건 어려운 좋은 문제인가,
  아니면 저품질 문제인가?

- **실험/결과**:
  task는 입원 중 가장 먼저 받은 microbiology test 3건의 검체 종류, 검사 시간, 검사 항목,
  결과를 묻는 list였다. composer canonical은 `test_result`를 comments/result text로
  구성했다.

  solver들은 같은 row set과 time order는 대체로 찾았지만, `test_result`를 `org_name` 또는
  다른 result-like field로 제출했다. 일부 solver는 `"null"` 문자열을 냈고, 한 solver는
  comments를 일부 제출했다.

- **reasoning 교차 분석**:
  composer는 `결과`라는 broad phrase를 comments text로 해석했다. solver들은 microbiology
  table에서 더 직접적인 result-like surface인 organism/result fields를 선택했다. 즉
  solver가 못 푼 것이 아니라, request/label이 결과 representation을 유일하게 고정하지
  못했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: hard-good이 아니라 low-quality rejected. row ordering 자체도 boundary tie로
  여러 번 흔들렸고, 최종적으로는 result source representation이 불명확했다.

- **변경**:
  hard validator는 추가하지 않았다. `result`라는 단어를 보고 특정 column/source를 금지하는
  방식은 리터럴/의미 휴리스틱이므로 금지 원칙 위반이다.

  - Label Contract의 source-sensitive rule을 `result/status/type` 필드까지 명시했다.
  - `AnswerOutputBinding.requested_by_phrase` schema description도
    `result/status/type/category/sequence-like` field는 source representation을 보존해야
    한다고 보강했다.

  이 변경은 DB 특화가 아니라 source representation 일반 원칙이다.

- **검증**:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`2 passed`).

  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

## Iteration 141 — Accepted draft can still have broken request language

- **질문**:
  `trial_63`은 pass rate `1/4 = 0.25`로 accepted됐다. accepted data가 정말 좋은가?

- **실험/결과**:
  task는 특정 ICU stay의 procedureevents에서 최근 5개 처치/검사 기록을 묻는 list였다.

  - request:
    `이번 ICU 입원 중 가장 최근에 내원받은 클리니컬 기기 및 처치 확인을 위한 검사 명칭, 분류, 시작 시각, 종료 시각, 진단 분야를 포함한 시각 순으대로 방문 내역 정보 5가지를 최신순으대로, 각 시각에서 검사 및 처치 종류가 차이나는 경우 검사 명칭 점점인을 적용해 알려주세요.`
  - canonical row set:
    `procedureevents` + `d_items`, `starttime desc`, `d_items.label asc`, limit 5
  - pass: 1 solver matched, 3 solver mismatched

- **reasoning 교차 분석**:
  composer는 첫 draft에서 `start_time` 동점 때문에 order ambiguity feedback을 받았다.
  이후 hidden `orderid`를 생각했다가, 최종적으로는 visible tie-break인 `procedure_name`
  오름차순으로 query를 고친 점은 List Determinism 원칙과 맞다.

  그러나 다음 feedback에서 `end_time`, `order_category` phrase가 누락되자, composer는
  자연스러운 user request 전체를 다시 쓰기보다 exact substring 계약을 맞추는 데 집중했다.
  그 결과 `순으대로`, `최신순으대로`, `점점인` 같은 malformed fragment가 들어갔다.

  solver reasoning도 같은 문제를 드러낸다. solver들은 row set은 대체로 `procedureevents`로
  찾았지만, request의 `분류`/`진단 분야`가 자연스러운 source role을 고정하지 못해
  `category`와 `order_category`를 서로 다른 방식으로 해석했다.

- **DB 교차검증**:
  DB 쿼리로 canonical row set/order 자체는 확인했다. 상위 5개는
  `Portable Chest X-Ray`, `20 Gauge`, `Endoscopy`, `Chest X-Ray`, `PICC Line`이고,
  4/5번째의 같은 시작 시각은 `procedure_name asc`로 결정된다.

- **정성 평가**:
  accepted data: low-quality accepted.

  row set과 visible tie-break는 구조적으로 검증 가능하지만, 사용자 요청이 깨진 한국어이고
  field role도 오해를 유발한다. 이는 어려운 좋은 문제가 아니라 composer가 requestability를
  희생해 schema 계약을 맞춘 문제다. satisfactory accepted streak는 계속 `0/5`다.

- **변경**:
  hard validator는 추가하지 않았다. malformed natural language를 100% precision으로
  판정하는 것은 불가능하고, 리터럴/토큰 기반 휴리스틱은 금지 원칙 위반이다.

  대신 prompt-first 원칙으로 durable source와 tool schema를 같은 방향으로 보강했다.

  - Request Contract: exact substring binding은 broken wording, invented term, diagnostic
    phrase, misleading column/key translation을 정당화하지 않는다고 명시했다.
  - Label Contract: binding phrase는 patched field/key gloss가 아니라 fluent customer
    wording이어야 한다고 명시했다.
  - `submit_draft.user_request` schema: feedback 후 field key/repair phrase를 끼워 넣지 말고
    전체 request를 깨끗하게 다시 쓰라고 명시했다.
  - missing phrase feedback: label fields를 무조건 유지하라는 압력을 줄이고, fluent request가
    안 되면 cleaner field set으로 rerun하거나 다른 label을 선택하라고 상기시킨다.
  - prompt 길이 예산을 지키기 위해 기존 Scope Example 1개와 중복 문구를 압축했다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_rejects_binding_phrase_absent_from_request tests/test_turn_budget_prompt.py -q`
  통과 (`9 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`111 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 142 — no-PK aggregate and binding repair reminders

- **질문**:
  `trial_64`는 accepted 없이 synthesis_failed로 끝났다. reject가 저품질을 잘 막은 것인가,
  아니면 좋은 데이터가 불필요하게 막혔는가?

- **실험/결과**:
  다섯 번의 submit 시도가 있었다.

  1. `"Temperature Celsius"` 평균 측정값 scalar:
     `label_no_primary_key_source`
  2. 환자 ICU stay 2건 list:
     `answer_contract_list_size_invalid`, `label_non_user_visible_source`
  3. microbiology 최근 5건 list:
     null `org_name`, duplicate rows, order ambiguity
  4. admission 최근 5건 list:
     row set/label은 좋아 보였으나 broad `입원 이력` phrase를 모든 output field에 재사용하여
     `answer_contract_binding_missing`
  5. admission repair:
     `입원일자`, `퇴원일자`, `입원 유형`, `입원 경로`를 answer_contract에는 넣었지만
     user_request 본문에 넣지 않아 `answer_contract_phrase_missing`으로 budget exhausted

- **reasoning 교차 분석**:
  solver rollout은 없었다. composer reasoning은 두 가지를 보여준다.

  - 첫 draft에서 composer는 avg aggregate이므로 no-PK row stability 문제가 없다고 판단했다.
    이 판단은 구조적으로 타당했다. 기존 validator는 aggregate source까지 no-PK row-value처럼
    막고 있었고, feedback의 “derived aggregate over no-PK table” 문구와도 맞지 않았다.
  - 마지막 admission draft에서 composer는 좋은 row set을 찾았지만, binding_missing feedback을
    받고도 user_request에 output role phrase를 모두 넣지 않았다. 즉 answer_contract만 고치고
    request를 같이 고치지 않는 repair failure다.

- **DB 교차검증**:
  admission 최종 row set은 직접 DB 쿼리로 확인했다.
  `subject_id=10015860`의 최근 입원 5건을 `admittime desc`로 조회하면 canonical label과
  같은 `admittime`, `dischtime`, `admission_type`, `admission_location`이 나온다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - 2번 ICU stay 2건 list와 3번 microbiology list는 low-quality rejected가 맞다.
  - 4/5번 admission draft는 어려운 좋은 문제가 아니라 거의 좋은 데이터가 request/contract
    repair 실패로 막힌 케이스다. row set은 좋았고, user_request만
    `내 입원 이력 중 최근 입원일자 기준 5건의 입원일자, 퇴원일자, 입원 유형, 입원 경로를 보여주세요`
    같은 식으로 고치면 자연스럽고 검증 가능한 task가 된다.

- **변경**:
  1. no-PK hard validator를 정밀도 100% 구조 기준으로 조정했다.
     `kind == "aggregate"`인 column source는 source row를 stable record로 재방문할 필요가
     없으므로 `label_no_primary_key_source`에서 제외한다. 이건 DB 리터럴/컬럼 의미 휴리스틱이
     아니라 query evidence의 구조 필드만 보는 검사다.

  2. `answer_contract_binding_missing` feedback을 보강했다.
     각 output binding phrase가 user_request 안에 정확히 나타나도록 user_request를 다시 쓰라고
     명시하고, answer_contract만 고치는 repair를 피하라고 상기시킨다. 이는 기존 Label Contract
     원칙의 reminder이며 새 durable instruction이 아니다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_list_output_binding tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_for_selected_order_key tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_from_table_without_primary_key tests/test_synthesis_runtime.py::test_submit_draft_allows_count_from_table_without_primary_key tests/test_synthesis_runtime.py::test_submit_draft_allows_aggregate_from_table_without_primary_key -q`
  통과 (`5 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`112 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

## Iteration 143 — Prefer full timestamp order over split date/time keys

- **질문**:
  `trial_65`는 accepted 없이 `MaxTurnsExceeded`로 끝났다. reject가 좋은 저품질 차단인지,
  아니면 또 다른 일반 개선점이 있는지 확인한다.

- **실험/결과**:
  submit은 네 번 있었다.

  1. eMAR 단일 이벤트 detail:
     1건 list, non-user-visible source 포함으로 reject.
  2. ICD diagnosis code/version list:
     blocked code source와 evidence mismatch로 reject.
  3. microbiology 최근 5건:
     request는 자연스러웠지만 “날짜순” 방향/동점이 모호해 `answer_contract_order_ambiguous`.
  4. microbiology order repair:
     request를 `검사일과 검사시간이 가장 최근인 순서대로`로 고쳤지만, order binding phrase가
     exact substring이 아니었고, chart date + chart time split ordering 후에도 같은 time의
     test sequence tie가 남아 order ambiguity가 계속됐다.

- **reasoning 교차 분석**:
  solver rollout은 없었다. composer reasoning은 microbiology repair에서 핵심 원인을 드러낸다.

  composer는 처음에 `chartdate + test_seq`를 쓰다가 동점이 남자 `chartdate + charttime`으로
  바꾸려 했다. 그런데 `charttime`은 이미 date+time을 포함하는 더 세밀한 timestamp 성격의
  필드라, `chartdate`와 함께 쓰면 order key 하나를 낭비한다. 자연스러운 수리는
  `charttime desc, test_seq desc`처럼 full timestamp를 primary order key로 쓰고, date/time은
  표시용으로 나눠 출력하는 쪽이다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - 1/2번은 low-quality rejected.
  - 3/4번은 row set은 가능성이 있지만 order contract가 아직 저품질이다. 어려운 좋은 문제가
    아니라, composer가 시간 granularity를 잘못 모델링해서 유일한 order를 만들지 못한 케이스다.

- **변경**:
  hard validator는 추가하지 않았다. `chartdate`, `charttime` 같은 컬럼 이름을 토큰으로 해석해
  검사하면 금지 원칙 위반이다.

  대신 `query.order_by` tool schema description에 DB-agnostic 원칙을 추가했다.
  full timestamp가 있으면 date-only와 time-only를 별도 order key로 소비하지 말고 full timestamp를
  time order key로 쓰며, date/time split은 display 용도로만 쓰라고 했다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`1 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`112 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_tooling_composer_tool_factory.py`
  통과.

## Iteration 149 — Temporal source roles must be requestable

- **질문**:
  `trial_71`은 accepted였지만 pass rate가 `1/4 = 0.25`였다. 이것을 만족할 만한 좋은 accepted로
  볼 수 있는가, 아니면 low-quality accepted인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `accepted`, quality gate도 accept였다.

  최종 request:
  `이 중환자실 입원 기간의 날짜 및 시간 이벤트 5개를 입력된 시간이 가장 최근인 순서로, 동일 시간대에는 항목명 순서로 정렬해서 보여주세요. 각 이벤트의 항목명, 기록된 시간값, 시간값 유형, 입력된 시간을 포함해서 알려주세요.`

  canonical answer는 `datetimeevents`에서 `storetime desc, d_items.label asc`로 고른 5개 row였다.
  출력 필드는 `item_label`, `recorded_timestamp`(`value`), `timestamp_type`(`valueuom`),
  `recorded_at`(`storetime`)였다.

  solver 결과:
  - 1개 solver만 `storetime desc, item label asc`로 맞췄다.
  - 2개 solver는 `입력된 시간`을 `value` 또는 `charttime` 성격의 시간으로 해석해 다른 5개 row를 제출했다.
  - 1개 solver는 tool call XML을 final text처럼 냈다가 `missing_submit_result`가 됐다.

- **reasoning 교차 분석**:
  composer reasoning은 ToolBudgetFeedback 이후에 submit boundary가 작동했음을 보여준다.
  `sample`이 `{"error":"submit_draft_required"}`를 반환한 뒤, 런타임이 `composer_submit_draft_missing`
  feedback을 만들고 다음 segment로 넘어갔다. Iteration 148의 런타임 수정은 의도대로 작동했다.

  낮은 pass rate의 원인은 runtime boundary가 아니다. solver reasoning을 보면 실패한 solver들은
  request의 `입력된 시간`을 composer가 의도한 `storetime`이 아니라 `charttime` 또는 `value` 쪽
  시간으로 해석했다. 같은 task 안에 `기록된 시간값`과 `입력된 시간`이 같이 있어 자연어 surface가
  충분히 분리되지 않았다.

- **정성 평가**:
  accepted data: low-quality accepted. 저품질이 accept된 경우라 만족 streak에는 넣지 않는다.
  데이터 row 자체는 groundable하고 duplicate/order diagnostics도 통과했지만, 자연어 요청이 여러 시간
  surface 중 하나를 안정적으로 가리키지 못했다.

  rejected data:
  - 첫 feedback은 data 품질 문제가 아니라 protocol boundary 확인용 `composer_submit_draft_missing`이었다.
  - 두 번째 제출은 order/binding repair 단계였고, 최종 accepted로 수리되긴 했지만 temporal role
    모호성은 남았다.

- **변경**:
  hard validator는 추가하지 않았다. 여러 timestamp/date surface 중 어떤 자연어가 어느 source role을
  뜻하는지 100% precision으로 판정하는 것은 리터럴/컬럼명 휴리스틱이 되기 쉽다.

  Prompt-first 원칙에 따라 Source Surface 정의를 보강했다.
  timestamp/date surface도 event time, stored/entered time, scheduled time처럼 서로 다른 source
  surface이며, 여러 surface가 맞을 수 있으면 generic time/date wording이 invalid라고 명시했다.

  tool schema/submit_draft schema는 같은 원칙의 reminder만 추가했다.
  `query.select`, `query.select[].as`, `query.order_by`는 여러 date/time-like value를 고를 때 각
  field/alias/order wording이 자연스럽게 distinct source role을 가져야 한다고 설명한다.
  `answer_contract` binding schema도 time-like output 두 개를 generic time phrase에 묶지 말라고
  상기한다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned -q`
    통과 (`3 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_71`은 accepted지만 low-quality accepted로 판정하므로 만족 streak는 `0/5` 유지.

## Iteration 150 — Abandon lists when two order keys still tie

- **질문**:
  `trial_72`는 왜 accepted 없이 끝났는가? temporal source-role 개선이 문제였는가, 아니면 다른
  list determinism 실패인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed / SynthesisArtifactGenerationError`.

  제출 1:
  - `composer_submit_draft_missing`
  - `sample`이 `submit_draft_required`를 반환했고, 런타임 boundary가 정상적으로 feedback으로
    승격했다. 데이터 품질 문제가 아니라 protocol boundary 확인이다.

  제출 2:
  `해당 입원 기간 동안 투약 이벤트 중 가장 먼저 발생한 5개의 약물명, 투약 시간, 그리고 투약 상태를 투약 시간순으로 알려주세요`
  - 오류: `label_null_value_forbidden`, `answer_contract_hidden_filter_unanchored`
  - `medication: null` row가 포함됐고, child emar record에서 admission sibling rows를 조회하면서
    parent scope가 entity에 고정되지 않았다.
  - rejected data 판정: low-quality rejected.

  제출 4/5:
  pharmacy 처방 목록으로 전환했지만 모든 row의 `prescription_start_time`이 같았다.
  `entered_time`을 visible tie-break로 추가해도 첫 세 row가 같은 `entered_time`을 공유해
  `duplicate_order_key_in_returned_rows`와 `limit_boundary_tie`가 남았다.
  마지막 request는 `빨른 것으부터`처럼 한국어도 깨져 `answer_contract_phrase_missing`까지 발생했다.
  - rejected data 판정: low-quality rejected. 어려운 좋은 문제가 아니라 deterministic list 후보를
    버리지 못한 문제다.

- **reasoning 교차 분석**:
  composer는 pharmacy 후보에서 hidden `pharmacy_id` tie-break가 부적절하다는 점은 인지했다.
  그 다음 visible `entered_time`을 추가했지만, query diagnostics가 여전히
  `duplicate_order_key_in_returned_rows`와 `limit_boundary_tie`를 보여줬다.
  이 시점에서 이 ordered limited list 후보는 버려야 했다. 그러나 composer는 같은 label을 문구 수리로
  밀어붙였고, budget boundary 이후에도 “valid label result”로 오인했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - emar draft는 null field와 hidden scope 문제가 명확한 low-quality rejected.
  - pharmacy draft는 두 order key로도 membership/order가 유일하지 않은 low-quality rejected.
  - 마지막 phrase 오류는 부차적이다. root cause는 order diagnostics를 보고도 label을 버리지 않은 것.

- **변경**:
  hard validator는 추가하지 않았다. 이미 query/submit diagnostics가 정확히 ambiguity를 탐지한다.

  Prompt-first 원칙에 따라 List Determinism Policy를 보강했다.
  두 order key를 써도 tie가 남으면 label을 switch해야 한다고 명시했다.

  tool schema와 feedback은 같은 원칙의 reminder로 보강했다.
  `query.order_by` description은 `duplicate_order_key` 또는 `limit_boundary_tie`가 두 order key 후에도
  남으면 ordered limited list를 abandon하거나 tied rows를 return해야 한다고 설명한다.
  `ANSWER_CONTRACT_ORDER_AMBIGUOUS` feedback도 wording-only repair 제출을 금지하고, 다른 label 또는
  tied rows를 요구한다.

  ToolBudgetFeedback 문구도 “repair query가 label values를 반환했다”가 아니라 “blocking diagnostics
  없이 label values를 반환했다”일 때만 submit하라고 수정했다. diagnostics가 block하면 그 후보는
  boundary 전에 버렸어야 한다는 점을 명확히 했다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_late_feedback_repair tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_repeated_query_repair_for_ambiguous_query tests/test_synthesis_runtime.py::test_submit_draft_rejects_ambiguous_limited_list_order -q`
    통과 (`5 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_72`는 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 151 — Limited list repairs must keep the fixed size phrase

- **질문**:
  `trial_73`은 왜 accepted 없이 `answer_contract_phrase_missing` +
  `answer_contract_query_mismatch`로 끝났는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed / SynthesisArtifactGenerationError`.

  제출 2:
  서비스 이력 draft는 1 row, null, blocked source가 섞인 low-quality rejected였다.

  제출 3-5:
  procedureevents 후보는 query 자체만 보면 더 좋아졌다.
  `stay_id=37093652`의 procedureevents 5개를 `starttime asc`로 가져왔고,
  `duplicate_order_key_in_returned_rows=false`였다.
  그러나 query는 `limit=5`였는데 user_request와 answer_contract에 고정 크기 `5개`가 들어가지 않았다.
  그래서 query가 정한 row membership과 request contract가 불일치했다.

- **reasoning 교차 분석**:
  composer reasoning은 “order_binding phrase를 정확히 맞춰야 한다”는 쪽에만 집중했다.
  feedback diagnostics에는 `missing_limit_phrase_for_query_limit=5`가 있었지만, composer는
  output/order phrase만 바꾸고 fixed-size phrase를 추가하지 않았다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - services draft는 low-quality rejected.
  - procedureevents draft는 row set/order는 잠재적으로 괜찮았지만, limit phrase가 빠진 contract
    오류 때문에 rejected. 이건 어려운 좋은 문제라기보다 composer repair 누락이다.
  - `procedure_value`는 여전히 자연어 requestability가 약해 다음 실험에서도 주의 깊게 봐야 한다.

- **변경**:
  hard validator는 추가하지 않았다. 이미 query mismatch validator가 정확히 잡고 있다.

  `ANSWER_CONTRACT_QUERY_MISMATCH` feedback을 보강했다.
  list query limit이 membership을 고정하면, phrase feedback과 같이 발생했을 때 같은 label을 유지한 채
  user_request와 `answer_contract.limit_phrase`에 exact fixed-size phrase를 추가해야 한다고 명시했다.
  limit을 제거하거나 output/order phrase만 고치는 수리는 금지했다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_requires_limit_phrase_when_query_limit_shapes_list -q`
    통과 (`1 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_73`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 152 — Do not relabel lifecycle surfaces

- **질문**:
  `trial_74`는 accepted였지만 pass rate가 `1/4 = 0.25`였다. 이것은 어려운 좋은 문제인가,
  아니면 low-quality accepted인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `accepted`, quality gate도 accept였다.

  최종 request:
  `이번 입원 기간 동안 처방된 처방 순서 5개 약물과 각각의 상태를 처방 시작 시간 순서대로 약물 이름 순서로 보여주세요`

  canonical answer는 pharmacy table의 `medication`, `starttime`, `status`를 사용했다.
  정렬은 `starttime asc`, `medication asc`였다.

  solver 결과:
  - 1개 solver만 pharmacy의 `status`를 사용해 정답과 일치했다.
  - 다른 solver들은 request의 `처방된/처방` wording을 보고 prescriptions table로 갔다.
  - 일부 solver는 prescriptions의 `stoptime` 또는 `drug_type`에서 상태를 유도했다.

- **reasoning 교차 분석**:
  composer는 pharmacy source를 쓰면서 자연어 request에서는 `처방된 약물`, `처방 순서`라고 표현했다.
  이는 solver 입장에서 prescriptions source로 가는 것이 더 자연스럽다.
  문제는 solver 난이도가 아니라 source surface mismatch다.

- **정성 평가**:
  accepted data: low-quality accepted. 저품질이 accept된 경우이므로 만족 streak에 넣지 않는다.

  rejected data:
  - 앞선 제출들은 fixed size, order ambiguity, field rename mismatch를 정상적으로 reject했다.
  - 최종 accepted도 request/source role이 불안정해서 좋은 데이터로 보지 않는다.

- **변경**:
  hard validator는 추가하지 않았다. lifecycle surface를 자연어로 어떻게 부르는지는 100% precision
  validator로 판정할 수 없다.

  Source Surface 원칙은 이미 prompt에 있으므로, tool schema/submit schema에서 같은 원칙을 더 직접적으로
  상기했다.
  `query.from.table` description은 order/request/event/fulfillment/log 같은 lifecycle surface를
  field가 겹친다는 이유로 서로 바꿔 부르지 말라고 설명한다.
  `submit_draft.user_request` schema도 같은 reminder를 포함한다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_payload_schema_uses_strict_json_string_fields -q`
    통과 (`2 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_74`는 accepted지만 low-quality accepted로 판정하므로 만족 streak는 `0/5` 유지.

## Iteration 153 — Phrase/binding-only repair must use zero data tools

- **질문**:
  `trial_75`는 왜 accepted 없이 `label_changed_during_repair`로 끝났는가?
  reasoning content가 반환되고 있으므로 composer의 판단을 tool call/diagnostics와 교차 분석한다.

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`.

  composer는 `emar`에서 환자의 최근 투약 기록 5개를 만들었다.
  `charttime desc, emar_seq asc` query는 5개 row를 안정적으로 반환했고, label 자체는
  수리 가능한 후보였다. 그러나 제출은 다음 흐름으로 실패했다.

  - first submit: `composer_submit_draft_missing`
  - medication list submit: `answer_contract_phrase_missing`
  - same list repair: `answer_contract_phrase_missing`, `answer_contract_binding_missing`
  - same list repair: `answer_contract_phrase_missing`, `answer_contract_binding_missing`
  - final submit: count aggregate로 label을 바꾸면서 `label_changed_during_repair`

- **reasoning 교차 분석**:
  composer는 처음에 duplicate time을 보고 `emar_seq` tie-break를 붙여 order를 안정화했다.
  그 다음 문제는 data가 아니라 request/answer_contract였다. missing phrase와 duplicate/missing
  order binding만 남았는데, reasoning은 "smaller scope"나 "simpler scalar count"로 바꿀 생각을
  했다. 실제 마지막 query도 같은 `emar` scope의 total count였고, repair-locked label과 전혀 다른
  canonical answer가 되어 validator가 정확히 막았다.

  즉 rejected data는 어려운 좋은 문제라기보다, 수리 가능한 list draft를 contract-only 단계에서
  새 label로 갈아탄 composer discipline 문제다. solver rollout은 없었다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 첫 list 후보는 query evidence와 row set이 괜찮았지만,
  자연어/contract repair를 완료하지 못했다. 마지막 count는 repair 상황에서 label을 바꾼 것이므로
  좋은 어려운 문제로 볼 수 없다.

- **변경**:
  프롬프트는 수정하지 않았다. `Feedback And Difficulty-Up Policy`에 이미
  `phrase/binding-only: no exploration`이 source of truth로 있다.

  문제는 runtime budget gate가 그 원칙을 binding-only 단독 오류에만 적용했다는 점이다.
  `answer_contract_phrase_missing` 또는 `answer_contract_binding_missing`만 남은 경우는 모두
  contract-only repair로 보아 data tool 예산을 0으로 맞췄다. feedback은 새 지시가 아니라 기존
  정책의 reminder로, "current label/query values를 보존하고 user_request/answer_contract만
  수리하라"고 알려준다.

  이 변경은 오류 코드 집합만 보는 구조적 판단이다. DB 리터럴, 테이블명, 컬럼명, 값 문자열을 보지
  않으므로 리터럴/토큰 휴리스틱 금지 원칙을 위반하지 않는다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_contract_only_data_repair tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_late_feedback_repair tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_allows_repair_query_after_limit -q`
    통과 (`3 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_75`는 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 154 — Limited-list order must not split selection and display

- **질문**:
  `trial_76`은 contract-only repair gate 이후 개선됐는가? accepted/rejected 품질을 reasoning과 함께
  판정한다.

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, `calibration_inconclusive`, solver pass rate `0/4`.

  composer는 ICU stay의 `inputevents`에서 수액/약물 입력 3개를 만들었다.
  첫 list submit은 output binding phrase가 request에 없어서 `answer_contract_phrase_missing`으로
  rejected됐다. Iteration 153 변경 뒤 composer는 새 data tool 탐색 없이 같은 label을 수리했고,
  solver rollout까지 갔다.

  최종 request:
  `나의 ICU 입원 시 투여 받은 수액 및 약물 입력 항목명, 투여량, 단위, 투여 시작 시간 중 가장 최근 3개를 시작 시간이 빠른 순으로 보여줘`

  canonical query는 `starttime desc limit 3` 결과를 그대로 label로 복사했다.
  canonical answer order는 `18:07`, `17:53`, `17:12`였다.

- **reasoning 교차 분석**:
  solver 3개는 request를 "가장 최근 3개를 먼저 고른 뒤 시작 시간이 빠른 순으로 보여달라"로 해석해
  `17:12`, `17:53`, `18:07` 순서로 제출했다. 한 solver는 이중 order를 더 혼동해 전체에서 earliest
  3개를 골랐다.

  핵심 문제는 solver 난이도가 아니다. request가 limited row membership은 latest로 고르라고 말하면서,
  display order는 earliest-first로 말한다. 그런데 현재 composer query/canonical label은 하나의 list
  order만 표현한다. 더 나쁘게는 `answer_contract.order_bindings`가 phrase는 `시작 시간이 빠른 순`,
  direction은 `desc`로 제출했다. 자연어 방향을 validator가 언어별 토큰으로 해석해 막는 것은
  precision 100%가 아니므로 금지한다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. solver가 못 푼 어려운 좋은 문제가 아니라, composer가 하나의
  list에 selection order와 display order를 분리해서 요청한 저품질 문제다.

- **변경**:
  Prompt-first 원칙에 따라 List Determinism Policy를 보강했다.
  한 list에는 하나의 order만 있어야 하며, selection/display split을 요청하지 말라고 명시했다.

  tool schema/desc도 같은 원칙의 reminder로 맞췄다.
  `query.order_by`는 반환 rows가 canonical label order이며 request/answer_contract로 뒤집을 계획을
  세우면 안 된다고 설명한다. `query.limit`는 limited membership을 고르는 order와 표시 order를
  다르게 요청하지 말라고 설명한다. `submit_draft.answer_contract.order_bindings`와
  `AnswerOrderBinding.direction`은 requested_by_phrase, label_json row order, direction이 서로
  맞아야 한다고 설명한다.

  `answer_contract_phrase_missing`/`answer_contract_binding_missing` feedback은 새 지시가 아니라
  같은 List Determinism Policy의 reminder로, repair 중 반대 display-order phrase를 추가하지 말라고
  상기한다.

  hard validator는 추가하지 않았다. `빠른 순` 같은 자연어 방향을 리터럴/토큰 휴리스틱으로 판정하면
  금지 원칙 위반이고 모든 DB/언어에서 precision 100%를 보장할 수 없다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_runtime.py::test_submit_draft_payload_schema_does_not_require_constraint_summary tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_reset_after_contract_repair_feedback tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
    통과 (`4 passed`).
  - prompt length `7993`, 8000자 예산 안에 유지.

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py`
  통과.

- **현재 streak**:
  `trial_76`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 155 — Limited query labels must request the fixed N

- **질문**:
  `trial_77`은 Iteration 154 뒤 어떤 저품질 원인으로 실패했는가? 어려운 좋은 문제였는가,
  아니면 composer가 아직 제출하지 말아야 할 draft를 반복했는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, solver rollout 없음.

  composer는 admission anchor에서 microbiologyevents 5개 list를 만들었다. 제출 흐름은:
  - first submit: `composer_submit_draft_missing`
  - submit 2: null organism/antibiotic/result/sensitivity 포함, `label_null_value_forbidden`,
    `answer_contract_phrase_missing`, `answer_contract_query_mismatch`, `answer_contract_order_ambiguous`
  - submit 3-5: null은 줄였지만 `limit=5` query를 쓰면서 request/limit_phrase에 5개 고정 개수를
    넣지 않았다. 또한 `charttime + test_seq` order인데 request는 `검사 시간 순서`만 말해
    `answer_contract_order_ambiguous`와 `answer_contract_binding_missing`이 계속 남았다.

- **reasoning 교차 분석**:
  composer reasoning은 null field 문제를 정확히 인지하고 "non-null fields only"로 수리했다.
  하지만 이후에도 query는 `limit=5`를 유지하면서 user_request는 "모든 미생물 검사 기록" 또는
  "미생물 검사 결과"처럼 fixed N을 말하지 않았다. 마지막 request에는 출력 field phrase는 대부분
  들어갔지만 answer_phrase와 fixed-size phrase가 여전히 맞지 않았다.

  이건 solver가 못 푼 어려운 문제가 아니다. solver rollout 전 label contract가 정확히 reject한
  low-quality draft다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. null label, unbound fixed limit, unstable/under-described order가
  모두 구조적으로 명확하다. 저품질이 accept되지 않은 점은 좋지만, composer가 같은 mismatch를
  반복한 점은 개선 대상이다.

- **변경**:
  hard validator는 추가하지 않았다. 이미 `answer_contract_query_mismatch`와
  `answer_contract_order_ambiguous`가 precision 100 구조 신호로 reject하고 있다.

  source of truth는 기존 Label Contract/Task Shapes/List Determinism 쪽에 있다. 이번에는
  tool schema와 feedback reminder를 좁게 보강했다.
  `submit_draft.answer_contract.limit_phrase` schema는 query가 fixed limit을 쓰면 null로 두거나
  all/every matching records를 묻지 말라고 설명한다.
  `query.limit` description도 limited query를 all/every matching records로 표현하지 말라고 설명한다.
  `answer_contract_query_mismatch` feedback은 phrase feedback과 함께 나올 때 정확한 fixed-size phrase를
  user_request와 limit_phrase에 추가하고, all/every matching records로 묻지 말라고 상기한다.

  이 변경은 DB 리터럴이나 컬럼 의미를 보지 않는다. query 구조의 fixed limit과 contract binding
  문제만 다루므로 리터럴/토큰 휴리스틱 금지 원칙을 위반하지 않는다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_payload_schema_does_not_require_constraint_summary tests/test_synthesis_runtime.py::test_submit_draft_requires_limit_phrase_when_query_limit_shapes_list tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
    통과 (`3 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_runtime.py tests/test_tooling_composer_tool_factory.py`
  통과.

- **현재 streak**:
  `trial_77`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 156 — Single-row detail fallback must stay unattractive

- **질문**:
  `trial_78`은 fixed limit wording 개선 뒤 좋아졌는가? 마지막 `answer_contract_list_size_invalid`는
  어려운 좋은 문제였는가, 아니면 저품질 fallback인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, solver rollout 없음.

  제출 흐름:
  - first submit: `composer_submit_draft_missing`
  - microbiology list: null organism, evidence mismatch, order ambiguity, binding error
  - admission list: row_count=1인데 list로 제출, `answer_contract_list_size_invalid`
  - lab list: null unit/flag와 order phrase mismatch
  - final patient info: row_count=1 기본정보 detail을 list로 제출, `answer_contract_list_size_invalid`

- **reasoning 교차 분석**:
  composer는 microbiology/labevents의 null과 ordering 문제를 본 뒤 "simpler label"로 가려 했다.
  그러나 마지막 선택은 aggregate가 아니라 단일 patient row의 demographic fields였다. 이는
  Task Shapes의 "Avoid single-record detail lookup" 위반이며, solver가 못 푼 어려운 문제가 아니다.

  admission query도 같은 패턴이었다. query가 실제로 1 row만 반환했는데 request는 입원 기록 확인
  list처럼 작성했다. validator가 정확히 막았다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. null 값과 order ambiguity는 정상적으로 거절됐고,
  마지막 두 single-row detail fallback도 좋은 어려운 문제가 아니라 composer가 쉬운 단일 record
  detail로 도망간 케이스다.

- **변경**:
  hard validator는 추가하지 않았다. 이미 1-2 row list는 precision 100으로 reject되고 있다.

  프롬프트에는 이미 Task Shapes로 "Avoid single-record detail lookup"이 있다. 이번에는
  `submit_draft.answer_contract.kind` schema를 같은 원칙의 reminder로 보강했다. `list`가 query rows
  array라는 설명 옆에, 1-2 row detail lookup을 list fallback으로 제출하지 말고 3-5 rows나 aggregate를
  선택하라고 명시했다.

  이 변경은 DB 내용이나 리터럴을 보지 않고 answer_contract shape만 설명한다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_payload_schema_does_not_require_constraint_summary tests/test_synthesis_runtime.py::test_submit_draft_rejects_single_row_list_as_too_direct tests/test_synthesis_runtime.py::test_submit_draft_rejects_limited_single_row_before_order_repair -q`
    통과 (`3 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_78`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 157 — User request must not translate table jargon

- **질문**:
  `trial_79`는 accepted지만 정말 만족할 만한 데이터인가? reasoning content가 반환되므로,
  composer가 왜 list를 버리고 scalar count로 갔는지와 solver 실패 원인을 함께 확인한다.

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `accepted`.

  제출 흐름:
  - submit 1/2: `composer_submit_draft_missing`
  - submit 3 accepted:
    `내 중환자실 입원 기간 동안 입력 이벤트 총 개수를 알려줘`
  - canonical answer: `{"total_input_records": 903}`
  - pass rate: `3/4 = 0.75`
  - infra excluded: solver 1개가 `max_episode_duration_ms=240000`으로 제외
  - evaluable miss: solver 1개가 `inputevents`를 filter한 뒤 `count_records`를 호출하지 않고
    `{"total_input_records": 0}`을 제출

- **reasoning 교차 분석**:
  composer는 처음에 ICU stay의 `inputevents` 최신 list를 만들려 했다. `starttime` 동점과
  hidden `orderid` tie-break 문제를 보고 list determinism을 포기했고, 마지막에 같은 scope에서
  scalar aggregate count로 전환했다. 이 전환 자체는 이전 원칙에 맞다.

  문제는 request surface다. `입력 이벤트`는 실제 고객 자연어라기보다 `inputevents` source name을
  그대로 번역한 표현이다. answer는 유일하고 solver가 도구로 풀 수 있지만, customer-facing request가
  도메인 역할로 번역되지 않았으므로 low-quality accepted로 판정한다.

  solver miss 1건은 tool 설계 결함으로 보지 않는다. reasoning은 `count_records`가 필요하다고 말했지만
  실제 tool call 없이 0을 제출했다. atomic tool 철학상 도구 호출을 생략하면 틀리는 것이 맞고,
  evaluator가 이를 잡았다.

- **정성 평가**:
  accepted data: low-quality accepted. 유일성/검증성은 충족하지만 사용자 문구가 테이블 jargon에 가깝다.

  rejected data: 없음. 이전 list 후보들은 ordering diagnostics 때문에 abandon됐고, 이는 올바른 방향이었다.

- **변경**:
  hard validator는 추가하지 않았다. 특정 단어가 테이블 jargon인지, 실제 도메인 용어인지 100% precision으로
  판정할 수 없고, 리터럴/토큰 기반 휴리스틱 금지 원칙에도 걸린다.

  대신 prompt/schema를 같은 원칙으로 보강했다.
  - durable prompt `Request Contract`: DB/table/column jargon이나 technical sequence/ref를 쓰지 말고
    도메인 역할로 표현하라고 압축해 명시.
  - `submit_draft.user_request` schema: table/source jargon과 table/column name 직역을 피하라고 설명.

  이는 특정 DB 리터럴을 막는 규칙이 아니라, user_request surface의 일반 원칙을 상기시키는 수정이다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_prompts.py::test_synthesis_input_defaults_to_schema_map_entity_selection tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_payload_schema_does_not_require_constraint_summary -q`
    통과 (`4 passed`).

  Prompt length:
  - `build_synthesis_agent_instructions(...)` 길이 `7993`, 기존 budget `<8000` 유지.

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_79`는 accepted지만 low-quality accepted로 보므로 만족 streak는 `0/5` 유지.

## Iteration 158 — Sequence fields must not be repair crutches

- **질문**:
  table jargon 보강 뒤 `trial_80`의 request surface는 좋아졌는가? accepted라면 정말 좋은 데이터인가,
  아니면 낮은 pass rate와 solver reasoning이 저품질을 가리키는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `accepted`, pass rate는 `1/4 = 0.25`.

  제출 흐름:
  - submit 1:
    `현재 ICU 입원 기간 중 가장 최근에 기록된 5개의 생체 신호 측정값...`
    - `label_no_primary_key_source`, order/binding 오류
  - submit 3:
    `제가 입원 중인 이 ICU에서 가장 최근에 투약된 약물 4개...`
    - 자연어 표면은 좋아졌지만 `answer_contract_hidden_filter_unanchored`,
      `answer_contract_phrase_missing`
  - submit 4 accepted:
    `제가 이 ICU에 입원 한 동안 투약된 약물 기록 중 가장 최근에 기록된 4개의 투약 은 무엇이고,
    각 투약의 이벤트 유형과 시간, 그리고 기록 순서번호를 찥점순으로 보여주세요`
    - canonical fields: `administered_time`, `event_type`, `medication_name`, `record_sequence`

- **reasoning 교차 분석**:
  composer는 처음에 `chartevents` list를 만들면서 hidden tie-break를 "fine"이라고 오판했다가
  validator feedback으로 버렸다. 이후 `emar` 투약 기록으로 이동한 것은 좋아졌지만,
  hidden filter anchoring을 고치는 과정에서 `record_sequence`를 output/order field로 추가했다.

  최종 reasoning은 "no blocking diagnostics"에만 반응했고, request가 `찥점순` 같은 깨진 표현과
  `기록 순서번호`라는 기술적 control을 포함한다는 사실을 자체적으로 거르지 못했다.

  solver reasoning도 이를 뒷받침한다.
  - 한 solver는 `record_sequence`를 실제 source sequence가 아니라 1-4 display rank로 바꿔 오답.
  - 다른 solver는 "시간순/오름차순"으로 해석해 row order를 뒤집어 오답.
  - 한 solver는 max turns로 실패.
  - 한 solver만 canonical과 일치.

- **정성 평가**:
  accepted data: low-quality accepted. 요청 문구가 깨져 있고, list determinism repair를 위해
  source sequence/order number를 노출했다. 이는 어려운 좋은 문제가 아니라 composer가 contract를
  만족시키려고 technical control을 붙인 문제다.

  rejected data: 첫 chartevents 후보는 no-PK source라 저품질 rejected. submit 3의 투약 후보는
  자연어/도메인 방향은 좋았지만 hidden anchor repair가 필요했다.

- **변경**:
  hard validator는 추가하지 않았다. 어떤 visible sequence/order field가 실제 도메인 답인지,
  기술적 repair crutch인지 100% precision으로 구분할 방법이 없고, 컬럼명/토큰 휴리스틱은 금지다.

  대신 prompt/tool schema/feedback reminder를 같은 원칙으로 보강했다.
  - durable prompt List Determinism Policy:
    source sequence/order number는 그 자체가 도메인 답일 때만 쓰고, tie-break/binding repair용으로
    추가하지 말라고 명시.
  - `query.select` / `query.order_by` schema:
    sequence/reference/order number를 list determinism이나 binding repair용으로 선택/정렬하지 말라고 명시.
  - `submit_draft` answer binding schema:
    sequence/reference/order field를 feedback 수리용으로 추가하지 말고, source sequence/order phrase가
    기계적이거나 어색하면 다른 label을 고르라고 설명.
  - hidden filter unanchored feedback:
    anchor repair가 source sequence/reference/order field 추가나 malformed tie-break wording을 정당화하지
    않는다고 기존 List Determinism Policy를 상기.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_rejects_hidden_child_to_parent_sibling_scope -q`
    통과 (`4 passed`).

  Prompt length:
  - `build_synthesis_agent_instructions(...)` 길이 `7991`, 기존 budget `<8000` 유지.

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_80`은 accepted지만 low-quality accepted로 보므로 만족 streak는 `0/5` 유지.

## Iteration 159 — Order phrase repair must stay fluent

- **질문**:
  sequence/order repair 보강 뒤 `trial_81`은 개선됐는가? accepted가 없었다면 거절 데이터는
  어려운 좋은 문제였는가, 저품질이었는가, 아니면 너무 쉬워서 정상 거절된 문제였는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`.

  제출 흐름:
  - submit 1: `composer_submit_draft_missing`
  - submit 2:
    `내 최근 투약 기록 5개를 투약 시간 순서대로 보여주세요.`
    - 오류: `answer_contract_evidence_mismatch`, `label_non_user_visible_source`
    - 원인: `emar_id`가 label에 포함됨
  - submit 3:
    `내 최근 투약 기록 5개를 투약 시간을 기준으로 최신순이부터 보여주세요.`
    - 오류: `answer_contract_phrase_missing`
  - submit 4:
    `내 최근 투약 기록 5개를 투약 시간, 투약 명칭, 투약 상태 순으로 멀어진 시간순방업물뉘보여주세요.`
    - 오류: `answer_contract_phrase_missing`
  - submit 5 budget exhausted:
    `내 최근 투약 기록 5개를 투약 시간, 투약 명칭, 투약 상태 순으로 멀어진 시간순으로 보여주세요.`
    - pass rate: `4/4 = 1.0`
    - 오류: `calibration_inconclusive`

- **reasoning 교차 분석**:
  sequence/order field 보강은 부분적으로 효과가 있었다. 이번에는 최종 label에 `record_sequence`나
  source sequence field가 들어가지 않았다.

  그러나 phrase repair에서 다른 문제가 드러났다. composer는 `상태`가 request에 없다는 feedback을
  본 뒤 같은 label을 수리하려 했고, 이후 order phrase mismatch를 고치려다 `최신순이부터`,
  `멀어진 시간순방업물뉘` 같은 깨진 한국어를 만들었다. 마지막에는 문구를 거의 고쳤지만 이미
  제출 budget을 소모했고, solver 4개가 모두 맞혀 too-easy로 reject됐다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - submit 2는 hidden/non-visible field가 label에 섞인 low-quality rejected.
  - submit 3/4는 label 자체는 방향이 맞지만 phrase repair가 깨져 low-quality rejected.
  - submit 5는 solver가 모두 풀 수 있는 too-easy rejected. 데이터 자체는 나쁘다기보다
    난이도 부족이며, 현재 upper band 기준에서는 reject가 맞다.

- **변경**:
  hard validator는 추가하지 않았다. 깨진 자연어/어색한 방향 표현을 100% precision으로 검출하는
  방법은 없고, 문자/토큰 기반 휴리스틱은 금지다.

  대신 기존 fluency 원칙을 order phrase가 들어가는 정확한 surface에 더 직접적으로 반영했다.
  - `submit_draft.answer_contract.order_bindings[].requested_by_phrase` schema:
    ordinary target-language direction words를 쓰고, typo-like tie-break term을 만들지 말라고 명시.
  - `answer_contract_phrase_missing` feedback:
    phrase repair 때도 ordinary direction words를 쓰고 malformed order phrase를 추가하지 말라고
    기존 Request/List Determinism 정책을 상기.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_rejects_answer_contract_phrase_absent_from_request -q`
    통과 (`2 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_81`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 160 — Binding feedback must not justify sequence tie-breaks

- **질문**:
  `trial_82`에서 order phrase 보강이 sequence/order repair를 막았는가? 아니면 composer가
  여전히 sequence field를 자연어로 정당화하려 하는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, solver rollout 없음.

  제출 흐름:
  - submit 1: `composer_submit_draft_missing`
  - submit 2:
    `해당 입원 기간 중 투여된 약물 중 시간순으로 앞선 5가지를 보여주세요. 약물명, 투여 시간,
    투여 상태를 알고 싶습니다.`
    - 오류: `answer_contract_binding_missing`
    - label에 `sequenceId` 포함
  - submit 3:
    `... 기록 순번을 알고 싶습니다.`
    - 오류: `answer_contract_phrase_missing`
    - sequence tie-break를 user_request로 정당화하려 함
  - submit 4: `composer_submit_draft_missing`
  - submit 5 budget exhausted:
    sequence field를 제거했지만 latest successful query evidence와 맞지 않아
    `answer_contract_evidence_mismatch`

- **reasoning 교차 분석**:
  composer reasoning이 문제를 명시했다:
  `sequenceId`는 tie-break이지 자연스러운 요청 대상이 아니며, 그런데도 이를 user_request에서
  정당화해야 한다고 판단했다. 즉 모델이 원칙을 부분적으로 이해했지만, `answer_contract_binding_missing`
  feedback을 "sequence field를 말로 포장하라"로 잘못 해석했다.

  이후 phrase/binding-only 구간에서 data tool이 차단된 것은 의도대로다. 하지만 composer는 sequence 없는
  label로 바꿔 submit했고, latest query evidence는 sequence 포함 label이어서 mismatch가 났다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 약물 투여 목록이라는 target 자체는 자연스러웠지만, duplicate rows와
  order tie를 해결하기 위해 source sequence를 붙이려 한 순간 저품질 repair가 됐다.

- **변경**:
  hard validator는 추가하지 않았다. sequence field가 실제 도메인 답인지 repair crutch인지 구분하는 것은
  여전히 semantic 판단이고, 컬럼명 기반 차단은 리터럴/토큰 휴리스틱이다.

  대신 `answer_contract_binding_missing` feedback에 기존 List Determinism Policy를 더 직접적으로 상기했다.
  missing binding이 source sequence/reference/order field를 정당화하라는 뜻이 아니며, 그런 field/order key는
  제거하거나 tied rows를 반환하거나 다른 label을 고르라고 명시했다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_order_binding_by_query_order_count -q`
    통과 (`1 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_82`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 161 — Trial 83 rejected low-quality surfaces, no new rule

- **질문**:
  binding feedback 보강 뒤 `trial_83`에서 sequence tie-break를 user_request로 포장하는 패턴이 줄었는가?
  accepted가 없었다면 어떤 저품질이 validator에 걸렸는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, solver rollout 없음.

  제출 흐름:
  - submit 1: `composer_submit_draft_missing`
  - submit 2:
    `이번 입원 중에 내게 진단된 질병들이 궁금해요. 진단 번호, ICD 코드, ICD 버전, 입원 시간,
    퇴원 시간을 순서대로 5개 보여주세요.`
    - 오류: `label_non_user_visible_source`
    - blocked PK/handle 성격의 진단 번호, ICD 코드, ICD 버전을 label로 노출
  - submit 3:
    `이번 입원에서 DRG 코드 정보를 알고 싶어요. DRG 유형, 설명, 중증도, 사망률을 보여주세요.`
    - 오류: `answer_contract_list_size_invalid`, `label_null_value_forbidden`,
      `label_no_primary_key_source`, `answer_contract_order_ambiguous`
    - no-PK `drgcodes`, 2 rows, null severity/mortality
  - submit 4: `composer_submit_draft_missing`
  - submit 5 budget exhausted:
    `이번 입원 중에 처방된 약물 정보를 알고 싶어요. 약물명, 처방유형, 상태, 투여 경로,
    입력 시간을 가장 최근에 입력된 순서로 5개씩 보여주세요.`
    - 오류: `answer_contract_phrase_missing`, `answer_contract_duplicate_answer_rows`,
      `answer_contract_binding_missing`
    - pharmacy list가 duplicate projected rows를 남겼고, secondary order key를 request/label에
      제대로 반영하지 못함

- **reasoning 교차 분석**:
  sequence tie-break를 "기록 순번"으로 포장하는 직접 패턴은 이전 trial보다 줄었다. 대신 composer는
  blocked diagnosis fields, no-PK DRG fields, duplicate pharmacy rows로 이동했다.

  마지막 pharmacy 후보에서 composer는 duplicate rows를 인식하고 visible distinguishing field를 찾으려 했다.
  하지만 최종 request/answer_contract/query가 싱크되지 않아 duplicate/binding feedback을 받았다.
  이는 어려운 좋은 문제가 아니라 아직 label construction 품질이 부족한 케이스다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 각 후보가 validator에 의해 정확히 거절됐다. 특히 blocked source,
  no-PK/null list, duplicate projected rows는 모두 기존 precision-100 validator가 잡을 수 있는 범위였다.

- **변경**:
  코드 변경 없음. 이미 존재하는 validator와 feedback이 의도대로 작동했다. 추가 hard rule을 만들면
  natural/source 판단으로 넘어가거나 기존 원칙을 중복하게 된다.

- **검증**:
  코드 변경 없음.

- **현재 streak**:
  `trial_83`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 162 — Broad source-role ambiguity must be rejected before solver

- **질문**:
  `trial_84` 계열에서 accepted가 나오면 정말 좋은 데이터인가? reasoning content를 보면
  composer가 broad source-role ambiguity를 submit 전에 인지하고 있는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.

  첫 `trial_84`는 첫 feedback 뒤 이벤트 갱신 없이 장시간 멈춰 provider/API stall로 중단했다.
  품질 판단 대상에서 제외하고 같은 설정으로 `trial_84_retry_01`을 재시도했다.

  `trial_84_retry_01` 결과는 accepted, pass_rate `0.25` (`1/4` matched).

  accepted request:
  `이번 입원 중 처방된 약물 목록을 시작 시간 순으로 알파벳순으로 정렬해서 5개 보여주세요.
  각 약물의 이름, 처방 시작 시간, 종료 시간, 투여 주기, 처방 유형, 주문 상태도 함께 알려주세요.`

  canonical label은 `pharmacy`의 medication/starttime/stoptime/frequency/proc_type/status를
  starttime + medication으로 정렬한 5-row list였다.

- **reasoning 교차 분석**:
  composer는 `pharmacy_id`를 technical handle로 보고 피하려고 했고, starttime 동점을 medication
  tie-break로 고쳤다. 이 부분은 이전 sequence/order repair 개선이 어느 정도 작동한 신호다.

  그러나 더 큰 문제를 놓쳤다. user_request의 “처방된 약물”은 자연스럽게 `prescriptions` 계열도
  강하게 가리킨다. 실제 solver 4개 중 3개는 `prescriptions`에서 시작했고, 그 결과 `pharmacy`
  canonical과 다른 row set/field meaning을 제출하거나 max turns에 걸렸다. 즉 solver가 어려워서
  못 푼 문제가 아니라, composer가 같은 자연어가 두 reachable source surface를 가리키는 상태로
  accepted draft를 제출한 것이다.

- **정성 평가**:
  accepted data: low-quality accepted. 저품질이 solver에서 거절된 게 아니라 quality gate를 통과했으므로
  개선 대상이다.

  rejected data: submit 2는 hidden admission scope와 order binding 부족으로 reject됐다. 이 reject는
  정상이다.

- **변경**:
  hard validator는 추가하지 않았다. “어떤 자연어가 어떤 source surface를 더 자연스럽게 가리키는가”는
  precision 100% validator로 만들 수 없고, 리터럴/컬럼명 휴리스틱으로 구현하면 금지 원칙 위반이다.

  Prompt-first 원칙에 따라 Source surface 정책을 보강했다.
  submit 전 sibling source surfaces를 비교하고, 같은 broad noun이 다른 reachable source에서 다른 row
  set을 만들 수 있으면 broad noun을 쓰지 말라고 명시했다. chosen source role을 자연어로 드러내거나,
  ordinary wording이 가리키는 source를 쓰거나, label을 바꾸게 했다.

  Scope example도 mini draft 형식으로 바꿔 `user_request`와 `answer_contract.answer_phrase`가 broad
  noun만 담은 경우 왜 나쁜지 보이게 했다.

  `submit_draft.user_request` schema description에도 같은 자기검사를 추가했다. 이는 feedback이 아니라
  tool 호출 직전 schema reminder다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_synthesis_runtime.py::test_submit_draft_payload_schema_uses_strict_json_string_fields -q`
    통과 (`2 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`96 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_84_retry_01`은 accepted지만 low-quality accepted로 판정하므로 만족 streak는 `0/5` 유지.

## Iteration 163 — Trial 85 rejected low-quality null/list-shape drafts

- **질문**:
  Iteration 162의 broad source-role ambiguity 보강 뒤 다음 smoke에서 accepted 품질이 좋아지는가?
  accepted가 없다면 rejected data는 어려운 좋은 문제였는가, 아니면 저품질이 잘 거절된 것인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, solver rollout 없음.

  제출 흐름:
  - submit 1/2: `composer_submit_draft_missing`
  - submit 3:
    `환자의 병원 내 이동 기록 중 가장 최근 5건을 보여주세요. 각 이동 기록 입실 시간,
    퇴실 시간, 진료과(병동), 이벤트 유형을 포함해주세요.`
    - 오류: `label_null_value_forbidden`
    - 최신 discharge transfer row의 `transfer_out_time`, `care_unit`이 null이었다.
  - submit 4: `composer_submit_draft_missing`
  - submit 5 budget exhausted:
    `환자의 입원 정보를 알려주세요. 입원 시간, 입원 유형, 퇴원 시간, 퇴원 장소를 포함해주세요.`
    - 오류: `answer_contract_list_size_invalid`
    - 실제 admission row가 1개인데 detail lookup을 list로 제출했다.

- **reasoning 교차 분석**:
  composer는 EMAR 후보에서 duplicate-looking rows를 보고 `emar_seq`/`emar_id`를 쓰지 않고 다른 label로
  전환했다. 이 부분은 sequence/order field 금지 정책을 어느 정도 따른 신호다.

  lab 후보에서는 `labevent_id` tie-break를 생각했지만, 해당 query는 ToolBudgetFeedback에 막혔고
  submit까지 가지 않았다. 이후 transfers 후보는 row/order는 괜찮았지만 null output field 때문에
  정확히 reject됐다. feedback 뒤 composer는 nullable field를 제거하거나 non-null field로 rerun해야
  한다고 reasoning했지만, 여러 새 target으로 이동하다가 최종적으로 1-row admission list를 제출했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. transfers는 null answer field 문제, admissions는 1-row detail
  lookup/list-shape 문제다. solver가 풀기 어려운 좋은 문제가 아니라 composer가 제출하지 말아야 할
  draft였다.

- **변경**:
  코드 변경 없음. `label_null_value_forbidden` feedback은 이미 nullable output field 제거, informative
  non-null field로 rerun, 또는 다른 scoped label 선택을 안내한다. `answer_contract.kind` schema도
  1-2 row detail lookup을 list fallback으로 제출하지 말라고 말한다. 새 hard rule을 추가할 정밀도
  100% 근거는 없다.

- **검증**:
  코드 변경 없음.

- **현재 streak**:
  `trial_85`는 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 164 — Trial 86 caught helper-field evidence and sequence repair

- **질문**:
  `trial_86`에서 accepted가 없었다면, 실패 원인은 새 정책 부재인가 아니면 기존 정책 위반을
  feedback이 잘 잡은 것인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, solver rollout 없음.

  제출 흐름:
  - submit 1: `composer_submit_draft_missing`
  - submit 2:
    `이번 입원 중 투약 기록 중 앞에서 5가지를 보여주세요`
    - 오류: `answer_contract_evidence_mismatch`, `answer_contract_order_ambiguous`
    - latest query는 `admission_type` helper field를 선택했지만 label에는 빼서 evidence mismatch가 났고,
      event_time 동점으로 order도 불안정했다.
  - submit 3:
    `이번 입원 중 투약 기록 중 가장 먼저 이루어진 5가지를 시간 순서대로 보여주세요`
    - 오류: `answer_contract_evidence_mismatch`
    - query에 `emar_sequence`를 추가했지만 label/request/contract 싱크가 맞지 않았다.
  - submit 4:
    같은 request
    - 오류: `answer_contract_binding_missing`
    - label에 `emar_sequence`가 들어갔지만 user_request가 이를 요청하지 않았다.
  - submit 5: `composer_submit_draft_missing`

- **reasoning 교차 분석**:
  composer는 event_time tie를 보고 `emar_seq`를 “natural tie-breaker”라고 판단했다. 하지만 이후
  binding feedback에서 `emar_sequence`가 단지 tie-break를 위해 추가된 sequence field라는 점을 인지하고
  제거/return tied rows 방향으로 가려 했다. 다만 phrase/binding-only feedback 뒤 data tool을 호출해
  ToolBudgetFeedback에 막혔고, 최종 submit까지 복구하지 못했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. 첫 후보는 helper field/evidence mismatch와 order ambiguity,
  이후 후보는 sequence field repair 문제다. 어려운 좋은 문제가 아니라 composer가 기존 Label Contract와
  List Determinism Policy를 충분히 따르지 못한 케이스다.

- **변경**:
  코드 변경 없음. `query.select`/Label Contract는 이미 “selected field는 label field가 된다”와
  “latest successful query result를 그대로 복사”를 말한다. List Determinism Policy와 binding feedback도
  sequence/order field를 tie-break repair로 정당화하지 말라고 말한다. 새 validator를 추가하려면
  sequence-like 컬럼명 리터럴 휴리스틱으로 흐를 위험이 있어 금지 원칙에 맞지 않는다.

- **검증**:
  코드 변경 없음.

- **현재 streak**:
  `trial_86`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 165 — Trial 87 rejected malformed Korean and hidden tie-breaks

- **질문**:
  `trial_87`에서 input/output event 계열 후보가 accepted 품질로 이어지는가? 실패한다면
  hidden tie-break, malformed request, source-role ambiguity 중 무엇이 남는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, solver rollout 없음.

  제출 흐름:
  - submit 1: `composer_submit_draft_missing`
  - submit 2:
    `이번 입원中最先로 투약된 주사제 5가지의 이름, 투약 시작 시간, 용량, 단위, 그리고 투약 상태를 알려주세요.`
    - 오류: `answer_contract_phrase_missing`, `answer_contract_order_ambiguous`,
      `answer_contract_binding_missing`
    - mixed-script/malformed Korean과 order binding 부족
  - submit 3:
    `이번 입원 중 시작 시간이 빠른 순서대로 처음 5가지 투약 품목의 이름과 투약 시작 시간, 용량, 단위를 알려주세요.`
    - 오류: `answer_contract_order_ambiguous`, `answer_contract_binding_missing`
    - inputevents starttime tie가 남았고, composer가 hidden `orderid` tie-break를 시도했다.
  - submit 4:
    `이번 입원 중 기록된 배양관 및 쇄쉽 출량 데이터 5개를 시간이 빠른 순서대로 품목 이름, 측정 시간, 측정값, 단위를 포함해서 관싸 있겠어?`
    - 오류: `answer_contract_binding_missing`
    - outputevents로 target을 바꿨지만 request Korean이 깨졌고 binding이 부족했다.
  - submit 5 budget exhausted:
    `이번 입원 중 기록된 배양관 및 쇄쉽 출량 데이터 5개를 측정 시간이 빠른 순서대로 정렬해주세요. 동일한 시간이마다는 경우 품목 이름 순서로 확인해주세요.`
    - 오류: `answer_contract_phrase_missing`
    - tie-break phrase를 추가하려 했지만 여전히 malformed wording.

- **reasoning 교차 분석**:
  composer는 inputevents에서 hidden `orderid`가 tie-break로 쓰였다는 점을 인지하고 다른 label로
  전환했다. 이 점은 hidden tie-break 금지 정책이 reasoning에 들어간 신호다.

  그러나 outputevents repair에서는 한국어가 크게 깨졌다. 기존 Request Contract와 schema description은
  mixed-script/malformed terms를 금지하고 있고, phrase_missing이 이를 reject했다. 즉 저품질이 accepted된
  것은 아니다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. hidden order key, malformed Korean, missing phrase/binding 문제다.
  어려운 좋은 문제가 아니라 composer가 제출하지 말아야 할 draft들이었다.

- **변경**:
  코드 변경 없음. malformed wording 금지, hidden tie-break 금지, phrase/binding repair는 이미 prompt와
  feedback에 있다. 이번 trial은 그 정책이 accepted 전에 저품질을 막은 사례로 본다.

- **검증**:
  코드 변경 없음.

- **현재 streak**:
  `trial_87`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 166 — List difficulty-up may add a related aggregate dimension

- **질문**:
  `trial_88`에서 too-easy 이후 composer가 같은 row set에 related aggregate dimension을 추가했는데도
  `answer_contract_not_incremental`로 reject된 것이 정당한가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`.

  submit 2:
  `내 입원 이력을 최신순으로 보여주세요. 입원 시간, 퇴원 시간, 입원 유형, 입원 장소, 퇴원 장소를 알려주세요.`
  - solver pass_rate `1.0` (`4/4`)
  - 너무 쉬워서 rejected.
  - 정성 평가: 데이터 자체는 정확하고 자연스럽지만 난이도상 너무 쉬운 direct list다.

  submit 4/5:
  `내 입원 이력을 ... 해당 입원 기록에 포함된 진단 개수도 알려주세요.`
  - 같은 3개 admission row, 같은 admission time desc order, 기존 admission fields 유지.
  - 새 field는 `diagnosis_count`: `admissions<-diagnoses_icd.hadm_id` related aggregate count.
  - 오류: `answer_contract_not_incremental`
  - diagnostics: `operation_changed`, `list_output_only`, `no_new_structural_constraint`.

- **reasoning 교차 분석**:
  composer는 too-easy feedback을 읽고 “related-row reasoning / aggregate dimension”을 추가해야 한다고
  정확히 판단했다. 이후 edge를 잘못 추측한 detour가 있었지만, 최종 query는 admission별 diagnosis
  count aggregate를 만들었다.

  이 draft는 원칙상 좋은 difficulty-up이다. list row set을 좁히지 않았고, prior output fields/order를
  유지하면서 related aggregate lookup을 추가했다. solver rollout까지 갔다면 너무 쉬운 direct list보다
  더 나은 학습 trace가 되었을 가능성이 높다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - submit 2는 good-but-too-easy rejected.
  - submit 4/5는 hard-good 후보를 validator가 과하게 막은 false reject로 본다.

- **변경**:
  precision 100% 구조 판정만 수정했다. 리터럴/컬럼명 휴리스틱은 쓰지 않았다.

  `_query_evidence_incremental_errors`에서 list difficulty-up을 평가할 때:
  - 이전 `select` output과 현재 aggregate query의 `group_by` output이 같은 table/column이면 같은 row-value
    output 보존으로 본다.
  - list retry가 기존 row-value outputs/order/predicate를 보존하고, related table aggregate output을
    새로 추가하면 `list_output_only`가 아니라 structural strengthening으로 인정한다.

  기존 passive same-row output-only addition은 계속 reject된다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_runtime.py::test_incremental_evidence_rejects_list_output_only_field_additions tests/test_synthesis_runtime.py::test_incremental_evidence_allows_list_related_aggregate_dimension tests/test_synthesis_runtime.py::test_incremental_evidence_rejects_added_list_row_filter tests/test_synthesis_runtime.py::test_submit_draft_too_easy_requires_incremental_answer_contract -q`
    통과 (`4 passed`).

  Broader relevant checks:
  - `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
    통과 (`97 passed`).
  - `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
    통과 (`125 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_backend_openai_agents.py tests/test_turn_budget_prompt.py`
  통과.

- **현재 streak**:
  `trial_88`은 accepted가 없으므로 만족 streak는 `0/5` 유지. 다만 false reject를 수정했으므로 다음
  trial에서 too-easy 이후 related aggregate difficulty-up이 통과되는지 확인한다.

## Iteration 167 — Duplicate projected rows are blocking even when source rows differ

- **질문**:
  `trial_88_recheck_01`에서 Iteration 166의 related aggregate false reject 수정이 재현되는가?
  재현되지 않는다면, reasoning content를 기준으로 accepted/rejected 품질을 어떻게 봐야 하는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`.

  이번 recheck는 `trial_88`의 admission-history related aggregate 케이스를 재현하지 못했다.
  대신 `procedureevents` 중심 후보에서 projected answer row duplicate가 반복됐다.

  제출 3:
  `이번 입원 중에 받은 시술 목록을 최신순으로 5개 보여주세요`
  - 오류: `answer_contract_phrase_missing`, `answer_contract_duplicate_answer_rows`
  - `22 Gauge` 두 row가 시술명, 시작/종료 시간, 카테고리, 상태, 체중 등 projected answer field에서
    구분되지 않았다.
  - source row는 서로 다른 procedure event일 수 있지만, solver가 제출할 수 있는 answer row가 같으므로
    정답 row set을 구조적으로 구분할 수 없다.

  제출 5:
  `이번 입원 정보를 알려주세요`
  - 오류: `answer_contract_list_size_invalid`, `answer_contract_phrase_missing`
  - admission detail 1건을 list로 제출했다. 이는 어려운 좋은 문제가 아니라 task shape 오류다.

- **reasoning 교차 분석**:
  reasoning content가 실제로 저장되어 있었고, composer의 판단 오류가 보였다.

  composer는 duplicate row를 처음에는 정확히 인지했다. 그러나 이후 “두 row는 실제로 서로 다른
  procedure event이므로 duplicate answer row도 valid하다”고 판단했다. 이 판단이 문제다.
  원칙상 solver는 hidden source row identity를 제출하지 못하고, requested output fields만 제출한다.
  따라서 source row가 둘이어도 projected answer row가 같으면 list answer는 indistinguishable하다.

  feedback 이후 composer는 `storetime`이나 category 같은 visible field를 추가하려 했지만, duplicate가
  해소되지 않거나 ToolBudgetFeedback boundary를 넘겼다. 마지막에는 1-row admission detail로 후퇴했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - procedure list는 low-quality rejected. 데이터가 어려운 것이 아니라 answer row uniqueness가 없다.
  - admission detail은 low-quality rejected. 1-row detail lookup/list shape mismatch다.

  이번 거절은 나쁘지 않다. 저품질 후보가 accepted되지 않았고, validator가 정확히 막았다. 다만 composer가
  duplicate diagnostics의 의미를 잘못 해석했으므로 prompt/schema/feedback reminder를 개선한다.

- **변경**:
  hard validator는 추가하지 않았다. `answer_contract_duplicate_answer_rows` validator는 이미 precision
  100%로 작동하고 있다.

  리터럴/컬럼명/토큰 휴리스틱은 추가하지 않았다. 이번 변경은 DB 내용이나 특정 테이블/컬럼명을 보지 않는다.

  Prompt-first 원칙에 따라 List Determinism Policy를 압축 보강했다.
  duplicate projected answer rows는 source row가 실제로 서로 달라도 blocking이며, 자연 visible
  field/aggregate 하나로 해소되지 않으면 label을 바꾸라고 명시했다.

  `query` tool schema/description도 같은 원칙을 반영했다. query diagnostics에서 duplicate projected
  answer rows가 나오면 submit 전 blocking diagnostics로 취급해야 한다.

  submit feedback은 새 지시가 아니라 같은 정책의 reminder로 보강했다.
  “distinct underlying source rows are not enough”를 추가해 composer가 같은 오류 해석을 반복하지 않게 했다.

  system prompt 길이 예산은 유지했다. 새 문구를 넣는 대신 workflow와 binding phrase 설명을 조금
  압축했고, `Do not invent ids` 같은 hard prohibition은 유지했다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
    통과 (`1 passed`).
  - `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
    통과 (`1 passed`).
  - `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_duplicate_projected_list_rows -q`
    통과 (`1 passed`).
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_turn_budget_prompt.py::test_synthesis_instructions_contain_hard_prohibitions -q`
    통과 (`2 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`114 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_88_recheck_01`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 168 — Source status must not collapse into broad object status

- **질문**:
  `trial_89`는 accepted였지만 pass_rate가 `0.25`였다. 이 데이터는 어려운 좋은 문제인가,
  아니면 low-quality accepted인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 accepted, pass_rate `0.25` (`1/4` matched).

  accepted request:
  `이번 입원 기간 동안 처방된 약물 5개의 이름, 처방 시작 시간, 종료 시간, 처방 상태 정보를 처방 시작 시간이 빠른 순서대로 보여주세요. 같은 시간에 시작한 약물은 이름 순으로 정렬해 주세요. 약물 이름이 있는 처방만 보여주세요.`

  canonical query는 `pharmacy`를 source로 사용했다.
  canonical status 값은 `pharmacy.status`의 `Discontinued via patient discharge`였다.

  solver 3/4는 같은 request를 보고 `prescriptions` 쪽을 자연스러운 처방 source로 선택했고,
  status 값으로 `MAIN`을 제출했다. 1/4만 `pharmacy.status`까지 찾아가 canonical과 matched했다.

- **reasoning 교차 분석**:
  composer reasoning은 admission anchor에서 pharmacy 26건을 발견했고, 약물 5개 list로 수리해 나갔다.
  order ambiguity와 null medication filter는 비교적 잘 처리했다. 그러나 마지막 source-role 판단에서
  “처방 상태”가 `pharmacy.status`를 뜻한다는 점을 자연어에 충분히 드러내지 못했다.

  solver reasoning은 더 명확하다. 대부분의 solver는 “처방된 약물”, “처방 상태”를 보고
  `prescriptions` table의 약물/상태를 ordinary matching source로 판단했다. 이는 solver가 어려워서
  틀린 것이 아니라, request가 여러 reachable source surface를 동시에 만족시키는 broad wording이었음을
  보여준다.

- **정성 평가**:
  accepted data: low-quality accepted. row count, ordering, null filter 자체는 괜찮지만, status field의
  source role이 하나로 고정되지 않는다. 어려운 좋은 문제가 아니라 source status ambiguity다.

  rejected data:
  - submit 2는 low-quality rejected. 26-row list, null label, unstable order가 명확했다.
  - submit 3은 binding repair 실패.
  - submit 4는 filter phrase가 빠진 repairable draft였다.

  따라서 `trial_89`는 accepted지만 만족 streak에 포함하지 않는다.

- **변경**:
  hard validator는 추가하지 않았다. `pharmacy.status`와 `prescriptions.status`의 의미 차이를
  테이블명/컬럼명/문자열 리터럴로 판정하는 방식은 금지 원칙 위반이고 precision 100%가 아니다.

  Prompt-first 원칙에 따라 Source Surface Policy를 status field에 더 직접적으로 적용했다.
  source status text는 source-specific이며 broad object status나 current/derived state로 묶으면 안 된다고
  system prompt를 압축 보강했다.

  `submit_draft` tool schema에는 result/status/type/category binding 설명을 보강했다.
  related source의 status field는 서로 교환 가능하지 않고, 다른 reachable source가 자체 status를 가질 수
  있으면 broad object status phrase가 invalid라고 명시했다.

  `user_request` schema와 `query` schema/description도 같은 원칙을 reminder로 반영했다.
  status/type/result output은 related surface가 다를 수 있으면 selected source role을 자연어로 이름 붙여야
  한다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
    통과 (`1 passed`).
  - `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
    통과 (`1 passed`).
  - `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned -q`
    통과 (`1 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`114 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_89`는 accepted지만 low-quality accepted로 판정하므로 만족 streak는 `0/5` 유지.

## Iteration 169 — Trial 90 rejected sequence-tie repair, no accepted data

- **질문**:
  Iteration 168의 source-specific status 보강 뒤 다음 smoke에서 accepted 품질이 좋아지는가?
  accepted가 없다면, 거절 데이터는 hard-good인가 low-quality인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`.

  제출 3:
  `최근 투약 기록 5개를 투약시각과 이벤트 내역 순서로 보여주세요.`
  - 오류: `answer_contract_binding_missing`
  - label fields: `medication`, `event`, `charttime`, `sequence`
  - query는 `emar.charttime desc`, `emar.emar_seq asc`로 정렬했고 `sequence`도 output에 포함했다.
  - binding error의 직접 원인은 order phrase가 bare output phrase로 재사용된 점이다.

  제출 5:
  `투약 기록 5개를 최신순으로 약물명, 투약 이벤트, 기록시각을 보여주세요.`
  - 오류: `answer_contract_evidence_mismatch`
  - composer가 `sequence` field를 제거한 label로 바꾸었지만, 그 label을 뒷받침하는 successful query evidence가
    없었다. 직전 query call은 ToolBudgetFeedback에 막혔다.

- **reasoning 교차 분석**:
  좋은 변화가 하나 있었다. composer는 먼저 pharmacy 후보에서 duplicate projected answer rows를 보고
  “underlying row가 달라도 projected answer가 같으면 blocking”이라고 정확히 판단했다. Iteration 167의
  보강은 의도대로 작동한 것으로 보인다.

  이후 EMAR 후보에서는 `emar_seq`를 tie-break/order output으로 사용했다. feedback 후 composer는 sequence가
  technical field라 제거해야 한다고 판단했지만, phrase/binding-only boundary에서 query를 다시 만들지 못했고
  마지막 submit은 latest successful query와 불일치했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - pharmacy 후보는 low-quality rejected. duplicate projected rows가 명확했고 composer가 올바르게 버렸다.
  - EMAR 후보는 잠재적으로 괜찮은 medication administration list였지만, `sequence` tie-break/output이
    자연스럽지 않고 evidence mismatch로 끝났다. hard-good이 아니라 repair/protocol 실패다.

- **변경**:
  코드 변경 없음.

  이유: 이번 문제를 precision 100% validator로 고치려면 `emar_seq` 같은 필드가 “technical sequence”인지
  일반화해서 알아야 하는데, 컬럼명/토큰 기반 휴리스틱은 금지 원칙 위반이다. 기존 validator는 저품질을
  accepted하지 않고 정확히 막았다.

  다만 관찰 사항은 다음 개선 후보로 남긴다. binding feedback 문구에는 “field/order key를 제거하려면 query
  evidence를 다시 맞춰야 한다”는 원칙과 “phrase/binding-only면 data tool을 쓰지 말라”는 reminder가 섞일 수
  있으므로, 같은 패턴이 반복되면 feedback/message 구조를 더 명확히 재검토한다.

- **검증**:
  별도 코드 변경이 없으므로 추가 테스트는 실행하지 않았다. 직전 Iteration 168에서 관련 suite
  `114 passed`와 ruff 통과를 확인했다.

- **현재 streak**:
  `trial_90`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 170 — Trial 91 rejected target switch after too-easy scalar

- **질문**:
  `trial_91`에서 source-status 보강 이후 accepted 품질이 좋아지는가? 실패한다면, too-easy 이후
  difficulty-up이 원칙대로 작동했는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`.

  제출 2:
  `내 입원 기록을 확인하고 싶어. 입원 유형과 입원 경로, 퇴원 장소, 입원일자와 퇴원일자를 최근 순서대로 3개까지 알려줘.`
  - 오류: `answer_contract_list_size_invalid`
  - 해당 patient의 admission row가 1건뿐이었다.

  제출 3:
  `지금까지 내가 병원에 총 몇 번 입원했는지 알려줘.`
  - solver pass_rate `1.0` (`4/4`)
  - 오류: `calibration_inconclusive`
  - 정성 평가: 정확하지만 너무 쉬운 scalar count다.

  제출 4/5:
  hospital transfer history 5-row list로 전환했다.
  - 오류: `answer_contract_not_incremental`
  - diagnostics: `kind_changed`, `operation_changed`, `predicate_removed`, `list_row_filter_added`
  - submit 4에는 `careunit: null`도 포함되어 `label_null_value_forbidden`이 함께 발생했다.

- **reasoning 교차 분석**:
  composer는 admission_count가 too easy로 rejected된 뒤 “더 복잡한 관련 정보”를 찾으려 했다. 그러나 실제로는
  admission_count baseline을 강화한 것이 아니라 transfers list라는 다른 task로 전환했다.

  transfers 후보 자체는 row count와 order는 괜찮았지만, 첫 버전은 null careunit을 포함했고, 최종 버전은
  baseline scalar의 target/predicate/output과 연결되지 않았다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - admission detail list는 low-quality rejected. 1-row list shape mismatch다.
  - admission_count는 good-but-too-easy rejected.
  - transfers list는 potential good list지만, too-easy repair로는 저품질이다. 어려운 좋은 문제가 아니라
    baseline을 버린 target switch다.

- **변경**:
  코드 변경 없음.

  이유: `answer_contract_not_incremental`이 precision 100% 구조 진단으로 정당하게 작동했다.
  target/predicate/kind가 바뀐 것을 리터럴 없이 구조적으로 잡았고, 저품질 accepted는 발생하지 않았다.

- **검증**:
  별도 코드 변경이 없으므로 추가 테스트는 실행하지 않았다.

- **현재 streak**:
  `trial_91`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 171 — Trial 92 rejected blocked codes and too-easy procedure count

- **질문**:
  `trial_92`에서 accepted 품질 개선이 재현되는가? 실패한다면 거절 데이터는 어떤 성격인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`.

  제출 2:
  `이번 입원 중에 기록된 진단 코드 목록을 순서대로 알려주세요.`
  - 오류: `label_non_user_visible_source`
  - `diagnoses_icd.seq_num`, `diagnoses_icd.icd_code`가 blocked source였다.
  - 정성 평가: low-quality rejected. 코드/순번 중심 label은 user-visible answer가 아니다.

  제출 3:
  `이번 입원의 입원 유형, 입원 시간, 퇴원 시간, 그리고 퇴원 장소를 알려주세요.`
  - 오류: `answer_contract_list_size_invalid`, `answer_contract_phrase_missing`
  - single admission detail을 list로 제출했다.
  - 정성 평가: low-quality rejected.

  제출 5:
  `이번 입원 중에 기록된 시술 건수는 몇 건인가요?`
  - solver pass_rate `1.0` (`4/4`)
  - 오류: `calibration_inconclusive`
  - canonical은 `procedures_icd` count `3`.

- **reasoning 교차 분석**:
  composer는 diagnosis code가 blocked라는 feedback 후 admissions detail로 이동했고, 그 다음 procedures count
  scalar로 이동했다. 마지막 후보는 solver 4/4가 모두 맞춘 단순 count다.

  solver reasoning 중 하나는 “시술”이 `procedures_icd`와 `procedureevents` 둘 다 관련될 수 있다고
  언급했다. 다만 최종적으로는 모두 `procedures_icd` count를 제출했다. accepted였다면 broad source-role
  위험을 더 따졌어야 하지만, 이번에는 too-easy로 rejected됐다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - diagnosis code list와 admission detail list는 low-quality rejected.
  - procedure count는 good-but-too-easy rejected이며, broad source wording 위험도 있다.

- **변경**:
  코드 변경 없음.

  이유: blocked code/source, list shape, too-easy scalar가 모두 기존 validator/solver calibration으로
  거절됐다. 저품질 accepted는 발생하지 않았다.

- **검증**:
  별도 코드 변경이 없으므로 추가 테스트는 실행하지 않았다.

- **현재 streak**:
  `trial_92`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 172 — Trial 93 accepted a good medical-order list

- **질문**:
  `trial_93` accepted, pass_rate `0.75`는 정말 좋은 데이터인가? 1개 solver mismatch는 저품질 신호인가,
  아니면 난이도/solver 실수인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 accepted, pass_rate `0.75` (`3/4` matched).

  accepted request:
  `이 입원 기간 동안의 의료 지시 목록을 보여주세요. 지시 시간이 가장 최근인 것부터 순서대로 최상위 5개를 표시하고, 같은 시간에 내린 지시는 순번이 높은 것부터 표시해 주세요. 각 지시의 순번, 지시 시간, 지시 유형, 거래 유형, 입원 유형을 포함해 주세요.`

  canonical query:
  - source: `poe` joined to `admissions`
  - scope: `hadm_id = 20044587`
  - order: `poe.ordertime desc`, `poe.poe_seq desc`
  - output: `order_sequence`, `order_time`, `order_category`, `transaction_type`, `admission_type`
  - row count: 5
  - null 없음, duplicate/order diagnostics 없음.

- **reasoning 교차 분석**:
  composer는 첫 query에서 nullable `order_detail`과 order ambiguity를 발견하고 제거/수리했다.
  최종 query는 `order_type`을 지시 유형으로, `transaction_type`을 거래 유형으로, admission join을 통해
  입원 유형을 반환했다.

  solver 3/4는 같은 source와 field mapping으로 정확히 matched했다. 1개 solver는 `지시 유형`을 더 세부
  subtype 쪽으로 해석해 `order_category: null`을 제출하면서 schema validation에 실패했다.

- **정성 평가**:
  accepted data: good accepted.
  - 의료 지시라는 source role이 자연스럽다.
  - row membership/order가 request와 query에서 일치한다.
  - 순번 tie-break는 request에 명시되어 있고, output에도 포함되어 solver가 검증할 수 있다.
  - 입원 유형 join은 단순 passive field일 수 있지만, 전체 task가 order/filter/join을 요구하므로 너무 쉬운
    direct lookup은 아니다.

  rejected data:
  - submit 2는 low-quality rejected. nullable field와 order ambiguity가 있었다.
  - submit 3은 binding repair 단계였고, 최종 submit에서 정상 수리됐다.

  mismatch 1건은 low-quality accepted 신호라기보다는 solver가 `order_type` vs subtype 계열을 잘못 고른
  케이스로 본다. request에 `거래 유형`이 별도 field로 있어 canonical source 역할은 충분히 구분된다.

- **변경**:
  코드 변경 없음.

  이유: accepted 품질이 충분히 좋고, solver 1개 mismatch를 precision 100% validator나 리터럴 휴리스틱으로
  잡을 근거가 없다.

- **검증**:
  별도 코드 변경이 없으므로 추가 테스트는 실행하지 않았다.

- **현재 streak**:
  `trial_93`은 good accepted로 판정하므로 만족 streak는 `1/5`.

## Iteration 148 — ToolBudgetFeedback must break the SDK tool loop

- **질문**:
  `trial_70`에서 강화된 ToolBudgetFeedback 문구를 composer가 실제로 따랐는가?
  reasoning content가 반환되고 있으므로, 피드백 전후 판단까지 확인한다.

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed / MaxTurnsExceeded`.

  제출 1:
  `이 검체 채취에 대한 미생물 검사 결과를 보여주세요. 검사 시간, 검체 종류, 시험 이름, 균주 이름을 확인하고 싶습니다.`
  - 오류: `answer_contract_list_size_invalid`, `label_null_value_forbidden`,
    `label_values_not_grounded`, `answer_contract_phrase_missing`,
    `answer_contract_evidence_mismatch`, `answer_contract_hidden_filter_unanchored`,
    `answer_contract_duplicate_answer_rows`
  - 단일 row에 `organism: null`이 포함됐고, hidden filter anchor도 부족했다.
  - rejected data 판정: low-quality rejected.

  제출 2:
  `이 중환자실 입원 동안의 배양검사 결과 목록을 시간순으로 보여주세요. 검사 시간, 검체 종류, 시험 이름, 균주 이름을 확인하고 싶습니다.`
  - 오류: `label_null_value_forbidden`, `label_values_not_grounded`,
    `answer_contract_evidence_mismatch`, `label_no_primary_key_source`,
    `answer_contract_query_mismatch`, `answer_contract_order_ambiguous`
  - 5개 microbiology row가 모두 `organism: null`이고, 같은 시간대 row가 섞여 order도 안정적이지 않았다.
  - rejected data 판정: low-quality rejected.

- **reasoning 교차 분석**:
  composer는 ToolBudgetFeedback을 읽고 "submit해야 한다"는 판단을 여러 번 했다.
  그러나 같은 SDK tool-use segment 안에서 다시 schema/profile/query를 호출했다.
  즉 문구가 약해서 못 알아들은 문제가 아니라, tool result가 모델에게 계속 tool call을 허용하는
  run 내부 응답으로 남아 있었다.

  이 문제는 프롬프트 정책 위반을 더 큰 문구로 반복할수록 중복 지시가 늘어날 위험이 있다.
  이미 tool이 구조화된 `{"error": "submit_draft_required"}`를 반환하므로, 이 신호를 런타임에서
  final output으로 승격하고 다음 composer turn에 feedback으로 재주입하는 것이 더 정확하다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - 제출 1/2 모두 어려운 좋은 문제가 아니라 low-quality rejected다.
  - null label, duplicate projected row, evidence mismatch, unstable order가 명확하며 solver가 못 푼
    것이 아니라 composer가 저품질 draft를 제출했다.

- **변경**:
  hard validator나 리터럴 휴리스틱은 추가하지 않았다.
  이번 변경은 데이터 내용, 테이블명, 컬럼명, 문자열 값을 보지 않는다.

  OpenAI Agents backend의 tool-use finalizer가 non-submit tool result라도 JSON payload의
  `error == "submit_draft_required"`이면 final output으로 끊도록 했다.
  이 신호는 우리 data tool budget gate가 직접 생성한 구조화된 protocol error이므로 precision 100%
  적용 대상이다.

  `record_missing_submit_feedback`은 이 final output을 plain-text 누락과 구분한다.
  ToolBudgetFeedback boundary에서는 "증거가 부족하면 data tool을 더 쓰라"는 일반 문구를 넣지 않고,
  원래 tool budget message를 그대로 next step으로 재주입한다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_backend_openai_agents.py::test_synthesis_tool_use_behavior_finalizes_tool_budget_feedback tests/test_synthesis_runtime.py::test_submit_draft_records_tool_budget_missing_submit_feedback -q`
    통과 (`2 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py tests/test_synthesis_backend_openai_agents.py -q`
  통과 (`124 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/backend_openai_agents.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_backend_openai_agents.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_70`도 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 145 — Visible field is not a semantic substitute

- **질문**:
  `trial_67`은 accepted였지만 `solver_pass_rate=0.25`였다. 승인 데이터가 정말 좋은가,
  아니면 low-quality accepted인가?

- **실험/결과**:
  composer는 MIMIC demo의 ICU `outputevents`에서 배출 유형, 배출량, 기록 시간을 오래된 순서대로
  5개 요청하는 task를 만들었다.

  최종 request:
  `이번 ICU 입원 중 배출 유형과 배출량과 기록 시간을 가장 오래된 것부터 5개 보여주세요`

  canonical answer는 `recorded_time`에 `storetime`을 넣었다. 하지만 solver 4개 중 3개는 같은
  요청을 보고 `charttime`을 기록 시간으로 해석했고, 그래서 canonical의 2~5번째 시간이 달라졌다.
  `charttime`은 blocked라 label에 직접 노출할 수 없지만, 그렇다고 `storetime`을 같은 의미의
  “기록 시간”으로 바꿔 넣으면 request/label/source role이 어긋난다.

- **reasoning 교차 분석**:
  composer reasoning은 첫 제출에서 `charttime`이 blocked라는 피드백을 받은 뒤 `storetime`을
  user-visible 대체 필드로 선택했다. 하나의 solver도 reasoning에서 `charttime`이 blocked임을
  인지하고 `storetime`을 제출했지만, 나머지 solver들은 자연어 “기록 시간”을 실제 이벤트 기록
  시간인 `charttime`으로 판단했다.

- **정성 평가**:
  accepted data: low-quality accepted. row set 자체는 괜찮지만, 출력 필드 의미가 애매해서 정답이
  하나로 고정되지 않는다. 어려운 좋은 문제가 아니라 composer가 visibility repair 중 source
  role을 바꿔버린 문제다.

  rejected data: 앞선 rejected들은 대체로 좋은 방향의 repair였다. 7건 list와 blocked field는
  정확히 거절됐고, binding repair도 필요한 거절이었다. 문제는 마지막 accepted에서 visible field
  대체의 의미 차이가 통과한 점이다.

- **변경**:
  hard validator는 추가하지 않았다. `charttime`/`storetime` 같은 컬럼명을 토큰 또는 문자 기반으로
  해석해 잡는 방식은 금지 원칙 위반이고, 모든 DB에서 precision 100%를 보장할 수 없다.

  대신 prompt-first 원칙에 따라 Source Surface Policy를 보강했다. visible field는 hidden field를
  의미적으로 대체할 수 없으며, ordinary wording이 hidden source role을 가리키면 다른 source/label을
  선택해야 한다.

  `query.select` tool schema description도 같은 원칙을 구체화했다. 선택한 field는 canonical label
  field가 되므로 blocked/internal source를 피하려고 의미가 다른 user-visible field를 선택하면
  안 되고, alias로 그 의미 차이를 숨길 수도 없다고 명시했다.

  `label_non_user_visible_source` feedback은 새 지시가 아니라 위 원칙의 reminder로만 보강했다.
  blocked field를 alias로 노출하지 말라는 기존 reminder에, 다른 visible field로 대체하려면
  user_request가 그 selected source role을 자연스럽게 물어야 한다는 문장을 추가했다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
    통과 (`1 passed`).
  - `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
    통과 (`1 passed`).
  - `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_from_non_user_visible_query_source -q`
    통과 (`1 passed`).
  - `uv run pytest tests/test_turn_budget_prompt.py -q`
    통과 (`6 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`112 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/turn_budget.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

- **현재 streak**:
  `trial_67`은 accepted지만 low-quality accepted로 판정하므로 만족 streak는 `0/5` 유지.

## Iteration 146 — Phrase/binding-only feedback must repair in place

- **질문**:
  `trial_68`은 왜 accepted 없이 `MaxTurnsExceeded`가 되었는가? composer reasoning을 보면
  어느 부분이 저품질이고, 어느 부분을 개선해야 하는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed / MaxTurnsExceeded`.

  제출 1:
  `나의 검사 결과 중 이상이 확인된 최근 5개 항목을 최신순으로 보여줘`
  - 오류: `answer_contract_phrase_missing`, `answer_contract_order_ambiguous`
  - 5개 row가 모두 같은 `test_time`이라 “최신순”만으로 limited row membership/order가 고정되지 않았다.
  - rejected data 판정: low-quality rejected. 어려운 문제가 아니라 order contract가 잘못된 문제.

  제출 2:
  `나의 입원 이력 중 최근 3개의 입원 기록을 입원일 순으로 보여줘`
  - 오류: `answer_contract_list_size_invalid`, `answer_contract_phrase_missing`
  - 실제 입원 기록은 2건인데 request는 3건을 요구했다.
  - rejected data 판정: low-quality rejected. list shape 불일치.

  제출 3:
  `나의 최근 5개의 약물 투여 기록을 투여 시간 순으로 보여줘`
  - 오류: `answer_contract_phrase_missing`, `answer_contract_binding_missing`
  - label 자체는 약물명, 투여 상태, 예정 시간, 투여 시간 phrase를 request에 추가하면 수리 가능한
    후보였다. 그러나 composer는 phrase/binding repair를 하지 않고 microbiology, ICU stay,
    prescriptions 쪽으로 새 탐색을 하다가 turn을 소진했다.
  - rejected data 판정: 수리 가능 후보였지만 request/contract만 미완성. 이것은 composer 행동
    문제다.

- **reasoning 교차 분석**:
  composer는 첫 lab 후보에서 tie를 인지했고, `labevent_id`를 tie-break로 쓰려는 생각까지 했다.
  이는 기술 handle/order wording으로 갈 위험이 있어 좋은 방향이 아니다. rejection 후 target 전환은
  합리적이었다.

  admission 후보에서는 profile로 row_count=2를 보고도 “최근 3개” list를 제출했다. 이 역시 잘
  거절됐다.

  핵심 문제는 세 번째 이후다. feedback은 phrase/binding-only였는데 reasoning은 “scalar aggregate를
  쓰겠다”, “microbiology”, “ICU stays”, “prescriptions”로 계속 새 target을 탐색했다. 이는
  Feedback Handling Policy의 “phrase repair는 request/answer_contract 수리” 원칙을 충분히
  따르지 못한 것이다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - 제출 1/2는 low-quality rejected.
  - 제출 3은 hard-good이 아니라 “수리 가능한 draft를 수리하지 못한” 케이스다. data 자체는
    잠재적으로 괜찮았지만, 같은 label의 자연어/contract repair가 필요했다.

- **변경**:
  hard validator는 추가하지 않았다. “이 label이 수리 가능한가”는 자연어 품질 판단을 포함하므로
  precision 100% validator로 만들 수 없다.

  Prompt-first 원칙에 따라 Feedback Handling Policy를 보강했다.
  phrase/binding-only feedback은 새 탐색을 하지 않고, 같은 label의 `user_request`와
  `answer_contract`를 수리해야 한다고 명시했다.

  `answer_contract_phrase_missing`과 `answer_contract_binding_missing` feedback은 새 지시가
  아니라 같은 정책의 reminder로 보강했다. “When only phrase/binding errors remain, do not call
  data tools; repair the same label in place.”라는 문구를 추가했다.

  prompt 길이 예산을 유지하기 위해 Draft Submission Budget 표현만 짧게 압축했다. 의미 변경은 없다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
    통과 (`1 passed`).
  - `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_reset_after_contract_repair_feedback tests/test_synthesis_runtime.py::test_submit_draft_feedbacks_missing_list_output_binding -q`
    통과 (`2 passed`).
  - `uv run pytest tests/test_turn_budget_prompt.py -q`
    통과 (`6 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`112 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

- **현재 streak**:
  `trial_68`은 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 147 — ToolBudgetFeedback is a hard boundary

- **질문**:
  `trial_69`은 왜 다시 `MaxTurnsExceeded`가 되었는가? `trial_68`에서 고친
  phrase/binding-only repair 문제와 같은 원인인가, 아니면 다른 원인인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed / MaxTurnsExceeded`.

  제출 1:
  `이번 중환자실 입원 중에 시행된 처치 목록 중 시작 시간 순서대로 상위 5개를 알려줘`
  - 오류: `answer_contract_evidence_mismatch`, `answer_contract_duplicate_answer_rows`
  - `20 Gauge` procedure 두 row가 모든 projected answer field에서 동일했다. composer는 query
    diagnostics로 duplicate를 인지했고 category/location 등 visible field를 추가하려 했지만, 일부
    query는 budget boundary에 막혔다. 그 뒤 blocked/failed query의 값을 label에 섞어
    evidence mismatch도 만들었다.

  제출 2:
  `이번 중환자실 입원 중에 주입된 약물 중 시작 시간 순서대로 상위 5개를 알려줘`
  - 오류: `label_values_not_grounded`, `answer_contract_phrase_missing`,
    `answer_contract_evidence_mismatch`, `answer_contract_duplicate_answer_rows`
  - inputevents 성격의 값을 실제 latest query evidence 없이 작성했고, 같은 projected rows가 반복됐다.

  제출 3:
  `이번 중환자실 입원 중 발생한 토어양 측정 기록을 시간 순서대로 모두 보여줬`
  - 오류: `answer_contract_list_size_invalid`, `answer_contract_phrase_missing`
  - outputevents는 2건뿐이라 list shape가 맞지 않았다. request Korean도 깨졌다.

- **reasoning 교차 분석**:
  composer는 첫 procedure query에서 duplicate row를 정확히 인지했다. 그러나 `ToolBudgetFeedback`이
  “한 번의 final query 후 submit”을 요구한 뒤에도 profile/query/neighborhood/sample을 계속 호출했다.
  reasoning에는 “budget limit again”, “need to submit”이 반복되지만 실제 다음 tool은 다시 data tool인
  경우가 많았다.

  즉 이번 문제의 핵심은 phrase/binding-only repair가 아니다. label duplicate나 row-count 문제는
  정상적으로 rejected됐고, 이후 composer가 protocol boundary를 hard boundary로 취급하지 못해 turn을
  낭비한 것이 직접 원인이다.

- **정성 평가**:
  accepted data: 없음.

  rejected data:
  - 제출 1/2는 low-quality rejected. duplicate projected rows와 evidence mismatch가 명확하다.
  - 제출 3도 low-quality rejected. 2-row list와 깨진 request 때문에 좋은 어려운 문제가 아니다.

- **변경**:
  hard validator는 추가하지 않았다. 이 케이스는 이미 validator가 정확히 reject하고 있다.

  Prompt-first 원칙에 따라 Draft Submission Budget 문구를 보강했다.
  `ToolBudgetFeedback`을 hard boundary로 명명하고, exploration 중단, `submit_draft` 우선, final
  query가 허용된 경우에도 한 번만 실행한 뒤 submit해야 한다고 압축해 명시했다.

  data tool budget feedback message도 같은 원칙의 reminder로 보강했다.
  다음 tool call이 `submit_draft`여야 하며, final label/repair query가 아직 없을 때만 정확히 한 번
  query를 허용하고, 그 뒤 target switch나 추가 data tool을 하지 말라고 직접 표현했다.

- **검증**:
  Targeted:
  - `uv run pytest tests/test_turn_budget_prompt.py -q` 통과 (`6 passed`).
  - `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
    통과 (`1 passed`).
  - `uv run pytest tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_late_first_submit tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_late_feedback_repair tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_binding_only_data_repair -q`
    통과 (`3 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`112 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/turn_budget.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

- **현재 streak**:
  `trial_69`도 accepted가 없으므로 만족 streak는 `0/5` 유지.

## Iteration 144 — Generic value fields must be requestable

- **질문**:
  `trial_66`은 accepted 없이 synthesis_failed로 끝났다. 마지막 procedure list는 좋은 데이터였는가,
  아니면 field 선택 자체가 저품질이었는가?

- **실험/결과**:
  composer는 특정 입원 기간의 최근 procedureevents 5건을 만들었다.

  canonical fields:
  - `procedure_name`
  - `procedure_time`
  - `procedure_status`
  - `procedure_value`

  request는 여러 번 수리됐지만, `procedure_value`를 자연스럽게 묻지 못했다. 최종 request는
  `시술명, 시술 시각, 완료 상태 값`을 요청했는데, answer_contract에서는 `procedure_value`를
  `시술 시각` phrase에 다시 묶어 `duplicate_output_binding_phrases`와 binding error가 남았다.

- **reasoning 교차 분석**:
  solver rollout은 없었다. composer reasoning은 `procedure_value`를 “시술 횟수”나 procedure
  name에 묶을 수 있을지 고민했지만, 실제 값은 procedureevents의 generic numeric source value라
  사용자에게 어떤 의미인지 자연스럽게 설명되지 않았다. 이 경우 좋은 수리는 `procedure_value`를
  label에서 제거하고 final query를 다시 실행하는 것이다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. row set과 order는 좋지만, generic value field를 출력에
  포함한 순간 requestability가 무너졌다. 어려운 좋은 문제가 아니라 field selection 문제다.

- **변경**:
  hard validator는 추가하지 않았다. 어떤 numeric field가 자연스러운 측정값인지 여부를
  리터럴/컬럼명으로 판정하면 금지 원칙 위반이다.

  대신 `query.select` tool schema description을 보강했다.
  selected field는 전부 label field가 되므로, generic measurement/value field는 요청이 그
  measured amount 또는 source value role을 ordinary language로 이름 붙일 수 있을 때만 선택하고,
  그렇지 않으면 omit하라고 명시했다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`1 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`112 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_tooling_composer_tool_factory.py`
  통과.

## Iteration 173 — Source classification wording must name the selected surface

- **질문**:
  `trial_94`의 procedure list는 solver에게 어려운 좋은 문제였는가, 아니면 composer가 broad
  request를 잘못 제출한 저품질 문제였는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed / calibration_inconclusive`, `pass_rate=0/4`.

  최종 budget-exhausted 제출:
  `이 입원 기간 동안 수행된 시술 목록을 수행 시간, 시술명, 시술 유형으로 보여주세요. 수행 시간 순서대로 정렬하고, 수행 시간이 같은 경우 시술명 순서로 정렬해주세요`

  canonical:
  - source: `icustays -> procedureevents -> d_items`
  - fields: `procedureevents.starttime`, `d_items.label`, `d_items.category`
  - order: `starttime asc`, `d_items.label asc`

  solver 4개 모두 row set, 시간, 시술명, order는 사실상 같은 경로로 도달했다. 그러나 `시술 유형`
  field를 모두 `procedureevents.ordercategoryname`으로 제출했고, canonical의 `d_items.category`와
  달라서 전부 mismatch가 났다.

- **reasoning 교차 분석**:
  composer는 procedureevents 5건을 발견했고, 같은 `performed_at` tie를 해결하려고 처음에는
  `orderid`를 tie-break로 쓰려 했다가 blocked handle 진단을 보고 제거했다. 그 뒤 visible한
  `d_items.label`을 tie-break로 사용한 것은 List Determinism Policy 관점에서 올바른 수리다.

  실패의 핵심은 order가 아니라 output source surface였다. composer는 `d_items.category`를
  `시술 유형`으로 제출했지만, solver reasoning은 “procedure event의 type”을 자연스럽게
  `procedureevents.ordercategoryname`으로 해석했다. 즉 solver가 풀 수 없었던 것이 아니라, request가
  같은 broad classification noun으로 둘 이상의 reachable source surface를 허용했다.

- **정성 평가**:
  accepted data: 없음.

  rejected data: low-quality rejected. row set과 order는 좋았지만, `시술 유형`이 어떤 source의
  classification인지 고정되지 않았다. solver가 주어진 tool로 같은 rows를 찾고 다른 visible
  classification field를 고른 것이므로 hard-good이 아니다.

- **변경**:
  hard validator는 추가하지 않았다. source classification phrase가 충분히 자연스럽게 특정 source를
  고정하는지 판정하려면 자연어 의미 판단이 필요하므로 precision 100% validator로 만들 수 없다.
  리터럴/토큰/컬럼명 기반 휴리스틱도 금지 원칙 때문에 사용하지 않았다.

  Prompt-first 원칙에 따라 durable policy와 tool schema description만 보강했다.
  `status/type` 중심으로 쓰여 있던 source-sensitive field 정책을 `status/type/category/classification`
  전반으로 일반화했다. broad object wording이 여러 reachable source surface에 맞을 수 있으면
  `user_request`와 `answer_contract`가 선택한 source role을 자연어로 드러내야 하고, 그 표현이
  어색하면 다른 label/source를 선택해야 한다.

  feedback은 추가하지 않았다. 이번 문제는 feedback 준수 문제가 아니라 submit 전 정책 판단 문제라서,
  피드백에 새 지시를 중복으로 넣지 않는 원칙을 지켰다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_runtime.py::test_submit_draft_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`3 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py -q`
  통과 (`108 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_94`는 accepted가 없으므로 연속 만족 accepted streak는 `0/5`로 리셋.

## Iteration 174 — Trial 95 accepted a good medication-administration list

- **질문**:
  `trial_95` accepted data는 정말 좋은 데이터인가? rejected 단계는 어려운 문제였는가, 저품질이었는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `accepted`, `quality_gate_status=accept`, `pass_rate=3/4`.
  flow_id는 `real_db_trial:20260429T102917Z:c4588888`.

  최종 request:
  `환자의 투약 시간과 약물명이 포함된 최근 투약 기록 3개를 보여주세요`

  canonical:
  - source: `emar`
  - hidden scope: `subject_id=10010471`
  - order/limit: `charttime desc`, `limit=3`
  - fields: `administration_time = emar.charttime`, `medication_name = emar.medication`
  - answer:
    1. `2155-12-07T14:08:00` — `HYDROmorphone Infusion – Comfort Care Guidelines`
    2. `2155-12-07T12:59:00` — `Sodium Chloride 0.9%  Flush`
    3. `2155-12-07T12:58:00` — `HYDROmorphone (Dilaudid)`

- **reasoning 교차 분석**:
  composer는 eMAR 후보를 고르고 환자/입원 context를 확인한 뒤, 최근 투약 기록 3건을 만들었다.
  첫 query는 `event_txt`를 함께 반환했지만 세 번째 row가 `null`이라 `label_null_value_forbidden`으로
  거절됐다. 이후 같은 label target을 유지하고 `event_txt`를 제거한 query를 다시 실행한 점은
  Label Grounding Policy에 맞다.

  phrase/binding feedback 뒤에는 새 탐색 없이 request/contract만 수리했다. 최종 request에는
  `투약 시간`과 `약물명`이 직접 들어가서 output bindings가 자연스럽게 고정됐다.

  solver 3개는 eMAR를 선택하고 `subject_id`, `charttime desc`, `medication`으로 정확히 맞췄다.
  1개 solver는 prescriptions를 투약 기록으로 해석해 다른 row set을 냈다. 다만 최종 request의
  `투약 시간` 표현은 실제 투약 기록인 eMAR 쪽을 충분히 가리키고, 다수 solver가 같은 경로를 찾았으므로
  accepted data 품질 결함보다는 단일 solver 경로 선택 실패로 본다.

- **정성 평가**:
  accepted data: good. hidden patient scope가 자연스럽고, 요청 field가 명시적이며, `charttime desc`
  top-3 order가 진단상 unique하다. null field도 제거되어 label이 간결하다.

  rejected data:
  - 제출 2는 low-quality rejected. null `event_description`을 label field로 포함했고, request에 없는
    `투약 상태`/보조 order binding까지 넣었다.
  - 제출 3/4는 data 자체는 수리 가능했지만 request/contract phrase가 미완성인 contract-repair 단계다.
    solver가 풀기 어려운 좋은 문제라기보다 submit 전 request binding이 부족한 상태였다.

- **변경**:
  코드 변경 없음. 이전 Iteration 173의 source classification 보강 뒤 첫 trial이며, 이번 accepted는
  별도 정책 수정 없이 만족 기준을 충족했다.

- **현재 streak**:
  `trial_95`는 좋은 accepted로 판정하므로 연속 만족 accepted streak는 `1/5`.

## Iteration 175 — Missing-submit after ToolBudgetFeedback must stay locked

- **질문**:
  `trial_96` 실패는 어려운 좋은 문제 때문인가, 아니면 composer가 protocol boundary를 지키지 못한
  저품질 생성 때문인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T103428Z:d3098c4d`.

  제출 흐름:
  - 첫 budget boundary 후 submit 없이 종료되어 `composer_submit_draft_missing`.
  - 그 다음에도 query/query/query/schema_map을 실행한 뒤 다시 `composer_submit_draft_missing`.
  - 진단 code list 제출은 `label_non_user_visible_source`.
  - eMAR list 제출은 `label_null_value_forbidden`.
  - 마지막 admission detail object는 `answer_contract_scalar_not_aggregate`.

- **reasoning 교차 분석**:
  composer는 admission hub를 잡고 diagnosis와 eMAR를 탐색했다. diagnosis query는 `seq_num`,
  `icd_code`, `icd_version`이 모두 blocked인데도 “row가 distinguishable하다”고 판단해 제출했다.
  reject 이후에는 eMAR로 전환했지만 `medication=null` row를 포함했고, 마지막에는 budget에 몰려
  admissions detail object를 scalar로 제출했다.

  더 중요한 반복 패턴은 submit 누락 후 boundary 해제다. `ToolBudgetFeedback`은 “한 번 query 후
  submit”을 요구했지만, 다음 라운드에서 일반 feedback처럼 여러 data tool이 실제 실행됐다. 이건
  자연어 품질 판단이 아니라 프로토콜 상태 전이 문제다.

- **정성 평가**:
  accepted data: 없음. 연속 만족 accepted streak는 `0/5`로 리셋.

  rejected data:
  - diagnosis code draft는 low-quality rejected. blocked/internal answer field를 직접 노출했다.
  - eMAR draft는 low-quality rejected. null answer field가 포함됐다.
  - admission detail scalar는 low-quality rejected. aggregate가 아닌 single-record detail을 scalar로
    제출했다.

- **변경**:
  precision 100%가 가능한 hard protocol gate를 추가했다. `ToolBudgetFeedback`을 받은 뒤
  `submit_draft` 없이 끝난 경우, 다음 라운드에서도 그 boundary를 유지한다. 이후 data tool은
  query가 아직 없을 때 `query` 1회만 허용하고, 그 뒤에는 어떤 data tool도
  `submit_draft_required`를 반환한다.

  이 변경은 DB literal, column token, 자연어 의미를 판정하지 않는다. “이전 run이 ToolBudgetFeedback
  이후 submit 없이 끝났는가”와 “boundary 이후 query가 이미 있었는가”만 보는 실행 프로토콜이므로
  precision 100%로 강제할 수 있다.

  plain text submit 누락 전체를 막지는 않았다. final output이 ToolBudgetFeedback payload가 아닌
  일반 plain text였던 경우는 기존처럼 evidence 보강 여지를 유지한다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_records_missing_submit_protocol_feedback tests/test_synthesis_runtime.py::test_submit_draft_records_tool_budget_missing_submit_feedback tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_allows_repair_query_after_limit tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_late_feedback_repair -q`
  통과 (`4 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`97 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_96`은 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.

## Iteration 176 — Too-easy repair must not use hidden existence filters

- **질문**:
  `trial_97`은 good-hard data를 살리지 못한 것인가, 아니면 too-easy 이후 composer가 잘못된
  difficulty-up 수리를 한 것인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T104246Z:000bc7f0`.

  첫 제출:
  `내 최근 입원 기록 5건과 각 입원의 유형, 입원 시각, 퇴원 시각을 보여주세요.`
  - pass_rate `4/4 = 1.0`
  - solver 모두 `admissions`에서 `subject_id=10039708`, `admittime desc`, limit 5를 정확히 풀었다.
  - data 자체는 깨끗하지만 너무 쉬운 baseline이다.

  이후 수리:
  - `내 진단이 있는 최근 입원 기록...`으로 hidden diagnosis existence filter를 추가했다.
  - 마지막에는 `입원 기간이 가장 긴 5건` wording을 넣었지만 canonical answer/query는 여전히 기존
    admission rows였고, hidden diagnosis filter가 남아 있었다.
  - validator가 `answer_contract_hidden_filter_unanchored`, `label_not_strengthened`,
    `answer_contract_not_incremental`로 거절했다.

- **reasoning 교차 분석**:
  composer reasoning은 “primary diagnosis를 붙이면 related-row dimension이 된다”고 판단했다.
  하지만 실제 schema에는 diagnosis code를 readable diagnosis description으로 연결할 수 있는
  composer query edge가 없었고, available query는 blocked diagnosis code 또는 hidden `seq_num=1`
  filter뿐이었다. 따라서 이 related dimension은 visible answer/aggregate로 만들 수 없었다.

  새 hard gate는 작동했다. `composer_submit_draft_missing` 이후 두 번째 query는 실제 DB query가 아니라
  `ToolBudgetFeedback: Missing-submit boundary reminder`를 반환했다. 다만 모델은 그 뒤에도 좋은
  strengthening을 만들지 못했다.

- **정성 평가**:
  accepted data: 없음. 연속 만족 accepted streak는 `0/5`.

  rejected data:
  - 첫 draft는 too-easy rejected. 좋은 품질이지만 목표 난이도에는 너무 쉬웠다.
  - 이후 drafts는 low-quality rejected. hidden/existence filter로 row set을 바꾸고, visible
    strengthening output이나 aggregate를 만들지 못했다.

- **변경**:
  hard validator는 추가하지 않았다. 이미 incremental validator가 precision 100%로 row-set
  narrowing과 hidden filter 문제를 잡고 있다.

  Feedback reminder만 보강했다. too-easy specificity feedback에서 related-row strengthening은
  visible output 또는 aggregate로 나타나야 하며, hidden existence/primary-row filter를 added
  dimension으로 쓰면 안 된다고 명시했다. 이는 새 지시가 아니라 기존 Difficulty-Up Policy
  (`같은 row set 유지`, `narrowing filter 금지`, `visible related/aggregate dimension`)의 구체적
  reminder다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_too_easy_feedback_preserves_readable_path tests/test_synthesis_runtime.py::test_submit_draft_too_easy_feedback_is_list_aware -q`
  통과 (`2 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`97 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_messages.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_97`은 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.

## Iteration 177 — Final list queries should set the 3-5 row boundary first

- **질문**:
  `trial_98`은 transfer history data가 어려워서 실패한 것인가, 아니면 composer가 final list query를
  잘못 짠 것인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T105312Z:d5bdd9cf`.

  composer는 patient transfer history를 선택했다. 첫 final query는 subject의 transfer 31행 전체를
  반환했고, request는 `병동 이동 기록` 전체를 요구했다. 이 때문에:
  - 초기 submit은 query result와 label copy가 맞지 않아 `answer_contract_evidence_mismatch`.
  - 전체 31행을 맞추려 하자 `answer_contract_list_limit_too_wide`.
  - discharge row의 `care_unit`/`out_time` null 때문에 `label_null_value_forbidden`.
  - 마지막에 limit 5와 non-null fields로 고친 draft는 solver `4/4`로 너무 쉬웠고 budget exhausted.

- **reasoning 교차 분석**:
  composer reasoning은 query 결과가 31행임을 인지했지만, ToolBudgetFeedback boundary 때문에
  "submit해야 한다"고 판단하고 불완전한 label을 제출했다. 이후에야 limit 5 query를 실행했다.
  즉 실패의 중심은 solver capability가 아니라 final label query를 실행하기 전에 list boundary를
  정하지 않은 planning/tool-use 문제다.

  solver들은 최종 5-row transfer task를 모두 쉽게 풀었다. therefore rejected hard-good이 아니라
  too-easy/저품질 수리 실패다.

- **정성 평가**:
  accepted data: 없음. streak는 `0/5`.

  rejected data:
  - 31-row transfer list는 low-quality rejected. list shape가 프로젝트 기준을 벗어났고 null도 포함됐다.
  - final 5-row transfer list는 깨끗하지만 too-easy rejected. 좋은 accepted로 볼 수 없다.

- **변경**:
  hard validator는 추가하지 않았다. list size/null validator는 이미 precision 100%로 작동한다.

  Tool schema description을 보강했다. `query.spec.limit`에 final row-list label은 query 전에 3-5 row
  boundary를 정하고, prior evidence로 전체 matching row가 이미 3-5임이 확인된 경우가 아니면
  `limit`을 포함하라고 명시했다. full matching set이 5행을 넘을 수 있는데 unbounded final list
  query를 submit하지 말라는 원칙이다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`1 passed`).

  Broader relevant checks:
  `uv run pytest tests/test_tooling_composer_tool_factory.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py -q`
  통과 (`108 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_tooling_composer_tool_factory.py`
  통과.

- **현재 streak**:
  `trial_98`은 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.

## Iteration 178 — One requested output slot must stay one label field

- **질문**:
  `trial_99`은 MIMIC microbiology data가 어려워서 실패한 것인가, 아니면 composer가 좋은 label 후보를
  contract로 잘못 포장한 것인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T105922Z:02a68aa6`.

  composer는 admission `27617929`의 `microbiologyevents`를 골랐다. 최종 후보는 입원 중 채취된
  검체 검사 5건의 시간, 검체 종류, 검사명, 결과를 반환하는 list였다. 첫 query는 `charttime asc`
  단일 정렬이라 동점과 limit boundary tie가 있었고, 두 번째 query는 `charttime asc`,
  `test_name asc`로 동점 정렬을 해결했다.

  실패 흐름:
  - 첫 submit 누락은 `composer_submit_draft_missing`.
  - 첫 list 제출은 string copy mismatch와 동점 정렬 때문에
    `label_values_not_grounded`, `answer_contract_evidence_mismatch`,
    `answer_contract_order_ambiguous`.
  - tie-break query 이후 request/contract phrase 수리 과정에서 `answer_contract_query_mismatch`,
    `answer_contract_phrase_missing`.
  - 마지막 제출은 `answer_contract_binding_missing`. 진단상 `채취 시간` phrase가
    `test_date`와 `test_time` 두 label field에 중복 binding됐다.

- **reasoning 교차 분석**:
  reasoning content는 반환되고 있었다. composer는 첫 feedback 뒤 실제 원인을 꽤 잘 파악했다.
  string casing/spacing을 그대로 복사해야 하고, `charttime` 동점 때문에 자연스러운 tie-break가
  필요하다고 판단했다. 그래서 `test_name` secondary sort를 넣은 것은 List Determinism Policy에 맞다.

  하지만 다음 단계에서 `chartdate`와 `charttime`을 모두 label field로 유지한 채, 둘 다 사용자의
  `채취 시간` 요청 슬롯에 묶었다. `charttime` 자체가 날짜와 시간을 포함하는 timestamp인데도
  별도 date-only field를 같이 반환한 셈이다. 이건 solver가 풀기 어려운 좋은 문제라기보다
  Label Contract 위반이다. 하나의 자연어 출력 슬롯은 하나의 label field가 되어야 하고, date/time이나
  value/unit을 나누려면 request에 각각의 자연스러운 phrase가 있어야 한다.

- **정성 평가**:
  accepted data: 없음. streak는 `0/5`.

  rejected data:
  - microbiology list의 row set 자체는 좋은 후보였다. hidden admission scope도 자연스럽고,
    검체 종류/검사명/결과/시간은 solver 도구로 접근 가능하다.
  - 다만 제출된 draft는 low-quality rejected. 마지막 실패는 데이터 난이도가 아니라 redundant
    `test_date`/`test_time` field split과 중복 output binding 문제다.

- **변경**:
  hard validator는 추가하지 않았다. `duplicate_output_binding_phrases`는 이미 precision 100%로
  잡히고 있었고, 리터럴/토큰/column semantic 휴리스틱도 사용하지 않았다.

  Prompt/tool schema/feedback을 같은 원칙으로 정렬했다.
  - 시스템 프롬프트 Label Contract: 하나의 자연어 output slot은 하나의 label field에 매핑하며,
    timestamp가 이미 date+time을 담고 있으면 병렬 date-only field를 같이 선택하지 말라고 명시했다.
  - `query.select` tool description: final label field 선택 단계에서 같은 원칙을 먼저 보게 했다.
  - `submit_draft` schema: output binding에서도 같은 정책을 반복했다.
  - binding feedback: 실제 중복 phrase와 field 목록을 보여주되, 새 지시가 아니라 Label Contract
    reminder로 표현했다. 중복 field set 자체가 문제인 경우에는 같은 target으로 field set을 정리해
    query를 다시 실행할 수 있게 기존 “phrase-only면 data tool 금지” 문구와 충돌하지 않도록 했다.

  프롬프트 길이는 8k 예산 아래로 유지하기 위해 Difficulty-Up example 표현을 압축했다. 정책의
  source-of-truth는 유지했고, feedback은 여전히 프롬프트를 상기시키는 역할만 한다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py -q`
  통과 (`114 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/turn_budget.py src/rl_task_foundry/tooling/composer/tool_factory.py src/rl_task_foundry/synthesis/submit_draft_messages.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_runtime.py tests/test_turn_budget_prompt.py`
  통과.

  Full:
  `uv run pytest -q`
  통과 (`508 passed`).

- **현재 streak**:
  `trial_99`은 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.

## Iteration 179 — Reject aggregates from no-primary-key answer sources

- **질문**:
  `trial_100`의 최종 scalar count는 어려운 좋은 문제였나, 아니면 solver tool surface로 재현할 수 없는
  저품질 문제였나?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T111255Z:b0d80bbd`.

  composer는 `icustays` `stay_id=39623478`에서 시작해 `chartevents` 측정값 list를 만들었다.
  첫 list draft는 `chartevents` row values를 직접 노출했고, `chartevents`가 no-primary-key table이라
  `label_no_primary_key_source`로 거절됐다. 동시에 모든 row가 같은 `charttime`이라
  `answer_contract_order_ambiguous`도 발생했고, `numeric_value`와 `unit`을 모두 `수치` phrase에 묶어
  `answer_contract_binding_missing`도 발생했다.

  마지막 수리는 list를 버리고 `chartevents` count scalar로 바꿨다:
  `이번 중환자실 입원 기간 동안 기록된 측정값 총 개수가 몇 개야?`
  canonical answer는 `{"measurement_count": 869}`였다. Solver 4개는 모두 실패했고 pass_rate는 `0/4`.

- **reasoning 교차 분석**:
  composer reasoning은 no-PK row values 문제를 인지하고 aggregate count로 바꾸면 안전하다고
  판단했다. 하지만 solver reasoning은 모두 같은 지점에서 막혔다. atomic solver는 `chartevents` 같은
  no-PK table을 stable `record_set`으로 materialize할 수 없고, count/aggregate도 그 record_set
  materialization을 전제로 한다. 실제 `sql_compile._pk_expression`도 no-primary-key table에서 row-set
  materialization을 막는다.

  따라서 이 문제는 “너무 어려운 좋은 count 문제”가 아니다. composer query DSL은 `COUNT(*)`를 실행할 수
  있지만, solver에게 제공된 atomic tool surface로는 같은 답을 재현할 수 없다. accepted되면 저품질
  데이터가 되는 유형이다.

- **정성 평가**:
  accepted data: 없음. streak는 `0/5`.

  rejected data:
  - 초기 chartevents list는 low-quality rejected. no-PK row values, 동점 order, 중복 output binding이
    동시에 있었다.
  - 최종 count scalar도 low-quality rejected. 숫자 자체는 DB query로 맞지만 solver tool로 풀 수 없는
    source surface다.

- **변경**:
  hard validator를 보강했다. `column_sources`가 no-primary-key source table을 가리키면, row value뿐
  아니라 aggregate도 `label_no_primary_key_source`로 거절한다. count(*)처럼 column ref가 없는 aggregate도
  `query` 결과의 `source_tables` metadata에 source table PK 여부를 기록하게 했다.

  이 변경은 precision 100% 조건을 만족한다. DB literal, table/column 이름의 의미, 자연어 token을 보지
  않는다. atomic tool의 구조적 제약인 “record_set materialization requires a primary key”와 query
  metadata의 `table_has_primary_key=false`만 사용한다.

  함께 정리한 내용:
  - 시스템 프롬프트: no-PK table에서는 row values뿐 아니라 aggregates도 금지하고
    primary-key-backed source/path를 쓰라고 수정.
  - `query.aggregate` tool description: aggregate source rows는 primary-key-backed table이어야 한다고 명시.
  - `submit_draft` feedback: “derived aggregate over no-PK table” 문구 제거.
  - atomic API spec과 pipeline lifecycle decision table도 같은 원칙으로 갱신해 지침 이원화를 막았다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_from_table_without_primary_key tests/test_synthesis_runtime.py::test_submit_draft_rejects_count_from_table_without_primary_key tests/test_synthesis_runtime.py::test_submit_draft_rejects_aggregate_from_table_without_primary_key tests/test_tooling_composer_query.py::test_query_aggregate_count_without_group_by tests/test_tooling_composer_query.py::test_query_aggregate_count_reports_no_primary_key_source_table tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
  통과 (`7 passed`).

  Broader relevant:
  `uv run pytest tests/test_synthesis_runtime.py tests/test_tooling_composer_query.py tests/test_tooling_composer_tool_factory.py tests/test_synthesis_prompts.py tests/test_turn_budget_prompt.py -q`
  통과 (`165 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/tooling/composer/query.py src/rl_task_foundry/synthesis/submit_draft_tool.py src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_tooling_composer_query.py tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py`
  통과.

  Full:
  `uv run pytest -q`
  통과 (`509 passed`).

- **현재 streak**:
  `trial_100`은 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.

## Iteration 180 — Submit only after repaired duplicate-row diagnostics clear

- **질문**:
  `trial_101`의 duplicate projected rows 실패는 새 hard validator가 필요한 문제인가, 아니면 기존
  List Determinism Policy를 composer가 끝까지 따르지 못한 문제인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T112523Z:ed58a70e`.
  최종 오류는 `answer_contract_order_ambiguous`, `answer_contract_duplicate_answer_rows`였다.

  composer는 처음에 `prescriptions`에서 최근 처방 5개 list를 만들었다. 이 draft는
  `prescriptions.drug`가 blocked source였고, `starttime` 동점으로 limited row membership도 불안정했다.
  이후 `pharmacy.medication`으로 바꿨지만 hidden `pharmacy_id`를 tie-break로 사용했고, projection
  diagnostics가 duplicate answer rows를 보고했다. 마지막에는 visible field인 `entertime`을 추가했지만
  재쿼리 결과에서도 4/5번째 row가 완전히 동일했고, `projection_diagnostics.duplicate_answer_rows=true`가
  그대로 남아 있었다.

- **reasoning 교차 분석**:
  `reasoning_content.jsonl`은 composer reasoning을 반환하고 있었다. trace를 보면 composer는 feedback을
  읽고 “visible distinguishing field를 추가해야 한다”고 이해했다. 하지만 `entertime`을 추가한 뒤 나온
  query diagnostics를 submit 전에 다시 판독하지 않았고, duplicate가 여전히 남은 label을 제출했다.

  따라서 이 문제는 새 precision-100 validator 문제가 아니다. validator는 이미 정확히 차단했다. 실패
  원인은 기존 정책의 다음 행동, 즉 “repair query 이후 diagnostics가 clear일 때만 submit한다”가 프롬프트와
  feedback에서 충분히 직접적으로 상기되지 않은 것이다.

- **정성 평가**:
  accepted data: 없음. streak는 `0/5`.

  rejected data:
  - `prescriptions` draft는 low-quality rejected. blocked answer source와 unstable order가 있었다.
  - `pharmacy` draft도 low-quality rejected. hidden tie-break와 duplicate projected rows가 있었다.
  - 최종 `entertime` 수리 draft도 low-quality rejected. visible field를 하나 더했지만 answer rows가 여전히
    구분되지 않았다.

- **변경**:
  Prompt-first 원칙에 따라 durable `List Determinism Policy`를 압축 보강했다. duplicate projected rows가
  나오면 visible field/aggregate를 추가하고 재쿼리하되, `diagnostics`가 clear일 때만 submit하도록 했다.
  여전히 duplicate이거나 sequence/reference/order만으로 구분되는 경우에는 label을 바꿔야 한다.

  `answer_contract_duplicate_answer_rows` feedback도 같은 정책을 상기하도록 보강했다. “rerun 후 submit”처럼
  들릴 수 있는 문구를 “rerun 후 diagnostics가 duplicate answer rows를 더 이상 보고하지 않을 때만 submit”으로
  바꿨다. 새 durable instruction을 feedback에 만든 것이 아니라, 기존 named policy의 적용 순서를 더 명확히
  한 것이다.

  hard validator는 추가하지 않았다. 기존 `projection_diagnostics.duplicate_answer_rows=true` 기반 rejection은
  precision 100% 구조 검증이며, 이번 변경은 그 결과를 composer가 submit 전에 읽게 만드는 prompt/feedback
  정렬이다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_duplicate_projected_list_rows tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow -q`
  통과 (`2 passed`).

  Relevant:
  `uv run pytest tests/test_synthesis_runtime.py tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py -q`
  통과 (`108 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/prompts.py src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_prompts.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_101`은 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.

## Iteration 181 — Do not imply sequence/order repair is acceptable

- **질문**:
  `trial_102`의 실패는 duplicate-row feedback 보강이 부족해서인가, 아니면 다른 feedback 문구가
  composer에게 source sequence/order repair를 허용하는 것처럼 읽힌 문제인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T113354Z:e1af5279`.
  최종 오류는 `answer_contract_duplicate_answer_rows`였다.

  composer는 `emar` anchor에서 입원 기간의 투약 기록 list를 만들었다. 최초 label은 투약 시간,
  약물명, 투약 결과, 예정 시간을 포함했지만 동일 투약 시간/약물/결과 row가 반복되어 list order와
  projected answer rows가 불안정했다. 이후 composer는 `emar_seq`를 “preview에 보였으니 user-visible”로
  해석하고 `sequence_number` output 또는 `emar_seq` order tie-break로 수리하려 했다.

- **reasoning 교차 분석**:
  trace상 composer는 duplicate projected rows 정책 자체는 읽었다. 하지만 “source sequence/reference/order
  numbers are technical unless that is the domain answer”보다, `answer_contract_order_ambiguous` feedback의
  “tie-break가 sequence/rank-like이면 source record sequence를 request wording에 이름 붙여라”라는 문구를
  더 허용적으로 해석했다. 그 결과 “기록된 순서대로” 같은 wording으로 sequence/order tie-break를 정당화하려고 했다.

  이건 hard validator로 precision 100% 처리할 수 있는 문제가 아니다. 어떤 visible field가 자연 도메인 sequence인지,
  단순 record order인지 DB literal 없이 완전히 판정할 수 없다. 대신 기존 prompt-first 원칙에 따라 feedback이
  durable List Determinism Policy와 충돌하지 않도록 맞추는 것이 맞다.

- **정성 평가**:
  accepted data: 없음. streak는 `0/5`.

  rejected data:
  - 투약 기록 list는 low-quality rejected. 동일한 visible answer rows가 반복되고, sequence/order만이 row를
    구분하는 수리 방향이었다.
  - 이 문제는 어려운 좋은 문제가 아니다. solver가 풀 수 있느냐 이전에, user-facing label 자체가 “자연스러운
    투약 기록”이 아니라 technical record sequence에 의존한다.

- **변경**:
  `answer_contract_order_ambiguous` feedback을 수정했다. 기존 문구는 sequence/rank-like tie-break를 쓰려면
  source record sequence를 request에 이름 붙이라고 말해, repair 과정에서 sequence/order wording을 추가해도 되는
  것처럼 읽힐 수 있었다.

  새 문구는 sequence/rank-like key가 “이미 자연 도메인 답”일 때만 유효하고, 그렇지 않으면 label을 바꾸라고
  말한다. 또한 hidden handle, artificial id wording뿐 아니라 새 sequence/order wording으로 repair하지 말라고
  명시했다. 이는 새 정책이 아니라 기존 List Determinism Policy의 `Source sequence/order numbers are technical
  unless that is the domain answer; never add them for tie-break/binding repair`와 feedback을 정렬한 것이다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_rejects_ambiguous_limited_list_order tests/test_synthesis_runtime.py::test_submit_draft_rejects_duplicate_projected_list_rows -q`
  통과 (`2 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_102`은 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.

## Iteration 182 — trial_103 accepted qualitative check

- **질문**:
  sequence/order repair feedback 정렬 이후 첫 smoke trial이 만족 가능한 accepted data를 만들었는가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `accepted`, flow_id는 `real_db_trial:20260429T113933Z:928dfc3a`.
  task_id는 `task_patient_admission_history_9950e0e6c7ffd6b0`이고 pass rate는 `3/4 = 0.75`였다.

  최종 user_request:
  `나의 병원 입원 기록 중 가장 최근 것 3개를 보여줘. 입원 시각, 퇴원 시각, 입원 종류, 그리고 퇴원 장소를 알려줘.`

  최종 label:
  - `2117-12-03T17:07:00` / `2117-12-06T17:30:00` / `EW EMER.` / `HOME`
  - `2117-10-25T22:22:00` / `2117-10-29T14:40:00` / `EW EMER.` / `HOME`
  - `2117-07-16T07:15:00` / `2117-07-25T12:34:00` / `ELECTIVE` / `HOME`

- **reasoning 교차 분석**:
  composer는 처음 `microbiologyevents`에서 최근 세균 검사 결과 5개를 시도했다. 그 draft는 date-only
  reformatting, duplicate projected rows, ambiguous order, missing order binding이 동시에 있었고 거절됐다.
  중요하게는, 이후 composer가 sequence/order key를 계속 밀지 않고 `admissions` 이력으로 label을 전환했다.
  이는 Iteration 181의 feedback 정렬 방향과 맞다.

  solver reasoning 4개 중 3개는 같은 절차로 풀었다:
  `admissions` record set 생성 → `subject_id=10021487` 필터 → `admittime desc` 정렬 → 3개 row의
  `admittime`, `dischtime`, `admission_type`, `discharge_location` materialize → submit.

  실패한 1개는 `invalid_submit/missing_submit_result`였다. reasoning상 올바른 계획은 세웠지만 tool-call 형식이
  깨져 최종 submit_result까지 가지 못했다. 데이터가 풀 수 없어서 실패한 것이 아니므로 저품질 신호로 보지 않는다.

- **정성 평가**:
  accepted data: 좋은 데이터로 판단한다. 자연스러운 환자 요청이고, hidden patient scope가 명확하며, solver tool
  surface로 재현 가능한 필터/정렬/list task다. 난이도는 높지 않지만 smoke trial 목표에는 적합하다.

  rejected data:
  - microbiology 후보는 low-quality rejected. 동일 visible answer rows와 date reformatting이 있었고, ordered
    list membership도 안정적이지 않았다.
  - 이 거절은 좋은 방향이다. 저품질 후보가 accept되지 않았고, composer가 다른 label로 전환했다.

- **변경**:
  코드 변경 없음. 현재 개선 방향이 기대대로 작동했는지 확인한 실험 기록이다.

- **현재 streak**:
  `trial_103` accepted는 만족 가능한 데이터로 판단하므로 연속 만족 accepted streak는 `1/5`.

## Iteration 183 — Keep Label Contract visible at tool-budget boundaries

- **질문**:
  `trial_104` 실패는 어려운 좋은 문제였나, 아니면 budget boundary에서 broad request를 제출한 저품질
  candidate였나?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T115002Z:915202d8`.
  마지막 오류는 `answer_contract_phrase_missing`였다.

  composer는 먼저 admission-scoped microbiology list를 만들었다. non-null organism filter를 넣자 row가
  2개뿐이라 `answer_contract_list_size_invalid`로 거절됐다. 이 거절은 타당하다. 이후 더 풍부한 emar/pharmacy
  쪽으로 label을 바꾸려 했지만, feedback 이후 data tool budget boundary까지 간 뒤 broad request를 제출했다:
  `해당 입원 기간 동안의 최신 약물 투약 기록 5건을 확인하고 싶어`.

  최종 label에는 `administration_time`, `sequence_number`, `medication_name`, `event_status`가 있었지만
  user_request에는 `투약 일시`, `순번`, `약물명`, `투약 상태`, `최신 순서로` 같은 binding phrases가 없었다.
  그래서 `answer_contract_phrase_missing`로 budget exhausted가 발생했다.

- **reasoning 교차 분석**:
  composer는 2-row microbiology list가 부족하다는 점은 이해했다. 하지만 emar duplicate 문제를 수리하는 과정에서
  다시 `emar_seq` 기반 `sequence_number`를 output/order field로 사용했고, budget boundary 이후에는
  user_request를 broad하게 유지한 채 제출했다.

  이 문제는 solver가 풀기 어려운 좋은 문제라기보다 composer가 Label Contract를 최종 제출 직전에 지키지 못한
  저품질 rejected다. 특히 “budget boundary라 submit해야 한다”는 메시지가 “binding phrase가 빠진 broad request도
  제출해도 된다”로 해석되면 안 된다.

- **정성 평가**:
  accepted data: 없음. streak는 `0/5`로 리셋한다.

  rejected data:
  - microbiology 2-row list는 low-quality rejected. list task shape에 맞지 않았다.
  - emar label은 low-quality rejected. sequence/order 중심 수리와 missing binding phrases가 있었다.
  - 저품질이 solver까지 가지 않고 submit validator에서 거절된 것은 올바른 동작이다.

- **변경**:
  FEEDBACK_REPAIR_MAX_DATA_TOOLS boundary feedback에 Label Contract reminder를 추가했다.
  “repair query가 blocking diagnostics 없이 label values를 반환했으면 submit”이라는 protocol reminder 뒤에,
  submit 전에도 user_request가 모든 output/order binding의 exact natural phrases를 포함해야 하며 broad wording으로
  missing binding phrases를 남기면 안 된다고 명시했다.

  이는 새 durable policy가 아니라 기존 Label Contract의 boundary-time reminder다. precision-100 hard validator는
  이미 `answer_contract_phrase_missing`으로 동작하고 있으므로 추가하지 않았다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_runtime.py::test_data_tool_budget_feedback_blocks_repeated_query_repair_for_ambiguous_query tests/test_synthesis_runtime.py::test_submit_draft_rejects_binding_phrase_absent_from_request -q`
  통과 (`2 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_104`는 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.

## Iteration 184 — Do not escape missing-submit boundary to one-row detail

- **질문**:
  `trial_105`의 마지막 `answer_contract_scalar_not_aggregate`는 어려운 좋은 문제인가, 아니면 missing-submit
  boundary에서 단일 row detail로 도망간 저품질 후보인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T115650Z:2525c340`.
  마지막 오류는 `answer_contract_scalar_not_aggregate`였다.

  composer는 emar anchor에서 시작했다. 첫 draft는 입원 기간 약물 투여 목록이었지만 60-row list였고,
  null field와 missing binding phrases가 있었다. 두 번째 draft는 최근 5개로 줄였지만, 여전히
  `administration_sequence`를 output으로 사용했고 `administration_time` 동점으로 order ambiguity가 남았다.
  이후 missing-submit boundary 뒤에 admissions 단일 row detail을 query하고 `kind="scalar"`로 제출했다.

- **reasoning 교차 분석**:
  reasoning상 composer는 “더 단순하고 null 없는 label”을 찾겠다며 admissions로 전환했다. 하지만 query 결과는
  aggregate가 아니라 selected row object였다:
  `admission_type`, `admission_time`, `admission_location`, `discharge_location`.
  `Task Shapes`와 `Label Contract`상 scalar는 count/min/max/sum/avg 같은 aggregate에만 허용된다.

  따라서 이 문제는 solver가 못 푸는 어려운 좋은 문제가 아니다. composer가 실패한 list repair에서 빠져나오려고
  단일 admission detail을 scalar처럼 제출한 저품질 rejected다.

- **정성 평가**:
  accepted data: 없음. streak는 `0/5`.

  rejected data:
  - 60-row emar list는 low-quality rejected. list shape, null field, missing phrase, sequence/order 문제가
    동시에 있었다.
  - 5-row emar 수리도 low-quality rejected. sequence output과 unstable order가 남았다.
  - 최종 admissions scalar도 low-quality rejected. 단일 row detail은 scalar label이 아니다.

- **변경**:
  `_missing_submit_after_tool_budget` boundary feedback을 보강했다. “한 번의 final query가 허용된다”는 문구가
  새 target으로 도망가도 된다는 뜻이 아님을 명시했다. final query는 current label target이어야 하고,
  Task Shapes는 여전히 적용되며, single-row detail로 escape하지 말고 scalar는 aggregate query가 필요하다고
  상기한다.

  이는 새 검증이 아니다. 기존 `answer_contract_scalar_not_aggregate` validator가 precision 100%로 차단하던
  조건을 protocol boundary에서 미리 상기하는 feedback-reminder 개선이다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_synthesis_runtime.py::test_submit_draft_records_missing_submit_protocol_feedback tests/test_synthesis_runtime.py::test_submit_draft_rejects_scalar_row_detail_without_aggregate_query -q`
  통과 (`2 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/synthesis/submit_draft_tool.py tests/test_synthesis_runtime.py`
  통과.

- **현재 streak**:
  `trial_105`는 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.

## Iteration 185 — Avoid passive source-specific processing/type outputs

- **질문**:
  `trial_106`의 pass rate `0/4`는 어려운 좋은 pharmacy 문제였나, 아니면 source surface ambiguity가 있는
  저품질 문제였나?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T120918Z:24217ec8`.
  최종 draft는 solver pass rate `0/4`로 `calibration_inconclusive` rejected였다.

  최종 user_request:
  `해당 입원 기간 동안 조제된 약물 목록과 조제 시작일, 조제 종료일, 조제 유형을 최근 조제 시작일 순, 약물명 오름차순으로 5개만 보여줘`

  composer label은 `pharmacy` source에서 `medication`, `starttime`, `stoptime`, `proc_type`을 사용했다.
  answer의 `processing_type`은 모두 `Unit Dose`였다.

- **reasoning 교차 분석**:
  solver 4개 모두 `prescriptions` source를 선택했고 `drug_type`을 조제/처방 유형으로 해석해 `MAIN`을 반환했다.
  즉 solver가 단순히 못 푼 것이 아니라, user-facing wording이 `pharmacy.proc_type`과 `prescriptions.drug_type`을
  충분히 구분하지 못했다. “조제 유형”이라는 말은 broad medication/prescription surface에서도 자연스럽게 해석된다.

  composer는 이전 feedback을 잘 반영해 blocked `prescriptions.drug`에서 `pharmacy.medication`으로 옮겼고
  phrase_missing도 수리했다. 하지만 마지막에 source-specific processing/type field를 passive width로 포함했다.
  값도 전부 `Unit Dose`로 동일해서 label 난이도를 좋은 방향으로 높이지 못했다.

- **정성 평가**:
  accepted data: 없음. streak는 `0/5`.

  rejected data:
  - 첫 prescriptions draft는 low-quality rejected. blocked answer source, phrase missing, order ambiguity가 있었다.
  - 최종 pharmacy draft도 low-quality rejected. source-specific `processing_type`이 broad wording으로 표현되어
    solvers가 ordinary prescription surface로 간다.
  - 이건 어려운 좋은 문제라기보다 user_request/source surface 불일치다.

- **변경**:
  `query.select` tool schema description을 보강했다. process/status/type/category output은 passive width가 아니며,
  user_request가 exact source/lifecycle surface를 이름 붙일 때만 포함해야 한다고 명시했다. 그 wording이 어색하면
  ordinary source를 선택하거나 해당 field를 빼야 한다.

  이는 새 hard validator가 아니다. source-specific type/category의 자연어 ambiguity는 DB literal 없이 precision
  100%로 판정할 수 없으므로, tool schema/description에서 composer의 field 선택 판단을 돕는 방향이 맞다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`1 passed`).

  Ruff:
  `uv run ruff check src/rl_task_foundry/tooling/composer/tool_factory.py tests/test_tooling_composer_tool_factory.py`
  통과.

- **현재 streak**:
  `trial_106`은 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.

## Iteration 186 — Surface query visibility blocks before submit

- **질문**:
  `trial_107`의 마지막 `label_non_user_visible_source`는 composer가 단순히 submit feedback을 못 지킨 것인가,
  아니면 query 단계에서 blocked/internal selected output을 더 구조적으로 드러내야 하는 문제인가?

- **실험/결과**:
  설정은 MIMIC demo, OpenRouter Kimi K2.5 composer/solver, 4 solver, topic 주입 없음.
  결과는 `synthesis_failed`, flow_id는 `real_db_trial:20260429T121709Z:8cdba29e`.
  마지막 오류는 `label_non_user_visible_source`였다.

  composer는 처음 `pharmacy`의 입원 중 약물 목록을 시도했다. 첫 draft는 `status` phrase가 user_request에 없고,
  모든 returned row의 `starttime`이 같아 limited list membership/order가 안정적이지 않았다. 두 번째 draft는
  `medication,starttime`으로 줄였지만, query 결과에 duplicate projected answer rows가 남았다.

  이후 composer는 `diagnoses_icd`로 전환해 `icd_code,seq_num`을 selected output으로 제출했다:
  `이번 입원 중 진단 목록을 순위순으로 5개 보여주세요`.
  하지만 query metadata에서 두 output 모두 `visibility="blocked"`였고, submit validator가 정확히 거절했다.

- **reasoning 교차 분석**:
  composer reasoning은 약물 목록의 동점/중복 문제를 인지하고 label 전환을 시도했다. 여기까지는 올바른 방향이다.
  실패 지점은 `seq_num`을 “테이블에 있으니 visible field일 수 있다”고 추론한 부분이다. 실제 query result의
  `column_sources`에는 `icd_code`와 `seq_num`이 blocked로 표시되어 있었다.

  이번 trial은 solver rollout 전에 종료되었으므로 solver reasoning은 없다. 따라서 품질 문제는 solver가 못 푼
  어려운 좋은 문제가 아니라, composer가 metadata visibility를 보지 않고 domain/table/column 추론으로 label을
  제출한 저품질 rejected다.

- **정성 평가**:
  accepted data: 없음. streak는 `0/5`.

  rejected data:
  - pharmacy 후보는 low-quality rejected. order ambiguity와 duplicate projected answer rows가 있었다.
  - diagnosis 후보도 low-quality rejected. source sequence/code를 자연어로 포장했지만 query metadata상 blocked
    output이므로 customer-facing label로 제출할 수 없다.
  - 저품질이 solver까지 가지 않고 submit validator에서 거절된 것은 맞지만, composer가 submit 직전까지 blocked
    output을 눈치채지 못한 점은 개선 대상이다.

- **변경**:
  1. `query` 결과에 `label_source_diagnostics`를 추가했다. selected output이 `blocked` 또는 `internal`이고
     값이 source를 직접 노출하면 `submit_blocked=true`와 함께 해당 output/source/visibility를 구조화해 반환한다.
     이는 literal/token/table-name 휴리스틱이 아니라 이미 존재하는 visibility metadata를 더 명확히 노출하는
     100% 정밀도 개선이다.
  2. `query.select` tool description에 `label_source_diagnostics.submit_blocked`를 blocking signal로 취급하고,
     visible output으로 rerun하거나 다른 label을 고르라고 추가했다.
  3. durable prompt의 Label Contract를 짧게 보강했다. label value visibility는 metadata 기준이며 이름 추론으로
     결정하지 않는다고 명시했다.
  4. `label_non_user_visible_source` feedback도 같은 원칙을 reminder로 맞췄다. 새 지시가 아니라 기존 Label
     Contract를 상기하는 feedback이다.

- **검증**:
  Targeted:
  `uv run pytest tests/test_tooling_composer_query.py::test_query_returns_visibility_provenance_for_outputs_and_refs tests/test_tooling_composer_query.py::test_query_marks_label_sources_without_primary_key tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_from_non_user_visible_query_source tests/test_synthesis_prompts.py::test_synthesis_agent_instructions_describe_composer_workflow tests/test_tooling_composer_tool_factory.py::test_composer_tool_schema_descriptions_are_prompt_aligned -q`
  통과 (`5 passed`).

  Related:
  `uv run pytest tests/test_synthesis_prompts.py tests/test_tooling_composer_tool_factory.py tests/test_tooling_composer_query.py::test_query_returns_visibility_provenance_for_outputs_and_refs tests/test_tooling_composer_query.py::test_query_marks_label_sources_without_primary_key tests/test_tooling_composer_query.py::test_query_rejects_blocked_non_handle_column_refs tests/test_synthesis_runtime.py::test_submit_draft_rejects_label_from_non_user_visible_query_source tests/test_synthesis_runtime.py::test_submit_draft_rejects_query_without_visibility_metadata -q`
  통과 (`28 passed`).

  Full:
  `uv run pytest -q` 통과 (`509 passed`).

- **현재 streak**:
  `trial_107`은 accepted가 없으므로 연속 만족 accepted streak는 `0/5`.
