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

## Cross-Iteration Summary (iter 1-7, extended through iter 13)

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

- **Attempts observed**: N / max_generation_attempts
- **Submissions**: M / N (how many attempts actually reached submit_draft)
- **Pass-rate trajectory**: [0.0, 0.67, 0.33, ...] per submitted attempt
- **Terminal status**: accepted | reject_too_easy (band still above) | reject_too_hard (band still below) | MaxTurnsExceeded | other
- **Ladder climb observed**: which rungs were visible in each submission's label structure
- **Regression signals**: agent over-exploring after rejection, repeating same rung, weakening label, etc.
