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

## Cross-Iteration Summary (iter 1-7, extended through iter 20)

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
| 20 | 1 | — | **3연속 accept + voice fix 검증.** Qualitative 평가에서 발견한 iter18/19의 staff-voice / mixed-voice 문제를 prompt의 rewrite instruction 강화로 해결(1인칭 ask 명시 + 금지 구절 리스트). 결과: city(Toulouse) 앵커에서 "저는 툴루즈에 거주하는 고객입니다. 제 대여 기록..." 1인칭 customer-ask voice 완벽 준수. 부수 효과로 **첫 clean 2-hop join task**(city→address→customer→rental) 성공 — iter15 multi-hop overshoot 실패가 누적 수정들로 해소. 단 Toulouse customer 수가 2+면 semantic ambiguity 가능성 별도 검증 필요 |

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
