# Independent DQS Evaluator Subagent Protocol

이 문서는 Codex에서 독립 평가자 subagent를 만들어 builder self-bias를 줄이는 방법을 정의한다. 목표는 완전한 객관성이 아니라, 구현자 자기정당화 편향을 낮추고 평가 해킹 가능성을 드러내는 것이다.

실험용 builder architecture에서는 SDK agent 종류, 수, context 공유/분리 방식을 자유롭게 바꿀 수 있다. 이 문서는 promotion 평가자에 대한 기본 독립성 프로토콜만 고정한다.

## When to use

다음 상황에서는 독립 evaluator report가 필요하다.

- 실험 노드를 `candidate`, `frontier`, `promoted`로 올리려는 경우
- phase target 달성 여부를 판단하는 경우
- solver band, rubric, architecture처럼 evaluation 결과에 큰 영향을 주는 변경을 검토하는 경우
- accepted data 품질에 이견이 있는 경우

Codex subagent는 사용자가 subagent 평가를 명시적으로 승인했거나 요청했을 때 만든다. 승인 없이 subagent를 만들 수 없으면 멈추고 사용자에게 평가 승인을 요청한다.

## Independence rules

- evaluator는 파일을 수정하지 않는다.
- evaluator는 promotion 결정을 내리지 않고 권고만 한다.
- evaluator는 builder의 의도보다 artifact, generated drafts, solver traces, DB evidence, DQS-v1 rubric을 우선한다.
- 가능하면 builder hypothesis와 implementation summary는 primary rubric labeling 이후에 본다.
- evaluator는 `accepted_low_quality` 의심을 보수적으로 보고한다. 이 판정은 promotion hard gate다.
- builder는 반론을 쓸 수 있지만 단독으로 evaluator report를 override할 수 없다.

## Input packet

Evaluator에게 주는 입력은 최소한 다음을 포함한다.

- `docs/experiments/first_principles.md`
- `docs/experiments/rubric_dqs_v1.md`
- `docs/experiments/tree.md`
- experiment node id, parent id, branch, commit
- evaluation policy: DB, solver model/count, band, batch size, repetitions
- artifact paths
- generated draft summaries
- accepted/rejected outcomes
- solver logs/traces/reasoning summaries if available
- query evidence or canonical answer evidence summaries
- baseline comparison results if available

Builder implementation details와 hypothesis는 가능한 한 뒤에 둔다.

## Codex subagent spawn prompt

Codex에서 evaluator subagent를 만들 때는 전체 builder 대화 맥락을 fork하지 않는 것이 기본이다. 독립성을 위해 필요한 파일 경로와 artifact 경로만 제공한다.

Suggested spawn settings:

```text
agent_type: default
fork_context: false
task: DQS-v1 independent evaluation
write_scope: none
```

`fork_context: false`가 기본이다. 사용자가 특정 실험에서 shared-context evaluator를 승인하면 바꿀 수 있지만, report에 그 사실과 이유를 기록해야 한다. Builder와 evaluator가 context를 공유한 경우 blind-ish 평가가 아니므로 confidence를 낮추거나 별도 재평가를 요청한다.

Suggested prompt:

```text
You are an independent DQS-v1 evaluator for rl-task-foundry.

Do not edit files. Do not promote the experiment. Your job is to evaluate the
provided experiment artifacts under:
- docs/experiments/first_principles.md
- docs/experiments/rubric_dqs_v1.md
- docs/experiments/tree.md

Focus on real dataset quality, not evaluator acceptance. Treat low-quality
accepted data as a hard failure. Look for DB-specific literal leakage, token
heuristics, non-precision-100 rejectors, source-of-truth duplication, and mode
collapse.

Evaluate every accepted and rejected draft. Classify each draft as one of:
ACCEPTED_GOOD, ACCEPTED_BORDERLINE, ACCEPTED_LOW_QUALITY,
REJECTED_GOOD_TOO_EASY, REJECTED_GOOD_TOO_HARD, REJECTED_LOW_QUALITY,
REJECTED_INFRA, REJECTED_UNKNOWN.

Return a structured report with counts, diversity score, hard gate results,
per-draft rationale, adversarial review, and recommendation. If evidence is
insufficient, say so explicitly instead of guessing.
```

## Report schema

Evaluator report는 `docs/experiments/evaluations/<node-id>.md`에 저장한다.

Required sections:

```yaml
rubric_version: DQS-v1
experiment_node: N0000
evaluator: independent-dqs-v1
overall_verdict: promote_candidate | reject | inconclusive

counts:
  accepted_good: 0
  accepted_borderline: 0
  accepted_low_quality: 0
  rejected_good_too_easy: 0
  rejected_good_too_hard: 0
  rejected_low_quality: 0
  rejected_infra: 0
  rejected_unknown: 0

diversity:
  score: 0
  mode_collapse: false
  rationale: ""

hard_gates:
  accepted_low_quality_zero: true
  no_literal_hack: true
  precision_100_rejectors_only: true
  no_source_of_truth_duplication: true
  db_readiness_not_forced: true
  no_mode_collapse: true
  evaluation_policy_recorded: true
  llm_judge_components_versioned: true
  llm_semantic_not_structural_reject: true

production_viability:
  trial_timeout_s: 300
  accepted_sample_count: 3
  productive_budget_seconds: 900
  three_accepts_reached_within_budget: true
  productive_seconds_per_accepted: null
  productive_seconds_per_accepted_within_budget: true
  excludes_one_time_db_startup_and_warmup: true
  excludes_provider_issue_trials_from_average: true
  elapsed_seconds_recorded_for_all_trials: true

per_draft:
  - draft_id: ""
    label: REJECTED_UNKNOWN
    rationale: ""
    risks: []

adversarial_review:
  possible_eval_hacking: []
  concerns: []

recommendation:
  decision: promote_candidate | reject | inconclusive
  reason: ""
```

## Builder response

Builder가 evaluator 판단에 이견이 있으면 `docs/experiments/evaluations/<node-id>_builder_response.md`에 짧게 남긴다.

Builder response must include:

- disputed evaluator claim
- evidence path
- whether the dispute affects a hard gate
- requested action: accept evaluator finding, request re-evaluation, ask user to adjudicate

Accepted low-quality 관련 이견은 사용자 판단 또는 재평가 없이는 promotion할 수 없다.
