# Experiment Graph

이 프로젝트의 실험은 greedy merge가 아니라 탐색 그래프로 운영한다. Git commit graph가 실제 실험 graph이고, `docs/experiments/registry.yaml`이 의미/평가 metadata의 source of truth다.

## Branch model

- `main`: 안정 snapshot. 실험을 직접 누적하지 않는다.
- baseline node: 현재 연구 기준선. registry의 `current_baseline`으로 지정한다.
- `exp/<node-id>-<slug>`: 실험 노드.
- worktree: 활성 노드 작업공간. worktree 자체는 기록 단위가 아니다.

## Node requirements

각 노드는 registry에 다음을 기록해야 한다.

- id
- branch
- commit
- parent
- parent_commit
- hypothesis
- architecture_summary
- change_summary
- principles_touched
- agent_topology
- evaluation_policy
- quality_eval
- branching_notes
- decision/status

노드는 직접 부모만 기록한다. 전체 lineage는 도구나 사람이 parent chain을 따라 계산한다.

`commit`은 해당 노드의 artifact snapshot이다. 그 노드를 registry에 기록하는 metadata commit은 artifact commit의 descendant일 수 있다. Active experiment worktree는 `AGENTS.md`와 `docs/experiments/*` governance docs를 포함한 snapshot에서 만들어야 하며, governance docs 이전의 raw code commit에서 직접 분기하지 않는다.

## Status values

- `current`: registry의 현재 기준선
- `running`: 실험 중
- `candidate`: 유망하지만 baseline 승격 전
- `frontier`: 현재 가장 유망한 탐색 경로
- `promoted`: baseline으로 승격됨
- `abandoned`: 더 진행하지 않음
- `local_minimum`: 이 경로가 막혔지만 기록 가치 있음
- `regressed`: baseline 대비 악화
- `infra_fail`: 인프라 실패로 평가 불가

## Promotion protocol

좋은 실험을 즉시 `main`에 merge하지 않는다.

1. node를 `candidate` 또는 `frontier`로 기록한다.
2. 동일 evaluation policy로 반복 검증한다.
3. `docs/experiments/evaluator_subagent.md` 프로토콜로 독립 DQS evaluator report를 만든다.
4. DQS-v1 hard gates와 적대적 리뷰를 통과한다.
5. 사용자 확인 후 baseline으로 promote한다.
6. 충분히 안정된 baseline snapshot만 `main`에 반영한다.

## Branching protocol

후속 실험은 부모 노드 commit에서 새 branch/worktree로 만든다. 한 경로가 local minimum에 빠지면 임의의 이전 노드로 돌아가 새 분기를 만들 수 있다.

권장 branch 이름:

```text
exp/N0001-label-first-thin-slice
exp/A0001-verifier-first-generator
exp/R0001-band-calibration-035
```

권장 node prefix:

- `P`: prompt/feedback/validator patch
- `T`: tool surface/tool semantics
- `A`: architecture variant
- `R`: rubric/evaluation policy
- `B`: baseline

## Agent topology

SDK agent 종류/수/context 공유 방식은 실험 설계자가 자유롭게 바꿀 수 있다. Composer/solver를 없애거나 planner/verifier/naturalizer/judge로 나누는 것도 허용된다. 단, 각 노드는 다음을 기록해야 한다.

- builder agent 종류와 수
- agent 간 shared context 여부
- isolated context 또는 blind-ish 평가 context 여부
- LLM judge/validator 사용 여부
- 각 LLM judge/validator의 model, prompt/version, context scope, decision authority
- 어떤 판단이 semantic quality signal이고, 어떤 판단이 precision-100 structural reject인지

LLM judge/validator는 사용할 수 있지만, precision-100 structural rejector인 것처럼 기록하면 안 된다.

## Independent evaluator reports

Promotion 후보 노드는 builder self-check만으로 승격할 수 없다. 독립 evaluator report를 `docs/experiments/evaluations/<node-id>.md`에 저장하고, 필요하면 builder response를 `docs/experiments/evaluations/<node-id>_builder_response.md`에 저장한다.

Evaluator는 code diff 자체보다 실험 artifact와 DQS-v1 rubric을 먼저 본다. 구현자의 hypothesis는 가능하면 primary labeling 이후에 비교한다.

## Phase stop rule

단계별 목표를 metric 기준으로 달성하면 Codex는 자동 개선을 멈추고 사용자에게 다음 결정을 요청해야 한다.

선택지:

1. 현재 baseline freeze/promote
2. 다음 phase로 진행
3. 다른 노드에서 새 분기
4. metric/rubric 수정
