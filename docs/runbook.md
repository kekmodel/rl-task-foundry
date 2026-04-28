# Rewrite Runbook

## Production Training Freeze

status: `ACTIVE`

owner:

- release owner / rewrite lead가 freeze 상태 유지와 해제 판단을 맡는다

effective_from:

- `2026-04-11`

scope:

- synthesis-agent hybrid rewrite가 proof environment와 quality gate를 통과하기 전까지 유지

## Prohibited Actions

- 기존 path-centric baseline artifact를 production RL training source로 사용하는 것
- rewrite 이전 generator에서 나온 dataset을 accepted production corpus로 승격하는 것
- proof task 이전에 production task registry를 운영 시작하는 것
- review pack 같은 별도 qualitative snapshot surface를 다시 도입하는 것

## Allowed Actions

- spec / plan / core contract 작업
- synthetic proof task 구축
- registration policy / runtime isolation 구현
- artifact 기반 정성 평가
- infra regression test 유지

## Exit Criteria

freeze는 아래가 모두 충족될 때만 해제 후보가 된다.

- proof task vertical slice 완료
- hybrid verifier A/B/C/D mandatory gate 구현
- self-scoped task consistency와 solver pass-rate quality filter 구현
- artifact 기반 정성 평가에서 반복적으로 품질 기준을 충족

## Operator Checklist

rewrite 진행 중에는 아래를 확인한다.

- baseline regression이 깨지지 않았는가
- 새 artifact가 training source로 오인되어 누적되지 않았는가
- qualitative review가 registry bundle과 debug traces 기준으로 수행되고 있는가
- production acceptance / registry commit은 milestone 달성 전까지 막혀 있는가

## Mandatory Experiment Quality Audit

프로젝트 코드가 완성될 때까지, 모든 synthesis/prompt/tool/feedback 개선
실험 분석 원칙으로 아래 정성 비교를 반드시 수행한다. 이 정성 비교는
정량 결과와 별도로 기록한다.

- Accepted data audit:
  - accepted count와 pass rate만으로 품질을 판단하지 않는다.
  - 각 accepted task의 `task.yaml`/`task.json`, `instance.json`,
    `canonical_answer.json`, 마지막 `phase_monitors.jsonl` submission을
    확인한다.
  - user request, topic, entity scope, canonical query path, label fields,
    ordering, tie-break, hidden filter가 서로 맞는지 판정한다.
  - accepted task를 `clean`, `borderline`, `low-quality accepted`, `topic
    drift` 중 하나로 분류한다.

- Rejected/failed data audit:
  - `too_hard`, low pass rate, `reject_too_hard`, `MaxTurnsExceeded`를 그대로
    저품질로 단정하지 않는다.
  - solver tools로 풀 수 있는데 solver가 실패한 경우는 hard-good으로
    분류한다.
  - request/label/query가 불일치하거나, source surface가 흔들리거나,
    hidden row-set/order/filter에 의존하거나, topic drift/difficulty jump가
    발생한 경우는 low-quality로 분류한다.
  - failed task를 `hard-good`, `low-quality`, `infra/provider failure`,
    `inconclusive` 중 하나로 분류한다.
  - low-quality draft가 solver pass-rate / quality gate에서 reject되면
    필터가 정상 작동한 것으로 본다. 이것은 개선 실패의 핵심 문제가
    아니다.
  - 핵심 문제는 low-quality draft가 accepted/registry commit까지 통과하는
    경우다. 이 경우는 accept rate가 좋아 보여도 품질 회귀로 간주한다.

- Batch comparison:
  - 새 실험은 이전 relevant baseline과 raw accept rate뿐 아니라 clean
    accepted count, borderline accepted count, low-quality accepted count,
    hard-good rejected count, low-quality rejected count를 비교한다.
  - 결론에는 "숫자로는 개선처럼 보이나 품질상 개선이 아닌 경우"를 반드시
    별도로 적는다.
  - low-quality rejected count는 필터가 막은 잡음으로 해석하고, low-quality
    accepted count를 가장 높은 위험 신호로 본다.
  - 이 정성 비교를 생략한 실험은 개선 근거로 사용하지 않는다.

## Artifact Inspection

정성 평가는 review pack이 아니라 아래 artifact를 본다.

- synthesis loop / rejection 흐름:
  - [`artifacts/.../debug/phase_monitors.jsonl`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
- synthesis agent 대화 / tool 사용:
  - [`artifacts/.../debug/traces/synthesis/transcripts/*.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
  - [`artifacts/.../debug/traces/synthesis/tool_traces/*.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
- solver run 행동:
  - [`artifacts/.../debug/traces/transcripts/*.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
  - [`artifacts/.../debug/traces/tool_traces/*.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
- accepted task 내용:
  - [`artifacts/.../bundle/tasks/*/task.yaml`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
  - [`artifacts/.../bundle/tasks/*/instance.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)
  - [`artifacts/.../bundle/tasks/*/canonical_answer.json`](/Users/jd/Documents/workspace/rl-data-harness/artifacts)

눈으로 데이터 생성 과정을 볼 때는 보통 `phase_monitors.jsonl`부터 보고, 이상한 draft가 있으면 대응하는 `synthesis/transcripts/*.json`와 solver transcript를 같이 본다.
