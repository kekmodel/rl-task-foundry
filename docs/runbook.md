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
- proof environment 이전에 production environment registry를 운영 시작하는 것

## Allowed Actions

- spec / plan / core contract 작업
- synthetic proof environment 구축
- registration policy / runtime isolation 구현
- review pack 생성 및 정성 평가
- infra regression test 유지

## Exit Criteria

freeze는 아래가 모두 충족될 때만 해제 후보가 된다.

- proof environment vertical slice 완료
- hybrid verifier A/B/C/D mandatory gate 구현
- cross-instance consistency와 solver pass-rate quality filter 구현
- rewrite review pack이 success criteria를 반복적으로 충족

## Operator Checklist

rewrite 진행 중에는 아래를 확인한다.

- baseline regression이 깨지지 않았는가
- 새 artifact가 training source로 오인되어 누적되지 않았는가
- review pack은 qualitative spot-check 용도로만 쓰이고 있는가
- production acceptance / registry commit은 milestone 달성 전까지 막혀 있는가
