#!/bin/bash
# judge31 — 통로가 열린 골격 위에서 루프를 돌린다.
#
# 무엇이 바뀐 런인가
#
#   v0 = combo (고정 골격, 편집 대상 아님)
#     judge27 의 v0 는 `[Scoring Rules]` 가 선언형이었다 — "결과가 순위여야 한다" 고만 하고
#     순위를 만드는 절차를 주지 않는다. 그래서 모델이 실제로 판단하는 것은 앞쪽 몇 자리뿐이고,
#     `[Core Principles]` 에 규칙을 더해도 그 몇 자리에만 닿았다(부검 깊이 변화가 8·13위에서
#     전부 ±0.01). 손실의 절반 이상이 ≤3 구간이고 그 구간은 깊은 순위를 쓴다.
#     combo 는 절차를 두 겹으로 준다 — 다섯 등급으로 굵게 가른 뒤 **각 등급 안에서 하나씩**
#     고르게 하고, 맨 아래 등급도 서열을 매기라고 못 박는다. dev 500·3벌에서 v0 대비
#     Δ +0.0064 [+0.0005, +0.0121] 이고 다섯 구간 전부 양수였다. 등급 단독(+0.0026)과 절차문
#     단독(+0.0037)의 합(+0.0063)과 거의 같아 **두 기제가 가산적**이다.
#     이 런에서 그것은 성과가 아니라 **전달 경로**다 — 같은 종류의 편집이 이제 깊은 자리까지
#     닿을 수 있는지를 본다. `[Scoring Rules]` 는 편집 대상에서 그대로 제외된다.
#
#   관문: 기준선 3벌 + 1차 평균 > 0 선별 + 2차 3벌 하한 > 0
#     한 벌 채점의 sd 가 dev 500 에서 약 0.004 다. 양쪽 1벌이면 Δ 의 반폭이 0.011 이라 하한 > 0
#     이 Δ +0.011 을 요구하는데, 단일 편집의 실효는 그보다 작다. 양쪽 3벌이면 반폭이 0.0053 이다.
#     그리고 기준선 한 벌은 후보 전부가 공유하므로 그 벌의 오차가 모든 후보 Δ 에 같은 방향으로
#     얹힌다 — v0 를 여섯 번 뽑아 본 실측에서 판정에 쓰던 한 벌이 나머지 평균보다 0.007 낮았고,
#     그 편향이 "후보 대부분이 양수" 라는 그림을 만들었다. `--baseline-draws 3` 이 그것을 지운다.
#     1차 문턱을 평균 > 0 으로 내리는 이유: 기준선을 제대로 뽑으면 하한 > 0 은 거의 안 나와
#     2차가 한 번도 안 돈다. 1차는 선별, 2차가 판정이다.
#
#   역할 3 × finding 최대 3, 형태가 맞는 짝만
#     역할은 judge27 과 같게 둔다(`fallback`·`single_small`·`prune`) — 1차 지표가 judge27 대비
#     깊이 변화라서 역할을 바꾸면 비교가 흐려진다.
#     `--candidates-cross` 가 이제 발견의 형태(`kind`)로 짝을 가린다: 이항 발견(`order`)은
#     선호문 역할로, 단항 발견(`check`)은 결속문 역할로만 간다. `prune` 은 발견을 구현하지
#     않으므로 이터당 하나다. 그래서 후보는 이터당 최대 4개다(3 + prune).
#
# 무엇으로 성공을 판정하나 — 관문 통과보다 앞에 놓는 것
#
#   1차  부검 깊이 배수 변화 8·13위가 ±0.01 을 벗어나는가. judge27 은 전부 그 안이었다.
#        Δ 는 우연히 커질 수 있지만 편집이 깊은 순위를 흔들기 시작하면 그것은 통로가 열린 것이다.
#   2차  `order` 종류 후보의 Δ 가 `check` 종류보다 큰가 (로그에 후보별 역할·finding 이 남는다).
#   3차  2차 3벌 확인을 통과한 이터가 하나라도 있는가.
#
# 비용: 이터당 후보 4개 $7 + 2차 통과자당 $3.5 + 기준선 3벌 $5 + 최종 test 3벌 두 프롬프트 $12.
# 3이터에 $60~70 을 본다. 통과자가 없는 이터는 2차가 0회 채점으로 지나간다.
#
#   tmux new-session -d -s judge31 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge31/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V0=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge27/prompt_v0_combo.txt
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge31.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge31 \
    --prompt "$V0" \
    --candidate-roles fallback,single_small,prune --candidates-cross \
    --candidates-cap 9 --findings-max 3 --full-score-max 9 \
    --screen-n 0 \
    --baseline-draws 3 --confirm-draws 3 --final-draws 3 \
    --gate-rule mean --adopt-rule lo --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations "${ITER:-3}" \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-100}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge31 exit=$rc" >> "$LOG"
