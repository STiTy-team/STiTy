#!/bin/bash
# judge27 — 1차/2차 문턱과 3벌 검증의 효과를 본다. v0 도 새로 생성한다.
#
# 이 런의 목적은 "좋은 개정을 찾는 것" 이 아니라 **관문 설계가 작동하는지 보는 것**이다.
#
# 왜 이렇게 바꾸는가 (judge24~26 실측)
#
#   기준선 한 벌이 모든 후보 Δ 를 같이 밀어 올린다
#     v0 를 dev 500 에서 네 번 뽑으니 0.5514 / 0.5578 / 0.5588 / 0.5591 이었다. judge24·25·26 이
#     판정에 쓴 첫 벌(run30 캐시)이 나머지 셋의 평균보다 0.0072 낮다. 후보는 전부 그 한 벌에
#     대고 재므로 **세 런의 후보 Δ 가 통째로 +0.007 올려 잡혔다.** judge25 이터 2 에서 편집이
#     서로 다른 후보 넷 중 셋이 동시에 하한 > 0 이던 것, judge26 이터 2 에서 여덟이 전부 양수던
#     것이 그 자국이다. → `--baseline-draws 3`
#
#   한 벌로 고른 1등은 다시 재면 사라진다
#     judge26 이터 2 의 1차 +0.0143 [+0.0029] 가 새 벌에서 −0.0019, 이터 3 의 +0.0182 [+0.0087]
#     이 +0.0049 였다. judge25 는 같은 자리에서 채택했고 test 560 에서 −0.0070 이었다.
#     → 1차는 선별로 쓰고 판정은 다벌 평균이 한다. `--confirm-draws 3`
#
#   정밀도 계산
#     한 벌 채점의 sd 는 dev 500 에서 약 0.0041. 양쪽 1벌이면 Δ 의 sd 가 0.0058 이라 하한 > 0 은
#     Δ +0.011 을 요구한다 — 지금 보이는 진짜 효과(+0.005~+0.007)보다 크다. 양쪽 3벌이면
#     sd 0.0033, 문턱 +0.0065 로 내려와 비로소 가릴 수 있는 크기가 된다.
#
#   1차 문턱을 평균 > 0 으로 내린다
#     기준선을 제대로 뽑으면 후보 Δ 가 0.007 내려앉는다. 세 런 후보 66개를 그 보정으로 다시 세면
#     하한 > 0 은 1개(1.5%), 평균 > 0 은 13개(20%)다. 1차를 하한으로 두면 2차가 한 번도 안 돈다.
#     → `--gate-rule mean`, 상위 개수 제한 없이 통과자 **전부**를 2차로 올린다.
#
#   역할 3 × finding 3 = 후보 9, induce 제외
#     judge26 이 후보를 가르는 것은 역할이 아니라 **Critic 이 무엇을 지목했는지**임을 보여줬다 —
#     같은 finding 안에서는 네 역할이 한 덩어리로 움직였고(이터 2 finding 1: +0.0128 / +0.0143 /
#     +0.0134 / +0.0099), finding 이 다르면 갈렸다(이터 3 finding 0: +0.0182 / +0.0049 / +0.0034
#     / +0.0081). 그래서 finding 폭을 넓힌다. `induce` 는 judge24 에서 gap 낮은 묶음이 −0.0126 ~
#     −0.0146 이었고 judge26 에서도 중간이라 뺀다. `prune` 은 남긴다 — 규칙이 쌓이면 짧은 지연과
#     긴 지연의 이득이 충돌하고, 버릴 쪽을 고를 수 있는 역할이 그것뿐이다.
#
#   v0 를 새로 생성한다
#     judge21~26 은 v0 한 개에 고정돼 있었다. 지금까지의 역할·finding 결론이 **그 v0 한 개의
#     성질인지** 가릴 수 없다. 통제 비교는 끊기지만 그 질문에 답한다.
#
#   최종 test 도 3벌
#     판정을 다벌로 해 놓고 최종 숫자만 한 벌로 재면 결론이 다시 ±0.011 에 묻힌다.
#
# 비용: 1차 9후보 $15.8/이터 + 2차 통과자당 $3.5 + 기준선 3벌 $5.3 + v0 생성 $6 + 최종 3벌 $12.
# 통과자 2~4개면 총 $110~130, 9개 전부 통과하는 최악이면 $180 이다. 예산은 중간에 끊기지
# 않도록 넉넉히 잡되, $150 을 넘어서면 통과자 수를 보고 원인을 가린다.
#
#   tmux new-session -d -s judge27 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge27/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge27.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run30 --run-id en2x/en-multi/judge27 \
    --generate-v0 --v0-candidates 2 \
    --candidate-roles fallback,single_small,prune --candidates-cross \
    --candidates-cap 9 --findings-max 3 --full-score-max 9 \
    --screen-n 0 \
    --baseline-draws 3 --confirm-draws 3 --final-draws 3 \
    --gate-rule mean --adopt-rule lo --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 100 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations "${ITER:-3}" \
    --workers 720 --score-workers 8 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-200}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge27 exit=$rc" >> "$LOG"
