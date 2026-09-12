#!/bin/bash
# run21 — 관문·분량 수정 넷의 효과. run20 대비 이것들만 다르다.
#
#   1) 분량 목표를 한계의 80% 로 (SIZE_TARGET_RATIO). 38건 실측에서 출력이 한계 대비
#      중앙값 +11% 라 목표를 붙여 주면 지켜지지 않았다. 착지율 29% -> 95% 예상.
#   2) 표본분산(n-1) + --rule-min-matches 8 -> 5. 모분산은 작은 표본에서 t 를 폭발시켜
#      (n=1 에 t=-24.17) 문턱을 높게 잡을 수밖에 없었고, 그 문턱이 진짜 규칙 둘
#      (n=5 t=-6.94, n=6 t=-2.80)을 재보지도 않고 죽였다. 재료 0인 이터 3개 -> 1개.
#   3) supported_by 를 세지 말고 검증. run20 규칙 11개 중 8개가 인용을 부풀렸고 4개는
#      인용한 사례에 그 패턴이 아예 없었다.
#   4) 통과 규칙 0개면 개정 생략. 근거 없이 만든 후보가 -0.043 ~ -0.095 로 무너졌다.
#
# **v0·분할·라벨·캐시는 run20 과 동일하다.** iter 0 은 캐시 적중으로 비용 0 이고 수치가
# run18~20 과 글자 그대로 같아야 한다 (train 0.3715 / dev 0.4276).
#
# 볼 것:
#   - 분량 탈락이 run20 의 3건에서 줄었나 (후보가 이터당 3개로 유지되나)
#   - `규칙 관문 N/M` 에서 재료 0인 이터가 줄었나
#   - 채택이 1회를 넘나. run20 은 test 에서 v0 대비 +0.041 (2.1 se) 였다
set -u

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1

set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.

if [ -x .venv-autoseg/bin/python ]; then PY=.venv-autoseg/bin/python; else PY=.venv/bin/python; fi

S="core/meaning_segmentator/tools/autoseg_en2x/run21"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run21.log"
mkdir -p "$(dirname "$LOG")"

echo "== $(date '+%F %T') run21 start (PY=$PY)" >> "$LOG"

$PY -u -m core.meaning_segmentator.autoseg.loop_distill \
    --dataset fleurs-en-multi-x \
    --pair-id en2x/en-multi --run-id run21 \
    --split-from en2x/en-multi/run16 \
    --labels-from en2x/en-multi/run20 \
    --train 100 --train-pool 200 \
    --n-cases 24 \
    --rule-min-matches 5 --rule-min-t 2.0 --rule-min-support 2 \
    --revision-rounds 1 --pool-size 2 \
    --provider openai --model gpt-5-mini \
    --iterations 5 --budget 12 >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') loop_distill exit=$rc" >> "$LOG"
[ $rc -eq 0 ] && touch "$S/loop.done" || touch "$S/loop.failed"

echo "== $(date '+%F %T') cost_report" >> "$LOG"
$PY -u -m core.meaning_segmentator.autoseg.infra.cost_report \
    --run-id en2x/en-multi/run21 --budget 12 >> "$LOG" 2>&1

touch "$S/chain.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
