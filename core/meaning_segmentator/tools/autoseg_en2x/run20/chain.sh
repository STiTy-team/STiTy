#!/bin/bash
# run20 — Critic 출력 계약을 "사례별 진단"에서 "규칙"으로 바꾼 효과만 본다.
#
# **run19 와 다른 것은 이것 하나다** (커밋 dcf6e465). 계층화·라벨 관문·토큰 상한은
# run19 에도 이미 들어가 있었다.
#
#   출력 컨테이너가 `cases` -> `rules`. 항목이 문장 하나(`id`+`pos`)가 아니라 규칙 하나.
#   `supported_by` 로 뒷받침 사례를 대게 하고, 2개 미만이면 라벨을 재기 전에 떨어진다
#   (--rule-min-support 2, 기본값).
#
# run19 는 규칙 13개 중 9개가 train 1400 경계 중 0~2자리에만 걸려 탈락했고, 그 9개는
# 전부 한 문장에서 나온 조건이었다. PE 재료가 이터당 1~2개로 말라 dev Δ 가 +0.0002 /
# +0.0111 로 문턱을 못 넘고 채택 0회로 끝났다.
#
# **v0·분할·라벨·캐시는 run19 와 동일하다.** iter 0 은 캐시 적중으로 비용 0 이고 수치가
# run19·run18 과 글자 그대로 같아야 한다.
#
# 볼 것: `규칙 관문 N/M 통과` 의 M 과 통과율. run19 는 4/13 이었다. M 이 줄되 통과율이
# 올라야 PE 재료가 확보된다.
set -u

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1

set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.

if [ -x .venv-autoseg/bin/python ]; then PY=.venv-autoseg/bin/python; else PY=.venv/bin/python; fi

S="core/meaning_segmentator/tools/autoseg_en2x/run20"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run20.log"
mkdir -p "$(dirname "$LOG")"

echo "== $(date '+%F %T') run20 start (PY=$PY)" >> "$LOG"

$PY -u -m core.meaning_segmentator.autoseg.loop_distill \
    --dataset fleurs-en-multi-x \
    --pair-id en2x/en-multi --run-id run20 \
    --split-from en2x/en-multi/run16 \
    --labels-from en2x/en-multi/run19 \
    --train 100 --train-pool 200 \
    --n-cases 24 \
    --rule-min-matches 8 --rule-min-t 2.0 \
    --revision-rounds 1 --pool-size 2 \
    --provider openai --model gpt-5-mini \
    --iterations 5 --budget 12 >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') loop_distill exit=$rc" >> "$LOG"
[ $rc -eq 0 ] && touch "$S/loop.done" || touch "$S/loop.failed"

echo "== $(date '+%F %T') cost_report" >> "$LOG"
$PY -u -m core.meaning_segmentator.autoseg.infra.cost_report \
    --run-id en2x/en-multi/run20 --budget 12 >> "$LOG" 2>&1

touch "$S/chain.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
